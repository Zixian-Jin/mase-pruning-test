import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset

import utils
import rank_functions

class BertDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label


    
# print(len(loader))
# input_ids.shape, attention_mask.shape, token_type_ids.shape, labels
#模型试算
# out = pretrained(input_ids=input_ids,
#            attention_mask=attention_mask,
#            token_type_ids=token_type_ids)
# out.last_hidden_state.shape






class DownstreamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # print(len(dataset), dataset[0])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pretrained = BertModel.from_pretrained('bert-base-chinese')
        self.pretrained.to(device) # TODO
            # pretrained = BertModel.from_pretrained('tmp')  # local checkpoint
        # inactivate grad for Bert params
        for param in self.pretrained.parameters():
            param.requires_grad_(False)

        self.init_dataset()
        self.init_dataloader()
        self.fc1 = torch.nn.Linear(768, 192)
        self.fc2 = torch.nn.Linear(192, 2)
    
    def init_dataset(self):
        self.train_dataset = BertDataset('train')
        self.val_dataset = BertDataset('test')

    def init_dataloader(self):
        def collate_fn(data):
            tokenizer = self.tokenizer
            sents = [i[0] for i in data]
            labels = [i[1] for i in data]

            # encoding
            data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=500,
                                        return_tensors='pt',
                                        return_length=True)
            #input_ids: ids after encoding
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = torch.LongTensor(labels).to(device)
            #print(data['length'], data['length'].max())

            return input_ids, attention_mask, token_type_ids, labels

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                            batch_size=16,
                                            collate_fn=collate_fn,
                                            shuffle=True,
                                            drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                                batch_size=32,
                                                collate_fn=collate_fn,
                                                shuffle=True,
                                                drop_last=True)
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(self.train_dataloader):
            break
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = self.fc1(out.last_hidden_state[:, 0])
        out = self.fc2(out)
        out = out.softmax(dim=1)
        return out

    def downstream_train(self):
        optimizer = AdamW(self.parameters(), lr=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        self.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_dataloader):
            out = self(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 5 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                print(i, loss.item(), accuracy)
            if i == 300:
                break
        self.save_downstream_model()
        
    def downstream_test(self):
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids,
                    labels) in enumerate(self.val_dataloader):
                if i == 50: break
                out = self(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
                out = out.argmax(dim=1)
                correct += (out == labels).sum().item()
                total += len(labels)
        acc = 100 * correct / total
        return acc

    def save_downstream_model(self, p='./ckpts/downstream_model_unpruned.pth'):
        state_dict = self.state_dict()
        selected_params = {k: v for k, v in state_dict.items() if ('fc1' in k or 'fc2' in k)}
        torch.save(selected_params, p)
        print('INFO: saved downstream model to %s'%p)

    def load_downstream_model(self, p='./ckpts/downstream_model_unpruned.pth'):
        downstream_params = torch.load(p)
        model.load_state_dict(downstream_params, strict=False)
        print('INFO: loaded downstream model from %s'%p)

    def check_sparsity(self, module):
        w = module.weight.data
        # print(w)
        thres = 0.05
        max_val = w.abs().max().item()
        for thres in np.linspace(0.00, max_val, 10):
            mask = torch.where(torch.abs(w)<thres, 1, 0)
            print("Sparsity of current module with thres=%f = %f"%(thres, torch.sum(mask)/(w.shape[0]*w.shape[1])))
        
        values, indicies = utils.matrix_profiler(w, rank=0.1, scope='local')
        print('Top 10% elements in the analysed matrix:')
        print('Values=', values)
        print('Indicies=', indicies)
        
    def simple_prune(self, module, thres):
        print('INFO: Pruning...')
        # print('Weight before pruning:')
        # print(module.weight.data)
        mask = (torch.abs(module.weight.data) >= thres)
        module.weight.data *= mask.float()
        # print('Weight after pruning:')
        # print(module.weight.data)
        print('INFO: Finished pruning.')

    def structured_prune(self, module, silent=True):
        print('INFO: Pruning...')
        # module = getattr(self, prune_config['module'])
        data = module.weight.data
        sparsity = prune_config['sparsity']

        mask = rank_functions.block_rank_fn_local(data, prune_config, sparsity, silent=silent)
        mask = mask.to(device)
        module.weight.data *= mask
        print('INFO: Finished pruning.')

# model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).shape



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # TODO
    prune_config = {
            "module": None,
            "scope": "local",
            "block_num": 64,
            "sparsity": 0.5
    }
    model = DownstreamModel()
    model.to(device)
    prune_config['module'] = "fc1"
    
    # model.downstream_train()
    
    model.load_downstream_model()
    model.check_sparsity(module=model.fc1)
    acc_1 = model.downstream_test()
    
    for i in [0.7, 0.8, 0.9, 0.92, 0.95, 0.98, 1.00]:
    # for i in [0.6, 0.7]:
        print('========== Prune after training ===========')
        model.load_downstream_model()
        print("Sparsity=%f"%i)
        prune_config = {
            "module": 'fc1',
            "scope": "local",
            "block_num": 16,
            "sparsity": i
        }
        model.structured_prune(module=model.fc1)
        acc_2 = model.downstream_test()
        print(f"Before pruning: acc={acc_1}. After pruning: acc={acc_2}.")

