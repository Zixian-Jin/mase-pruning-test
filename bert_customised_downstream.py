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
        self.fc1.to(device)
        self.fc2.to(device)
    
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
                if i == 30: break
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


    def bert_attention_prune(self, layer_list, weight_list):
        '''
            Ex. layer_list = [0, 1, 3]
            weight_list = ['Q', 'K']
        '''
        modules_to_prune = []
        
        for layer in layer_list:
            for weight in weight_list:
                if weight == 'Q':
                    modules_to_prune.append(self.pretrained.encoder.layer._modules[str(layer)].attention.self.query)
                elif weight == 'K':
                    modules_to_prune.append(self.pretrained.encoder.layer._modules[str(layer)].attention.self.key)
                elif weight == 'V':
                    modules_to_prune.append(self.pretrained.encoder.layer._modules[str(layer)].attention.self.value)
                elif weight == 'W1':
                    modules_to_prune.append(self.fc1)
                elif weight == 'W2':
                    modules_to_prune.append(self.fc2)
                else:
                    print('WARNING: Invalid pruned weight has been ignored.')
        
        for module in modules_to_prune:
            self.structured_prune(module)

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
    
    model.load_downstream_model()
    # model.check_sparsity(module=model.fc1)
    
    
    base_sparsity = 0.2
    outstanding_sparsity = 0.4
    outstanding_layer = 1
    weights_to_prune = ['Q', 'K', 'V']
    
    print('Pruing Configuraitons:')
    print('Block_num=', prune_config['block_num'])
    # print('Layers to prune=', layers_to_prune)
    print('Weights to prune=', weights_to_prune)
    
    
    for outstanding_layer in range(12):
        print('========== Prune after training ===========')
        model.__init__()
        model.load_downstream_model()
        model.to(device)
        
        # Step 1: prune all layers with base sparsity. This is done outside the inner loop for saving time.
        if base_sparsity > 0:
            prune_config = {
                "module": 'fc1',
                "scope": "local",
                "block_num": 64,
                "sparsity": base_sparsity
            }
            model.bert_attention_prune(list(range(12)), weights_to_prune)
            acc_1 = model.downstream_test()
                    
        for outstanding_sparsity in [0.4, 0.5, 0.6, 0.7, 0.8]:
            print("Base Sparsity=%f, Outstanding Sparsity=%f"%(base_sparsity, outstanding_sparsity))
            # Step 2: prune the outstanding layer with outstanding sparsity
            prune_config = {
                "module": 'fc1',
                "scope": "local",
                "block_num": 64,
                "sparsity": outstanding_sparsity
            }        
            model.bert_attention_prune([outstanding_layer], weights_to_prune)
            
            acc_2 = model.downstream_test()
            print(f"Report: Outstanding layer={outstanding_layer}. After base pruning: acc={acc_1}. After outstanding pruning: acc={acc_2}.")



# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu' # TODO
#     prune_config = {
#             "module": None,
#             "scope": "local",
#             "block_num": 64,
#             "sparsity": 0.5
#     }
#     model = DownstreamModel()
#     model.to(device)
#     # prune_config['module'] = "fc1"
#     module = model.pretrained.encoder.layer._modules['0'].attention.self.query.weight.detach()
#     # model.downstream_train()
    
#     model.load_downstream_model()
#     # model.check_sparsity(module=model.fc1)
#     acc_1 = model.downstream_test()
    
#     layers_to_prune = [2, 4, 6, 8, 10]  # list(range(12))
#     weights_to_prune = ['Q', 'K', 'V']
    
#     print('Pruing Configuraitons:')
#     print('Block_num=', prune_config['block_num'])
#     print('Layers to prune=', layers_to_prune)
#     print('Weights to prune=', weights_to_prune)
#     for i in [0.5, 0.6, 0.7, 0.8]: # 0.9, 0.92, 0.95, 0.98, 1.00]:
#     # for i in [0.9, 0.95, 0.98, 1.00]:
#         print('========== Prune after training ===========')
#         model.load_downstream_model()
#         print("Sparsity=%f"%i)
#         prune_config = {
#             "module": 'fc1',
#             "scope": "local",
#             "block_num": 64,
#             "sparsity": i
#         }
#         # model.structured_prune(module=model.fc1)
#         model.bert_attention_prune(layers_to_prune, weights_to_prune)
#         acc_2 = model.downstream_test()
#         print(f"Before pruning: acc={acc_1}. After pruning: acc={acc_2}.")

