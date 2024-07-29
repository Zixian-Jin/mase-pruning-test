import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset

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


device = 'cuda' if torch.cuda.is_available() else 'cpu' # TODO



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
        optimizer = AdamW(model.parameters(), lr=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_dataloader):
            out = model(input_ids=input_ids,
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
        
    def downstream_test(self):
        model.eval()
        correct = 0
        total = 0

        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(self.val_dataloader):
            if i == 5:
                break
            print(i)
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            out = out.argmax(dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        print(correct / total)


# model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).shape



if __name__ == '__main__':
    model = DownstreamModel()
    model.to(device)
    model.downstream_train()
    model.downstream_test()