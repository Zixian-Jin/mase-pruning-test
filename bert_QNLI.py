
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
import pandas as pd

import logging
from tqdm.auto import tqdm, trange
import math
from sklearn.metrics import classification_report

# from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    BertTokenizerFast, 
    BertForSequenceClassification, 
    BertTokenizer, 
    BertConfig
)


class BertQNLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def add_targets(encodings, label):
    encodings.update({'label': label})

class BertQNLI():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name = 'bert-base-cased'
        model_path = './../BERT-MODELS/bert-QNLI-pretrained.bin'
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', model_max_length=128)
        config = BertConfig.from_pretrained("./../BERT-MODELS/bert-QNLI-config.json", num_labels=2) 
        self.model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        self.model.to(self.device)

        self.init_dataset(train_samples=25000, val_samples=500)
        self.init_dataloader(train_bs=20, eval_bs=32)

    
    def init_dataset(self, train_samples, val_samples):
        ################### Temp Variables ##################
        datasets = load_dataset('glue', 'qnli')
        train_data = pd.DataFrame(datasets['train']).drop(columns = ['idx'])
        val_data = pd.DataFrame(datasets['validation']).drop(columns = ['idx'])
        test_data = pd.DataFrame(datasets['test']).drop(columns = ['idx'])

        # conversion from datafram to list for tokenization
        train_questions, train_sentences, train_labels = train_data['question'][:train_samples].tolist(), train_data['sentence'][:train_samples].tolist(), train_data['label'][:train_samples].tolist()
        eval_question, eval_sentences, eval_labels = val_data['question'][:val_samples].tolist(), val_data['sentence'][:val_samples].tolist(), val_data['label'][:val_samples].tolist()

        # tokenization
        train_encodings = self.tokenizer(train_questions, train_sentences, padding='max_length', truncation=True)
        val_encodings = self.tokenizer(eval_question, eval_sentences, padding='max_length', truncation=True)
        add_targets(train_encodings, train_labels)
        add_targets(val_encodings, eval_labels)
        ################### End of Temp Variables ###############
        
        self.train_dataset = BertQNLIDataset(train_encodings)
        self.val_dataset = BertQNLIDataset(val_encodings)


    def init_dataloader(self, train_bs=10, eval_bs=10):
        data_collator = default_data_collator
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_bs)
        self.val_dataloader = DataLoader(self.val_dataset, collate_fn=data_collator, batch_size=eval_bs)


    def train(self):
        raise NotImplementedError("This method is currently deprecated.")
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
        
    def eval(self):
        
        print("***** Running eval *****")
        self.model.eval()
        self.model.to(self.device)
        labels = []
        predictions = []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_dataloader, desc="Eval Iteration")):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                predicted = outputs.logits.argmax(dim=-1)

                labels += batch["labels"].tolist()
                predictions += predicted.tolist()
                
        correct = (torch.Tensor(labels) == torch.Tensor(predictions)).sum().item()
        total = len(labels)
        acc = 100 * correct / total    
        return acc
    

# def main():

#     accuracy_metric = load_metric("accuracy")
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#     print(accuracy)

#     for line in classification_report(labels, predictions).split('\n'):
#         print(line)
