
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def add_targets(encodings, label):
    encodings.update({'label': label})



def main():

    # Data Preprocessing
    datasets = load_dataset('glue', 'qnli')

    train_data = pd.DataFrame(datasets['train']).drop(columns = ['idx'])
    validation_data = pd.DataFrame(datasets['validation']).drop(columns = ['idx'])
    test_data = pd.DataFrame(datasets['test']).drop(columns = ['idx'])

    train_questions, train_sentences, train_labels = train_data['question'][0:25000].tolist(), train_data['sentence'][0:25000].tolist(), train_data['label'][0:25000].tolist()
    eval_question, eval_sentences, eval_labels = validation_data['question'][0:3000].tolist(), validation_data['sentence'][0:3000].tolist(), validation_data['label'][0:3000].tolist()
    # 將dataframe格式轉成list(tokenize時使用)

    # Tokenizer Configurations
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', model_max_length=128)

    train_encodings = tokenizer(train_questions, train_sentences, padding='max_length', truncation=True)
    eval_encodings = tokenizer(eval_question, eval_sentences, padding='max_length', truncation=True)

    add_targets(train_encodings, train_labels)
    add_targets(eval_encodings, eval_labels)







    model_name = 'bert-base-cased'
    model_path = './../BERT-MODELS/bert-QNLI-pretrained.bin'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained("./../BERT-MODELS/bert-QNLI-config.json", num_labels=2) 
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    # model = BertForSequenceClassification.from_pretrained(
    #                                     model_name, 
    #                                     state_dict=torch.load(model_path, map_location=torch.device('cpu')), 
    #                                     num_labels=2)


    train_batch_size, eval_batch_size = 10, 10

    data_collator = default_data_collator

    train_dataset = Dataset(train_encodings)
    eval_dataset = Dataset(eval_encodings)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)


    print("***** Running eval *****")
    model.eval()

    labels = []
    predictions = []

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Eval Iteration")):
        outputs = model(**batch)
        predicted = outputs.logits.argmax(dim=-1)

        labels += batch["labels"].tolist()
        predictions += predicted.tolist()


    accuracy_metric = load_metric("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    print(accuracy)

    for line in classification_report(labels, predictions).split('\n'):
        print(line)

if __name__ == '__main__':
    main()