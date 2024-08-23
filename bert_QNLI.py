import os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
import pandas as pd
import copy
import logging
from tqdm.auto import tqdm, trange
from sklearn.metrics import classification_report

from utils import *
from registered_pruning import update_module_parametrization
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

        self.init_dataset(train_samples=25000, val_samples=512)
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
    





class BertQNLIPrunerProgram(BertQNLI):
    '''
        This class is an extended class of <BertQNLI> so that 
        it is capable of bert-specific pruning.
    '''
    
    
    # NOTE: this constant map is not expected to be amended anywhere.
    BERT_LAYER_TUNABLE_MATMUL_MAP = {
        'Q': ['attention', 'self', 'query'],
        'K': ['attention', 'self', 'key'],
        'V': ['attention', 'self', 'value'],
        'W0': ['attention', 'output', 'dense'],
        'W1': ['intermediate', 'dense'],
        'W2': ['output', 'dense'],
    }
    
    
    def __init__(self):
        self.last_bert_qnli_prune_cfg = {}
        self.bert_qnli_prune_cfg =  {}
        # NOTE: naming style is consistent with mase-dse: <Block>
        # prune_cfg is expected to be of the following format:
        # {
        #     '0': {'Q': None, 'K': None, 'V': None, 'W0': None, 'W1': None, 'W2': None},
        #     ## '1', '2', ... , '11'
        #     'pooler': {'Linear': None},
        #     'classifier': {'Linear': None}
        # }

        super().__init__()
        self.init_bert_qnli_configs()


    def init_bert_qnli_configs(self) -> None:
        '''
            This method only init self.bert_qnli_prune_cfg.
            self.last_bert_qnli_prune_cfg still remains as an empty dict,
            and will be initiated at the first time pruner is called.
        '''
        empty_sparse_cfg = {'block_num': 2, 'sparsity': 0}
        # BERT layer 0 - 11
        for layer in range(0, 12):
            self.bert_qnli_prune_cfg[str(layer)] = {}
            for module in ['Q', 'K', 'V', 'W0', 'W1', 'W2']:
                self.bert_qnli_prune_cfg[str(layer)][module] = empty_sparse_cfg
        # BERT pooler
        self.bert_qnli_prune_cfg['pooler'] = {'Linear': empty_sparse_cfg}
        # Downstream QNLI classfier
        self.bert_qnli_prune_cfg['classifier'] = {'Linear': empty_sparse_cfg}



    def get_bert_qnli_tunable_module(self, layer_name, matmul_name) -> nn.Module:
        '''
            returns an object for the module to be searched
            NOTE: returns <nn.Module>, not <nn.Module.weight>!!!
        '''
        root_obj = None
        attr_path = []
        
        # Step 1: find root_obj & attr_path
        if layer_name.isdigit():
            # '0', '1', ..., '11'
            root_obj = self.model.bert.encoder.layer._modules[layer_name]
            attr_path = self.BERT_LAYER_TUNABLE_MATMUL_MAP[matmul_name]
        elif layer_name == 'pooler':
            root_obj = self.model.bert.pooler.dense
            attr_path = []
        elif layer_name == 'classifier':
            root_obj = self.model.classifier
            attr_path = []
        else:
            raise NotImplementedError("Unrecognised layer.")

        # Step 2: get the attr
        module = get_nested_attr(root_obj, attr_path)
        return module
        
        
        
    def bert_qnli_prune(self, mask_root_dir='../BERT-QNLI-masks') -> None:
        '''
            mask_root_dir: pre-calculated masks, if applicable, can be retrieved from here
        '''
        if self.last_bert_qnli_prune_cfg == {}:
            self.last_bert_qnli_prune_cfg = copy.deepcopy(self.bert_qnli_prune_cfg)
            
        for layer, module_dict in self.bert_qnli_prune_cfg.items():
            for name, cfg in module_dict.items():
                if self.last_bert_qnli_prune_cfg[layer][name]== self.bert_qnli_prune_cfg[layer][name]:
                    pass
                else:
                    module = self.get_bert_qnli_tunable_module(str(layer), name)
                    
                    # find pre-calculated mask for current module & sparsity cfg
                    mask_tag = f'layer_{layer}_module_{name}_weight_bn_{cfg['block_num']}_sp_{int(cfg['sparsity']*100)}.pt'
                    local_mask_path = os.path.join(mask_root_dir, mask_tag)
                    if not os.path.exists(local_mask_path):
                        if not (cfg['sparsity'] == 0.0 or cfg['sparsity'] == 1.0):
                            # The "all zeros" and "all ones" mask are not designed to be pre-calculated.
                            print(f'WARNING: the mask {local_mask_path} does not exists.')
                        
                    update_module_parametrization(module, 'weight', cfg, local_mask_path)

        # update last_bert_prune_config
        self.last_bert_prune_config = copy.deepcopy(self.bert_qnli_prune_cfg)

        





        



# def main():

#     accuracy_metric = load_metric("accuracy")
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#     print(accuracy)

#     for line in classification_report(labels, predictions).split('\n'):
#         print(line)
