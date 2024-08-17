from bert_QNLI import BertQNLI

from rank_functions import block_rank_fn_local
from registered_pruning import update_module_parametrization
from utils import *
from torch import nn

def get_nested_attr(root_obj, attr_path: str):
    '''
        e.g., root_obj = model
              attr_path = ['fc1', 'weight']
    '''
    attr_full_name = '.'.join(attr_path)
    obj = root_obj
    for attr in attr_path:
        obj = getattr(obj, attr)
        assert (obj != None), f"the leaf attr {attr_full_name} to be accessed does not exist!"
    return obj



def find_bert_tunable_module(model: nn.Module, layer_name, matmul_name) -> nn.Module:
    '''
        returns an object for the module to be searched
        NOTE: returns <nn.Module>, not <nn.Module.weight>!!!
    '''
    root_obj = None
    attr_path = []
    
    # Step 1: find root_obj & attr_path
    if layer_name.isdigit():
        # '0', '1', ..., '11'
        root_obj = model.bert.encoder.layer._modules[layer_name]
        attr_path = BERT_LAYER_TUNABLE_MATMUL_MAP[matmul_name]
    elif layer_name == 'pooler':
        root_obj = model.bert.pooler.dense
        attr_path = []
    elif layer_name == 'classifier':
        root_obj = model.classifier
        attr_path = []
    else:
        raise NotImplementedError("Unrecognised layer.")

    # Step 2: get the attr
    module = get_nested_attr(root_obj, attr_path)
    return module
    
    
    
BERT_LAYER_TUNABLE_MATMUL_MAP = {
    'Q': ['attention', 'self', 'query'],
    'K': ['attention', 'self', 'key'],
    'V': ['attention', 'self', 'value'],
    'W0': ['attention', 'output', 'dense'],
    'W1': ['intermediate', 'dense'],
    'W2': ['output', 'dense'],
}

# NOTE: naming style is consistent with mase-dse: <Block>
BERT_PRUNE_CONFIGS = {
    '0': {'Q': None, 'K': None, 'V': None, 'W0': None, 'W1': None, 'W2': None},
    ## '1', '2', ... , '11'
    'pooler': {'Linear': None}
}

BERT_QNLI_PRUNE_CONFIGS = {'classifier': {'Linear': None}}


def init_bert_configs():
    empty_sparse_cfg = {'block_num': 2, 'sparsity': 0}
    # BERT layer 0 - 11
    for layer in range(0, 12):
        BERT_PRUNE_CONFIGS[str(layer)] = {}
        for module in ['Q', 'K', 'V', 'W0', 'W1', 'W2']:
            BERT_PRUNE_CONFIGS[str(layer)][module] = empty_sparse_cfg
    # BERT pooler
    BERT_PRUNE_CONFIGS['pooler']['Linear'] = empty_sparse_cfg
    # Downstream QNLI classfier
    BERT_QNLI_PRUNE_CONFIGS['classifier']['Linear'] = empty_sparse_cfg
    BERT_QNLI_PRUNE_CONFIGS.update(BERT_PRUNE_CONFIGS)

def bert_prune(model: nn.Module):
    init_bert_configs()
    print(BERT_QNLI_PRUNE_CONFIGS)
    layer = 0
    matmul = 'Q'
    new_cfg = {'block_num': 16, 'sparsity': 0.9}
    BERT_QNLI_PRUNE_CONFIGS[str(layer)]['Q'] = new_cfg
    module = find_bert_tunable_module(model, str(layer), 'Q')
    update_module_parametrization(module, 'weight', new_cfg)

def pruning_sensitivity_test(toy):
    
    
    prune_config = {
            "module": toy.model.fc1,
            "scope": "local",
            "block_num": 64,
            "sparsity": 0.5        
    }

    toy.load_model('./ckpts/mnist_cnn_model_unpruned.pth')
    acc_1, loss_1 = toy.eval()
    
    base_cfg = {'block_num': 10, 'sparsity': 0.0}
    outstanding_cfg = {'block_num': 10, 'sparsity': 0.7}
    
    tunable_layer_list = ['fc2', 'fc1']
    for layer in tunable_layer_list:
        for base_layer in tunable_layer_list:
            module = getattr(toy.model, base_layer)
            if base_layer != layer:
                update_module_parametrization(module, 'weight', base_cfg)
            else:
                update_module_parametrization(module, 'weight', outstanding_cfg)
        acc_2, loss_2 = toy.eval()
        print(f'Layer pruned = {layer}, sparsity = {outstanding_cfg['sparsity']}')
        print(f"Before pruning: acc={acc_1}, loss={loss_1}. After pruning: acc={acc_2}, loss={loss_2}")


def main():
    program = BertQNLI()
    acc_1 = program.eval()
    bert_prune(program.model)
    acc_2 = program.eval()
    print(f'Before pruning: acc={acc_1}. After pruning: acc={acc_2}.')
    
    
if __name__ == '__main__':
    main()