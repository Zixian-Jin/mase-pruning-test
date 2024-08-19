import os
import copy
from torch import nn

from bert_QNLI import BertQNLI
from rank_functions import block_rank_fn_local
from registered_pruning import update_module_parametrization
from utils import *


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

def bert_save_masks(output_root_dir='../BERT-QNLI-masks'):    
    program = BertQNLI()
    model = program.model
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
        
    init_bert_configs()
    prune_configs = copy.deepcopy(BERT_QNLI_PRUNE_CONFIGS)

    count = 0
    for layer, module_dict in prune_configs.items():
        for name, cfg in module_dict.items():
            module = find_bert_tunable_module(model, str(layer), name)
            param = module.weight
            for block_num in [16, 32, 64, 128, 256]:
                # for sparsity in [0.5, 0.6, 0.7, 0.8, 0.9]:
                for sparsity in [0.3, 0.4]:
                    sp_cfg = {'block_num': block_num, 'sparsity': sparsity}
                    new_mask = block_rank_fn_local(param.detach(), sp_cfg)
                    new_mask = new_mask.bool()
                    mask_tag = f'layer_{layer}_module_{name}_weight_bn_{block_num}_sp_{int(sparsity*100)}.pt'
                    save_path = os.path.join(output_root_dir, mask_tag)
                    torch.save(new_mask, save_path)
                    print(f'Tensor saved to {save_path}')
                    count += 1
    print(f'Successfully saved {count} mask tensors of torch.bool type.')


def bert_prune_unit(model: nn.Module, new_bert_prune_config, mask_root_dir='../BERT-QNLI-masks'):
    global g_last_bert_prune_config
    
    if g_last_bert_prune_config == {}:
        g_last_bert_prune_config = copy.deepcopy(new_bert_prune_config)
        
    for layer, module_dict in new_bert_prune_config.items():
        for name, cfg in module_dict.items():
            if g_last_bert_prune_config[layer][name]== new_bert_prune_config[layer][name]:
                pass
            else:
                module = find_bert_tunable_module(model, str(layer), name)
                
                # find pre-calculated mask for current module & sparsity cfg
                mask_tag = f'layer_{layer}_module_{name}_weight_bn_{cfg['block_num']}_sp_{int(cfg['sparsity']*100)}.pt'
                local_mask_path = os.path.join(mask_root_dir, mask_tag)
                if not os.path.exists(local_mask_path):
                    if not (cfg['sparsity'] == 0.0 or cfg['sparsity'] == 1.0):
                        # The "all zeros" and "all ones" mask are not designed to be pre-calculated.
                        print(f'WARNING: the mask {local_mask_path} does not exists.')
                    
                update_module_parametrization(module, 'weight', cfg, local_mask_path)
                
    g_last_bert_prune_config = copy.deepcopy(new_bert_prune_config)

def bert_prune_example(model: nn.Module):
    init_bert_configs()
    print(BERT_QNLI_PRUNE_CONFIGS)
    layer = 0
    matmul = 'Q'
    new_cfg = {'block_num': 16, 'sparsity': 0.9}
    BERT_QNLI_PRUNE_CONFIGS[str(layer)]['Q'] = new_cfg
    module = find_bert_tunable_module(model, str(layer), 'Q')
    update_module_parametrization(module, 'weight', new_cfg)


def bert_pruning_sensitivity_test_type1():
    '''
        A single-layer, all-module pruning test.
        For each trial (i.e., each sparsity cfg), prune 
        ALL six matmul moduels of a SINGLE layer.
        TODO: should pooler & classifier be pruned as well?
    '''
    init_bert_configs()
    print(BERT_QNLI_PRUNE_CONFIGS)

    program = BertQNLI()
    acc_1 = program.eval()

    base_cfg = {'block_num': 2, 'sparsity': 0.0}
    
    # factors of 768 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768]
    count = 0
    for block_num in [16, 32, 64, 128, 256]:
        for sparsity in [0.5, 0.6, 0.7, 0.8, 0.9]:
            outstanding_cfg = {'block_num': block_num, 'sparsity': sparsity}
            print(f'\n\n================= Trial #{count} ===================')
            print(f'oustanding_cfg = {outstanding_cfg}')
            count += 1
            last_layer = 11
            for layer in range(12):
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['Q'] = outstanding_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['K'] = outstanding_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['V'] = outstanding_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['W0'] = outstanding_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['W1'] = outstanding_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(layer)]['W2'] = outstanding_cfg 
                    
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['Q'] = base_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['K'] = base_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['V'] = base_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['W0'] = base_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['W1'] = base_cfg
                BERT_QNLI_PRUNE_CONFIGS[str(last_layer)]['W2'] = base_cfg
                
                bert_prune_unit(program.model, BERT_QNLI_PRUNE_CONFIGS)

                acc_2 = program.eval()
                print(f'Layer pruned = {layer}, config = {outstanding_cfg}')
                print(f'Before pruning: acc={acc_1}. After pruning: acc={acc_2}.')
                print()
                last_layer = layer
                
                

def bert_pruning_sensitivity_test_type2():
    '''
        A all-layer, single-module pruning test.
        For each trial (i.e., each sparsity cfg), prune 
        a SINGLE matmul moduel of ALL encoder layers
    '''
    init_bert_configs()
    print(BERT_QNLI_PRUNE_CONFIGS)

    program = BertQNLI()
    acc_1 = program.eval()

    dense_cfg = {'block_num': 2, 'sparsity': 0.0}
    
    # factors of 768 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768]
    count = 0
    for block_num in [16, 32, 64, 128, 256]:
        for sparsity in [0.5, 0.6, 0.7, 0.8, 0.9]:
            outstanding_cfg = {'block_num': block_num, 'sparsity': sparsity}
            print(f'\n\n================= Trial #{count} ===================')
            print(f'oustanding_cfg = {outstanding_cfg}')
            count += 1
            last_module = 'W2'
            for module in BERT_LAYER_TUNABLE_MATMUL_MAP.keys():
                # prune only one type of module for all encoder layers
                for layer in range(12):
                    BERT_QNLI_PRUNE_CONFIGS[str(layer)][module] = outstanding_cfg
                    BERT_QNLI_PRUNE_CONFIGS[str(layer)][last_module] = dense_cfg
                bert_prune_unit(program.model, BERT_QNLI_PRUNE_CONFIGS)

                acc_2 = program.eval()
                print(f'Module type pruned = {module}, config = {outstanding_cfg}')
                print(f'Before pruning: acc={acc_1}. After pruning: acc={acc_2}.')
                print()
                last_module = module  
    
if __name__ == '__main__':
    g_last_bert_prune_config = {}
    # bert_pruning_sensitivity_test_type2()
    bert_save_masks('../BERT-QNLI-masks')
