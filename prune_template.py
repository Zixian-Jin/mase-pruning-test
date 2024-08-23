import os
import copy
from torch import nn

from bert_QNLI import BertQNLIPrunerProgram
from rank_functions import block_rank_fn_local

from utils import *




def bert_save_masks(output_root_dir='../BERT-QNLI-masks'):    
    program = BertQNLIPrunerProgram()
    model = program.model
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
        
    prune_configs = copy.deepcopy(program.bert_qnli_prune_cfg)

    count = 0
    for layer, module_dict in prune_configs.items():
        for name, cfg in module_dict.items():
            module = program.get_bert_qnli_tunable_module(model, str(layer), name)
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


# def bert_prune_example(model: nn.Module):
#     init_bert_configs()
#     print(program.bert_qnli_prune_cfg)
#     layer = 0
#     matmul = 'Q'
#     new_cfg = {'block_num': 16, 'sparsity': 0.9}
#     program.bert_qnli_prune_cfg[str(layer)]['Q'] = new_cfg
#     module = find_bert_tunable_module(model, str(layer), 'Q')
#     update_module_parametrization(module, 'weight', new_cfg)


def bert_pruning_sensitivity_test_type1():
    '''
        A single-layer, all-module pruning test.
        For each trial (i.e., each sparsity cfg), prune 
        ALL six matmul moduels of a SINGLE layer.
        TODO: should pooler & classifier be pruned as well?
    '''

    program = BertQNLIPrunerProgram()
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
                program.bert_qnli_prune_cfg[str(layer)]['Q'] = outstanding_cfg
                program.bert_qnli_prune_cfg[str(layer)]['K'] = outstanding_cfg
                program.bert_qnli_prune_cfg[str(layer)]['V'] = outstanding_cfg
                program.bert_qnli_prune_cfg[str(layer)]['W0'] = outstanding_cfg
                program.bert_qnli_prune_cfg[str(layer)]['W1'] = outstanding_cfg
                program.bert_qnli_prune_cfg[str(layer)]['W2'] = outstanding_cfg 
                    
                program.bert_qnli_prune_cfg[str(last_layer)]['Q'] = base_cfg
                program.bert_qnli_prune_cfg[str(last_layer)]['K'] = base_cfg
                program.bert_qnli_prune_cfg[str(last_layer)]['V'] = base_cfg
                program.bert_qnli_prune_cfg[str(last_layer)]['W0'] = base_cfg
                program.bert_qnli_prune_cfg[str(last_layer)]['W1'] = base_cfg
                program.bert_qnli_prune_cfg[str(last_layer)]['W2'] = base_cfg
                
                program.bert_qnli_prune()

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
    program = BertQNLIPrunerProgram()
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
            for module in program.BERT_LAYER_TUNABLE_MATMUL_MAP.keys():
                # prune only one type of module for all encoder layers
                for layer in range(12):
                    program.bert_qnli_prune_cfg[str(layer)][module] = outstanding_cfg
                    program.bert_qnli_prune_cfg[str(layer)][last_module] = dense_cfg
                program.bert_qnli_prune()

                acc_2 = program.eval()
                print(f'Module type pruned = {module}, config = {outstanding_cfg}')
                print(f'Before pruning: acc={acc_1}. After pruning: acc={acc_2}.')
                print()
                last_module = module  
    
if __name__ == '__main__':
    bert_pruning_sensitivity_test_type2()
    # bert_save_masks('../BERT-QNLI-masks')
