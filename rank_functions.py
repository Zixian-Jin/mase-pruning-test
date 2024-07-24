import torch
import random
import numpy as np



def block_lp_norm(block: torch.Tensor, p:float):
    # for max norm: p = float('inf')
    return block.norm(p)


def get_block_mask_local(data: torch.Tensor, config: dict, sparsity: float) -> torch.Tensor:
    '''
        config = {
            block_num: 4,
            block_size: 3,
        }
    '''
    data_row, data_col = data.shape
    block_num = config['block_num']
    block_size = config['block_size']
    
    block_mask = torch.zeros((data_row, block_num), dtype=torch.bool)
    block_norms = torch.zeros((data_row, block_num), dtype=torch.float32)
    
    for i in range(data_row):  # row-wise scope for local sparsification
        # 1. get block-wise norms
        for b_id in range(block_num):
            block_norms[i, b_id] = block_lp_norm(data[i, block_size*b_id : block_size*(b_id+1)], 2)  # get the norm of a sliced block tensor
        # 2. rank blocks
        local_thres = torch.quantile(block_norms[i], sparsity)
        # 3. set block mask
        block_mask[i] = (block_norms[i] > local_thres)
    
    return block_mask

def block_rank_fn_local(data, config, sparsity: float, silent=True) -> torch.Tensor:
    '''
        config = {
            block_num: 4,
            block_size: 3,
        }
    '''
    
    row, col = data.shape
    assert (col % config['block_num'] == 0), f"The weight is not divisible by {config['block_num']}."
    assert (sparsity <= 1 and sparsity >= 0)
    config['block_size'] = int(col / config['block_num'])
    
    # 1. create a block-level mask
    block_mask = get_block_mask_local(data, config, sparsity)
    if not silent:
        print('Generated block-level mask:')
        print(block_mask)   
        
    # 2. transform block_mask into an elementwise mask
    element_mask = torch.ones((row, col))
    for i in range(row):
        for j in range(col):
            block_id = j // config['block_size']
            element_mask[i][j] = 0 if (block_mask[i][block_id] == 0) else 1
    
    return element_mask