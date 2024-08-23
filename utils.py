import torch
import numpy as np
import matplotlib.pyplot as plt
from rank_functions import *

def matrix_profiler(mat: torch.Tensor, rank: float, scope: str):
    ''' analyse the top k entries of each row of a matrix'''
    row, col = mat.shape
    # assert topk <= col, "topk must be no greater than dim1"
    assert (0 <= rank <= 1), "rank must be between 0 and 1"
    top_k = int(col*rank)
    
    if scope == 'local':
        values, indicies = torch.topk(mat, top_k, dim=1, largest=True, sorted=False)
    elif scope == 'global':
        pass
    else:
        pass
    
    return values, indicies
    
def plot_array_histogram(tensor: torch.Tensor, bins=50):

    data = tensor.abs().numpy()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    n1, bins1, patches1 = ax.hist(data, bins=bins, density=True, alpha=0.5, color='blue')

    ax.set_xlim([0, data.max()])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    plt.title('Distribution Histogram')
    # ax.legend(['Data 1', 'Data 2'])

    plt.show()


def check_sparsity(self, module):
    w = module.weight.data
    # print(w)
    thres = 0.05
    max_val = w.abs().max().item()
    for thres in np.linspace(0.00, max_val, 10):
        mask = torch.where(torch.abs(w)<thres, 1, 0)
        print("Sparsity of current module with thres=%f = %f"%(thres, torch.sum(mask)/(w.shape[0]*w.shape[1])))
    
    # values, indicies = utils.matrix_profiler(w, rank=0.1, scope='local')
    # print('Top 10% elements in the analysed matrix:')
    # print('Values=', values)
    # print('Indicies=', indicies)
        

def simple_prune(module, thres):
    print('WARNING: this function is deprecated. Use update_module_parametrizatoin instead.')
    print('INFO: Pruning...')
    # print('Weight before pruning:')
    # print(module.weight.data)
    mask = (torch.abs(module.weight.data) >= thres)
    module.weight.data *= mask.float()
    # print('Weight after pruning:')
    # print(module.weight.data)
    print('INFO: Finished pruning.')

def structured_prune(module, prune_cfg, silent=True):
    print('WARNING: this function is deprecated. Use update_module_parametrizatoin instead.')
    print(f'INFO: Pruning module {module}...')
    data = module.weight.datach()
    mask = block_rank_fn_local(data, prune_cfg, silent=silent)
    module.weight.data *= mask
    if not silent:
        print('INFO: Finished pruning.')

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
