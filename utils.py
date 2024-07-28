import torch
import numpy as np
import matplotlib.pyplot as plt


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
    