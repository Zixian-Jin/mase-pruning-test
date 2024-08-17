import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from rank_functions import *



class StructuredPruningMask(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        return x * self.mask.to(x.device)  # TODO

# def get_structured_mask(module, sparsity_cfg, silent=True) -> torch.Tensor:
#     return block_rank_fn_local(module.weight.detach(), sparsity_cfg, silent)


def update_module_parametrization(module: nn.Module, param_name: str, new_cfg: dict, silent=True):
    
    assert param_name in ['weight', 'bias'], "param_name can only be weight or bias."
    assert new_cfg != None, "new_cfg cannot be empty!"
    
    if not hasattr(module, 'parametrizations'):
        # the module has never been pruned before: only retrieve the param tensor
        original_param = getattr(module, param_name)
    else:
        # the module was pruned: remove previous registration
        original_param = getattr(module.parametrizations, param_name).original
        parametrize.remove_parametrizations(module, param_name, leave_parametrized=False)  # NOTE: important!
    
    new_mask = block_rank_fn_local(original_param.detach(), new_cfg)    
    parametrize.register_parametrization(module, param_name, StructuredPruningMask(new_mask))
    if not silent:
        print(f'Successfully re-registered parametrization with new_cfg={new_cfg}')



