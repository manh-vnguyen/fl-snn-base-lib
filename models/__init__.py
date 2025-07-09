# models package initialization
from .svgg import SVGG

def init_model(exp, device):
    return SVGG(**exp.model_params).to(device)

__all__ = ['init_model', 'SVGG'] 