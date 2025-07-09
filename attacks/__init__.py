# attacks package initialization
from .atks import *

__all__ = ['IPM', 'MinMax', 'Fang', 'Mimic', 'GaussRandom', 'LIE', 'MinSum', 'LabelFlip', 'SignFlip'] 

def get_attack_class(attack_type):
    """Dynamically get attack class by name."""
    if attack_type not in __all__:
        raise ImportError(f"Attack class '{attack_type}' not found in attacks module. Available attacks: {__all__}")
    return globals()[attack_type] 