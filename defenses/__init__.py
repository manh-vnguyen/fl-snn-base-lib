# defenses package initialization
from .defs import *

__all__ = ['Mean', 'DnC', 'Krum', 'RFA', 'CenterClipping', 'SignGuard', 'NormClipping', 'TopK'] 

def get_defense_class(defense_type):
    """Dynamically get defense class by name."""
    if defense_type not in __all__:
        raise ImportError(f"Defense class '{defense_type}' not found in defenses module. Available defenses: {__all__}")
    return globals()[defense_type] 