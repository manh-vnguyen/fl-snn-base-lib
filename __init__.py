# fl_lib package initialization
from .datasets import get_dataset, IIDPartitioner
from .models import init_model

# Import commonly used classes from submodules
from .defenses import Mean, DnC, Krum, RFA, CenterClipping, SignGuard, NormClipping, TopK
from .attacks import IPM, MinMax, Fang, GaussRandom, LIE, MinSum, LabelFlip, SignFlip

# Import trainers
from .trainers import FLTrainer, SoloTrainer

__all__ = [
    # Core functionality
    'get_dataset', 'IIDPartitioner', 'init_model',
    
    # Defenses/Aggregators
    'Mean', 'DnC', 'Krum', 'RFA', 'CenterClipping', 'SignGuard', 'NormClipping', 'TopK',
    
    # Attacks
    'IPM', 'MinMax', 'Fang', 'GaussRandom', 'LIE', 'MinSum', 'LabelFlip', 'SignFlip',
    
    # Trainers
    'FLTrainer', 'SoloTrainer'
] 