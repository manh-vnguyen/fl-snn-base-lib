# fed package initialization
from .fed import FLTrainer
from .solo import SoloTrainer
from .utils import test_model

__all__ = ['FLTrainer', 'SoloTrainer', 'test_model']
