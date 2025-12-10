import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

__all__ = [
    "augmentation",
    "inference",
    "model",
    "model_loader",
    "preprocessing",
    "train"
]
