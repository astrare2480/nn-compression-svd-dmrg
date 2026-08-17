"""再利用可能なモデル定義。"""

from .cnn import FashionMNISTCNN
from .mlp import MNISTMLP

__all__ = ["MNISTMLP", "FashionMNISTCNN"]
