"""実験の再現性を高めるための乱数シード処理。

参照元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Python・NumPy・PyTorch の乱数源を固定する。

    CUDAでは決定的アルゴリズムを優先する。DataLoader の shuffle 順は
    別途 ``make_torch_generator`` で固定する必要がある。
    """
    # 決定的 CUDA のため、CUDA カーネルが起動する前に設定する。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_torch_generator(seed: int = 0) -> torch.Generator:
    """DataLoader のshuffleや ``random_split`` 用のGeneratorを返す。

    学習用 DataLoader を ``next(iter(loader))`` で先に読むと、この
    Generator は消費されて学習順が変わる点に注意する。
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
