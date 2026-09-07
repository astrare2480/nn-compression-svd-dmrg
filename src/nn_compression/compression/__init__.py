"""ニューラルネットワークの圧縮処理。"""

from .conv_svd import (
    conv2d_weight_matrix,
    factorize_Conv2d_layer,
    factorize_conv2d_layer,
    factorize_named_conv2d,
)
from .conv_tucker import (
    build_tucker2_conv,
    build_tucker2_conv_from_components,
    tucker2_decompose_conv_weight,
    tucker2_effective_weight,
    tucker2_hooi,
    tucker2_hooi_sweep,
)
from .hooi import core_from_factors, has_converged, hooi, hooi_sweep
from .linear_svd import (
    RebuildSVD,
    factorize_linear_layer,
    factorize_named_linear,
    rebuild_linear_from_svd,
)
from .mlp_svd import make_one_layer_svd_model, make_two_layer_svd_model
from .named_layers import factorize_named_layers
from .rank_sweep import (
    sweep_conv2d_ranks,
    sweep_conv_svd_ranks,
    sweep_layer_ranks,
)
from .svd import SVD, retained_energy, retained_energy_from_matrix, truncated_svd
from .tt import tt_num_parameters, tt_reconstruct, tt_svd, tt_svd_exact, tt_unfold
from .tucker import (
    compression_factor,
    hosvd,
    parameter_ratio,
    reconstruct_tucker,
    tucker_parameter_count,
)

__all__ = [
    "truncated_svd",
    "retained_energy",
    "retained_energy_from_matrix",
    "rebuild_linear_from_svd",
    "factorize_linear_layer",
    "factorize_named_linear",
    "factorize_conv2d_layer",
    "factorize_named_conv2d",
    "factorize_named_layers",
    "sweep_layer_ranks",
    "sweep_conv2d_ranks",
    "sweep_conv_svd_ranks",
    "conv2d_weight_matrix",
    "make_two_layer_svd_model",
    "make_one_layer_svd_model",
    "hosvd",
    "reconstruct_tucker",
    "tucker_parameter_count",
    "parameter_ratio",
    "compression_factor",
    "has_converged",
    "hooi_sweep",
    "core_from_factors",
    "hooi",
    "build_tucker2_conv",
    "build_tucker2_conv_from_components",
    "tucker2_decompose_conv_weight",
    "tucker2_hooi",
    "tucker2_hooi_sweep",
    "tucker2_effective_weight",
    "tt_unfold",
    "tt_svd_exact",
    "tt_svd",
    "tt_reconstruct",
    "tt_num_parameters",
    # 旧名
    "SVD",
    "RebuildSVD",
    "factorize_Conv2d_layer",
]
