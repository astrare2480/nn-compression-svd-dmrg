"""汎用の TT / MPS / TT-matrix。

ニューラルネットワークへの適用（``TTLinear``、``TTConv2d``）は
``nn_compression.compression``。物理系の DMRG は ``nn_compression.dmrg``。
"""

from .canonical import tt_canonicality_errors, tt_canonicalize, tt_move_center
from .contraction import tt_fro_norm, tt_inner
from .core import (
    tt_num_parameters,
    tt_physical_shape,
    tt_ranks,
    tt_reconstruct,
    tt_svd,
    tt_svd_error_bound,
    tt_svd_exact,
    tt_svd_ranks,
    tt_unfold,
)
from .matrix import (
    dense_to_tt_matrix_cores,
    dense_to_tt_matrix_tensor,
    tt_cores_to_tt_matrix_cores,
    tt_matrix_cores_to_tt_cores,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from .matrix_two_site import (
    TTMatrixPairSplit,
    canonicalize_tt_matrix_cores,
    merge_tt_matrix_cores,
    merged_tt_matrix_to_matrix,
    replace_tt_matrix_core_pair,
    split_merged_tt_matrix,
    tt_matrix_canonicality_errors,
)
from .modes import (
    ordered_factorizations,
    tt_matrix_num_parameters_from_ranks,
    tt_matrix_physical_sizes,
    tt_matrix_shape_imbalance,
)
from .rounding import tt_bond_singular_values, tt_round, tt_truncate_bond

__all__ = [
    "tt_unfold",
    "tt_svd_exact",
    "tt_svd",
    "tt_svd_error_bound",
    "tt_svd_ranks",
    "tt_reconstruct",
    "tt_num_parameters",
    "tt_physical_shape",
    "tt_ranks",
    "tt_canonicalize",
    "tt_canonicality_errors",
    "tt_move_center",
    "tt_truncate_bond",
    "tt_bond_singular_values",
    "tt_round",
    "tt_inner",
    "tt_fro_norm",
    "tt_matrix_ranks",
    "tt_matrix_num_parameters",
    "tt_matrix_to_dense",
    "dense_to_tt_matrix_tensor",
    "tt_cores_to_tt_matrix_cores",
    "tt_matrix_cores_to_tt_cores",
    "dense_to_tt_matrix_cores",
    "TTMatrixPairSplit",
    "merge_tt_matrix_cores",
    "merged_tt_matrix_to_matrix",
    "split_merged_tt_matrix",
    "canonicalize_tt_matrix_cores",
    "tt_matrix_canonicality_errors",
    "replace_tt_matrix_core_pair",
    "ordered_factorizations",
    "tt_matrix_physical_sizes",
    "tt_matrix_shape_imbalance",
    "tt_matrix_num_parameters_from_ranks",
]
