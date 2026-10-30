"""2-site DMRG の再利用可能な実装。

構成:

- DMRG 本体（TFIM・dense Hamiltonian・厳密 energy に依存しない）:
  ``environment`` / ``effective_hamiltonian`` / ``one_site`` / ``two_site`` / ``sweep`` /
  ``convergence``。
- 実験支援（本体からは呼ばれない）: ``initialization``（初期 MPS）、``diagnostics``
  （dense 化・全残差・energy 分散）。
- 模型固有: ``models.tfim``。

MPS core は ``(r_left, physical, r_right)``、MPO core は
``(w_left, physical_out, physical_in, w_right)``。
"""

from .convergence import (
    DMRGResult,
    DMRGRoundTripRecord,
    DMRGRoundTripResult,
    run_two_site_dmrg,
    run_two_site_dmrg_round_trip,
)
from .diagnostics import (
    ResidualVarianceDiagnostics,
    compute_dense_residual_and_variance,
    mpo_to_dense,
)
from .effective_hamiltonian import (
    apply_one_site_effective_hamiltonian,
    apply_two_site_effective_hamiltonian,
    build_one_site_effective_hamiltonian,
    build_two_site_effective_hamiltonian,
)
from .environment import (
    build_left_environments,
    build_right_environments,
    mps_mpo_energy,
    mps_mpo_expectation,
    normalize_mps,
    update_left_environment,
    update_right_environment,
)
from .initialization import make_random_mps
from .models import build_tfim_dense_hamiltonian, build_tfim_mpo
from .one_site import OneSiteUpdateResult, update_one_site_core
from .sweep import SweepPassResult, SweepStepLog, two_site_sweep
from .two_site import (
    TwoSiteSplitResult,
    TwoSiteUpdateResult,
    merge_two_sites,
    split_two_site_tensor,
    update_two_site_pair,
)

__all__ = [
    # DMRG 本体
    "update_left_environment",
    "update_right_environment",
    "build_left_environments",
    "build_right_environments",
    "mps_mpo_expectation",
    "mps_mpo_energy",
    "normalize_mps",
    "build_one_site_effective_hamiltonian",
    "apply_one_site_effective_hamiltonian",
    "update_one_site_core",
    "merge_two_sites",
    "build_two_site_effective_hamiltonian",
    "apply_two_site_effective_hamiltonian",
    "split_two_site_tensor",
    "update_two_site_pair",
    "two_site_sweep",
    "run_two_site_dmrg_round_trip",
    "run_two_site_dmrg",
    "OneSiteUpdateResult",
    "TwoSiteSplitResult",
    "TwoSiteUpdateResult",
    "SweepStepLog",
    "SweepPassResult",
    "DMRGRoundTripResult",
    "DMRGRoundTripRecord",
    "DMRGResult",
    # 実験支援・模型固有
    "make_random_mps",
    "mpo_to_dense",
    "compute_dense_residual_and_variance",
    "ResidualVarianceDiagnostics",
    "build_tfim_mpo",
    "build_tfim_dense_hamiltonian",
]
