r"""開境界の横磁場 Ising 模型（TFIM）の MPO と dense Hamiltonian。

参照元:
    notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb

模型（スピン 1/2、開境界）:

    H = -J * sum_{i=0}^{L-2} Z_i Z_{i+1} - g * sum_{i=0}^{L-1} X_i

``X``, ``Z`` は Pauli 行列、``J`` は ``coupling``、``g`` は ``transverse_field``。
site 数 ``L`` は引数で、固定値を持たない。dtype / device も引数で、module-level の
既定値を持たない。

この module は **模型固有** の処理で、DMRG 本体（``nn_compression.dmrg`` の environment /
局所更新 / sweep / 収束判定）からは呼ばれない。dense 版は小規模系の検証 oracle 専用。
"""

from __future__ import annotations

import math
import numbers

import torch

from ...compression.svd import coerce_integer_scalar
from ...compression.tt_validation import validate_tt_dtype


def _coefficient_tensor(value: float, dtype: torch.dtype, device: torch.device, name: str) -> torch.Tensor:
    """係数を対象 dtype / device の 0 階 Tensor にする。

    Python float として有限でも、float32 では ``1e100`` が ``inf``、``1e-46`` が ``0`` に
    なる。変換後の値が非有限、または非零な入力が 0 になる場合はモデルを作らず
    ``ValueError``（メッセージに引数名を含める）。厳密な 0 は受け付ける。
    """
    converted = torch.tensor(value, dtype=dtype, device=device)
    if not bool(torch.isfinite(converted)):
        raise ValueError(f"{name} を {dtype} へ変換すると有限ではなくなります: {value!r}")
    if value != 0.0 and bool(converted == 0):
        raise ValueError(f"{name} を {dtype} へ変換すると 0 になります: {value!r}")
    return converted


def _validate_tfim_arguments(
    length, coupling, transverse_field, dtype, device
) -> tuple[int, torch.Tensor, torch.Tensor, torch.device]:
    """TFIM builder 共通の引数検査。``length`` は 2 以上の整数（bool 不可）、係数は有限な実数。"""
    size = coerce_integer_scalar(length, name="length")
    if size < 2:
        raise ValueError(f"length は 2 以上である必要があります: {length!r}")
    coefficients = []
    for name, value in (("coupling", coupling), ("transverse_field", transverse_field)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} は実数である必要があります: {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} は有限である必要があります: {value!r}")
        coefficients.append(float(value))
    validate_tt_dtype(torch.empty(0, dtype=dtype), name="dtype")
    target_device = torch.device(device)
    return (
        size,
        _coefficient_tensor(coefficients[0], dtype, target_device, "coupling"),
        _coefficient_tensor(coefficients[1], dtype, target_device, "transverse_field"),
        target_device,
    )


def _pauli_matrices(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(identity, pauli_x, pauli_z)`` を指定 dtype / device で作る。"""
    identity = torch.eye(2, dtype=dtype, device=device)
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    return identity, pauli_x, pauli_z


def build_tfim_mpo(
    length: int,
    coupling: float,
    transverse_field: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[torch.Tensor]:
    r"""開境界 TFIM ``H = -J sum Z_i Z_{i+1} - g sum X_i`` の MPO core 列を作る。

    length:
        site 数 ``L``（2 以上の整数、bool 不可）。
    coupling, transverse_field:
        ``J`` と ``g``（有限な実数、bool 不可）。対象 dtype へ変換した値が非有限、
        または非零な値が 0 になる場合は ``ValueError``。厳密な 0 は受け付ける。
    dtype, device:
        ``torch.float32`` / ``torch.float64`` と作成先 device。

    戻り値:
        長さ ``L`` の MPO core 列。各 core の shape は ``(w_left, 2, 2, w_right)``
        （両端 core の外側 bond は 1、内部 bond は 3）。MPO bond 次元 3 の bulk 行列は

        ```text
        [[ I,       0,        0 ],
         [ Z,       0,        0 ],
         [-g X,   -J Z,       I ]]
        ```

        で、左端は最下行、右端は最左列を取る。物理次元は 2 固定（スピン 1/2）。
    """
    size, coupling_value, field_value, target_device = _validate_tfim_arguments(
        length, coupling, transverse_field, dtype, device
    )
    identity, pauli_x, pauli_z = _pauli_matrices(dtype, target_device)
    zero = torch.zeros_like(identity)
    x_term = -field_value * pauli_x
    z_term = -coupling_value * pauli_z

    # 行 = 左 MPO bond、列 = 右 MPO bond。各要素は (physical_out, physical_in) の 2x2 行列。
    bulk = torch.stack(
        [
            torch.stack([identity, zero, zero], dim=-1),
            torch.stack([pauli_z, zero, zero], dim=-1),
            torch.stack([x_term, z_term, identity], dim=-1),
        ],
        dim=0,
    )
    left_boundary = torch.stack([x_term, z_term, identity], dim=-1).unsqueeze(0)
    right_boundary = torch.stack([identity, pauli_z, x_term], dim=0).unsqueeze(-1)
    cores = [left_boundary, *[bulk.clone() for _ in range(size - 2)], right_boundary]
    if any(not bool(torch.isfinite(core).all()) for core in cores):
        raise ValueError("build_tfim_mpo の結果に有限でない要素があります。")
    return cores


def _kron_all(matrices: list[torch.Tensor]) -> torch.Tensor:
    """行列列の Kronecker 積（先頭を最上位の桁とする）。"""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def build_tfim_dense_hamiltonian(
    length: int,
    coupling: float,
    transverse_field: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    r"""開境界 TFIM ``H = -J sum Z_i Z_{i+1} - g sum X_i`` の dense Hamiltonian を作る。

    引数の意味は ``build_tfim_mpo`` に同じ。戻り値は shape ``(2**L, 2**L)``。site 0 が
    最上位の桁（``mpo_to_dense(build_tfim_mpo(...))`` と同じ index 規約）。

    **小規模系の検証 oracle 専用**: メモリは ``4**L`` 要素に比例して指数的に増える。DMRG 本体
    からは呼ばれない。
    """
    size, coupling_value, field_value, target_device = _validate_tfim_arguments(
        length, coupling, transverse_field, dtype, device
    )
    identity, pauli_x, pauli_z = _pauli_matrices(dtype, target_device)
    dimension = 2**size
    hamiltonian = torch.zeros((dimension, dimension), dtype=dtype, device=target_device)

    for site in range(size - 1):
        factors = [identity] * size
        factors[site] = pauli_z
        factors[site + 1] = pauli_z
        hamiltonian = hamiltonian - coupling_value * _kron_all(factors)
    for site in range(size):
        factors = [identity] * size
        factors[site] = pauli_x
        hamiltonian = hamiltonian - field_value * _kron_all(factors)
    if not bool(torch.isfinite(hamiltonian).all()):
        raise ValueError("build_tfim_dense_hamiltonian の結果に有限でない要素があります。")
    return hamiltonian
