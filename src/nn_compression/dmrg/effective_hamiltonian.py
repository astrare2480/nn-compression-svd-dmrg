r"""1-site / 2-site の有効 Hamiltonian（明示行列と matrix-free 作用）。

参照元:
    notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb

左右の environment ``L``, ``R`` と中心の MPO core から、局所問題の有効 Hamiltonian
``H_eff`` を作る。``H_eff`` の添字は ``(bra, ket)`` の順で、行は bra 側の局所 index
（左 bond, 物理 index, 右 bond）を C 順に平坦化したもの、列は ket 側のそれ。

- 明示行列版 (``build_*``): ``m x m``（``m`` は局所自由度の数）を作る。局所 tensor の
  次元 ``m`` に対し ``m^2`` 要素のメモリが必要になる。
- matrix-free 版 (``apply_*``): ``H_eff`` を作らず、入力 tensor に作用させた結果だけを
  返す。明示行列版に同じ入力を掛けた結果と一致する。

局所問題が標準固有値問題になるのは、``L`` と ``R`` が（中心を除いて）正準形の MPS から
作られている場合である。この module は正準性を検査しない。

autograd: ``detach()`` / ``no_grad()`` は使わず、入力が ``requires_grad=True`` なら graph を
維持する。
"""

from __future__ import annotations

import torch

from .validation import _validate_one_site_local_operands, _validate_two_site_local_operands

# 明示 H_eff の Hermite 性検査に使う相対許容（eps の倍数）。
_HERMITIAN_EPS_FACTOR = 1000.0


def build_one_site_effective_hamiltonian(
    left_environment: torch.Tensor,
    center_mpo_core: torch.Tensor,
    right_environment: torch.Tensor,
) -> torch.Tensor:
    r"""中心 1 site の有効 Hamiltonian を明示的な ``(m, m)`` 行列として作る。

    left_environment:
        shape ``(r_left, w_left, r_left)``。
    center_mpo_core:
        shape ``(w_left, n, n, w_right)``。
    right_environment:
        shape ``(r_right, w_right, r_right)``。

    戻り値:
        shape ``(m, m)``、``m = r_left * n * r_right``。

    $$
    (H_{\mathrm{eff}})_{(a d i),(c e k)}=\sum_{b,j}L_{abc}\,W_{bdej}\,R_{ijk}
    $$

    メモリは局所次元 ``m`` の 2 乗に比例するため、大規模問題を反復固有値 solver へ渡す
    用途には ``apply_one_site_effective_hamiltonian`` を使うこと。入力は変更しない。
    """
    _validate_one_site_local_operands(left_environment, center_mpo_core, right_environment)
    local_dimension = left_environment.shape[0] * center_mpo_core.shape[1] * right_environment.shape[0]
    tensor = torch.einsum("abc,bdej,ijk->adicek", left_environment, center_mpo_core, right_environment)
    return tensor.reshape(local_dimension, local_dimension)


def apply_one_site_effective_hamiltonian(
    left_environment: torch.Tensor,
    center_mpo_core: torch.Tensor,
    right_environment: torch.Tensor,
    input_core: torch.Tensor,
) -> torch.Tensor:
    r"""1-site 有効 Hamiltonian を ``input_core`` へ matrix-free に作用させる。

    ``input_core`` は shape ``(r_left, n, r_right)``（ket 側の局所 tensor）。戻り値は同じ
    shape で、``build_one_site_effective_hamiltonian(...) @ input_core.reshape(-1)`` を
    reshape したものと一致する。``H_eff`` を作らず、環境・MPO・入力を 3 回の縮約へ
    分けて作用させる。入力は変更しない。
    """
    _validate_one_site_local_operands(
        left_environment, center_mpo_core, right_environment, input_core=input_core
    )
    # c, e, k: ket の (左 bond, 物理, 右 bond)
    step = torch.einsum("cek,ijk->ceij", input_core, right_environment)
    step = torch.einsum("ceij,bdej->cbdi", step, center_mpo_core)
    return torch.einsum("cbdi,abc->adi", step, left_environment)


def build_two_site_effective_hamiltonian(
    left_environment: torch.Tensor,
    left_mpo_core: torch.Tensor,
    right_mpo_core: torch.Tensor,
    right_environment: torch.Tensor,
) -> torch.Tensor:
    r"""隣接 2 site の有効 Hamiltonian を明示的な ``(m, m)`` 行列として作る。

    left_environment:
        shape ``(r_left, w_left, r_left)``。
    left_mpo_core / right_mpo_core:
        shape ``(w_left, n_left, n_left, w_mid)`` / ``(w_mid, n_right, n_right, w_right)``。
    right_environment:
        shape ``(r_right, w_right, r_right)``。

    戻り値:
        shape ``(m, m)``、``m = r_left * n_left * n_right * r_right``。行と列の添字順は
        ``merge_two_sites`` が返す 4 階 tensor の C 順平坦化に一致する。

    メモリは局所次元 ``m`` の 2 乗に比例する。大規模な反復固有値 solver 向けではなく、
    小さい系の厳密対角化・検証用である。matrix-free に作用だけ必要なら
    ``apply_two_site_effective_hamiltonian`` を使う。入力は変更しない。
    """
    _validate_two_site_local_operands(left_environment, left_mpo_core, right_mpo_core, right_environment)
    local_dimension = (
        left_environment.shape[0]
        * left_mpo_core.shape[1]
        * right_mpo_core.shape[1]
        * right_environment.shape[0]
    )
    # p: 左 MPO bond, q: 2 つの MPO core 間の bond, r: 右 MPO bond
    step = torch.einsum("apb,psuq->absuq", left_environment, left_mpo_core)
    step = torch.einsum("absuq,qtvr->abstuvr", step, right_mpo_core)
    tensor = torch.einsum("abstuvr,crd->astcbuvd", step, right_environment)
    return tensor.reshape(local_dimension, local_dimension)


def apply_two_site_effective_hamiltonian(
    left_environment: torch.Tensor,
    left_mpo_core: torch.Tensor,
    right_mpo_core: torch.Tensor,
    right_environment: torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    r"""2-site 有効 Hamiltonian を ``input_tensor`` へ matrix-free に作用させる。

    ``input_tensor`` は shape ``(r_left, n_left, n_right, r_right)``（``merge_two_sites`` の
    出力と同じ並び）。戻り値は同じ shape で、
    ``build_two_site_effective_hamiltonian(...) @ input_tensor.reshape(-1)`` を reshape
    したものと一致する。``(m, m)`` 行列は作らず、4 回の縮約で作用させる。入力は変更しない。
    """
    _validate_two_site_local_operands(
        left_environment,
        left_mpo_core,
        right_mpo_core,
        right_environment,
        input_tensor=input_tensor,
    )
    # b, u, v, d: ket の (左 bond, 左物理, 右物理, 右 bond)
    step = torch.einsum("buvd,crd->buvcr", input_tensor, right_environment)
    step = torch.einsum("buvcr,qtvr->bucqt", step, right_mpo_core)
    step = torch.einsum("bucqt,psuq->bcpst", step, left_mpo_core)
    return torch.einsum("bcpst,apb->astc", step, left_environment)


def _lowest_eigenpair(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """実対称（Hermite）行列の最小固有値と、その正規化された固有ベクトルを返す。

    ``torch.linalg.eigh`` は片側の三角成分しか読まないため、非対称な入力を黙って
    誤った固有対へ変換しないよう、先に対称性を検査する（``eps`` の定数倍の相対許容）。
    非有限値・正方でない入力も ``ValueError``。局所 solver は dense 対角化であり、
    Lanczos / Davidson は使わない。autograd graph は維持する（ただし縮退した固有値では
    ``eigh`` の勾配が不安定になる）。
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"局所 Hamiltonian は正方行列である必要があります: shape={tuple(matrix.shape)}")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("局所 Hamiltonian に NaN / inf が含まれています。")
    scale = matrix.abs().amax()
    asymmetry = (matrix - matrix.mH).abs().amax()
    tolerance = _HERMITIAN_EPS_FACTOR * torch.finfo(matrix.dtype).eps * scale
    if not bool(asymmetry <= tolerance):
        raise ValueError(
            "局所 Hamiltonian が対称（Hermite）ではありません: "
            f"max|H - H^T|={asymmetry.item():.3e}, 許容={tolerance.item():.3e}。"
            "MPO が Hermitian であることを確認してください。"
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvalues[0], eigenvectors[:, 0]
