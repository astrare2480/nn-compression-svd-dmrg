r"""DMRG の入力 validation（アルゴリズム本体ではない）。

参照元:
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb

Notebook 02 の ``validate_mps_mpo_pair`` は ``zip`` で長さの短い側に黙って合わせて
いたため、MPS と MPO の site 数の違いを bond の検査より前に検出できなかった。ここでは
site 数を先に比べ、その後で各 core・隣接 bond・物理次元・dtype / device を検査する。

規約:

- MPS core: ``(r_left, physical, r_right)``。
- MPO core: ``(w_left, physical_out, physical_in, w_right)``。
- 開境界なので、先頭の left bond と末尾の right bond は 1。
- 対応 dtype は既存 TT API と同じ ``torch.float32`` / ``torch.float64``。complex、
  float16、bfloat16、整数、bool は ``TypeError``。
- 型・dtype の誤りは ``TypeError``、shape・値・範囲の誤りは ``ValueError``。
- NaN / inf: core 列全体を受け取る入口（``_validate_mps_mpo_pair`` など）は
  ``check_finite=True`` で ``ValueError`` にする。単一 core を縮約する低レベル API は
  有限性を検査せず、NaN / inf はそのまま結果へ伝播する（検査のたびに host への
  同期が起きるため）。

この module の helper はすべてパッケージ内部用で、公開 API ではない。
"""

from __future__ import annotations

import math
import numbers

import torch

from ..compression.svd import coerce_integer_scalar
from ..compression.tt_validation import validate_tt_dtype

_DIRECTIONS = ("left_to_right", "right_to_left")


def _validate_direction(direction) -> str:
    """``direction`` が ``"left_to_right"`` / ``"right_to_left"`` のどちらかであることを確かめる。"""
    if not isinstance(direction, str):
        raise TypeError(f"direction は str である必要があります: {direction!r}")
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction は {_DIRECTIONS} のどちらかである必要があります: {direction!r}")
    return direction


def _validate_max_bond(max_bond) -> int | None:
    """``max_bond`` が ``None`` または 1 以上の整数（bool 不可）であることを確かめる。"""
    if max_bond is None:
        return None
    value = coerce_integer_scalar(max_bond, name="max_bond")
    if value < 1:
        raise ValueError(f"max_bond は None または 1 以上の整数である必要があります: {max_bond!r}")
    return value


def _validate_tolerance(value, *, name: str) -> float:
    """許容誤差が有限で 0 以上の実数（bool 不可）であることを確かめて ``float`` にする。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} は実数である必要があります: {value!r} ({type(value)!r})")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} は有限で 0 以上である必要があります: {value!r}")
    return result


def _validate_positive_int(value, *, name: str) -> int:
    """``value`` が 1 以上の整数（bool 不可）であることを確かめる。"""
    result = coerce_integer_scalar(value, name=name)
    if result < 1:
        raise ValueError(f"{name} は 1 以上の整数である必要があります: {value!r}")
    return result


def _validate_index(value, upper: int, *, name: str) -> int:
    """``0 <= value < upper`` の整数 index（bool 不可）であることを確かめる。"""
    result = coerce_integer_scalar(value, name=name)
    if not 0 <= result < upper:
        raise ValueError(f"{name}={result} は 0 <= {name} < {upper} を満たす必要があります。")
    return result


def _validate_tensor(tensor, *, name: str, ndim: int) -> None:
    """``torch.Tensor`` で、階数が ``ndim``、全 dimension が正、dtype が対応範囲であることを確かめる。"""
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(tensor)!r}")
    validate_tt_dtype(tensor, name=name)
    if tensor.ndim != ndim:
        raise ValueError(f"{name} は{ndim}階である必要があります: shape={tuple(tensor.shape)}")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} の各 dimension は正である必要があります: shape={tuple(tensor.shape)}")


def _validate_same_dtype_device(*named_tensors: tuple[str, torch.Tensor]) -> None:
    """渡された Tensor の dtype / device がすべて先頭と一致することを確かめる。"""
    first_name, first = named_tensors[0]
    for name, tensor in named_tensors[1:]:
        if tensor.dtype != first.dtype:
            raise TypeError(
                f"{name}.dtype={tensor.dtype} は {first_name}.dtype={first.dtype} と一致する必要があります。"
            )
        if tensor.device != first.device:
            raise ValueError(
                f"{name}.device={tensor.device} は {first_name}.device={first.device} と一致する必要があります。"
            )


def _require_finite(tensors: list[torch.Tensor], *, name: str) -> None:
    """全 Tensor が有限であることを、host 同期 1 回で確かめる。"""
    flags = torch.stack([torch.isfinite(tensor).all() for tensor in tensors])
    if not bool(flags.all()):
        bad = [index for index, flag in enumerate(flags.tolist()) if not flag]
        raise ValueError(f"{name} に NaN / inf が含まれています: index={bad}")


def _validate_core_sequence(cores, *, name: str, ndim: int, bond_axes: tuple[int, int]) -> None:
    """MPS / MPO の core 列に共通する検査（型・階数・端の bond・隣接 bond・dtype / device）。"""
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise ValueError(f"{name} は空でない list / tuple である必要があります: {cores!r}")
    for index, core in enumerate(cores):
        _validate_tensor(core, name=f"{name}[{index}]", ndim=ndim)
    left_axis, right_axis = bond_axes
    if cores[0].shape[left_axis] != 1:
        raise ValueError(f"{name}[0] の left bond は 1 である必要があります: {cores[0].shape[left_axis]}")
    if cores[-1].shape[right_axis] != 1:
        raise ValueError(f"{name}[-1] の right bond は 1 である必要があります: {cores[-1].shape[right_axis]}")
    for index in range(len(cores) - 1):
        if cores[index].shape[right_axis] != cores[index + 1].shape[left_axis]:
            raise ValueError(
                f"{name}[{index}] の right bond ({cores[index].shape[right_axis]}) と "
                f"{name}[{index + 1}] の left bond ({cores[index + 1].shape[left_axis]}) が一致しません。"
            )
    for index, core in enumerate(cores[1:], start=1):
        _validate_same_dtype_device((f"{name}[0]", cores[0]), (f"{name}[{index}]", core))


def _validate_mps_cores(mps_cores, *, name: str = "mps_cores", check_finite: bool = True) -> None:
    """MPS core 列（各 ``(r_left, physical, r_right)``）を検査する。"""
    _validate_core_sequence(mps_cores, name=name, ndim=3, bond_axes=(0, 2))
    if check_finite:
        _require_finite(list(mps_cores), name=name)


def _validate_mpo_cores(mpo_cores, *, name: str = "mpo_cores", check_finite: bool = True) -> None:
    """MPO core 列（各 ``(w_left, physical_out, physical_in, w_right)``）を検査する。"""
    _validate_core_sequence(mpo_cores, name=name, ndim=4, bond_axes=(0, 3))
    if check_finite:
        _require_finite(list(mpo_cores), name=name)


def _validate_mps_mpo_pair(mps_cores, mpo_cores, *, check_finite: bool = True) -> None:
    """MPS と MPO が同じ chain を表せること（Hamiltonian の期待値を取れること）を検査する。

    site 数を最初に比べ、``zip`` による黙示的な短縮が起きないようにする。そのあと
    各列の構造、MPO の物理次元が MPS と一致すること（``physical_out == physical_in
    == physical``）、MPS と MPO の dtype / device の一致を確かめる。
    """
    if not isinstance(mps_cores, (list, tuple)) or len(mps_cores) == 0:
        raise ValueError(f"mps_cores は空でない list / tuple である必要があります: {mps_cores!r}")
    if not isinstance(mpo_cores, (list, tuple)) or len(mpo_cores) == 0:
        raise ValueError(f"mpo_cores は空でない list / tuple である必要があります: {mpo_cores!r}")
    if len(mps_cores) != len(mpo_cores):
        raise ValueError(
            f"MPS と MPO の site 数が一致しません: len(mps_cores)={len(mps_cores)}, "
            f"len(mpo_cores)={len(mpo_cores)}"
        )
    _validate_mps_cores(mps_cores, check_finite=False)
    _validate_mpo_cores(mpo_cores, check_finite=False)
    _validate_same_dtype_device(("mps_cores[0]", mps_cores[0]), ("mpo_cores[0]", mpo_cores[0]))
    for site, (mps_core, mpo_core) in enumerate(zip(mps_cores, mpo_cores)):
        physical = mps_core.shape[1]
        if mpo_core.shape[1] != physical or mpo_core.shape[2] != physical:
            raise ValueError(
                f"site {site} の物理次元が一致しません: MPS={physical}, "
                f"MPO(out, in)=({mpo_core.shape[1]}, {mpo_core.shape[2]})"
            )
    if check_finite:
        # 構造の検査がすべて通ってから、host 同期 1 回で有限性を確かめる。
        _require_finite([*mps_cores, *mpo_cores], name="mps_cores / mpo_cores")


def _validate_environment(
    environment,
    *,
    name: str,
    mps_bond: int,
    mpo_bond: int,
) -> None:
    """environment が ``(mps_bond, mpo_bond, mps_bond)``（bra, MPO, ket の順）であることを確かめる。"""
    _validate_tensor(environment, name=name, ndim=3)
    expected = (mps_bond, mpo_bond, mps_bond)
    if tuple(environment.shape) != expected:
        raise ValueError(
            f"{name} の shape が一致しません: expected={expected}, actual={tuple(environment.shape)}"
        )


def _validate_update_operands(
    environment,
    mps_core,
    mpo_core,
    *,
    side: str,
) -> None:
    """environment 1 site 分の更新（``update_left/right_environment``）の入力を検査する。

    ``side`` は ``"left"``（environment は MPS core の left bond 側）か ``"right"``。
    """
    _validate_tensor(mps_core, name="mps_core", ndim=3)
    _validate_tensor(mpo_core, name="mpo_core", ndim=4)
    _validate_tensor(environment, name="environment", ndim=3)
    _validate_same_dtype_device(("environment", environment), ("mps_core", mps_core), ("mpo_core", mpo_core))
    physical = mps_core.shape[1]
    if mpo_core.shape[1] != physical or mpo_core.shape[2] != physical:
        raise ValueError(
            f"物理次元が一致しません: MPS={physical}, "
            f"MPO(out, in)=({mpo_core.shape[1]}, {mpo_core.shape[2]})"
        )
    if side == "left":
        _validate_environment(
            environment, name="environment", mps_bond=mps_core.shape[0], mpo_bond=mpo_core.shape[0]
        )
    else:
        _validate_environment(
            environment, name="environment", mps_bond=mps_core.shape[2], mpo_bond=mpo_core.shape[3]
        )


def _validate_one_site_local_operands(
    left_environment,
    center_mpo_core,
    right_environment,
    *,
    input_core=None,
) -> None:
    """1-site 局所問題（``L``, ``W``, ``R`` と任意の入力 core）の整合を検査する。"""
    _validate_tensor(center_mpo_core, name="center_mpo_core", ndim=4)
    _validate_tensor(left_environment, name="left_environment", ndim=3)
    _validate_tensor(right_environment, name="right_environment", ndim=3)
    _validate_same_dtype_device(
        ("left_environment", left_environment),
        ("center_mpo_core", center_mpo_core),
        ("right_environment", right_environment),
    )
    if center_mpo_core.shape[1] != center_mpo_core.shape[2]:
        raise ValueError(
            f"center_mpo_core の物理次元 (out, in) が一致しません: {tuple(center_mpo_core.shape[1:3])}"
        )
    left_bond = left_environment.shape[0]
    right_bond = right_environment.shape[0]
    _validate_environment(
        left_environment, name="left_environment", mps_bond=left_bond, mpo_bond=center_mpo_core.shape[0]
    )
    _validate_environment(
        right_environment, name="right_environment", mps_bond=right_bond, mpo_bond=center_mpo_core.shape[3]
    )
    if input_core is not None:
        _validate_tensor(input_core, name="input_core", ndim=3)
        _validate_same_dtype_device(("left_environment", left_environment), ("input_core", input_core))
        expected = (left_bond, center_mpo_core.shape[2], right_bond)
        if tuple(input_core.shape) != expected:
            raise ValueError(
                f"input_core の shape が一致しません: expected={expected}, actual={tuple(input_core.shape)}"
            )


def _validate_two_site_local_operands(
    left_environment,
    left_mpo_core,
    right_mpo_core,
    right_environment,
    *,
    input_tensor=None,
) -> None:
    """2-site 局所問題（``L``, ``W_n``, ``W_{n+1}``, ``R`` と任意の入力 tensor）の整合を検査する。"""
    _validate_tensor(left_mpo_core, name="left_mpo_core", ndim=4)
    _validate_tensor(right_mpo_core, name="right_mpo_core", ndim=4)
    _validate_tensor(left_environment, name="left_environment", ndim=3)
    _validate_tensor(right_environment, name="right_environment", ndim=3)
    _validate_same_dtype_device(
        ("left_environment", left_environment),
        ("left_mpo_core", left_mpo_core),
        ("right_mpo_core", right_mpo_core),
        ("right_environment", right_environment),
    )
    for name, core in (("left_mpo_core", left_mpo_core), ("right_mpo_core", right_mpo_core)):
        if core.shape[1] != core.shape[2]:
            raise ValueError(f"{name} の物理次元 (out, in) が一致しません: {tuple(core.shape[1:3])}")
    if left_mpo_core.shape[3] != right_mpo_core.shape[0]:
        raise ValueError(
            f"隣接する MPO core の bond が一致しません: {left_mpo_core.shape[3]} != {right_mpo_core.shape[0]}"
        )
    left_bond = left_environment.shape[0]
    right_bond = right_environment.shape[0]
    _validate_environment(
        left_environment, name="left_environment", mps_bond=left_bond, mpo_bond=left_mpo_core.shape[0]
    )
    _validate_environment(
        right_environment, name="right_environment", mps_bond=right_bond, mpo_bond=right_mpo_core.shape[3]
    )
    if input_tensor is not None:
        _validate_tensor(input_tensor, name="input_tensor", ndim=4)
        _validate_same_dtype_device(("left_environment", left_environment), ("input_tensor", input_tensor))
        expected = (left_bond, left_mpo_core.shape[2], right_mpo_core.shape[2], right_bond)
        if tuple(input_tensor.shape) != expected:
            raise ValueError(
                f"input_tensor の shape が一致しません: expected={expected}, actual={tuple(input_tensor.shape)}"
            )
