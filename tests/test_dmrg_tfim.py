"""``dmrg.models.tfim``（開境界 TFIM の MPO と dense Hamiltonian）のテスト。"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import nn_compression.dmrg.models.tfim as tfim_module
from _dmrg_testing import DTYPES, available_devices, dense_state, kron_all, random_mps, tolerances
from nn_compression.dmrg import (
    build_tfim_dense_hamiltonian,
    build_tfim_mpo,
    mpo_to_dense,
    mps_mpo_energy,
    run_two_site_dmrg,
)
from nn_compression.dmrg.models import build_tfim_dense_hamiltonian as models_dense
from nn_compression.dmrg.models import build_tfim_mpo as models_mpo


def _explicit_oracle(length, coupling, field, dtype=torch.float64):
    """site ごとの演算子を Kronecker 積で並べる、テスト内の独立 oracle。"""
    identity = torch.eye(2, dtype=dtype)
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    total = torch.zeros(2**length, 2**length, dtype=dtype)
    for site in range(length - 1):
        ops = [identity] * length
        ops[site], ops[site + 1] = pauli_z, pauli_z
        total = total - coupling * kron_all(ops)
    for site in range(length):
        ops = [identity] * length
        ops[site] = pauli_x
        total = total - field * kron_all(ops)
    return total


@pytest.mark.parametrize("length", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("coupling,field", [(1.0, 0.7), (0.0, 1.0), (1.3, 0.0), (-0.5, 2.0)])
def test_mpo_dense_and_oracle_agree_for_variable_length(length, coupling, field):
    """site 数は引数。MPO を dense 化したもの、dense builder、独立 oracle が一致する。"""
    mpo = build_tfim_mpo(length, coupling, field, dtype=torch.float64, device="cpu")
    dense = build_tfim_dense_hamiltonian(length, coupling, field, dtype=torch.float64, device="cpu")
    oracle = _explicit_oracle(length, coupling, field)
    assert len(mpo) == length
    torch.testing.assert_close(dense, oracle, **tolerances(torch.float64))
    torch.testing.assert_close(mpo_to_dense(mpo), oracle, **tolerances(torch.float64))
    torch.testing.assert_close(dense, dense.T)


def test_mpo_core_shapes_and_open_boundary():
    two = build_tfim_mpo(2, 1.0, 1.0, dtype=torch.float64, device="cpu")
    assert [tuple(core.shape) for core in two] == [(1, 2, 2, 3), (3, 2, 2, 1)]
    five = build_tfim_mpo(5, 1.0, 1.0, dtype=torch.float64, device="cpu")
    assert [tuple(core.shape) for core in five] == [(1, 2, 2, 3), *[(3, 2, 2, 3)] * 3, (3, 2, 2, 1)]
    # bulk core は独立した Tensor（1 つを書き換えても他に影響しない）
    five[1][0, 0, 0, 0] = 42.0
    assert five[2][0, 0, 0, 0].item() != 42.0


def test_two_site_ground_energy_has_closed_form_and_open_boundary():
    """open boundary の 2 site TFIM: E0 = -sqrt(J^2 + 4 g^2)。periodic の項は含まない。"""
    for coupling, field in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.6, 1.7)]:
        dense = build_tfim_dense_hamiltonian(2, coupling, field, dtype=torch.float64, device="cpu")
        assert torch.linalg.eigvalsh(dense)[0].item() == pytest.approx(-math.hypot(coupling, 2 * field), abs=1e-12)
    # 3 site で境界の結合 Z_0 Z_2 が無いこと（open）: field=0 の基底 energy は -2J
    dense = build_tfim_dense_hamiltonian(3, 1.0, 0.0, dtype=torch.float64, device="cpu")
    assert torch.linalg.eigvalsh(dense)[0].item() == pytest.approx(-2.0)


@pytest.mark.parametrize("length", [3, 6])
def test_mps_mpo_energy_with_tfim_matches_dense(length):
    mpo = build_tfim_mpo(length, 1.0, 0.8, dtype=torch.float64, device="cpu")
    dense = build_tfim_dense_hamiltonian(length, 1.0, 0.8, dtype=torch.float64, device="cpu")
    mps = random_mps([2] * length, 3, seed=2)
    psi = dense_state(mps)
    torch.testing.assert_close(
        mps_mpo_energy(mps, mpo), psi @ dense @ psi / (psi @ psi), **tolerances(torch.float64)
    )


def test_dmrg_on_tfim_reaches_exact_ground_energy():
    """TFIM は DMRG 本体の外側で作り、本体へ MPO として渡すだけ。"""
    length = 6
    mpo = build_tfim_mpo(length, 1.0, 0.9, dtype=torch.float64, device="cpu")
    dense = build_tfim_dense_hamiltonian(length, 1.0, 0.9, dtype=torch.float64, device="cpu")
    result = run_two_site_dmrg(
        random_mps([2] * length, 2, seed=3),
        mpo,
        max_bond=None,
        min_round_trips=2,
        max_round_trips=20,
        energy_tolerance=1e-11,
        required_consecutive=2,
        atol=1e-8,
        rtol=1e-8,
    )
    assert result.stop_reason == "converged"
    assert result.history[-1].energy == pytest.approx(torch.linalg.eigvalsh(dense)[0].item(), abs=1e-9)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dtype_and_device_follow_arguments(dtype, device):
    """(19)(20)(21)"""
    mpo = build_tfim_mpo(4, 1.0, 0.5, dtype=dtype, device=device)
    dense = build_tfim_dense_hamiltonian(4, 1.0, 0.5, dtype=dtype, device=device)
    assert all(core.dtype == dtype and core.device.type == device.type for core in mpo)
    assert dense.dtype == dtype and dense.device.type == device.type
    torch.testing.assert_close(mpo_to_dense(mpo), dense, **tolerances(dtype))
    # device は文字列でも torch.device でもよい
    assert build_tfim_mpo(2, 1.0, 1.0, dtype=dtype, device=str(device))[0].device.type == device.type


def test_no_module_level_tensors_defaults_or_fixed_dtype():
    """module-level に Pauli 行列・dtype・device の既定値を持たず、dtype / device は必須引数。"""
    for name, value in vars(tfim_module).items():
        assert not isinstance(value, (torch.Tensor, torch.device)), name
        if name.isupper():
            raise AssertionError(f"module-level の定数があります: {name}")
    for function in (build_tfim_mpo, build_tfim_dense_hamiltonian):
        parameters = inspect.signature(function).parameters
        assert parameters["dtype"].default is inspect.Parameter.empty
        assert parameters["device"].default is inspect.Parameter.empty
        assert parameters["dtype"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY


def test_models_subpackage_reexports_the_same_functions():
    assert models_mpo is build_tfim_mpo and models_dense is build_tfim_dense_hamiltonian


@pytest.mark.parametrize("builder", [build_tfim_mpo, build_tfim_dense_hamiltonian])
def test_invalid_arguments_are_rejected(builder):
    """(23)(24)"""
    good = {"dtype": torch.float64, "device": "cpu"}
    for length in (1, 0, -3):
        with pytest.raises(ValueError, match="length"):
            builder(length, 1.0, 1.0, **good)
    for length in (True, 2.0, "3", None):
        with pytest.raises(TypeError):
            builder(length, 1.0, 1.0, **good)
    for bad in (True, "1.0", None):
        with pytest.raises(TypeError):
            builder(3, bad, 1.0, **good)
        with pytest.raises(TypeError):
            builder(3, 1.0, bad, **good)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            builder(3, bad, 1.0, **good)
        with pytest.raises(ValueError):
            builder(3, 1.0, bad, **good)
    for dtype in (torch.complex128, torch.float16, torch.bfloat16, torch.int64, torch.bool):
        with pytest.raises(TypeError):
            builder(3, 1.0, 1.0, dtype=dtype, device="cpu")


def _assert_model_finite(model) -> None:
    tensors = model if isinstance(model, list) else [model]
    assert all(torch.isfinite(tensor).all() for tensor in tensors)


@pytest.mark.parametrize("builder", [build_tfim_mpo, build_tfim_dense_hamiltonian])
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_tfim_coefficients_stay_finite_in_the_target_dtype(builder, device):
    """対象 dtype へ変換したあとに非有限・零化する係数は、引数名付きで拒否する。"""
    with pytest.raises(ValueError, match="coupling"):
        builder(3, 1e100, 1.0, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="transverse_field"):
        builder(3, 1.0, 1e100, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="coupling"):
        builder(3, 1e39, 0.5, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="transverse_field"):
        builder(3, 0.5, -1e39, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="coupling"):
        builder(3, 1e-46, 1.0, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="transverse_field"):
        builder(3, 1.0, 1e-50, dtype=torch.float32, device=device)
    # Python float がすでに inf になる float64 の上限の外側。
    with pytest.raises(ValueError, match="coupling"):
        builder(3, 1e309, 1.0, dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="transverse_field"):
        builder(3, 1.0, -1e309, dtype=torch.float64, device=device)

    _assert_model_finite(builder(3, 0.0, 0.0, dtype=torch.float32, device=device))
    _assert_model_finite(builder(3, -0.0, 0.0, dtype=torch.float64, device=device))
    _assert_model_finite(builder(3, 1.0, 0.7, dtype=torch.float64, device=device))
    _assert_model_finite(builder(3, 1e38, -0.5, dtype=torch.float32, device=device))
    _assert_model_finite(builder(3, -0.25, 1e-45, dtype=torch.float32, device=device))
    # 1e300 は float64 で有限で、長さ 3 の結合を足しても overflow しない。1e-320 は非零のまま残る。
    _assert_model_finite(builder(3, 1e300, 1e-320, dtype=torch.float64, device=device))
