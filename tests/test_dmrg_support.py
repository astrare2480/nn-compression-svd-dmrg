"""DMRG の実験支援（初期 MPS・dense 診断）と、本体と支援・模型固有処理の分離のテスト。"""

from __future__ import annotations

import ast
import copy
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import nn_compression.dmrg as dmrg
from _dmrg_testing import (
    DTYPES,
    available_devices,
    dense_state,
    random_chain_hamiltonian,
    random_mps,
    tolerances,
)
from nn_compression.dmrg import (
    ResidualVarianceDiagnostics,
    compute_dense_residual_and_variance,
    make_random_mps,
    mps_mpo_energy,
    mps_mpo_expectation,
    mpo_to_dense,
)

SRC_DMRG = Path(dmrg.__file__).resolve().parent


# ---- make_random_mps ------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,max_bond",
    [([2, 2, 2, 2], 3), ([3, 3, 3], 2), ([5, 4, 5, 2], 7), ([2, 3, 4, 2, 3], 1), ([4], 3), ([2, 2], 100)],
)
def test_make_random_mps_shapes_and_bond_cap(dims, max_bond):
    """(16)(17)(18) 任意の物理次元列・開境界・bond 上限。"""
    mps = make_random_mps(dims, max_bond, generator=torch.Generator().manual_seed(0), dtype=torch.float64, device="cpu")
    assert [core.shape[1] for core in mps] == dims
    assert mps[0].shape[0] == 1 and mps[-1].shape[2] == 1
    for left, right in zip(mps, mps[1:]):
        assert left.shape[2] == right.shape[0] <= max_bond
    for cut in range(1, len(dims)):
        capacity = min(torch.tensor(dims[:cut]).prod().item(), torch.tensor(dims[cut:]).prod().item())
        assert mps[cut].shape[0] == min(max_bond, capacity)


def test_make_random_mps_is_reproducible_and_does_not_touch_global_rng():
    """(25) 同じ generator 状態から同じ core 列。global RNG には触れない。"""
    torch.manual_seed(123)
    global_state = torch.get_rng_state()
    first = make_random_mps([2, 3, 2], 2, generator=torch.Generator().manual_seed(7), dtype=torch.float64, device="cpu")
    second = make_random_mps([2, 3, 2], 2, generator=torch.Generator().manual_seed(7), dtype=torch.float64, device="cpu")
    other = make_random_mps([2, 3, 2], 2, generator=torch.Generator().manual_seed(8), dtype=torch.float64, device="cpu")
    assert torch.equal(torch.get_rng_state(), global_state)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
    # 同じ generator を続けて使うと、状態が進んで別の値になる
    generator = torch.Generator().manual_seed(7)
    a = make_random_mps([2, 3, 2], 2, generator=generator, dtype=torch.float64, device="cpu")
    b = make_random_mps([2, 3, 2], 2, generator=generator, dtype=torch.float64, device="cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, first))
    assert not all(torch.equal(x, y) for x, y in zip(a, b))


def test_make_random_mps_does_not_normalize():
    mps = random_mps([3, 3, 3, 3], 3, seed=1)
    norm_squared = dense_state(mps).square().sum().item()
    assert abs(norm_squared - 1.0) > 1e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_make_random_mps_dtype_device(dtype, device):
    """(19)(20)(21)"""
    for core in random_mps([2, 3, 2], 2, dtype=dtype, device=device):
        assert core.dtype == dtype and core.device.type == device.type


def test_make_random_mps_rejects_invalid_arguments():
    """(23)(24)"""
    generator = torch.Generator().manual_seed(0)
    base = {"generator": generator, "dtype": torch.float64, "device": "cpu"}
    for bad in (True, False, 0, -1, 1.5, "2", None):
        with pytest.raises((TypeError, ValueError)):
            make_random_mps([2, 2], bad, **base)
    for bad_dims in ([], [2, 0], [2, -1], [2, True], [2, 2.0], "22", None, 3):
        with pytest.raises((TypeError, ValueError)):
            make_random_mps(bad_dims, 2, **base)
    with pytest.raises(TypeError, match="Generator"):
        make_random_mps([2, 2], 2, generator=0, dtype=torch.float64, device="cpu")
    for dtype in (torch.complex128, torch.float16, torch.bfloat16, torch.int64, torch.bool):
        with pytest.raises(TypeError):
            make_random_mps([2, 2], 2, generator=generator, dtype=dtype, device="cpu")
    with pytest.raises(ValueError, match="device"):
        make_random_mps([2, 2], 2, generator=generator, dtype=torch.float64, device="meta")


# ---- mpo_to_dense -----------------------------------------------------------------------


@pytest.mark.parametrize("dims", [[2, 2, 2], [3, 3, 3], [5, 4], [2, 3, 4, 2], [3]])
def test_mpo_to_dense_matches_independent_kron_construction(dims):
    """(16)(18) TFIM に限らない一般 MPO の dense 化。"""
    mpo, dense = random_chain_hamiltonian(dims, seed=2)
    torch.testing.assert_close(mpo_to_dense(mpo), dense, **tolerances(torch.float64))


def test_mpo_to_dense_handles_rectangular_site_operators():
    generator = torch.Generator().manual_seed(3)
    first = torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
    second = torch.randn(4, 3, 2, 1, generator=generator, dtype=torch.float64)
    dense = mpo_to_dense([first, second])
    assert dense.shape == (6, 6)
    expected = torch.einsum("oiw,pjw->opij", first[0], second[:, :, :, 0].permute(1, 2, 0)).reshape(6, 6)
    torch.testing.assert_close(dense, expected)
    third = torch.randn(1, 2, 5, 1, generator=generator, dtype=torch.float64)
    assert mpo_to_dense([third]).shape == (2, 5)


@pytest.mark.parametrize("dims", [[2, 2, 2], [3, 2, 4]])
def test_mpo_to_dense_agrees_with_mps_mpo_contraction(dims):
    """(27)"""
    mps = random_mps(dims, 3, seed=4)
    mpo, _ = random_chain_hamiltonian(dims, seed=5)
    psi = dense_state(mps)
    torch.testing.assert_close(
        psi @ mpo_to_dense(mpo) @ psi, mps_mpo_expectation(mps, mpo), **tolerances(torch.float64)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_mpo_to_dense_dtype_device_and_no_mutation(dtype, device):
    """(21)(22)"""
    mpo, dense = random_chain_hamiltonian([2, 3, 2], dtype=dtype, device=device)
    before = copy.deepcopy(mpo)
    result = mpo_to_dense(mpo)
    assert result.dtype == dtype and result.device.type == device.type
    torch.testing.assert_close(result, dense, **tolerances(dtype))
    assert all(torch.equal(a, b) for a, b in zip(mpo, before))


def test_mpo_to_dense_backward_is_finite_and_rejects_invalid_input():
    """(23)(26)"""
    mpo, _ = random_chain_hamiltonian([2, 3, 2])
    leaves = [core.clone().requires_grad_(True) for core in mpo]
    mpo_to_dense(leaves).square().sum().backward()
    assert all(torch.isfinite(core.grad).all() for core in leaves)

    with pytest.raises(ValueError):
        mpo_to_dense([])
    with pytest.raises(ValueError, match="4階"):
        mpo_to_dense([mpo[0][0], *mpo[1:]])
    with pytest.raises(ValueError, match="left bond"):
        mpo_to_dense(mpo[1:])
    with pytest.raises(ValueError, match="right bond"):
        mpo_to_dense(mpo[:-1])
    with pytest.raises(ValueError, match="bond"):
        mpo_to_dense([mpo[0], mpo[1][:2], mpo[2]])
    with pytest.raises(TypeError):
        mpo_to_dense([core.to(torch.complex128) for core in mpo])
    with pytest.raises(TypeError, match="dtype"):
        mpo_to_dense([mpo[0], mpo[1].float(), mpo[2]])
    bad = [core.clone() for core in mpo]
    bad[1][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN / inf"):
        mpo_to_dense(bad)


# ---- compute_dense_residual_and_variance ------------------------------------------------


def _symmetric(size, seed, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return ((raw + raw.T) / 2).to(dtype=dtype, device=device)


def test_dense_residual_squared_equals_energy_variance_and_definitions():
    """(28)"""
    hamiltonian = _symmetric(12, 1)
    generator = torch.Generator().manual_seed(2)
    state = torch.randn(12, generator=generator, dtype=torch.float64)
    result = compute_dense_residual_and_variance(state, hamiltonian)
    assert isinstance(result, ResidualVarianceDiagnostics)
    normalized = state / state.norm()
    energy = normalized @ hamiltonian @ normalized
    torch.testing.assert_close(result.energy, energy)
    torch.testing.assert_close(result.residual_norm, (hamiltonian @ normalized - energy * normalized).norm())
    torch.testing.assert_close(
        result.energy_variance, normalized @ hamiltonian @ hamiltonian @ normalized - energy**2, atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(result.residual_norm.square(), result.energy_variance, atol=1e-10, rtol=1e-9)
    for tensor in (result.energy, result.residual_norm, result.energy_variance):
        assert tensor.shape == ()


def test_dense_residual_is_scale_invariant_and_reshapes_state():
    """非正規化入力・多次元入力でも同じ値になる。"""
    hamiltonian = _symmetric(12, 3)
    state = torch.randn(12, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    reference = compute_dense_residual_and_variance(state, hamiltonian)
    scaled = compute_dense_residual_and_variance(1e3 * state, hamiltonian)
    reshaped = compute_dense_residual_and_variance(state.reshape(3, 4), hamiltonian)
    for other in (scaled, reshaped):
        torch.testing.assert_close(other.energy, reference.energy)
        torch.testing.assert_close(other.residual_norm, reference.residual_norm)
        torch.testing.assert_close(other.energy_variance, reference.energy_variance, atol=1e-9, rtol=1e-9)


def test_dense_residual_is_zero_for_eigenstate_and_variance_is_clamped_nonnegative():
    hamiltonian = _symmetric(10, 5)
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    for dtype in DTYPES:
        result = compute_dense_residual_and_variance(
            eigenvectors[:, 3].to(dtype), hamiltonian.to(dtype)
        )
        assert result.residual_norm.item() < tolerances(dtype)["atol"]
        assert result.energy_variance.item() >= 0.0
        torch.testing.assert_close(result.energy.double(), eigenvalues[3], atol=1e-4, rtol=1e-4)


def test_dense_residual_matches_mps_mpo_energy():
    dims = [2, 3, 2]
    mps = random_mps(dims, 2, seed=6)
    mpo, dense = random_chain_hamiltonian(dims, seed=7)
    result = compute_dense_residual_and_variance(dense_state(mps), mpo_to_dense(mpo))
    torch.testing.assert_close(result.energy, mps_mpo_energy(mps, mpo), **tolerances(torch.float64))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dense_residual_dtype_device(dtype, device):
    """(19)(20)(21)"""
    hamiltonian = _symmetric(8, 8, dtype, device)
    state = torch.randn(8, generator=torch.Generator().manual_seed(9), dtype=torch.float64).to(dtype=dtype, device=device)
    result = compute_dense_residual_and_variance(state, hamiltonian)
    for tensor in (result.energy, result.residual_norm, result.energy_variance):
        assert tensor.dtype == dtype and tensor.device.type == device.type
    torch.testing.assert_close(result.residual_norm.square(), result.energy_variance, atol=1e-3 if dtype == torch.float32 else 1e-9, rtol=1e-3 if dtype == torch.float32 else 1e-8)


def test_dense_diagnostics_are_invariant_to_global_scale():
    """状態全体のスカラー倍で energy・正規化残差・分散は変わらない。"""
    generator = torch.Generator().manual_seed(0)
    base_state = torch.randn(8, generator=generator, dtype=torch.float64)
    base_hamiltonian = _symmetric(8, 1)
    cases = (
        (torch.float32, (1.0, 1e30, 1e-30, -1.0, -1e30, -1e-30)),
        (torch.float64, (1.0, 1e200, 1e-200, -1.0, -1e200, -1e-200)),
    )
    for dtype, scales in cases:
        state = base_state.to(dtype=dtype)
        hamiltonian = base_hamiltonian.to(dtype=dtype)
        reference = compute_dense_residual_and_variance(state, hamiltonian)
        close = {"atol": 1e-4, "rtol": 1e-4} if dtype == torch.float32 else {"atol": 1e-8, "rtol": 1e-8}
        for scale in scales:
            scaled = state * scale
            state_before, hamiltonian_before = scaled.clone(), hamiltonian.clone()
            result = compute_dense_residual_and_variance(scaled, hamiltonian)
            assert torch.equal(scaled, state_before) and torch.equal(hamiltonian, hamiltonian_before)
            for name in ("energy", "residual_norm", "energy_variance"):
                value = getattr(result, name)
                assert value.dtype == dtype and torch.isfinite(value).all()
                torch.testing.assert_close(value, getattr(reference, name), **close)
    with pytest.raises(ValueError, match="ノルム"):
        compute_dense_residual_and_variance(torch.zeros(8, dtype=torch.float32), torch.eye(8))
    tiny = torch.zeros(8, dtype=torch.float32)
    tiny[0] = 1e-30
    tiny_result = compute_dense_residual_and_variance(tiny, torch.eye(8))
    assert torch.isfinite(tiny_result.energy) and tiny_result.residual_norm.item() == pytest.approx(0.0, abs=1e-5)


def test_dense_residual_does_not_mutate_inputs_and_rejects_invalid_input():
    """(22)(23)"""
    hamiltonian = _symmetric(6, 10)
    state = torch.randn(6, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    before = (hamiltonian.clone(), state.clone())
    compute_dense_residual_and_variance(state, hamiltonian)
    assert torch.equal(hamiltonian, before[0]) and torch.equal(state, before[1])

    with pytest.raises(ValueError, match="正方"):
        compute_dense_residual_and_variance(state, torch.ones(6, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="2階"):
        compute_dense_residual_and_variance(state, torch.ones(6, dtype=torch.float64))
    with pytest.raises(ValueError, match="一致しません"):
        compute_dense_residual_and_variance(state[:5], hamiltonian)
    with pytest.raises(TypeError, match="dtype"):
        compute_dense_residual_and_variance(state, hamiltonian.float())
    with pytest.raises(ValueError, match="device"):
        compute_dense_residual_and_variance(state, hamiltonian.to("meta"))
    with pytest.raises(ValueError, match="ノルム"):
        compute_dense_residual_and_variance(torch.zeros(6, dtype=torch.float64), hamiltonian)
    with pytest.raises(TypeError):
        compute_dense_residual_and_variance(state.to(torch.complex128), hamiltonian.to(torch.complex128))
    with pytest.raises(TypeError):
        compute_dense_residual_and_variance([1.0] * 6, hamiltonian)
    bad = state.clone()
    bad[0] = float("inf")
    with pytest.raises(ValueError, match="NaN / inf"):
        compute_dense_residual_and_variance(bad, hamiltonian)


# ---- 本体と支援・模型固有処理の分離 -------------------------------------------------------

CORE_MODULES = (
    "validation",
    "environment",
    "effective_hamiltonian",
    "one_site",
    "two_site",
    "sweep",
    "convergence",
)
FORBIDDEN_MODULE_PARTS = {"models", "tfim", "diagnostics", "initialization"}
FORBIDDEN_NAMES = {
    "build_tfim_mpo",
    "build_tfim_dense_hamiltonian",
    "mpo_to_dense",
    "compute_dense_residual_and_variance",
    "ResidualVarianceDiagnostics",
    "make_random_mps",
    "tt_reconstruct",
    "DTYPE",
    "DEVICE",
    "LENGTH",
    "H_DENSE",
}


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_modules_do_not_depend_on_tfim_dense_or_support_code(module_name):
    """(29) 本体 module は TFIM・dense 診断・初期化・dense 再構成・notebook 定数を参照しない。"""
    tree = ast.parse((SRC_DMRG / f"{module_name}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parts = set((node.module or "").split("."))
            assert not parts & FORBIDDEN_MODULE_PARTS, f"{module_name}: {node.module}"
            for alias in node.names:
                assert alias.name not in FORBIDDEN_NAMES, f"{module_name}: {alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not set(alias.name.split(".")) & FORBIDDEN_MODULE_PARTS
        elif isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, f"{module_name}: {node.id}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_NAMES, f"{module_name}: {node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "manual_seed", f"{module_name}: module-level seed"


def test_core_modules_do_not_hardcode_two_level_systems_or_dtype_device_defaults():
    """物理次元 2・固定 site 数・module-level の dtype / device 既定値を持たない。"""
    for module_name in (*CORE_MODULES, "initialization", "diagnostics"):
        tree = ast.parse((SRC_DMRG / f"{module_name}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id not in {"DTYPE", "DEVICE", "LENGTH", "SEED", "NUM_SITES"}
        source = (SRC_DMRG / f"{module_name}.py").read_text(encoding="utf-8")
        assert "torch.set_default" not in source
        assert "manual_seed" not in source


def test_public_api_boundary_is_exactly_the_documented_list():
    """(Public API) __all__ は指定された一覧と一致し、補助関数は公開しない。"""
    expected = {
        "update_left_environment", "update_right_environment", "build_left_environments",
        "build_right_environments", "mps_mpo_expectation", "mps_mpo_energy", "normalize_mps",
        "build_one_site_effective_hamiltonian", "apply_one_site_effective_hamiltonian",
        "update_one_site_core", "merge_two_sites", "build_two_site_effective_hamiltonian",
        "apply_two_site_effective_hamiltonian", "split_two_site_tensor", "update_two_site_pair",
        "two_site_sweep", "run_two_site_dmrg_round_trip", "run_two_site_dmrg",
        "OneSiteUpdateResult", "TwoSiteSplitResult", "TwoSiteUpdateResult", "SweepStepLog",
        "SweepPassResult", "DMRGRoundTripResult", "DMRGRoundTripRecord", "DMRGResult",
        "make_random_mps", "mpo_to_dense", "compute_dense_residual_and_variance",
        "ResidualVarianceDiagnostics", "build_tfim_mpo", "build_tfim_dense_hamiltonian",
    }  # fmt: skip
    assert set(dmrg.__all__) == expected and len(dmrg.__all__) == len(expected)
    for name in dmrg.__all__:
        assert hasattr(dmrg, name)
    for private in ("_lowest_eigenpair", "_next_stability_count", "_choose_stop_reason", "_directional_sweep", "_kron_all"):
        assert private not in dmrg.__all__


def test_every_top_level_definition_outside_all_is_private():
    """docs-sync 規約: ``__all__`` 外の top-level 関数・クラスは ``_`` 始まり。"""
    for source_path in sorted(SRC_DMRG.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                assert node.name in dmrg.__all__, f"{source_path.name}: {node.name}"


@pytest.mark.parametrize(
    "module_name",
    ["validation", "environment", "two_site", "sweep", "convergence", "diagnostics", "initialization", "models.tfim"],
)
def test_each_module_imports_standalone_without_circular_import(module_name):
    """循環 import がない: 各 module を単独で import できる。"""
    environment = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    completed = subprocess.run(
        [sys.executable, "-c", f"import nn_compression.dmrg.{module_name}"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    importlib.import_module(f"nn_compression.dmrg.{module_name}")


def test_make_random_mps_cpu_generator_matches_cpu_output_and_seed():
    """CPU generator と CPU 出力は従来どおり、同じ seed で同じ MPS になる。"""
    first = make_random_mps(
        [2, 3, 2], 2, generator=torch.Generator().manual_seed(11), dtype=torch.float64, device="cpu"
    )
    second = make_random_mps(
        [2, 3, 2], 2, generator=torch.Generator(device="cpu").manual_seed(11), dtype=torch.float64, device=torch.device("cpu")
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert all(core.device.type == "cpu" for core in first)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="CUDAが必要")
def test_make_random_mps_accepts_matching_cuda_device():
    """generator と出力先の CUDA index が一致すれば生成できる。"""
    generator = torch.Generator(device="cuda:0").manual_seed(5)
    cores = make_random_mps([2, 3], 2, generator=generator, dtype=torch.float32, device="cuda:0")
    repeated = make_random_mps(
        [2, 3], 2, generator=torch.Generator(device="cuda:0").manual_seed(5), dtype=torch.float32, device="cuda:0"
    )
    assert all(core.device == torch.device("cuda:0") for core in cores)
    assert all(torch.equal(left, right) for left, right in zip(cores, repeated))
    current = make_random_mps(
        [2, 2], 1, generator=torch.Generator(device="cuda").manual_seed(1), dtype=torch.float32, device="cuda"
    )
    assert current[0].device.type == "cuda"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="CUDA device が2台必要")
def test_make_random_mps_rejects_cuda_device_index_mismatch():
    """``cuda:0`` の generator で ``cuda:1`` へは出さない。``torch.randn`` の前に拒否する。"""
    generator = torch.Generator(device="cuda:0").manual_seed(0)
    with pytest.raises(ValueError, match=r"cuda:0"):
        make_random_mps([2, 2], 2, generator=generator, dtype=torch.float32, device="cuda:1")
    allowed = make_random_mps(
        [2, 2],
        2,
        generator=torch.Generator(device="cuda:1").manual_seed(3),
        dtype=torch.float32,
        device="cuda:1",
    )
    assert all(core.device == torch.device("cuda:1") for core in allowed)
