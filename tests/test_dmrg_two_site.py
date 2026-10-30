"""``dmrg.two_site``（結合・SVD 分割・2-site 局所更新）のテスト。"""

from __future__ import annotations

import copy

import pytest
import torch

from _dmrg_testing import (
    DTYPES,
    available_devices,
    projection_to_full_space,
    random_chain_hamiltonian,
    random_mps,
    sweep_tolerances,
    tolerances,
)
from nn_compression.compression import tt_canonicalize
from nn_compression.dmrg import (
    TwoSiteSplitResult,
    TwoSiteUpdateResult,
    build_left_environments,
    build_right_environments,
    merge_two_sites,
    mps_mpo_energy,
    split_two_site_tensor,
    update_two_site_pair,
)

DIRECTIONS = ("left_to_right", "right_to_left")
SHAPES = {
    "d2": (2, 2, 2, 2),
    "d3": (3, 3, 3, 3),
    "d5": (2, 5, 4, 3),
    "rank1": (1, 3, 2, 1),
    "wide_bonds": (4, 2, 3, 5),
}


def _random_theta(shape, *, seed=0, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype=dtype, device=device)


def _gram_errors(split: TwoSiteSplitResult):
    left = split.left_core
    right = split.right_core
    left_matrix = left.reshape(-1, left.shape[2])
    right_matrix = right.reshape(right.shape[0], -1)
    left_error = (left_matrix.T @ left_matrix - torch.eye(left_matrix.shape[1], dtype=left.dtype)).abs().max()
    right_error = (right_matrix @ right_matrix.T - torch.eye(right_matrix.shape[0], dtype=right.dtype)).abs().max()
    return left_error.item(), right_error.item()


def test_merge_two_sites_matches_tensordot_and_validates():
    left = _random_theta((2, 3, 4), seed=1)
    right = _random_theta((4, 2, 5), seed=2)
    theta = merge_two_sites(left, right)
    assert theta.shape == (2, 3, 2, 5)
    torch.testing.assert_close(theta, torch.tensordot(left, right, dims=([2], [0])))
    with pytest.raises(ValueError, match="bond"):
        merge_two_sites(left, right[:3])
    with pytest.raises(TypeError, match="dtype"):
        merge_two_sites(left, right.float())
    with pytest.raises(ValueError, match="device"):
        merge_two_sites(left, right.to("meta"))
    with pytest.raises(ValueError, match="3階"):
        merge_two_sites(left[0], right)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_exact_split_restores_normalized_tensor(shape, direction):
    """(8)(16)(17) max_bond=None で単位ノルムの Theta を復元する。"""
    theta = _random_theta(shape)
    theta = theta / torch.linalg.vector_norm(theta)
    split = split_two_site_tensor(theta, direction=direction, max_bond=None)
    restored = merge_two_sites(split.left_core, split.right_core)
    torch.testing.assert_close(restored, theta, **tolerances(torch.float64))
    assert split.retained_rank == min(shape[0] * shape[1], shape[2] * shape[3])
    assert split.left_core.shape == (shape[0], shape[1], split.retained_rank)
    assert split.right_core.shape == (split.retained_rank, shape[2], shape[3])
    assert split.discarded_weight.item() == 0.0
    assert split.discarded_squared_norm.item() == 0.0
    torch.testing.assert_close(split.retained_norm, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(split.truncated_tensor_before_normalization, theta, **tolerances(torch.float64))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_split_places_singular_values_on_the_correct_side(direction):
    """右向き: 左 core が左正準、右向き: 右 core が右正準。"""
    theta = _random_theta((3, 2, 3, 4))
    split = split_two_site_tensor(theta, direction=direction, max_bond=None)
    left_error, right_error = _gram_errors(split)
    if direction == "left_to_right":
        assert left_error < 1e-12
        assert right_error > 1e-3
    else:
        assert right_error < 1e-12
        assert left_error > 1e-3


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("max_bond", [1, 2, 3, 5])
def test_truncated_split_rank_cap_and_discarded_norm(direction, max_bond):
    """(9) 保持 rank の上限と捨てたノルムの定義。"""
    theta = _random_theta((3, 3, 3, 3), seed=5)
    split = split_two_site_tensor(theta, direction=direction, max_bond=max_bond)
    singular = torch.linalg.svdvals(theta.reshape(9, 9))
    rank = min(max_bond, 9)
    assert split.retained_rank == rank
    assert isinstance(split.retained_rank, int)
    assert split.left_core.shape[2] == split.right_core.shape[0] == rank
    torch.testing.assert_close(split.singular_values, singular)
    assert split.singular_values.shape == (9,)
    discarded = (singular[rank:] ** 2).sum()
    torch.testing.assert_close(split.discarded_squared_norm, discarded)
    torch.testing.assert_close(split.discarded_weight, discarded / (singular**2).sum())
    torch.testing.assert_close(split.retained_norm, (singular[:rank] ** 2).sum().sqrt())
    truncated = split.truncated_tensor_before_normalization
    torch.testing.assert_close(torch.linalg.vector_norm(truncated), split.retained_norm)
    assert torch.linalg.matrix_rank(truncated.reshape(9, 9)).item() == rank
    # 規約: 再結合は常に単位ノルムで、truncated / retained_norm と一致する。
    restored = merge_two_sites(split.left_core, split.right_core)
    torch.testing.assert_close(torch.linalg.vector_norm(restored), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(restored, truncated / split.retained_norm, **tolerances(torch.float64))
    # 切り捨て分は Eckart-Young の最良近似
    torch.testing.assert_close(
        torch.linalg.vector_norm(theta - truncated) ** 2, discarded, **tolerances(torch.float64)
    )


def test_split_of_unnormalized_tensor_follows_documented_convention():
    theta = 5.0 * _random_theta((2, 3, 3, 2), seed=6)
    split = split_two_site_tensor(theta, direction="left_to_right", max_bond=None)
    torch.testing.assert_close(split.truncated_tensor_before_normalization, theta, **tolerances(torch.float64))
    torch.testing.assert_close(split.retained_norm, torch.linalg.vector_norm(theta))
    restored = merge_two_sites(split.left_core, split.right_core)
    torch.testing.assert_close(restored, theta / torch.linalg.vector_norm(theta), **tolerances(torch.float64))


def test_split_max_bond_larger_than_available_rank_keeps_everything():
    theta = _random_theta((2, 2, 2, 2))
    split = split_two_site_tensor(theta, direction="left_to_right", max_bond=100)
    assert split.retained_rank == 4
    assert split.discarded_weight.item() == 0.0


def test_split_does_not_mutate_input():
    """(22)"""
    theta = _random_theta((3, 2, 2, 3))
    before = theta.clone()
    split_two_site_tensor(theta, direction="right_to_left", max_bond=2)
    assert torch.equal(theta, before)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, 2.0, "2", torch.tensor(2)])
def test_split_rejects_invalid_max_bond(bad):
    """(23)(24) bool を整数として受け付けない。"""
    theta = _random_theta((2, 2, 2, 2))
    with pytest.raises((TypeError, ValueError)):
        split_two_site_tensor(theta, direction="left_to_right", max_bond=bad)


def test_split_rejects_invalid_inputs():
    theta = _random_theta((2, 2, 2, 2))
    for direction in ("left", "right", "LEFT_TO_RIGHT", None, 0):
        with pytest.raises((TypeError, ValueError)):
            split_two_site_tensor(theta, direction=direction, max_bond=None)
    with pytest.raises(ValueError, match="4階"):
        split_two_site_tensor(theta[0], direction="left_to_right", max_bond=None)
    with pytest.raises(ValueError, match="ノルム"):
        split_two_site_tensor(torch.zeros_like(theta), direction="left_to_right", max_bond=None)
    bad = theta.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN / inf"):
        split_two_site_tensor(bad, direction="left_to_right", max_bond=None)
    for dtype in (torch.complex128, torch.float16, torch.bfloat16, torch.int64, torch.bool):
        with pytest.raises(TypeError):
            split_two_site_tensor(torch.ones(2, 2, 2, 2, dtype=dtype), direction="left_to_right", max_bond=None)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_split_dtype_device_preserved(dtype, device):
    """(19)(20)(21)"""
    theta = _random_theta((3, 2, 2, 3), dtype=dtype, device=device)
    theta = theta / torch.linalg.vector_norm(theta)
    split = split_two_site_tensor(theta, direction="left_to_right", max_bond=None)
    for tensor in (
        split.left_core,
        split.right_core,
        split.singular_values,
        split.discarded_squared_norm,
        split.discarded_weight,
        split.retained_norm,
        split.truncated_tensor_before_normalization,
    ):
        assert tensor.dtype == dtype and tensor.device.type == device.type
    torch.testing.assert_close(merge_two_sites(split.left_core, split.right_core), theta, **tolerances(dtype))


def test_split_backward_is_finite():
    """(26)"""
    theta = _random_theta((3, 2, 2, 3), seed=9).requires_grad_(True)
    split = split_two_site_tensor(theta, direction="left_to_right", max_bond=3)
    (split.right_core.square().sum() + split.discarded_weight + split.retained_norm).backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()


def _pair_setup(dims, max_bond, pair, *, dtype=torch.float64, device="cpu"):
    mpo, dense = random_chain_hamiltonian(dims, seed=11, dtype=dtype, device=device)
    mps = tt_canonicalize(random_mps(dims, max_bond, seed=12, dtype=dtype, device=device), pair)
    left = build_left_environments(mps, mpo)
    right = build_right_environments(mps, mpo)
    return mps, mpo, dense, left, right


CASES = {
    "d2": ([2, 2, 2, 2], 2),
    "d3": ([3, 3, 3, 3], 3),
    "mixed_rank1": ([2, 3, 4, 2, 3], 1),
    "mixed_rank3": ([2, 3, 4, 2, 3], 3),
}


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_untruncated_update_equals_dense_local_ground_state(dims, max_bond, direction):
    """(7)(10)(11)(16)(17)(18) max_bond=None の局所更新は P^T H P の基底状態に一致する。"""
    for pair in range(len(dims) - 1):
        mps, mpo, dense, left, right = _pair_setup(dims, max_bond, pair)
        result = update_two_site_pair(
            mps, mpo, pair, left[pair], right[pair + 2], direction=direction, max_bond=None
        )
        assert isinstance(result, TwoSiteUpdateResult) and isinstance(result.split, TwoSiteSplitResult)
        assert result.pair_start == pair and result.direction == direction

        projection = projection_to_full_space(mps, pair, 2)
        expected = torch.linalg.eigvalsh(projection.T @ dense @ projection)[0]
        torch.testing.assert_close(result.local_minimum_energy, expected, **tolerances(torch.float64))
        assert result.local_residual.item() <= 1e-10
        # 局所 energy は更新前以下、環境が正準なので更新後の全体 energy は局所最小値に一致する。
        before_global = mps_mpo_energy(mps, mpo)
        after_global = mps_mpo_energy(result.cores, mpo)
        torch.testing.assert_close(result.local_energy_before, before_global, **tolerances(torch.float64))
        assert result.local_minimum_energy.item() <= result.local_energy_before.item() + 1e-12
        assert after_global.item() <= before_global.item() + 1e-10
        torch.testing.assert_close(after_global, result.local_minimum_energy, **tolerances(torch.float64))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_truncated_update_respects_bond_cap_and_reports_discarded_weight(direction):
    dims = [3, 3, 3, 3]
    pair = 1
    mps, mpo, dense, left, right = _pair_setup(dims, 4, pair)
    result = update_two_site_pair(mps, mpo, pair, left[pair], right[pair + 2], direction=direction, max_bond=2)
    assert result.split.retained_rank <= 2
    assert result.cores[pair].shape[2] == result.cores[pair + 1].shape[0] <= 2
    # 切り捨て後 energy は切り捨て前の局所最小値以上
    assert mps_mpo_energy(result.cores, mpo).item() >= result.local_minimum_energy.item() - 1e-10
    assert result.split.discarded_weight.item() >= 0.0


def test_update_replaces_only_the_pair_and_does_not_mutate_inputs():
    """(22)"""
    dims = [2, 3, 2, 3, 2]
    pair = 2
    mps, mpo, _, left, right = _pair_setup(dims, 2, pair)
    operands = [*mps, *mpo, left[pair], right[pair + 2]]
    before = copy.deepcopy(operands)
    result = update_two_site_pair(
        mps, mpo, pair, left[pair], right[pair + 2], direction="left_to_right", max_bond=None
    )
    for a, b in zip(operands, before):
        assert torch.equal(a, b)
    for index, (new, old) in enumerate(zip(result.cores, mps)):
        if index not in (pair, pair + 1):
            assert torch.equal(new, old)
            assert new.data_ptr() != old.data_ptr()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_update_dtype_device_preserved(dtype, device):
    """(19)(20)(21)"""
    pair = 1
    mps, mpo, dense, left, right = _pair_setup([2, 3, 2, 3], 2, pair, dtype=dtype, device=device)
    result = update_two_site_pair(
        mps, mpo, pair, left[pair], right[pair + 2], direction="right_to_left", max_bond=None
    )
    for tensor in (*result.cores, result.local_energy_before, result.local_minimum_energy, result.local_residual):
        assert tensor.dtype == dtype and tensor.device.type == device.type
    projection = projection_to_full_space(mps, pair, 2)
    expected = torch.linalg.eigvalsh(projection.T @ dense @ projection)[0]
    torch.testing.assert_close(result.local_minimum_energy, expected, **tolerances(dtype))
    assert result.local_residual.item() <= sweep_tolerances(dtype)["atol"]


def test_update_rejects_invalid_arguments():
    """(23)(24)"""
    mps, mpo, _, left, right = _pair_setup([2, 3, 2, 3], 2, 1)
    kwargs = {"direction": "left_to_right", "max_bond": None}
    for pair in (True, 1.0):
        with pytest.raises(TypeError):
            update_two_site_pair(mps, mpo, pair, left[1], right[3], **kwargs)
    for pair in (-1, 3, 4):
        with pytest.raises(ValueError, match="pair_start"):
            update_two_site_pair(mps, mpo, pair, left[1], right[3], **kwargs)
    with pytest.raises(ValueError, match="direction"):
        update_two_site_pair(mps, mpo, 1, left[1], right[3], direction="left", max_bond=None)
    for bad in (True, 0, 1.5):
        with pytest.raises((TypeError, ValueError)):
            update_two_site_pair(mps, mpo, 1, left[1], right[3], direction="left_to_right", max_bond=bad)
    with pytest.raises(ValueError, match="shape"):
        update_two_site_pair(mps, mpo, 1, left[0], right[3], **kwargs)
    with pytest.raises(ValueError, match="shape"):
        update_two_site_pair(mps, mpo, 1, left[1], right[4], **kwargs)
    with pytest.raises(TypeError, match="dtype"):
        update_two_site_pair(mps, mpo, 1, left[1].float(), right[3], **kwargs)
    with pytest.raises(ValueError, match="device"):
        update_two_site_pair(mps, mpo, 1, left[1], right[3].to("meta"), **kwargs)
    with pytest.raises(ValueError, match="site 数"):
        update_two_site_pair(mps[:-1], mpo, 1, left[1], right[3], **kwargs)
    single_mps = [torch.ones(1, 2, 1, dtype=torch.float64)]
    single_mpo = [torch.ones(1, 2, 2, 1, dtype=torch.float64)]
    env = torch.ones(1, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="2 site"):
        update_two_site_pair(single_mps, single_mpo, 0, env, env, **kwargs)


def test_update_backward_is_finite_for_non_degenerate_problem():
    """(26)"""
    pair = 1
    mps, mpo, _, left, right = _pair_setup([2, 3, 2, 3], 2, pair)
    environment = left[pair].clone().requires_grad_(True)
    result = update_two_site_pair(
        mps, mpo, pair, environment, right[pair + 2], direction="left_to_right", max_bond=None
    )
    result.local_minimum_energy.backward()
    assert environment.grad is not None and torch.isfinite(environment.grad).all()


def _extreme_base_tensor() -> torch.Tensor:
    """最大絶対値が 1 で、特異値が十分離れた (2, 2, 2, 2) テンソル。"""
    generator = torch.Generator().manual_seed(123)
    base = torch.randn(2, 2, 2, 2, generator=generator, dtype=torch.float64)
    return base / base.abs().amax()


def _scale_safe_reference_split(theta: torch.Tensor, direction: str, max_bond: int | None):
    """float64 で、最大絶対値正規化してから計算する分割の参照値。"""
    matrix = theta.detach().reshape(theta.shape[0] * theta.shape[1], -1).double()
    entry_scale = matrix.abs().amax()
    u, relative, vh = torch.linalg.svd(matrix / entry_scale, full_matrices=False)
    relative_squared = relative.square()
    available = relative.shape[0]
    retained = available if max_bond is None else min(max_bond, available)
    weight = relative_squared[retained:].sum() / relative_squared.sum()
    retained_relative_norm = relative_squared[:retained].sum().sqrt()
    normalized = relative[:retained] / retained_relative_norm
    u_kept, vh_kept = u[:, :retained], vh[:retained]
    if direction == "left_to_right":
        left, right = u_kept, normalized.unsqueeze(1) * vh_kept
    else:
        left, right = u_kept * normalized.unsqueeze(0), vh_kept
    left = left.reshape(theta.shape[0], theta.shape[1], retained)
    right = right.reshape(retained, theta.shape[2], theta.shape[3])
    merged = torch.einsum("abc,cde->abde", left, right)
    discarded_norm = entry_scale * relative_squared[retained:].sum().sqrt()
    return {
        "merged": merged,
        "weight": weight,
        "retained_norm": entry_scale * retained_relative_norm,
        "discarded_squared_norm": discarded_norm.square(),
        "retained_rank": retained,
    }


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("max_bond", [None, 1])
@pytest.mark.parametrize(
    "scale,dtype",
    [
        (1e-30, torch.float32),
        (1e30, torch.float32),
        (1e-200, torch.float64),
        (1e200, torch.float64),
    ],
)
def test_split_extreme_scale_keeps_finite_cores_and_weight(scale, dtype, direction, max_bond):
    """二乗が dtype の範囲を外れても、分割・weight・再構成・backward は有限。"""
    theta = (_extreme_base_tensor() * scale).to(dtype=dtype)
    before = theta.clone()
    result = split_two_site_tensor(theta, direction=direction, max_bond=max_bond)
    assert torch.equal(theta, before)
    assert result.retained_rank == (4 if max_bond is None else 1)
    for tensor in (
        result.left_core,
        result.right_core,
        result.singular_values,
        result.discarded_weight,
        result.retained_norm,
        result.truncated_tensor_before_normalization,
    ):
        assert tensor.dtype == dtype and torch.isfinite(tensor).all()
    assert not torch.isnan(result.discarded_squared_norm)
    reference = _scale_safe_reference_split(theta, direction, max_bond)
    merged = merge_two_sites(result.left_core, result.right_core).double()
    close = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float32 else {"atol": 1e-8, "rtol": 1e-8}
    torch.testing.assert_close(merged, reference["merged"], **close)
    torch.testing.assert_close(result.discarded_weight.double(), reference["weight"], **close)
    torch.testing.assert_close(result.retained_norm.double(), reference["retained_norm"], **close)
    reference_square = reference["discarded_squared_norm"]
    if bool(torch.isfinite(reference_square.to(dtype))):
        torch.testing.assert_close(result.discarded_squared_norm.double(), reference_square, **close)
    elif bool(reference_square == 0) or float(reference_square) == 0.0:
        assert result.discarded_squared_norm.item() == 0.0
    else:
        # 絶対二乗和は dtype に収まらない。inf へ丸めても weight は相対値のまま有限。
        assert torch.isinf(result.discarded_squared_norm)
        assert result.discarded_weight.item() > 0.0
    if direction == "left_to_right":
        gram = result.left_core.reshape(-1, result.retained_rank)
        identity = torch.eye(result.retained_rank, dtype=dtype)
        orthogonality_error = (gram.T @ gram - identity).abs().max()
    else:
        gram = result.right_core.reshape(result.retained_rank, -1)
        identity = torch.eye(result.retained_rank, dtype=dtype)
        orthogonality_error = (gram @ gram.T - identity).abs().max()
    assert orthogonality_error.item() <= (1e-4 if dtype == torch.float32 else 1e-8)

    leaf = theta.detach().clone().requires_grad_(True)
    differentiated = split_two_site_tensor(leaf, direction=direction, max_bond=max_bond)
    (
        differentiated.left_core.square().sum()
        + differentiated.right_core.square().sum()
        + differentiated.discarded_weight
    ).backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize(
    "leading,trailing,dtype",
    [(1e20, 1e18, torch.float32), (1e160, 1e150, torch.float64)],
)
def test_truncation_when_leading_square_overflows_but_discarded_square_fits(leading, trailing, dtype, direction):
    """総二乗和は overflow するが、捨てた分の二乗和は dtype に収まる。"""
    matrix = torch.diag(torch.tensor([leading, trailing], dtype=torch.float64)).to(dtype=dtype)
    theta = matrix.reshape(1, 2, 2, 1)
    result = split_two_site_tensor(theta, direction=direction, max_bond=1)
    assert result.retained_rank == 1
    assert torch.isfinite(result.left_core).all() and torch.isfinite(result.right_core).all()
    assert torch.isfinite(result.discarded_squared_norm) and torch.isfinite(result.discarded_weight)
    leading_value = matrix[0, 0].double()
    trailing_value = matrix[1, 1].double()
    magnitude = leading_value.abs()
    relative_leading = leading_value / magnitude
    relative_trailing = trailing_value / magnitude
    expected_weight = relative_trailing.square() / (relative_leading.square() + relative_trailing.square())
    torch.testing.assert_close(result.discarded_weight.double(), expected_weight, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(
        result.discarded_squared_norm.double(),
        (magnitude * relative_trailing.abs()).square(),
        atol=0.0,
        rtol=1e-4,
    )
    torch.testing.assert_close(result.retained_norm.double(), magnitude * relative_leading.abs(), atol=0.0, rtol=1e-4)
    leaf = theta.detach().clone().requires_grad_(True)
    differentiated = split_two_site_tensor(leaf, direction=direction, max_bond=1)
    (differentiated.right_core.square().sum() + differentiated.discarded_weight).backward()
    assert torch.isfinite(leaf.grad).all()


def test_split_of_dtype_max_tensor_does_not_return_nan():
    """core や retained norm が dtype に収まらないときは NaN ではなく理由のある例外。"""
    theta = torch.full((2, 2, 2, 2), torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="表現"):
        split_two_site_tensor(theta, direction="left_to_right", max_bond=None)
