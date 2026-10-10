"""予測安定性 certificate（``metrics.prediction_stability``）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/04_classification_margin_argmax_stability.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/06_minimum_margin_batchwide_stability.ipynb
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from nn_compression.metrics import (
    BatchStabilityCertificate,
    MinimumMargin,
    batch_stability_certificate,
    classification_margins,
    minimum_classification_margin,
    mlp_sample_logits_error_bounds,
    sample_stability_certificates,
)

DTYPES = [torch.float32, torch.float64]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません"),
    ),
]


def _t(values, dtype=torch.float64, device="cpu") -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# classification_margins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_classification_margins_hand_computed_example(dtype):
    # sample 0: 3 - 1 = 2, sample 1: 5 と 5 が tie -> 0, sample 2: 2 - 1.5 = 0.5
    logits = _t([[3.0, 1.0, 0.0], [5.0, 5.0, 1.0], [0.0, 2.0, 1.5]], dtype)

    margins = classification_margins(logits)

    assert margins.shape == (3,)
    assert margins.dtype == dtype
    torch.testing.assert_close(margins, _t([2.0, 0.0, 0.5], dtype))


def test_classification_margins_notebook_example_and_class_position():
    # Notebook 04 の例: top1=2.4, top2=1.6 -> margin 0.8。top1 の位置は問わない。
    logits = _t([[2.4, 1.6, 0.2, -0.5, 1.2], [1.6, 0.2, -0.5, 1.2, 2.4]])

    margins = classification_margins(logits)

    torch.testing.assert_close(margins, _t([0.8, 0.8]))


def test_classification_margins_top_logit_tie_is_exactly_zero():
    logits = _t([[1.0, 1.0, 1.0], [0.1 + 0.2, 0.3 + 0.0, -1.0]])

    margins = classification_margins(logits)

    assert float(margins[0]) == 0.0
    assert float(margins[1]) >= 0.0  # 0.1+0.2 と 0.3 の丸め差でも負にならない


def test_classification_margins_two_classes_is_absolute_difference():
    logits = _t([[1.0, 4.0], [-2.0, -5.0]])

    torch.testing.assert_close(classification_margins(logits), _t([3.0, 3.0]))


def test_classification_margins_are_never_negative():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(200, 10, generator=generator, dtype=torch.float64)

    assert bool((classification_margins(logits) >= 0).all())


def test_classification_margins_do_not_mutate_input():
    logits = _t([[3.0, 1.0, 0.0], [5.0, 5.0, 1.0]])
    saved = logits.clone()

    classification_margins(logits)

    assert torch.equal(logits, saved)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_classification_margins_keep_dtype_and_device(dtype, device):
    margins = classification_margins(_t([[3.0, 1.0], [0.0, 2.0]], dtype, device))

    assert margins.dtype == dtype
    assert margins.device.type == torch.device(device).type


def test_classification_margins_keep_autograd_graph():
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5]]).requires_grad_(True)

    margins = classification_margins(logits)
    assert margins.requires_grad
    margins.sum().backward()

    expected = _t([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, expected)


def test_classification_margins_reject_insufficient_classes():
    with pytest.raises(ValueError, match="2以上"):
        classification_margins(_t([[1.0], [2.0]]))


def test_classification_margins_reject_bad_shapes():
    with pytest.raises(ValueError):
        classification_margins(_t([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        classification_margins(torch.zeros(2, 3, 4, dtype=torch.float64))
    with pytest.raises(ValueError):
        classification_margins(torch.zeros(0, 3, dtype=torch.float64))


def test_classification_margins_reject_dtype_and_type_problems():
    with pytest.raises(TypeError):
        classification_margins(torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        classification_margins(torch.ones(2, 3, dtype=torch.float16))
    with pytest.raises(TypeError):
        classification_margins([[1.0, 2.0]])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_classification_margins_reject_non_finite_values(bad):
    with pytest.raises(ValueError, match="NaN / inf"):
        classification_margins(_t([[1.0, bad], [0.0, 1.0]]))


# ---------------------------------------------------------------------------
# minimum_classification_margin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_minimum_classification_margin_value_and_bottleneck_index(dtype):
    result = minimum_classification_margin(_t([2.0, 0.75, 3.0, 1.0], dtype))

    assert isinstance(result, MinimumMargin)
    assert result.value.shape == () and result.value.dtype == dtype
    assert float(result.value) == 0.75
    assert result.sample_index.dtype == torch.int64
    assert int(result.sample_index) == 1


def test_minimum_classification_margin_ties_return_first_index():
    result = minimum_classification_margin(_t([2.0, 0.5, 0.5, 3.0]))

    assert int(result.sample_index) == 1


def test_minimum_classification_margin_single_sample_and_zero_margin():
    single = minimum_classification_margin(_t([1.25]))
    assert float(single.value) == 1.25 and int(single.sample_index) == 0

    zero = minimum_classification_margin(_t([1.0, 0.0, 2.0]))
    assert float(zero.value) == 0.0 and int(zero.sample_index) == 1


def test_minimum_classification_margin_does_not_mutate_and_keeps_graph():
    margins = _t([2.0, 0.5, 3.0]).requires_grad_(True)
    saved = margins.detach().clone()

    result = minimum_classification_margin(margins)

    assert torch.equal(margins.detach(), saved)
    result.value.backward()
    torch.testing.assert_close(margins.grad, _t([0.0, 1.0, 0.0]))


def test_minimum_classification_margin_reject_bad_inputs():
    with pytest.raises(ValueError):
        minimum_classification_margin(_t([[1.0, 2.0]]))  # 2階
    with pytest.raises(ValueError):
        minimum_classification_margin(torch.zeros(0, dtype=torch.float64))  # 空
    with pytest.raises(ValueError):
        minimum_classification_margin(_t([1.0, float("nan")]))
    with pytest.raises(TypeError):
        minimum_classification_margin(torch.ones(3, dtype=torch.int64))
    with pytest.raises(TypeError):
        minimum_classification_margin([1.0, 2.0])


# ---------------------------------------------------------------------------
# sample_stability_certificates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_sample_stability_certificates_hand_computed_scalar_bound(dtype):
    margins = _t([2.0, 0.0, 0.5], dtype)

    certified = sample_stability_certificates(margins, _t(0.4, dtype))

    # U=0.4: 0.4 < 1.0 は成立、0.4 < 0.0 は不成立、0.4 < 0.25 は不成立
    assert certified.dtype == torch.bool and certified.shape == (3,)
    assert certified.tolist() == [True, False, False]


@pytest.mark.parametrize("dtype", DTYPES)
def test_sample_stability_certificates_strict_inequality_at_boundary(dtype):
    margins = _t([1.0], dtype)
    half = _t(0.5, dtype)  # U = gamma / 2 ちょうど
    below = torch.nextafter(half, _t(0.0, dtype))
    above = torch.nextafter(half, _t(1.0, dtype))

    assert sample_stability_certificates(margins, half).tolist() == [False]
    assert sample_stability_certificates(margins, below).tolist() == [True]
    assert sample_stability_certificates(margins, above).tolist() == [False]


def test_sample_stability_certificates_tiny_margin_has_no_halving_round_off():
    # gamma が最小の subnormal のとき gamma/2 は 0 に丸められるため、
    # 「U < gamma/2」を浮動小数点で直接比べると U=0 が不成立と誤判定される
    # （数学的には 0 < gamma/2 で成立）。「2*U < gamma」の形ならこの誤判定が起きない。
    tiny = torch.nextafter(_t(0.0, torch.float32), _t(1.0, torch.float32))
    margins = tiny.reshape(1)

    assert sample_stability_certificates(margins, _t(0.0, torch.float32)).tolist() == [True]
    # U = gamma のとき 2*U < gamma は偽。
    assert sample_stability_certificates(margins, tiny).tolist() == [False]


def test_sample_stability_certificates_zero_error_and_tie():
    margins = _t([1.0, 0.0])

    certified = sample_stability_certificates(margins, 0.0)

    # 誤差0でも margin 0（top-1 tie）の sample は保証できない。
    assert certified.tolist() == [True, False]


def test_sample_stability_certificates_accept_scalar_forms_and_sample_wise_bounds():
    margins = _t([2.0, 1.0, 0.5])

    by_zero_dim = sample_stability_certificates(margins, _t(0.3))
    by_python_float = sample_stability_certificates(margins, 0.3)
    by_sample_wise = sample_stability_certificates(margins, _t([0.3, 0.3, 0.3]))

    assert by_zero_dim.tolist() == by_python_float.tolist() == by_sample_wise.tolist()
    assert by_zero_dim.tolist() == [True, True, False]

    # sample ごとに異なる上界: 0.9 < 1.0 は成立、0.6 < 0.5 は不成立、0.2 < 0.25 は成立
    mixed = sample_stability_certificates(margins, _t([0.9, 0.6, 0.2]))
    assert mixed.tolist() == [True, False, True]


def test_sample_stability_certificates_python_float_is_cast_to_margin_dtype():
    margins = _t([1.0], torch.float32)

    certified = sample_stability_certificates(margins, 0.25)

    assert certified.dtype == torch.bool
    assert certified.tolist() == [True]


def test_sample_stability_certificates_overflowing_bound_is_not_certified():
    margins = torch.tensor([3.0e38], dtype=torch.float32)
    huge = torch.tensor(3.0e38, dtype=torch.float32)  # 2*U が inf になる

    assert sample_stability_certificates(margins, huge).tolist() == [False]


def test_sample_stability_certificates_do_not_mutate_inputs():
    margins = _t([2.0, 1.0])
    bound = _t([0.3, 0.7])
    margins_saved, bound_saved = margins.clone(), bound.clone()

    sample_stability_certificates(margins, bound)

    assert torch.equal(margins, margins_saved)
    assert torch.equal(bound, bound_saved)


@pytest.mark.parametrize("device", DEVICES)
def test_sample_stability_certificates_keep_device(device):
    margins = _t([2.0, 1.0], device=device)

    certified = sample_stability_certificates(margins, _t([0.3, 0.7], device=device))

    assert certified.device.type == torch.device(device).type


def test_sample_stability_certificates_reject_bad_error_bounds():
    margins = _t([2.0, 1.0, 0.5])

    with pytest.raises(ValueError, match="非負"):
        sample_stability_certificates(margins, -0.1)
    with pytest.raises(ValueError, match="非負"):
        sample_stability_certificates(margins, _t([0.1, -0.1, 0.1]))
    with pytest.raises(ValueError, match="NaN / inf"):
        sample_stability_certificates(margins, float("nan"))
    with pytest.raises(ValueError, match="NaN / inf"):
        sample_stability_certificates(margins, _t([0.1, float("inf"), 0.1]))
    with pytest.raises(ValueError):
        sample_stability_certificates(margins, _t([0.1, 0.1]))  # (B,) と不一致
    with pytest.raises(ValueError):
        sample_stability_certificates(margins, _t([[0.1, 0.1, 0.1]]))  # 2階
    with pytest.raises(TypeError):
        sample_stability_certificates(margins, _t([0.1, 0.1, 0.1], torch.float32))  # dtype 不一致
    with pytest.raises(TypeError):
        sample_stability_certificates(margins, True)
    with pytest.raises(TypeError):
        sample_stability_certificates(margins, "0.1")
    with pytest.raises(TypeError):
        sample_stability_certificates(margins, None)


def test_sample_stability_certificates_reject_bad_margins():
    with pytest.raises(ValueError):
        sample_stability_certificates(_t([[1.0, 2.0]]), 0.1)
    with pytest.raises(ValueError):
        sample_stability_certificates(_t([1.0, float("inf")]), 0.1)
    with pytest.raises(TypeError):
        sample_stability_certificates(torch.ones(2, dtype=torch.int64), 0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_sample_stability_certificates_reject_device_mismatch():
    with pytest.raises(ValueError):
        sample_stability_certificates(_t([1.0, 2.0]), _t([0.1, 0.1], device="cuda"))


# ---------------------------------------------------------------------------
# batch_stability_certificate
# ---------------------------------------------------------------------------


def test_batch_stability_certificate_matches_composition_of_parts():
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5], [4.0, 0.0, 3.0]])  # margins 2, 0.5, 1

    result = batch_stability_certificate(logits, 0.3)

    assert isinstance(result, BatchStabilityCertificate)
    torch.testing.assert_close(result.margins, _t([2.0, 0.5, 1.0]))
    assert float(result.minimum_margin) == 0.5
    assert int(result.bottleneck_index) == 1
    # U=0.3: 0.3<1.0 成立, 0.3<0.25 不成立, 0.3<0.5 成立
    assert result.sample_certified.tolist() == [True, False, True]
    assert result.all_certified.dtype == torch.bool
    assert bool(result.all_certified) is False

    expected_margins = classification_margins(logits)
    expected_min = minimum_classification_margin(expected_margins)
    assert torch.equal(result.margins, expected_margins)
    assert torch.equal(result.minimum_margin, expected_min.value)
    assert torch.equal(result.bottleneck_index, expected_min.sample_index)
    assert torch.equal(
        result.sample_certified, sample_stability_certificates(expected_margins, 0.3)
    )


def test_batch_stability_certificate_all_certified_with_small_batch_bound():
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5]])

    result = batch_stability_certificate(logits, 0.2)  # 最小 margin 0.5 の半分 0.25 より小さい

    assert result.sample_certified.tolist() == [True, True]
    assert bool(result.all_certified) is True


def test_batch_stability_certificate_batchwide_bound_equals_minimum_margin_condition():
    # 全 sample 共通の上界 U では、all_certified と U < (最小 margin)/2 が同値。
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5], [4.0, 0.0, 3.0]])
    half_min = 0.5 / 2

    below = batch_stability_certificate(logits, half_min * 0.999)
    equal = batch_stability_certificate(logits, half_min)

    assert bool(below.all_certified) is True
    assert bool(equal.all_certified) is False


def test_batch_stability_certificate_accepts_sample_wise_bounds():
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5]])

    result = batch_stability_certificate(logits, _t([0.9, 0.3]))

    assert result.sample_certified.tolist() == [True, False]
    assert bool(result.all_certified) is False


def test_batch_stability_certificate_tie_sample_is_never_certified():
    logits = _t([[5.0, 5.0, 1.0], [3.0, 1.0, 0.0]])

    result = batch_stability_certificate(logits, 0.0)

    assert float(result.minimum_margin) == 0.0
    assert int(result.bottleneck_index) == 0
    assert result.sample_certified.tolist() == [False, True]


def test_batch_stability_certificate_does_not_mutate_inputs():
    logits = _t([[3.0, 1.0, 0.0], [0.0, 2.0, 1.5]])
    bound = _t([0.2, 0.2])
    logits_saved, bound_saved = logits.clone(), bound.clone()

    batch_stability_certificate(logits, bound)

    assert torch.equal(logits, logits_saved)
    assert torch.equal(bound, bound_saved)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_stability_certificate_keeps_dtype(dtype):
    result = batch_stability_certificate(_t([[3.0, 1.0], [0.0, 2.0]], dtype), _t(0.1, dtype))

    assert result.margins.dtype == dtype
    assert result.minimum_margin.dtype == dtype


def test_batch_stability_certificate_propagates_validation_errors():
    with pytest.raises(ValueError):
        batch_stability_certificate(_t([[1.0], [2.0]]), 0.1)
    with pytest.raises(ValueError):
        batch_stability_certificate(_t([[1.0, 2.0]]), -0.1)
    with pytest.raises(TypeError):
        batch_stability_certificate(torch.ones(2, 3, dtype=torch.int64), 0.1)


# ---------------------------------------------------------------------------
# 保証の実効性: certificate が成立した sample は、実際に argmax が変わらない
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_certified_samples_keep_argmax_under_bounded_logit_perturbation(dtype):
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(500, 6, generator=generator, dtype=dtype)
    margins = classification_margins(logits)

    # 各 sample で ||delta l||_2 = U_n = 0.99 * gamma_n / 2 となる摂動を作る（ℓ∞ <= ℓ2 <= U_n）。
    direction = torch.randn(500, 6, generator=generator, dtype=dtype)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    bounds = 0.99 * margins / 2
    perturbed = logits + direction * bounds[:, None]

    certified = sample_stability_certificates(margins, bounds)

    assert bool(certified.all())
    assert torch.equal(perturbed.argmax(dim=1), logits.argmax(dim=1))


@pytest.mark.parametrize("dtype", DTYPES)
def test_certificate_from_mlp_error_bounds_is_sound_on_real_forward(dtype):
    generator = torch.Generator().manual_seed(1)
    X = torch.randn(300, 8, generator=generator, dtype=dtype)
    W1 = torch.randn(16, 8, generator=generator, dtype=dtype)
    b1 = torch.randn(16, generator=generator, dtype=dtype)
    W2 = torch.randn(5, 16, generator=generator, dtype=dtype)
    b2 = torch.randn(5, generator=generator, dtype=dtype)
    delta_W1 = 0.02 * torch.randn(16, 8, generator=generator, dtype=dtype)

    logits = torch.relu(X @ W1.T + b1) @ W2.T + b2
    logits_tilde = torch.relu(X @ (W1 + delta_W1).T + b1) @ W2.T + b2
    bounds = mlp_sample_logits_error_bounds(X, delta_W1, W2)

    result = batch_stability_certificate(logits, bounds)

    changed = logits.argmax(dim=1) != logits_tilde.argmax(dim=1)
    # 十分条件なので、certified な sample では argmax が絶対に変わらない。
    assert not bool((changed & result.sample_certified).any())
    # 検査が空にならないよう、certified / 非 certified の両方が存在する設定にしてある。
    assert bool(result.sample_certified.any())
    assert not bool(result.sample_certified.all())


# ---------------------------------------------------------------------------
# Bugbot: 丸め・overflow で certificate / margin が危険側に倒れないこと
# ---------------------------------------------------------------------------


def _bound_forms(value: torch.Tensor):
    """同じ数値を Python float / 0階 Tensor / (B,) Tensor で渡す。"""
    python_value = float(value.detach().cpu())
    scalar = value.reshape(())
    vector = value.reshape(1)
    return python_value, scalar, vector


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_certificate_half_margin_boundary_for_every_bound_form(dtype, device):
    """U == γ/2 は不成立、その直下は成立、直上は不成立。"""
    margins = torch.tensor([1.0], dtype=dtype, device=device)
    half = torch.tensor(0.5, dtype=dtype, device=device)
    below = torch.nextafter(half, torch.zeros_like(half))
    above = torch.nextafter(half, torch.tensor(float("inf"), dtype=dtype, device=device))

    for value, expected in ((half, False), (below, True), (above, False)):
        for bound in _bound_forms(value):
            certified = sample_stability_certificates(margins, bound)
            assert certified.dtype == torch.bool
            assert certified.device == margins.device
            assert certified.tolist() == [expected]


@pytest.mark.parametrize("device", DEVICES)
def test_float32_subnormal_python_bound_is_not_a_false_certificate(device):
    """2 * 2**-150 == 2**-149（float32 の最小正 subnormal）なので、厳密には不成立。

    Python float や float64 scalar を float32 へ最近接丸めすると 2**-150 は 0 になり、
    誤って certificate が成立する。
    """
    margin = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32, device=device),
        torch.tensor(float("inf"), dtype=torch.float32, device=device),
    )
    margins = margin.reshape(1)
    scalar = torch.tensor(2.0**-150, dtype=torch.float64, device=device)

    assert sample_stability_certificates(margins, 2.0**-150).tolist() == [False]
    assert scalar.ndim == 0
    assert sample_stability_certificates(margins, scalar).tolist() == [False]
    # 境界の直下（2**-151）は成立したままでなければならない。
    assert sample_stability_certificates(margins, 2.0**-151).tolist() == [True]


def _assert_margin_is_finite_lower_bound(logits: torch.Tensor) -> torch.Tensor:
    margins = classification_margins(logits)
    assert margins.shape == (logits.shape[0],)
    assert margins.dtype == logits.dtype
    assert margins.device == logits.device
    assert bool(torch.isfinite(margins).all())
    assert bool((margins >= 0).all())

    top2 = torch.topk(logits, k=2, dim=1).values
    for index in range(logits.shape[0]):
        true_margin = Fraction(top2[index, 0].item()) - Fraction(top2[index, 1].item())
        assert Fraction(margins[index].item()) <= true_margin
    return margins


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_extreme_finite_logits_margin_is_safe_and_certificate_runs(dtype, device):
    fmax = torch.finfo(dtype).max
    logits = torch.tensor(
        [
            [fmax, -fmax],
            [-1.0, -fmax],
            [fmax, fmax],
            [3.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    saved = logits.clone()

    margins = _assert_margin_is_finite_lower_bound(logits)

    assert torch.equal(logits, saved)
    assert float(margins[2]) == 0.0  # tie
    normal = classification_margins(torch.tensor([[3.0, 1.0, 0.0]], dtype=dtype, device=device))
    torch.testing.assert_close(normal, torch.tensor([2.0], dtype=dtype, device=device))

    result = batch_stability_certificate(logits, 0.0)
    assert bool(torch.isfinite(result.margins).all())
    assert bool((result.margins >= 0).all())
    assert result.sample_certified.dtype == torch.bool
    # tie の行は誤差 0 でも保証できない。overflow した行は有限 margin なので落とさない。
    assert bool(result.sample_certified[2]) is False
    assert bool(result.sample_certified[0]) is True
    assert bool(result.all_certified) is False

    # cap した margin を超える上界では、inf margin のときと違い certificate は立たない。
    huge = torch.tensor(fmax, dtype=dtype, device=device)
    denied = batch_stability_certificate(logits[:1], huge)
    assert bool(denied.sample_certified[0]) is False


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_extreme_logits_do_not_cut_autograd_of_normal_rows(dtype, device):
    """通常行は ``top - second`` の勾配、overflow cap と 1 ulp 補正の行は勾配 0。"""
    fmax = torch.finfo(dtype).max
    logits = torch.tensor(
        [[3.0, 1.0], [fmax, -fmax], [-1.0, -fmax]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    margins = classification_margins(logits)
    assert bool(torch.isfinite(margins).all())
    cap = torch.tensor(fmax, dtype=dtype, device=device)
    # overflow した差は最大有限値そのもの。上向き丸めの行は 1 ulp 小さい。
    assert torch.equal(margins[1], cap)
    assert torch.equal(margins[2], torch.nextafter(cap, torch.zeros((), dtype=dtype, device=device)))

    margins.sum().backward()

    assert logits.grad is not None
    torch.testing.assert_close(
        logits.grad[0],
        torch.tensor([1.0, -1.0], dtype=dtype, device=device),
    )
    torch.testing.assert_close(logits.grad[1], torch.zeros(2, dtype=dtype, device=device))
    torch.testing.assert_close(logits.grad[2], torch.zeros(2, dtype=dtype, device=device))
