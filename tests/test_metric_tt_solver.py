"""device を選べる計量つき TT ソルバの、小さい問題での検算。"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.compression.metric_low_rank import gram_matrix, regularize_gram
from nn_compression.tt_mps.matrix import dense_to_tt_matrix_cores
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt import (
    _contract_left_group,
    _contract_right_group,
    metric_tt_hvp,
    metric_tt_jacobi_diagonal,
    metric_tt_rhs,
    move_tt_matrix_center,
    run_metric_tt_als,
)
from nn_compression.tt_mps.metric_tt_solver import (
    MetricTTSolverConfig,
    _conjugate_gradient,
    kronecker_metric_factors,
    kronecker_preconditioner_matrix,
    metric_tt_hvp_on_device,
    metric_tt_jacobi_diagonal_on_device,
    metric_tt_rhs_on_device,
    run_metric_tt_als_on_device,
    solve_metric_tt_site_on_device,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA がない")


def _problem(
    seed: int,
    out_modes: tuple[int, ...],
    in_modes: tuple[int, ...],
    rank: int,
    n_tokens: int,
):
    generator = torch.Generator().manual_seed(seed)
    out_features = math.prod(out_modes)
    in_features = math.prod(in_modes)
    weight = torch.randn(out_features, in_features, generator=generator, dtype=torch.float64)
    cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=rank)
    activations = torch.randn(n_tokens, in_features, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    return cores, weight, gram


def _reduced_operator(cores, site: int, gram: torch.Tensor, in_modes: tuple[int, ...]) -> torch.Tensor:
    left = _contract_left_group(cores, site)
    right = _contract_right_group(cores, site + 1)
    n_left = math.prod(in_modes[:site]) if site else 1
    n_site = in_modes[site]
    n_right = math.prod(in_modes[site + 1 :]) if site < len(cores) - 1 else 1
    coupled_left = torch.einsum("amn,bmp->anbp", left, left)
    coupled_right = torch.einsum("amx,bmy->axby", right, right)
    metric = gram.reshape(n_left, n_site, n_right, n_left, n_site, n_right)
    operator = torch.einsum("auAv,ujxvJy,bxBy->ajbAJB", coupled_left, metric, coupled_right)
    width = cores[site].shape[0] * n_site * cores[site].shape[3]
    return operator.reshape(width, width)


def _positive(seed: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(width, width, generator=generator, dtype=torch.float64)
    matrix = factor @ factor.T + 0.5 * torch.eye(width, dtype=torch.float64)
    return (matrix + matrix.T) * 0.5


def test_device_operators_match_the_cpu_reference():
    out_modes, in_modes = (4, 4, 4), (4, 4, 4)
    cores, weight, gram = _problem(0, out_modes, in_modes, rank=4, n_tokens=24)
    site = 1
    direction = torch.randn(cores[site].shape, dtype=torch.float64)
    expected_hvp = metric_tt_hvp(cores, site, direction, gram, out_modes, in_modes)
    expected_rhs = metric_tt_rhs(cores, site, weight, gram, out_modes, in_modes)
    expected_diagonal = metric_tt_jacobi_diagonal(cores, site, gram, out_modes, in_modes)
    actual_hvp = metric_tt_hvp_on_device(cores, site, direction, gram, out_modes, in_modes)
    actual_rhs = metric_tt_rhs_on_device(cores, site, weight @ gram, out_modes, in_modes)
    actual_diagonal = metric_tt_jacobi_diagonal_on_device(cores, site, gram, out_modes, in_modes)
    torch.testing.assert_close(actual_hvp, expected_hvp, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_rhs, expected_rhs, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_diagonal, expected_diagonal, rtol=1e-12, atol=1e-12)


@needs_cuda
def test_cuda_float64_operators_match_the_cpu_reference():
    out_modes, in_modes = (4, 4, 4), (4, 4, 4)
    cores, weight, gram = _problem(1, out_modes, in_modes, rank=4, n_tokens=24)
    site = 1
    direction = torch.randn(cores[site].shape, dtype=torch.float64)
    expected = metric_tt_hvp(cores, site, direction, gram, out_modes, in_modes)
    moved_cores = [core.cuda() for core in cores]
    actual = metric_tt_hvp_on_device(
        moved_cores, site, direction.cuda(), gram.cuda(), out_modes, in_modes
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-12, atol=1e-12)


def test_local_hessian_is_identity_along_the_output_mode():
    out_modes, in_modes = (4, 4), (4, 4)
    cores, _weight, gram = _problem(2, out_modes, in_modes, rank=4, n_tokens=20)
    site = 1
    operator = _reduced_operator(cores, site, gram, in_modes)
    rank_left, _m_site, n_site, rank_right = cores[site].shape
    vector = torch.randn(rank_left, n_site, rank_right, dtype=torch.float64)
    direction = torch.zeros_like(cores[site])
    direction[:, 0] = vector
    product = metric_tt_hvp_on_device(cores, site, direction, gram, out_modes, in_modes)
    expected = (operator @ vector.reshape(-1)).reshape(rank_left, n_site, rank_right)
    torch.testing.assert_close(product[:, 0], expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(product[:, 1:], torch.zeros_like(product[:, 1:]), atol=1e-10, rtol=0.0)


def test_kronecker_preconditioner_reproduces_an_exact_product_and_stays_positive():
    left = _positive(3, 4)
    middle = _positive(4, 2)
    right = _positive(5, 2)
    gram = torch.kron(torch.kron(left, middle), right)
    factors = kronecker_metric_factors(gram, 4, 2, 2)
    reconstructed = torch.kron(torch.kron(factors.left, factors.middle), factors.right)
    print(
        "exact",
        f"clipped={factors.clipped}",
        f"sign_flipped={factors.sign_flipped}",
        f"relative={factors.relative_error:.3e}",
    )
    torch.testing.assert_close(reconstructed, gram, rtol=1e-10, atol=1e-10)
    assert factors.clipped is False

    out_modes, in_modes = (2, 2, 4), (4, 2, 2)
    cores, _weight, _gram = _problem(6, out_modes, in_modes, rank=2, n_tokens=12)
    site = 1
    preconditioner = kronecker_preconditioner_matrix(cores, site, factors, out_modes, in_modes)
    operator = _reduced_operator(cores, site, gram, in_modes)
    scale = torch.linalg.matrix_norm(operator, ord="fro")
    gap = torch.linalg.matrix_norm(preconditioner - operator, ord="fro")
    assert float((gap / scale).item()) <= 1e-10

    generator = torch.Generator().manual_seed(7)
    width = 4 * 2 * 2
    draw = torch.randn(width, width, generator=generator, dtype=torch.float64)
    general = (draw @ draw.T) + 1e-3 * torch.eye(width, dtype=torch.float64)
    general = (general + general.T) * 0.5
    general_factors = kronecker_metric_factors(general, 4, 2, 2)
    general_preconditioner = kronecker_preconditioner_matrix(
        cores, site, general_factors, out_modes, in_modes
    )
    eigenvalues = torch.linalg.eigvalsh((general_preconditioner + general_preconditioner.T) * 0.5)
    print(
        "general",
        f"clipped={general_factors.clipped}",
        f"sign_flipped={general_factors.sign_flipped}",
        f"relative={general_factors.relative_error:.3e}",
        f"min_eig={float(eigenvalues.min().item()):.3e}",
    )
    assert float(eigenvalues.min().item()) > 0.0


def test_default_device_als_matches_the_cpu_q_trajectory():
    out_modes, in_modes = (4, 4, 4), (4, 4, 4)
    cores, weight, gram = _problem(8, out_modes, in_modes, rank=4, n_tokens=32)
    cpu_cores, cpu_history = run_metric_tt_als(
        cores,
        weight,
        gram,
        out_modes,
        in_modes,
        sweep_tolerance=-1.0,
        max_sweeps=3,
    )
    config = MetricTTSolverConfig(
        device="cpu",
        precision="float64",
        preconditioner="jacobi",
        cg_tolerance_mode="fixed",
        sweep_tolerance=-1.0,
        max_sweeps=3,
    )
    device_cores, device_history = run_metric_tt_als_on_device(
        cores, weight, gram, out_modes, in_modes, config
    )
    assert len(device_history) == len(cpu_history) == 3
    for device_record, cpu_record in zip(device_history, cpu_history, strict=True):
        assert device_record.q_lambda == pytest.approx(cpu_record.q_lambda, rel=1e-9, abs=1e-9)
    for left, right in zip(device_cores, cpu_cores, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-9)


@needs_cuda
def test_cuda_default_als_matches_the_cpu_q_trajectory():
    out_modes, in_modes = (4, 4, 4), (4, 4, 4)
    cores, weight, gram = _problem(9, out_modes, in_modes, rank=4, n_tokens=32)
    _cpu_cores, cpu_history = run_metric_tt_als(
        cores,
        weight,
        gram,
        out_modes,
        in_modes,
        sweep_tolerance=-1.0,
        max_sweeps=3,
    )
    config = MetricTTSolverConfig(
        device="cuda",
        precision="float64",
        preconditioner="jacobi",
        cg_tolerance_mode="fixed",
        sweep_tolerance=-1.0,
        max_sweeps=3,
    )
    _device_cores, device_history = run_metric_tt_als_on_device(
        cores, weight, gram, out_modes, in_modes, config
    )
    assert len(device_history) == 3
    for device_record, cpu_record in zip(device_history, cpu_history, strict=True):
        assert device_record.q_lambda == pytest.approx(cpu_record.q_lambda, rel=1e-9, abs=1e-9)


def test_float32_refinement_reaches_the_float64_residual_target():
    generator = torch.Generator().manual_seed(10)
    width = 16
    basis, _r_factor = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    eigenvalues = torch.logspace(-3, 3, width, dtype=torch.float64)
    gram = (basis * eigenvalues) @ basis.transpose(0, 1)
    gram = (gram + gram.transpose(0, 1)) * 0.5
    condition = float(eigenvalues.max().item() / eigenvalues.min().item())
    assert condition == pytest.approx(1e6, rel=0.01)
    out_modes, in_modes = (4, 4), (4, 4)
    weight = torch.randn(width, width, generator=generator, dtype=torch.float64)
    cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=4)
    cores = canonicalize_tt_matrix_cores(cores, out_modes, in_modes, 0)
    cores = move_tt_matrix_center(cores, out_modes, in_modes, 0, 1)
    config = MetricTTSolverConfig(
        device="cpu",
        precision="float32_refine",
        preconditioner="jacobi",
        cg_tolerance_mode="fixed",
        cg_relative_tolerance=1e-8,
        max_cg_iterations=500,
        max_refinements=8,
    )
    record = solve_metric_tt_site_on_device(cores, 1, weight, gram, out_modes, in_modes, config)
    assert record.final_relative_residual <= 1e-8
    assert record.refinements >= 1


def test_adaptive_tolerances_stay_inside_the_bounds_and_q_does_not_increase():
    out_modes, in_modes = (4, 4, 2), (4, 4, 2)
    cores, weight, gram = _problem(11, out_modes, in_modes, rank=4, n_tokens=24)
    config = MetricTTSolverConfig(
        device="cpu",
        precision="float64",
        preconditioner="jacobi",
        cg_tolerance_mode="adaptive",
        cg_relative_tolerance=1e-8,
        cg_tolerance_max=1e-2,
        adaptive_eta=0.1,
        sweep_tolerance=-1.0,
        max_sweeps=2,
    )
    _fitted, history = run_metric_tt_als_on_device(cores, weight, gram, out_modes, in_modes, config)
    previous = math.inf
    for record in history:
        assert record.q_lambda <= previous
        previous = record.q_lambda
        for tolerance, gradient in zip(record.cg_tolerances, record.local_gradient_norms, strict=True):
            assert config.cg_relative_tolerance <= tolerance <= config.cg_tolerance_max
            expected = min(config.cg_tolerance_max, max(config.cg_relative_tolerance, config.adaptive_eta * gradient))
            assert tolerance == pytest.approx(expected, rel=0.0, abs=0.0)


def test_every_config_returns_cpu_float64_and_leaves_the_inputs_unchanged():
    out_modes, in_modes = (2, 4), (2, 4)
    cores, weight, gram = _problem(12, out_modes, in_modes, rank=2, n_tokens=10)
    original_cores = [core.clone() for core in cores]
    original_weight = weight.clone()
    original_gram = gram.clone()
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        for precision in ("float64", "float32_refine"):
            for preconditioner in ("jacobi", "kronecker"):
                for mode in ("fixed", "adaptive"):
                    if device == "cuda" and not (
                        precision == "float64" and preconditioner == "jacobi" and mode == "fixed"
                    ):
                        continue
                    config = MetricTTSolverConfig(
                        device=device,
                        precision=precision,
                        preconditioner=preconditioner,
                        cg_tolerance_mode=mode,
                        max_cg_iterations=3,
                        max_sweeps=1,
                        sweep_tolerance=-1.0,
                    )
                    fitted, history = run_metric_tt_als_on_device(
                        cores, weight, gram, out_modes, in_modes, config
                    )
                    assert history[-1].stop_reason == "max_sweeps"
                    for core in fitted:
                        assert core.dtype == torch.float64
                        assert core.device.type == "cpu"
    for before, after in zip(original_cores, cores, strict=True):
        assert torch.equal(before, after)
    assert torch.equal(original_weight, weight)
    assert torch.equal(original_gram, gram)


def test_stop_reasons_are_recorded_on_the_last_sweep_only():
    out_modes, in_modes = (2, 4), (2, 4)
    cores, weight, gram = _problem(13, out_modes, in_modes, rank=2, n_tokens=10)

    def run(**overrides):
        settings = {
            "device": "cpu",
            "max_cg_iterations": 2,
            "sweep_tolerance": -1.0,
            "max_sweeps": 4,
        }
        settings.update(overrides)
        config = MetricTTSolverConfig(**settings)
        return run_metric_tt_als_on_device(cores, weight, gram, out_modes, in_modes, config)[1]

    by_sweeps = run(max_sweeps=2)
    assert [record.stop_reason for record in by_sweeps] == [None, "max_sweeps"]

    by_gradient = run(gradient_tolerance=1e6)
    assert by_gradient[0].stop_reason == "gradient_tolerance"
    assert len(by_gradient) == 1

    by_time = run(time_limit_seconds=0.0)
    assert by_time[0].stop_reason == "time_limit_seconds"
    assert len(by_time) == 1


def test_tf32_flags_are_restored_after_the_run():
    out_modes, in_modes = (2, 4), (2, 4)
    cores, weight, gram = _problem(14, out_modes, in_modes, rank=2, n_tokens=8)
    matmul_flag = torch.backends.cuda.matmul.allow_tf32
    cudnn_flag = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seen: dict[str, object] = {}

    def progress(record) -> None:
        del record
        seen["matmul"] = torch.backends.cuda.matmul.allow_tf32
        seen["cudnn"] = torch.backends.cudnn.allow_tf32
        seen["precision"] = torch.get_float32_matmul_precision()

    try:
        config = MetricTTSolverConfig(device="cpu", max_cg_iterations=2, max_sweeps=1, sweep_tolerance=-1.0)
        run_metric_tt_als_on_device(cores, weight, gram, out_modes, in_modes, config, progress=progress)
        assert seen["matmul"] is False
        assert seen["cudnn"] is False
        assert seen["precision"] == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_flag
        torch.backends.cudnn.allow_tf32 = cudnn_flag
        torch.set_float32_matmul_precision(precision)


def _old_environment_factor(group: torch.Tensor, metric_factor: torch.Tensor) -> torch.Tensor:
    """修正前の式。``(r, n, r, n)`` の中間配列を作る（小さい問題の参照用）。"""
    contracted = torch.einsum("amn,bmp->anbp", group, group)
    return torch.einsum("anbp,np->ab", contracted, metric_factor)


def test_environment_factor_matches_the_old_formula():
    from nn_compression.tt_mps.metric_tt_solver import _environment_factor

    generator = torch.Generator().manual_seed(21)
    group = torch.randn(7, 3, 5, generator=generator, dtype=torch.float64)
    metric = _positive(22, 5)
    expected = _old_environment_factor(group, metric)
    actual = _environment_factor(group, metric)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("precision", ["float64", "float32_refine"])
@pytest.mark.parametrize("preconditioner", ["jacobi", "kronecker"])
def test_preconditioner_is_built_once_per_local_problem(monkeypatch, precision, preconditioner):
    import nn_compression.tt_mps.metric_tt_solver as solver

    calls = {"jacobi": 0, "eigensystem": 0}
    original_jacobi = solver.metric_tt_jacobi_diagonal_on_device
    original_eigensystem = solver._kronecker_eigensystem

    def counted_jacobi(*args, **kwargs):
        calls["jacobi"] += 1
        return original_jacobi(*args, **kwargs)

    def counted_eigensystem(*args, **kwargs):
        calls["eigensystem"] += 1
        return original_eigensystem(*args, **kwargs)

    monkeypatch.setattr(solver, "metric_tt_jacobi_diagonal_on_device", counted_jacobi)
    monkeypatch.setattr(solver, "_kronecker_eigensystem", counted_eigensystem)

    out_modes, in_modes = (2, 2, 4), (4, 2, 2)
    cores, weight, gram = _problem(23, out_modes, in_modes, rank=3, n_tokens=40)
    cores = canonicalize_tt_matrix_cores(cores, out_modes, in_modes, 0)
    cores = move_tt_matrix_center(cores, out_modes, in_modes, 0, 1)
    config = MetricTTSolverConfig(
        device="cpu",
        precision=precision,
        preconditioner=preconditioner,
        cg_relative_tolerance=1e-10,
        max_refinements=5,
    )
    record = solve_metric_tt_site_on_device(cores, 1, weight, gram, out_modes, in_modes, config)
    assert record.iterations >= 2
    if precision == "float32_refine":
        assert record.refinements >= 2
    if preconditioner == "jacobi":
        assert calls == {"jacobi": 1, "eigensystem": 0}
    else:
        assert calls == {"jacobi": 0, "eigensystem": 1}
    assert record.seconds_preconditioner_setup >= 0.0
    assert record.seconds_preconditioner_apply > 0.0


def test_kronecker_als_matches_the_q_trajectory_with_the_old_environment(monkeypatch):
    import nn_compression.tt_mps.metric_tt_solver as solver

    out_modes, in_modes = (2, 2, 4), (4, 2, 2)
    cores, weight, gram = _problem(24, out_modes, in_modes, rank=3, n_tokens=40)
    config = MetricTTSolverConfig(
        device="cpu", preconditioner="kronecker", sweep_tolerance=-1.0, max_sweeps=3
    )
    _new_cores, new_history = run_metric_tt_als_on_device(cores, weight, gram, out_modes, in_modes, config)
    monkeypatch.setattr(solver, "_environment_factor", _old_environment_factor)
    _old_cores, old_history = run_metric_tt_als_on_device(cores, weight, gram, out_modes, in_modes, config)
    new_q = torch.tensor([record.q_lambda for record in new_history], dtype=torch.float64)
    old_q = torch.tensor([record.q_lambda for record in old_history], dtype=torch.float64)
    torch.testing.assert_close(new_q, old_q, rtol=1e-10, atol=0.0)
    assert all(record.seconds_preconditioner_apply >= 0.0 for record in new_history)


def test_environment_factor_fits_in_memory_at_the_t1_site2_shape():
    from nn_compression.tt_mps.metric_tt_solver import _environment_factor

    rank, m_left, n_left = 382, 4, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(25)
    group = torch.randn(rank, m_left, n_left, generator=generator, dtype=torch.float64, device=device)
    draw = torch.randn(n_left, n_left, generator=generator, dtype=torch.float64, device=device)
    metric = draw @ draw.T
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    result = _environment_factor(group, metric)
    assert result.shape == (rank, rank)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline
        print(f"peak_extra_bytes={peak}")
        assert peak < 1024**3


def _identity_preconditioner(vector: torch.Tensor) -> torch.Tensor:
    return vector


def test_cg_check_every_one_matches_the_default_als_trajectory():
    cores, weight, gram = _problem(21, (2, 3, 2), (2, 2, 3), rank=3, n_tokens=16)
    base = MetricTTSolverConfig(device="cpu", precision="float64", max_sweeps=2, max_cg_iterations=40)
    explicit = MetricTTSolverConfig(
        device="cpu", precision="float64", max_sweeps=2, max_cg_iterations=40, cg_check_every=1
    )
    base_cores, base_history = run_metric_tt_als_on_device(cores, weight, gram, (2, 3, 2), (2, 2, 3), base)
    explicit_cores, explicit_history = run_metric_tt_als_on_device(
        cores, weight, gram, (2, 3, 2), (2, 2, 3), explicit
    )
    for got, want in zip(explicit_cores, base_cores, strict=True):
        assert torch.equal(got, want)
    assert [record.q_lambda for record in explicit_history] == [record.q_lambda for record in base_history]
    assert [record.cg_iterations for record in explicit_history] == [record.cg_iterations for record in base_history]
    assert [record.cg_residuals for record in explicit_history] == [record.cg_residuals for record in base_history]


def test_blocked_cg_reaches_the_same_tolerance_on_a_diagonal_system():
    rhs = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    scale = torch.tensor([1.0, 4.0, 9.0, 2.0], dtype=torch.float64)

    def apply_hessian(vector: torch.Tensor) -> torch.Tensor:
        return vector * scale

    arguments = dict(
        apply_hessian=apply_hessian,
        apply_preconditioner=_identity_preconditioner,
        rhs=rhs,
        initial=torch.zeros_like(rhs),
        relative_tolerance=1e-8,
        max_iterations=40,
    )
    synced = _conjugate_gradient(**arguments, check_every=1)
    blocked = _conjugate_gradient(**arguments, check_every=10)
    assert synced[5] == "converged"
    assert blocked[5] == "converged"
    assert synced[2] <= 1e-8
    assert blocked[2] <= 1e-8
    assert blocked[1] >= synced[1]


def test_blocked_cg_drops_updates_after_nonpositive_curvature_or_nonfinite_values():
    rhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    scale = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)

    def apply_then_negative(vector: torch.Tensor, calls: dict[str, int]) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] >= 4:
            return -vector
        return vector * scale

    negative_calls = {"n": 0}
    synced = _conjugate_gradient(
        lambda vector: apply_then_negative(vector, negative_calls),
        _identity_preconditioner,
        rhs,
        torch.zeros_like(rhs),
        relative_tolerance=1e-12,
        max_iterations=10,
        check_every=1,
    )
    negative_calls["n"] = 0
    blocked = _conjugate_gradient(
        lambda vector: apply_then_negative(vector, negative_calls),
        _identity_preconditioner,
        rhs,
        torch.zeros_like(rhs),
        relative_tolerance=1e-12,
        max_iterations=10,
        check_every=10,
    )
    assert synced[5] == "nonpositive_curvature"
    assert blocked[5] == "nonpositive_curvature"
    assert torch.equal(blocked[0], synced[0])
    assert blocked[1] == synced[1]
    assert blocked[6] == 10 - synced[1]
    assert blocked[6] > 0
    assert torch.isfinite(blocked[0]).all()

    def apply_then_nan(vector: torch.Tensor, calls: dict[str, int]) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] >= 4:
            return torch.full_like(vector, float("nan"))
        return vector * scale

    nan_calls = {"n": 0}
    synced_nan = _conjugate_gradient(
        lambda vector: apply_then_nan(vector, nan_calls),
        _identity_preconditioner,
        rhs,
        torch.zeros_like(rhs),
        relative_tolerance=1e-12,
        max_iterations=10,
        check_every=1,
    )
    nan_calls["n"] = 0
    blocked_nan = _conjugate_gradient(
        lambda vector: apply_then_nan(vector, nan_calls),
        _identity_preconditioner,
        rhs,
        torch.zeros_like(rhs),
        relative_tolerance=1e-12,
        max_iterations=10,
        check_every=10,
    )
    assert synced_nan[5] == "nonfinite"
    assert blocked_nan[5] == "nonfinite"
    assert torch.equal(blocked_nan[0], synced[0])
    assert blocked_nan[1] == synced[1]
    assert blocked_nan[6] == 10 - synced[1]
    assert torch.isfinite(blocked_nan[0]).all()


def test_cg_reports_nonpositive_curvature_without_changing_the_initial_vector():
    initial = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
    solved = _conjugate_gradient(
        lambda vector: -vector,
        _identity_preconditioner,
        torch.ones_like(initial),
        initial,
        relative_tolerance=1e-8,
        max_iterations=5,
        check_every=1,
    )
    assert solved[5] == "nonpositive_curvature"
    assert solved[1] == 0
    assert solved[6] == 0
    assert torch.equal(solved[0], initial)


def test_blocked_cg_keeps_x_when_the_preconditioned_dot_is_zero_and_exact_solve_stays_converged():
    rhs = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    initial = torch.zeros_like(rhs)

    def rotate(vector: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(vector)
        out[0] = -vector[1]
        out[1] = vector[0]
        return out

    stalled = _conjugate_gradient(
        lambda vector: vector,
        rotate,
        rhs,
        initial,
        relative_tolerance=1e-12,
        max_iterations=10,
        check_every=10,
    )
    assert stalled[5] == "breakdown"
    assert stalled[1] == 0
    assert stalled[6] == 10
    assert torch.equal(stalled[0], initial)

    solved = _conjugate_gradient(
        lambda vector: vector,
        _identity_preconditioner,
        torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        relative_tolerance=1e-8,
        max_iterations=10,
        check_every=10,
    )
    assert solved[5] == "converged"
    assert solved[2] <= 1e-8
    assert torch.allclose(solved[0], torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64))


def test_timing_sync_false_matches_the_synced_solution():
    cores, weight, gram = _problem(22, (2, 3, 2), (2, 2, 3), rank=3, n_tokens=16)
    modes = ((2, 3, 2), (2, 2, 3))
    for check_every in (1, 10):
        synced = MetricTTSolverConfig(
            device="cpu",
            precision="float64",
            max_sweeps=2,
            max_cg_iterations=30,
            cg_check_every=check_every,
            timing_sync=True,
        )
        quiet = MetricTTSolverConfig(
            device="cpu",
            precision="float64",
            max_sweeps=2,
            max_cg_iterations=30,
            cg_check_every=check_every,
            timing_sync=False,
        )
        synced_cores, synced_history = run_metric_tt_als_on_device(cores, weight, gram, *modes, synced)
        quiet_cores, quiet_history = run_metric_tt_als_on_device(cores, weight, gram, *modes, quiet)
        for got, want in zip(quiet_cores, synced_cores, strict=True):
            assert torch.equal(got, want)
        for got, want in zip(quiet_history, synced_history, strict=True):
            assert got.q_lambda == want.q_lambda
            assert got.cg_iterations == want.cg_iterations
            assert got.cg_residuals == want.cg_residuals
