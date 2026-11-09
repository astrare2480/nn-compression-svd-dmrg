"""Qwen3 1層パイロットの、モデルを使わない検算。"""

from __future__ import annotations

import math

import pytest
import torch

from nn_compression.compression.metric_low_rank import (
    gram_matrix,
    max_matrix_rank_within_budget,
    output_error_ratios_from_activations,
    output_error_ratios_from_gram,
    regularize_gram,
    squared_output_error_from_activations,
    squared_output_error_from_gram,
    truncated_svd_weight,
    whitened_truncated_svd,
)
from nn_compression.llm.hf_loading import load_wikitext
from nn_compression.llm.text_documents import (
    document_sha256,
    hash_overlaps,
    is_wikitext_level1_heading,
    restore_wikitext_articles,
    select_token_prefix_sequences,
)
from nn_compression.metrics.bootstrap import bootstrap_ratio_difference_intervals
from nn_compression.metrics.pilot_decision import classify_pilot_outcome, evaluate_rho
from nn_compression.tt_mps import (
    dense_to_tt_matrix_cores,
    tt_matrix_num_parameters,
    tt_matrix_num_parameters_from_ranks,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from nn_compression.tt_mps.matrix_two_site import canonicalize_tt_matrix_cores
from nn_compression.tt_mps.metric_tt import (
    expand_tt_matrix_bonds,
    fit_diagonal_metric_tt,
    metric_tt_hvp,
    metric_tt_jacobi_diagonal,
    metric_tt_rhs,
    preconditioned_conjugate_gradient,
    q_lambda_of_cores,
    run_metric_tt_als,
    scale_matrix_columns,
    select_balanced_tt_matrix_modes,
    select_uniform_tt_rank,
    tt_matrix_core_adjoint,
)


def _column_major_vec(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.transpose(0, 1).contiguous().reshape(-1)


def _explicit_hessian(cores, site, gram, out_modes, in_modes) -> torch.Tensor:
    """P を基底の再構成で作り、P^T (G ⊗ I) P を返す。HVP は使わない。"""
    core = cores[site]
    dimension = core.numel()
    out_features = math.prod(out_modes)
    eye = torch.eye(dimension, dtype=torch.float64)
    mapping = torch.zeros(out_features * math.prod(in_modes), dimension, dtype=torch.float64)
    for index in range(dimension):
        direction = eye[index].reshape(core.shape)
        replaced = list(cores)
        replaced[site] = direction
        mapping[:, index] = _column_major_vec(tt_matrix_to_dense(replaced, out_modes, in_modes))
    kronecker = torch.kron(gram, torch.eye(out_features, dtype=torch.float64))
    return mapping.transpose(0, 1) @ kronecker @ mapping


def _random_tt(out_modes, in_modes, rank, generator):
    cores = []
    ranks = (1,) + tuple(rank for _ in range(len(out_modes) - 1)) + (1,)
    for site, (out_mode, in_mode) in enumerate(zip(out_modes, in_modes)):
        cores.append(
            torch.randn(
                ranks[site],
                out_mode,
                in_mode,
                ranks[site + 1],
                generator=generator,
                dtype=torch.float64,
            )
        )
    return cores


def test_exact_tt_svd_reproduces_weight_in_float64():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    out_modes = (3, 4)
    in_modes = (2, 4)
    cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=None)
    reconstructed = tt_matrix_to_dense(cores, out_modes, in_modes)
    relative = torch.linalg.vector_norm(reconstructed - weight) / torch.linalg.vector_norm(weight)
    assert float(relative.item()) <= 1e-10


def test_whitened_svd_matches_discarded_energy_and_beats_plain_svd():
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    activations = torch.randn(30, 5, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    rank = 2
    whitened = whitened_truncated_svd(weight, gram, rank)
    energy = whitened.discarded_singular_value_energy
    assert torch.allclose(whitened.q_lambda, energy, rtol=1e-8, atol=1e-8)
    plain = truncated_svd_weight(weight, rank)
    plain_q = squared_output_error_from_gram(weight - plain, gram)
    assert float(whitened.q_lambda.item()) <= float(plain_q.item()) + 1e-8


def test_hvp_matches_explicit_hessian_and_jacobi_diagonal():
    generator = torch.Generator().manual_seed(2)
    out_modes = (2, 2)
    in_modes = (2, 2)
    cores = canonicalize_tt_matrix_cores(
        _random_tt(out_modes, in_modes, 2, generator),
        out_modes,
        in_modes,
        0,
    )
    activations = torch.randn(20, 4, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    hessian = _explicit_hessian(cores, 0, gram, out_modes, in_modes)
    direction = torch.randn(cores[0].shape, generator=generator, dtype=torch.float64)
    applied = metric_tt_hvp(cores, 0, direction, gram, out_modes, in_modes).reshape(-1)
    reference = hessian @ direction.reshape(-1)
    assert torch.allclose(applied, reference, rtol=1e-8, atol=1e-8)
    diagonal = metric_tt_jacobi_diagonal(cores, 0, gram, out_modes, in_modes).reshape(-1)
    assert torch.allclose(diagonal, torch.diagonal(hessian), rtol=1e-8, atol=1e-8)


def test_canonical_local_hessian_min_eigenvalue_is_at_least_gram_min():
    generator = torch.Generator().manual_seed(3)
    out_modes = (2, 2)
    in_modes = (2, 2)
    cores = canonicalize_tt_matrix_cores(
        _random_tt(out_modes, in_modes, 2, generator),
        out_modes,
        in_modes,
        1,
    )
    activations = torch.randn(25, 4, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    hessian = _explicit_hessian(cores, 1, gram, out_modes, in_modes)
    hessian_min = torch.linalg.eigvalsh(hessian)[0]
    gram_min = torch.linalg.eigvalsh(gram)[0]
    assert float(hessian_min.item()) >= float(gram_min.item()) - 1e-8


def test_pcg_from_current_core_does_not_increase_local_q():
    generator = torch.Generator().manual_seed(4)
    out_modes = (2, 2)
    in_modes = (2, 2)
    cores = canonicalize_tt_matrix_cores(
        _random_tt(out_modes, in_modes, 2, generator),
        out_modes,
        in_modes,
        0,
    )
    weight = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    activations = torch.randn(18, 4, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    diagonal = metric_tt_jacobi_diagonal(cores, 0, gram, out_modes, in_modes)
    rhs = metric_tt_rhs(cores, 0, weight, gram, out_modes, in_modes)

    def apply_hessian(direction):
        return metric_tt_hvp(cores, 0, direction, gram, out_modes, in_modes)

    solved = preconditioned_conjugate_gradient(
        apply_hessian,
        rhs,
        cores[0],
        diagonal,
        relative_tolerance=1e-14,
        max_iterations=6,
        record_solutions=True,
    )
    assert solved.solution_history is not None
    assert len(solved.solution_history) >= 2
    values = []
    for solution in solved.solution_history:
        replaced = list(cores)
        replaced[0] = solution
        values.append(float(q_lambda_of_cores(replaced, weight, gram, out_modes, in_modes).item()))
    for previous, current in zip(values, values[1:]):
        assert current <= previous + 1e-8 * max(1.0, abs(previous))


def test_parameter_count_matches_rank_formula_and_zero_bond_padding_keeps_weight():
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    out_modes = (2, 2, 2)
    in_modes = (2, 2, 2)
    cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=2)
    ranks = tt_matrix_ranks(cores)
    assert tt_matrix_num_parameters(cores) == tt_matrix_num_parameters_from_ranks(
        out_modes, in_modes, ranks
    )
    target = tuple(1 if index in {0, len(ranks) - 1} else rank + 1 for index, rank in enumerate(ranks))
    expanded = expand_tt_matrix_bonds(cores, out_modes, in_modes, target)
    assert tt_matrix_ranks(expanded) == target
    assert tt_matrix_num_parameters(expanded) == tt_matrix_num_parameters_from_ranks(
        out_modes, in_modes, target
    )
    before = tt_matrix_to_dense(cores, out_modes, in_modes)
    after = tt_matrix_to_dense(expanded, out_modes, in_modes)
    # ゼロ埋め自体は値を変えない。縮約の順序が変わる分の float64 丸めだけを許す。
    assert torch.allclose(before, after, rtol=1e-12, atol=1e-12)


def test_wikitext_level1_headings_split_articles_and_keep_subsections_inside():
    lines = [
        "preface should be dropped\n",
        " = First = \n",
        "",
        "hello\n",
        " = = Section = = \n",
        "body\n",
        " = Second = \n",
        "tail\n",
    ]
    assert is_wikitext_level1_heading(" = Homarus gammarus = \n")
    assert not is_wikitext_level1_heading(" = = Description = = \n")
    assert not is_wikitext_level1_heading(" = = = Deep = = = \n")
    articles = restore_wikitext_articles(lines)
    assert len(articles) == 2
    assert articles[0].startswith(" = First = \n")
    assert " = = Section = = \n" in articles[0]
    assert "hello\n" in articles[0]
    assert "body\n" in articles[0]
    assert "preface" not in articles[0]
    assert articles[1].startswith(" = Second = \n")
    assert "tail\n" in articles[1]
    assert " = First = " not in articles[1]


def test_document_hash_overlap_and_prefix_selection_counts():
    left = document_sha256(" Same \r\n text ")
    right = document_sha256("Same \n text")
    assert left == right
    overlaps = hash_overlaps({"calibration": [left, "a"], "validation": [right, "b"]})
    assert overlaps["calibration ∩ validation"] == [left]

    documents = ["aa", "bbb", "c", "dddd", "eeeee"]

    def encode(text: str) -> list[int]:
        return list(range(len(text)))

    selected = select_token_prefix_sequences(
        documents,
        encode,
        source="synthetic",
        min_tokens=3,
        length=3,
        limit=2,
        seed=0,
    )
    assert selected.n_documents == 5
    assert selected.n_documents_eligible == 3
    assert selected.n_sequences == 2
    assert selected.n_tokens_kept == 6
    assert selected.n_tokens_total == 2 + 3 + 1 + 4 + 5
    assert selected.n_tokens_excluded == selected.n_tokens_total - selected.n_tokens_kept
    assert selected.n_special_tokens == 0
    assert all(len(sequence.token_ids) == 3 for sequence in selected.sequences)


def test_wikitext_test_split_is_refused_before_loading():
    with pytest.raises(ValueError, match="test"):
        load_wikitext({"wikitext": {"revision": "unused"}}, "wikitext-2-raw-v1", "test")


def test_uniform_rank_respects_budget_and_extra_diagonal():
    out_modes = (2, 2)
    in_modes = (2, 2)
    without_extra = select_uniform_tt_rank(out_modes, in_modes, budget=24, extra_parameters=0)
    with_extra = select_uniform_tt_rank(out_modes, in_modes, budget=24, extra_parameters=8)
    assert without_extra["total_parameters"] <= 24
    assert with_extra["total_parameters"] <= 24
    assert with_extra["rank_cap"] <= without_extra["rank_cap"]
    assert max_matrix_rank_within_budget(3072, 1024, 1_572_864) == 384
    assert max_matrix_rank_within_budget(3072, 1024, 786_432) == 192


def test_balanced_mode_tie_break_is_lexicographic():
    chosen = select_balanced_tt_matrix_modes(12, 12, 2)
    from nn_compression.tt_mps import ordered_factorizations, tt_matrix_shape_imbalance

    best = min(
        (
            (
                tt_matrix_shape_imbalance(out_modes, in_modes),
                out_modes,
                in_modes,
            )
            for out_modes in ordered_factorizations(12, 2)
            for in_modes in ordered_factorizations(12, 2)
        )
    )
    assert (chosen["imbalance"], chosen["out_modes"], chosen["in_modes"]) == best


def test_diagonal_metric_tt_exact_absorption_matches_the_deployed_pair():
    generator = torch.Generator().manual_seed(8)
    weight = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    activations = torch.randn(15, 8, generator=generator, dtype=torch.float64) + 0.5
    gram = gram_matrix(activations)
    out_modes = (3, 4)
    in_modes = (2, 4)
    fitted = fit_diagonal_metric_tt(weight, gram, 15, out_modes, in_modes, max_rank=None)
    scaled_dense = tt_matrix_to_dense(fitted.scaled_cores, out_modes, in_modes)
    deployed = scale_matrix_columns(scaled_dense, fitted.input_scale_inv)
    absorbed = tt_matrix_to_dense(fitted.absorbed_cores, out_modes, in_modes)
    weight_scaled = scale_matrix_columns(weight, fitted.scales)
    relative_scaled = torch.linalg.vector_norm(scaled_dense - weight_scaled) / torch.linalg.vector_norm(weight_scaled)
    relative_absorbed = torch.linalg.vector_norm(absorbed - deployed) / torch.linalg.vector_norm(deployed)
    assert float(relative_scaled.item()) <= 1e-10
    assert float(relative_absorbed.item()) <= 1e-10


def test_adjoint_matches_dense_inner_product():
    generator = torch.Generator().manual_seed(6)
    out_modes = (2, 3)
    in_modes = (2, 2)
    cores = _random_tt(out_modes, in_modes, 2, generator)
    probe = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    dense = tt_matrix_to_dense(cores, out_modes, in_modes)
    adjoint = tt_matrix_core_adjoint(cores, 1, probe, out_modes, in_modes)
    left = torch.sum(dense * probe)
    right = torch.sum(cores[1] * adjoint)
    assert torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def test_one_site_als_does_not_increase_q_lambda():
    generator = torch.Generator().manual_seed(7)
    out_modes = (2, 2)
    in_modes = (2, 2)
    weight = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    cores = dense_to_tt_matrix_cores(weight, out_modes, in_modes, max_rank=1)
    activations = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    gram = regularize_gram(gram_matrix(activations), 1e-6)
    initial = float(q_lambda_of_cores(cores, weight, gram, out_modes, in_modes).item())
    _updated, history = run_metric_tt_als(
        cores,
        weight,
        gram,
        out_modes,
        in_modes,
        max_cg_iterations=30,
        max_sweeps=2,
        sweep_tolerance=0.0,
    )
    assert history
    assert history[-1].q_lambda <= initial + 1e-8


def test_bootstrap_uses_the_same_document_draws_for_each_method():
    numerators = {
        "M3": [4.0, 1.0, 3.0, 2.0],
        "M2": [1.0, 1.0, 1.0, 1.0],
    }
    denominators = {
        "M3": [1.0, 1.0, 1.0, 1.0],
        "M2": [1.0, 1.0, 1.0, 1.0],
    }
    intervals = bootstrap_ratio_difference_intervals(
        numerators,
        denominators,
        [("M3", "M2")],
        n_resamples=200,
        seed=0,
    )
    interval = intervals[("M3", "M2")]
    assert interval["estimate"] == pytest.approx(1.5)
    assert interval["low"] <= interval["estimate"] <= interval["high"]


def _rho(domain, e2, e3, e5, low, high):
    return evaluate_rho(
        budget="25",
        tensorization="T2",
        domain=domain,
        e_m2=e2,
        e_m3=e3,
        e_m5=e5,
        g_ci_low=low,
        g_ci_high=high,
    )


def test_rho_is_withheld_when_the_gap_is_not_informative():
    crossing = evaluate_rho(
        budget="50",
        tensorization="T1",
        domain="validation_same",
        e_m2=1.0,
        e_m3=2.0,
        e_m5=1.2,
        g_ci_low=-0.1,
        g_ci_high=0.2,
    )
    assert crossing.rho_status == "denominator_not_informative"
    assert crossing.rho is None
    assert crossing.delta_tt == pytest.approx(0.8)
    small = evaluate_rho(
        budget="50",
        tensorization="T1",
        domain="calibration",
        e_m2=1.0,
        e_m3=1.04,
        e_m5=1.01,
        g_ci_low=0.01,
        g_ci_high=0.05,
    )
    assert small.rho_status == "denominator_not_informative"


def test_pilot_decision_labels_follow_the_fixed_thresholds():
    def grid(validation_rho_high: bool, calibration_rho_high: bool):
        cells = []
        for domain in ("calibration", "validation_same", "validation_other"):
            high = calibration_rho_high if domain == "calibration" else validation_rho_high
            e5 = 1.2 if high else 1.9
            cells.append(
                evaluate_rho(
                    budget="50",
                    tensorization="T2",
                    domain=domain,
                    e_m2=1.0,
                    e_m3=2.0,
                    e_m5=e5,
                    g_ci_low=0.5,
                    g_ci_high=1.5,
                )
            )
        return cells

    # high validation ρ: (2-1.2)/(2-1) = 0.8
    assert classify_pilot_outcome(
        grid(True, True),
        tensorizations=("T2",),
        budgets=("50",),
    ) == "本実験へ進む"
    # low validation ρ: (2-1.9)/(2-1) = 0.1, calibration also low
    assert classify_pilot_outcome(
        grid(False, False),
        tensorizations=("T2",),
        budgets=("50",),
    ) == "中止"
    assert classify_pilot_outcome(
        grid(False, True),
        tensorizations=("T2",),
        budgets=("50",),
    ) == "過適合が主な結果"
    assert classify_pilot_outcome([], tensorizations=("T2",), budgets=("50",)) == "判定保留"


def test_square_activation_is_not_read_as_a_gram_matrix():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(8, 1024, generator=generator, dtype=torch.float64)
    approximated = torch.randn(8, 1024, generator=generator, dtype=torch.float64)
    activation = torch.randn(1024, 1024, generator=generator, dtype=torch.float64)
    direct_numerator = squared_output_error_from_activations(weight - approximated, activation)
    direct_denominator = squared_output_error_from_activations(weight, activation)
    from_activation = output_error_ratios_from_activations(weight, approximated, activation)
    assert from_activation["numerator"] == pytest.approx(float(direct_numerator.item()))
    assert from_activation["denominator"] == pytest.approx(float(direct_denominator.item()))
    assert from_activation["e"] == pytest.approx(float((direct_numerator / direct_denominator).item()))
    gram = gram_matrix(activation)
    from_gram = output_error_ratios_from_gram(weight, approximated, gram)
    assert from_gram["e"] == pytest.approx(from_activation["e"], rel=1e-10, abs=1e-10)
    assert from_gram["numerator"] == pytest.approx(from_activation["numerator"], rel=1e-10, abs=1e-8)
