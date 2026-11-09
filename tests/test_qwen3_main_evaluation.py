"""本実験の評価（統計と、小さい Qwen3 での順伝播の部品）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nn_compression.llm import qwen3_main_evaluation as ev
from nn_compression.llm import qwen3_main_statistics as st
from nn_compression.llm.activation_capture import accumulate_input_grams
from nn_compression.llm.perplexity import sliding_window_perplexity

# ---------------------------------------------------------------- 統計


def test_bootstrap_counts_rows_sum_and_seed():
    counts = st.bootstrap_counts(7, n_resamples=50, seed=0)
    assert counts.shape == (50, 7)
    assert np.all(counts.sum(axis=1) == 7)
    assert st.counts_digest(counts) == st.counts_digest(st.bootstrap_counts(7, n_resamples=50, seed=0))


def test_ratio_uses_sums_not_mean_of_ratios():
    num = np.array([1.0, 9.0])
    den = np.array([1.0, 99.0])
    counts = np.array([[1.0, 1.0]])
    point, boot = st.ratio_point_and_boot(num, den, counts)
    assert point == pytest.approx(10.0 / 100.0)
    assert boot[0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(d_point=0.0, d_lo=-1, e3_point=1, h02_lo=1, h02_hi=1, h05_lo=1, h05_hi=1), "denominator_not_informative"),
        (dict(d_point=0.5, d_lo=-0.1, e3_point=1, h02_lo=1, h02_hi=1, h05_lo=1, h05_hi=1), "denominator_not_informative"),
        (dict(d_point=0.01, d_lo=0.005, e3_point=1, h02_lo=1, h02_hi=1, h05_lo=1, h05_hi=1), "denominator_not_informative"),
        (dict(d_point=0.5, d_lo=0.4, e3_point=1, h02_lo=0.1, h02_hi=0.2, h05_lo=0.0, h05_hi=0.1), "high"),
        (dict(d_point=0.5, d_lo=0.4, e3_point=1, h02_lo=0.0, h02_hi=0.2, h05_lo=-0.2, h05_hi=-0.01), "mid"),
        (dict(d_point=0.5, d_lo=0.4, e3_point=1, h02_lo=-0.3, h02_hi=-0.01, h05_lo=-0.5, h05_hi=-0.2), "low"),
        (dict(d_point=0.5, d_lo=0.4, e3_point=1, h02_lo=-0.1, h02_hi=0.1, h05_lo=-0.3, h05_hi=-0.1), "unresolved"),
        (dict(d_point=0.5, d_lo=0.4, e3_point=1, h02_lo=1, h02_hi=1, h05_lo=1, h05_hi=1, solver_unresolved=True), "not_classified"),
    ],
)
def test_classification_table(kwargs, expected):
    assert st.classify_from_intervals(**kwargs) == expected


def _cell_numerators(rng, n, e2, e3, e5, noise=0.002):
    den = rng.uniform(1.0, 2.0, size=n)
    def make(e):
        return den * (e + noise * rng.standard_normal(n))
    return {"M2": make(e2), "M3": make(e3), "M5": make(e5), "M3F": make(e3 - 0.01 * (e3 - e2))}, den


def test_evaluate_cell_high_and_low():
    rng = np.random.default_rng(1)
    counts = st.bootstrap_counts(40, n_resamples=2000)
    nums, den = _cell_numerators(rng, 40, 0.10, 0.30, 0.11)
    high = st.evaluate_cell(nums, den, counts)
    assert high["class"] == "high"
    assert high["rho"] == pytest.approx(0.95, abs=0.05)
    assert high["F_below_0.2"]
    nums, den = _cell_numerators(rng, 40, 0.10, 0.30, 0.29)
    assert st.evaluate_cell(nums, den, counts)["class"] == "low"
    nums, den = _cell_numerators(rng, 40, 0.299, 0.30, 0.20)
    assert st.evaluate_cell(nums, den, counts)["class"] == "denominator_not_informative"


def test_one_minus_rho_only_when_denominator_informative():
    # D / e(M3) ≈ 0.033 < 0.05 で、D の区間は 0 より上。分類は分母なし、1−ρ は出さない。
    rng = np.random.default_rng(5)
    counts = st.bootstrap_counts(40, n_resamples=500)
    nums, den = _cell_numerators(rng, 40, 0.29, 0.30, 0.292, noise=0.0001)
    row = st.evaluate_cell(nums, den, counts)
    assert row["D_lo"] > 0.0
    assert row["class"] == "denominator_not_informative"
    assert np.isfinite(row["rho"]) and np.isnan(row["one_minus_rho"])
    nums, den = _cell_numerators(rng, 40, 0.10, 0.30, 0.11)
    row = st.evaluate_cell(nums, den, counts)
    assert row["one_minus_rho"] == pytest.approx(1.0 - row["rho"])


def test_results_record_the_counts_actually_used():
    rng = np.random.default_rng(6)
    counts = st.bootstrap_counts(30, n_resamples=200, seed=0)
    other = st.bootstrap_counts(30, n_resamples=200, seed=1)
    nums, den = _cell_numerators(rng, 30, 0.10, 0.30, 0.11)
    assert st.evaluate_cell(nums, den, counts)["counts_digest"] == st.counts_digest(counts)
    assert st.evaluate_cell(nums, den, other)["counts_digest"] == st.counts_digest(other)
    out = st.nll_report({"orig": den, "M5": den * 1.01}, den, counts, pairs=[("M5", "orig")])
    assert out["counts_digest"] == st.counts_digest(counts)


def _evaluate_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_evaluate.py"
    spec = importlib.util.spec_from_file_location("qwen3_main_evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fingerprint_blocks_mixed_conditions(tmp_path):
    script = _evaluate_script()
    raw = tmp_path / "raw"
    raw.mkdir()
    first = {"model": {"revision": "a"}, "smoke": False, "unit_hashes": {"wt2": ["x", "y"]}, "files_sha256": {"units.json": "1"}}
    script.check_or_write_fingerprint(raw, first)  # 空なので書く
    (raw / "L14.pt").write_bytes(b"")
    script.check_or_write_fingerprint(raw, first)  # 同じ条件なら通る
    changed = {**first, "unit_hashes": {"wt2": ["y", "x"]}}  # 数は同じで順番だけ違う
    with pytest.raises(RuntimeError, match="unit_hashes"):
        script.check_or_write_fingerprint(raw, changed)
    (raw / script.FINGERPRINT_NAME).unlink()
    with pytest.raises(RuntimeError, match="ありません"):
        script.check_or_write_fingerprint(raw, first)  # 結果があるのに記録がない


def test_aggregate_statements_thresholds():
    rows = [{"class_wt2": "high", "class_c4": "high", "F_below_0.2_wt2": True} for _ in range(26)]
    rows += [{"class_wt2": "unresolved", "class_c4": "mid", "F_below_0.2_wt2": False} for _ in range(8)]
    result = st.aggregate_statements(rows)
    assert result["W"] and result["B"] and result["F"]
    rows[-1] = {"class_wt2": "mid", "class_c4": "low", "F_below_0.2_wt2": False}
    result = st.aggregate_statements(rows)
    assert result["W"] and not result["B"]
    rows[0] = {"class_wt2": "low", "class_c4": "high", "F_below_0.2_wt2": True}
    assert not st.aggregate_statements(rows)["W"]
    with pytest.raises(ValueError):
        st.aggregate_statements(rows[:-1])


def test_nll_report_difference_interval_contains_point():
    rng = np.random.default_rng(0)
    tokens = rng.integers(500, 1500, size=20).astype(float)
    base = tokens * 2.5
    sums = {"orig": base, "M5": base * 1.01}
    counts = st.bootstrap_counts(20, n_resamples=500)
    out = st.nll_report(sums, tokens, counts, pairs=[("M5", "orig")])
    point = out["dnll_M5_minus_orig"]
    assert out["dnll_M5_minus_orig_lo"] <= point <= out["dnll_M5_minus_orig_hi"]
    assert point == pytest.approx(0.025)


# ---------------------------------------------------------------- 小さい Qwen3


@pytest.fixture(scope="module")
def tiny_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).to(torch.float32).eval()
    return model


def _articles():
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (n,), generator=generator).tolist() for n in (1300, 700)]


def test_article_windows_match_sliding_window(tiny_model):
    articles = _articles()
    sums, tokens = ev.full_model_article_nll(tiny_model, articles, device="cpu")
    for index, ids in enumerate(articles):
        reference = sliding_window_perplexity(tiny_model, torch.tensor(ids), max_length=1024, stride=512)
        assert int(tokens[index]) == reference["n_tokens"]
        assert sums[index] == pytest.approx(reference["mean_nll"] * reference["n_tokens"], rel=1e-5)


@pytest.mark.parametrize("layer_index", [0, 1, 2])
def test_nll_from_layer_equals_full_model(tiny_model, layer_index):
    articles = _articles()
    full_sums, full_tokens = ev.full_model_article_nll(tiny_model, articles, device="cpu")
    caches = ev.cache_window_inputs(tiny_model, layer_index, articles, device="cpu")
    sums, tokens = ev.nll_from_layer(tiny_model, layer_index, caches, len(articles))
    assert np.array_equal(tokens, full_tokens)
    np.testing.assert_allclose(sums, full_sums, rtol=1e-6)


def _tiny_cell(model, layer_index, projection, scale):
    module = ev.module_at(model.model.layers[layer_index], ev.PROJECTION_PATHS[projection])
    weight = module.weight.detach().to(torch.float64).clone()
    generator = torch.Generator().manual_seed(3)
    matrices = {
        name: weight + scale * (i + 1) * torch.randn(weight.shape, generator=generator, dtype=torch.float64)
        for i, name in enumerate(ev.SUBLAYER_METHODS)
    }
    return ev.CellMethods(f"L{layer_index}_{projection}_50", layer_index, projection, 50, weight, matrices, {})


def test_unit_quadratic_pass_matches_grams(tiny_model):
    layer_index = 1
    generator = torch.Generator().manual_seed(2)
    chunks = [(0, torch.randint(0, 97, (40,), generator=generator)), (0, torch.randint(0, 97, (30,), generator=generator)), (1, torch.randint(0, 97, (50,), generator=generator))]
    cell = _tiny_cell(tiny_model, layer_index, "q", 0.01)
    delta = cell.weight - cell.matrices["M5"]
    products = {"attn_in": torch.stack([cell.weight.T @ cell.weight, delta.T @ delta])}
    results, totals, caches = ev.unit_quadratic_pass(tiny_model, layer_index, chunks, 2, products, device="cpu")
    modules = {"attn_in": tiny_model.model.layers[layer_index].self_attn.q_proj}
    per_unit = []
    for unit in (0, 1):
        tensors = [ids for owner, ids in chunks if owner == unit]
        gram = accumulate_input_grams(tiny_model, modules, tensors, device="cpu")["attn_in"]["total"]
        per_unit.append(gram)
    for unit, gram in enumerate(per_unit):
        expected = float(torch.sum((delta @ gram) * delta))
        assert results["attn_in"][unit, 1] == pytest.approx(expected, rel=1e-10)
        assert results["attn_in"][unit, 0] == pytest.approx(float(torch.sum((cell.weight @ gram) * cell.weight)), rel=1e-10)
    torch.testing.assert_close(totals["attn_in"], per_unit[0] + per_unit[1], rtol=1e-12, atol=1e-12)
    assert caches.unit_of_chunk == [0, 0, 1]
    assert len(caches.hidden) == 3 and len(caches.mlp_in) == 3


@pytest.mark.parametrize("projection", ["o", "up"])
def test_sublayer_pass_identity_and_positive_error(tiny_model, projection):
    layer_index = 2
    generator = torch.Generator().manual_seed(4)
    chunks = [(0, torch.randint(0, 97, (40,), generator=generator)), (1, torch.randint(0, 97, (40,), generator=generator))]
    group = ev.INPUT_GROUPS[projection]
    size = tiny_model.model.layers[layer_index].self_attn.o_proj.in_features if group == "o_in" else 64
    products = {group: torch.zeros((1, size, size), dtype=torch.float64)}
    _results, _totals, caches = ev.unit_quadratic_pass(tiny_model, layer_index, chunks, 2, products, device="cpu")
    cell = _tiny_cell(tiny_model, layer_index, projection, 0.02)
    nums, dens, gated, checks = ev.sublayer_pass(tiny_model, layer_index, caches, 2, [cell], device="cpu")
    assert all(item["ok"] for item in checks), checks
    assert np.all(dens["attn"] > 0) and np.all(dens["mlp"] > 0)
    errors = nums[cell.cell]
    assert np.all(errors["M1"] > 0)
    assert errors["M5"].sum() > errors["M1"].sum()
    if projection == "up":
        assert np.all(gated[cell.cell]["num_M1"] > 0) and np.all(gated[cell.cell]["den"] > 0)
    else:
        assert gated == {}
