"""全データ勾配の TTLinear 2-core スイープを、小さい合成データで検証する。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import assign_tt_linear_cores, factorize_named_tt_linear
from nn_compression.tt_mps import (
    canonicalize_tt_matrix_cores,
    merge_tt_matrix_cores,
    split_merged_tt_matrix,
    tt_matrix_canonicality_errors,
    tt_matrix_ranks,
)
from nn_compression.models import MNISTMLP
from nn_compression.training.loops import evaluate
from nn_compression.training.tt_linear_full_batch_sweep import (
    NonFiniteLocalUpdate,
    _advance_stopping,
    _relative_update,
    accumulate_tt_linear_pair_full_batch_gradients,
    load_condition_summary,
    load_tt_linear_sweep_checkpoint,
    run_resumable_condition_grid,
    save_tt_linear_full_batch_sweep_run,
    train_tt_linear_full_batch_sweeps,
    update_tt_linear_two_core_pair,
)

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)
# このテストの行列は 8 次元前後。float64 の還元 SVD と平均損失の加算順の差は
# 1e-8 より十分小さい、という前提で atol=rtol=1e-8 を使う。
CLOSE = {"rtol": 1e-8, "atol": 1e-8}


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


def _shape_rank_bounds(out_modes, in_modes, max_rank):
    """形状と rank 上限だけから決まる bond の上限。数値 rank ではない。"""
    left = 1
    remaining = math.prod(m * n for m, n in zip(out_modes, in_modes))
    ranks = [1]
    for index, (out_mode, in_mode) in enumerate(zip(out_modes, in_modes)):
        left *= out_mode * in_mode
        remaining //= out_mode * in_mode
        if index == len(out_modes) - 1:
            ranks.append(1)
        else:
            ranks.append(min(left, remaining, max_rank))
    return tuple(ranks)


def _loader(features, labels, batch_size):
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def _synthetic(n_train=7, n_val=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    train_x = torch.randn(n_train, 8, generator=generator, dtype=torch.float64)
    train_y = torch.randint(0, 3, (n_train,), generator=generator)
    val_x = torch.randn(n_val, 8, generator=generator, dtype=torch.float64)
    val_y = torch.randint(0, 3, (n_val,), generator=generator)
    return train_x, train_y, val_x, val_y


def _build_model(seed=0):
    torch.manual_seed(seed)
    model = factorize_named_tt_linear(TinyMLP(), "fc1", OUT_MODES, IN_MODES, max_rank=4)
    for core in model.fc1.cores:
        core.requires_grad_(False)
    return model


def _diagnostic(features, labels):
    # loader を消費せず、dataset の先頭 index を直接読むのと同じ並び。
    return features[:2].clone(), labels[:2].clone()


def _train_kwargs(model, train_x, train_y, val_x, val_y, **overrides):
    images, labels = _diagnostic(train_x, train_y)
    kwargs = {
        "layer_name": "fc1",
        "rank_cap": 4,
        "tt_pair_learning_rate": 1e-2,
        "dense_parameter_learning_rate": 1e-3,
        "max_sweeps": 1,
        "local_steps_per_pair": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "diagnostic_images": images,
        "diagnostic_labels": labels,
        "device": torch.device("cpu"),
        "center": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _state_values(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def test_fashion_mnist_modes_match_mlp_io_and_are_separate_from_numerical_rank():
    out_modes = (4, 2, 2, 2, 4, 4)
    in_modes = (2, 2, 2, 2, 7, 7)
    assert math.prod(out_modes) == 512
    assert math.prod(in_modes) == 784
    bounds = _shape_rank_bounds(out_modes, in_modes, max_rank=12)
    assert bounds == (1, 8, 12, 12, 12, 12, 1)

    torch.manual_seed(0)
    model = factorize_named_tt_linear(MNISTMLP(), "fc1", out_modes, in_modes, max_rank=12)
    assert model.fc1.in_features == 784
    assert model.fc1.out_features == 512
    assert tuple(core.shape[1] for core in model.fc1.cores) == out_modes
    assert tuple(core.shape[2] for core in model.fc1.cores) == in_modes
    actual = tt_matrix_ranks(list(model.fc1.cores))
    assert actual == bounds
    logits = model(torch.randn(2, 1, 28, 28))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_chunked_full_batch_gradients_match_for_t_and_dense_parameters():
    train_x, train_y, _, _ = _synthetic()
    model = _build_model()
    before = _state_values(model)
    merged = merge_tt_matrix_cores(model.fc1.cores[0], model.fc1.cores[1])
    full = _loader(train_x, train_y, batch_size=7)
    chunks = _loader(train_x, train_y, batch_size=3)
    leaf = merged.detach().clone().requires_grad_(True)
    grad_full, dense_full = accumulate_tt_linear_pair_full_batch_gradients(
        model, full, "fc1", 0, leaf, device=torch.device("cpu")
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
    leaf_chunks = merged.detach().clone().requires_grad_(True)
    grad_chunks, dense_chunks = accumulate_tt_linear_pair_full_batch_gradients(
        model, chunks, "fc1", 0, leaf_chunks, device=torch.device("cpu")
    )
    torch.testing.assert_close(grad_chunks, grad_full, **CLOSE)
    assert set(dense_chunks) == set(dense_full)
    assert "fc1.bias" in dense_full
    assert "fc2.weight" in dense_full
    assert "fc2.bias" in dense_full
    assert not any(name.startswith("fc1.cores.") for name in dense_full)
    for name in dense_full:
        torch.testing.assert_close(dense_chunks[name], dense_full[name], **CLOSE)
    assert torch.linalg.vector_norm(grad_full) > 0
    assert torch.linalg.vector_norm(dense_full["fc2.weight"]) > 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_update_uses_separate_learning_rates_and_leaves_other_cores_unchanged():
    train_x, train_y, _, _ = _synthetic()
    model = _build_model()
    candidate = copy.deepcopy(model)
    before_cores = [core.detach().clone() for core in model.fc1.cores]
    before_dense = _state_values(model)
    merged = merge_tt_matrix_cores(model.fc1.cores[0], model.fc1.cores[1]).detach()
    leaf = merged.clone().requires_grad_(True)
    loader = _loader(train_x, train_y, batch_size=3)
    grad_t, dense_grads = accumulate_tt_linear_pair_full_batch_gradients(
        model, loader, "fc1", 0, leaf, device=torch.device("cpu")
    )
    tt_lr = 2e-2
    dense_lr = 5e-3
    images, labels = _diagnostic(train_x, train_y)
    local_update = update_tt_linear_two_core_pair(
        model,
        candidate,
        loader,
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=8,
        tt_pair_learning_rate=tt_lr,
        dense_parameter_learning_rate=dense_lr,
        diagnostic_images=images,
        diagnostic_labels=labels,
        device=torch.device("cpu"),
    )
    for name, parameter in model.named_parameters():
        if name.startswith("fc1.cores."):
            continue
        expected = before_dense[name] - dense_lr * dense_grads[name]
        torch.testing.assert_close(parameter, expected, **CLOSE)
        wrong = before_dense[name] - tt_lr * dense_grads[name]
        assert not torch.allclose(parameter, wrong, rtol=1e-6, atol=1e-8)
    expected_t = merged - tt_lr * grad_t
    split = split_merged_tt_matrix(expected_t, max_bond=8, direction="right")
    restored = merge_tt_matrix_cores(model.fc1.cores[0], model.fc1.cores[1])
    reference = merge_tt_matrix_cores(split.left_core, split.right_core)
    torch.testing.assert_close(restored, reference, **CLOSE)
    assert torch.equal(model.fc1.cores[2], before_cores[2])
    assert candidate.fc2.weight.data_ptr() != model.fc2.weight.data_ptr()
    tt_norm = float(torch.linalg.vector_norm(merged))
    tt_grad_norm = float(torch.linalg.vector_norm(grad_t))
    dense_values = [
        before_dense[name]
        for name in before_dense
        if not name.startswith("fc1.cores.")
    ]
    dense_grad_values = [dense_grads[name] for name in before_dense if name in dense_grads]
    dense_norm = float(
        torch.stack([torch.linalg.vector_norm(value).square() for value in dense_values]).sum().sqrt()
    )
    dense_grad_norm = float(
        torch.stack(
            [torch.linalg.vector_norm(value).square() for value in dense_grad_values]
        ).sum().sqrt()
    )
    assert local_update.tt_parameter_norm == pytest.approx(tt_norm)
    assert local_update.tt_gradient_norm == pytest.approx(tt_grad_norm)
    assert local_update.tt_relative_update == pytest.approx(tt_lr * tt_grad_norm / tt_norm)
    assert local_update.dense_parameter_norm == pytest.approx(dense_norm)
    assert local_update.dense_gradient_norm == pytest.approx(dense_grad_norm)
    assert local_update.dense_relative_update == pytest.approx(dense_lr * dense_grad_norm / dense_norm)
    assert local_update.tt_relative_update > 0
    assert local_update.dense_relative_update > 0


def test_round_trip_update_count_center_and_latest_cores():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    events = []

    def observer(event):
        events.append(
            {
                "pair_index": event["pair_index"],
                "local_step": event["local_step"],
                "left_core": event["left_core"].clone(),
                "right_core": event["right_core"].clone(),
            }
        )

    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        observer=observer,
        **_train_kwargs(model, train_x, train_y, val_x, val_y, local_steps_per_pair=1),
    )
    # d=3 なので 1 往復は 2*(3-1)*1 = 4 回。中心は site 0 に戻る。
    assert result.stop_reason == "max_sweeps"
    assert result.completed_sweeps == 1
    assert result.successful_local_updates == 4
    assert result.incomplete_sweep_successes == 0
    assert result.center == 0
    ok_logs = [row for row in result.local_updates if row["status"] == "ok"]
    assert len(ok_logs) == 4
    for row in ok_logs:
        assert math.isfinite(row["tt_relative_update"])
        assert math.isfinite(row["dense_relative_update"])
        assert row["tt_parameter_norm"] > 0
        assert row["dense_parameter_norm"] > 0
    assert events[0]["pair_index"] == 0 and events[0]["local_step"] == 0
    assert events[1]["pair_index"] == 1
    assert not torch.equal(events[1]["left_core"], events[0]["left_core"])

    reference = _build_model()
    reference_candidate = copy.deepcopy(reference)
    images, labels = _diagnostic(train_x, train_y)
    update_tt_linear_two_core_pair(
        reference,
        reference_candidate,
        _loader(train_x, train_y, 3),
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=4,
        tt_pair_learning_rate=1e-2,
        dense_parameter_learning_rate=1e-3,
        diagnostic_images=images,
        diagnostic_labels=labels,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(events[1]["left_core"], reference.fc1.cores[1], **CLOSE)

    model_two = _build_model(seed=1)
    train_x, train_y, val_x, val_y = _synthetic(seed=1)
    result_two = train_tt_linear_full_batch_sweeps(
        model_two,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            model_two,
            train_x,
            train_y,
            val_x,
            val_y,
            local_steps_per_pair=2,
            max_sweeps=1,
            early_stopping_patience=10,
        ),
    )
    assert result_two.successful_local_updates == 8
    assert result_two.center == 0
    assert result_two.completed_sweeps == 1


def test_non_finite_diagnostic_does_not_commit_cores_or_dense_parameters():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    before = _state_values(model)
    images, labels = _diagnostic(train_x, train_y)
    images = torch.full_like(images, torch.nan)
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            diagnostic_images=images,
            diagnostic_labels=labels,
            max_sweeps=1,
        ),
    )
    assert result.stop_reason == "non_finite"
    assert result.successful_local_updates == 0
    assert result.completed_sweeps == 0
    assert result.best_sweep == 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_relative_update_rejects_zero_parameter_with_nonzero_gradient():
    assert _relative_update(1e-3, 0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="相対更新量"):
        _relative_update(1e-3, 1.0, 0.0)


@pytest.mark.parametrize("exc_type", [ValueError, AssertionError, RuntimeError])
def test_unexpected_exception_propagates_without_applying_the_failed_candidate(
    monkeypatch, exc_type
):
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    reference = _build_model()
    images, labels = _diagnostic(train_x, train_y)
    update_tt_linear_two_core_pair(
        reference,
        copy.deepcopy(reference),
        _loader(train_x, train_y, 3),
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=4,
        tt_pair_learning_rate=1e-2,
        dense_parameter_learning_rate=1e-3,
        diagnostic_images=images,
        diagnostic_labels=labels,
        device=torch.device("cpu"),
    )
    import nn_compression.training.tt_linear_full_batch_sweep as sweep_mod

    real_split = sweep_mod.split_merged_tt_matrix
    calls = {"n": 0}

    def fail_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise exc_type("forced implementation failure")
        return real_split(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "split_merged_tt_matrix", fail_on_second)
    with pytest.raises(exc_type, match="forced implementation failure"):
        train_tt_linear_full_batch_sweeps(
            model,
            _loader(train_x, train_y, 3),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 4),
            **_train_kwargs(model, train_x, train_y, val_x, val_y, max_sweeps=1),
        )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, reference.get_parameter(name), **CLOSE)


def test_direction_error_is_not_a_non_finite_update():
    train_x, train_y, _, _ = _synthetic()
    model = _build_model()
    before = _state_values(model)
    images, labels = _diagnostic(train_x, train_y)
    with pytest.raises(ValueError, match="direction") as caught:
        update_tt_linear_two_core_pair(
            model,
            copy.deepcopy(model),
            _loader(train_x, train_y, 3),
            layer_name="fc1",
            pair_index=0,
            direction="up",
            center=0,
            rank_cap=4,
            tt_pair_learning_rate=1e-2,
            dense_parameter_learning_rate=1e-3,
            diagnostic_images=images,
            diagnostic_labels=labels,
            device=torch.device("cpu"),
        )
    assert not isinstance(caught.value, NonFiniteLocalUpdate)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_svd_failure_keeps_last_success_and_is_not_non_finite(monkeypatch):
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    initial = _state_values(model)
    reference = _build_model()
    images, labels = _diagnostic(train_x, train_y)
    update_tt_linear_two_core_pair(
        reference,
        copy.deepcopy(reference),
        _loader(train_x, train_y, 3),
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=4,
        tt_pair_learning_rate=1e-2,
        dense_parameter_learning_rate=1e-3,
        diagnostic_images=images,
        diagnostic_labels=labels,
        device=torch.device("cpu"),
    )
    import nn_compression.training.tt_linear_full_batch_sweep as sweep_mod

    real_split = sweep_mod.split_merged_tt_matrix
    calls = {"n": 0}

    def fail_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise torch.linalg.LinAlgError("forced svd failure")
        return real_split(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "split_merged_tt_matrix", fail_on_second)
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(model, train_x, train_y, val_x, val_y, max_sweeps=1),
    )
    assert result.stop_reason == "svd_failure"
    assert result.stop_reason != "non_finite"
    assert "LinAlgError" in result.stop_detail
    assert result.completed_sweeps == 0
    assert result.successful_local_updates == 1
    assert result.incomplete_sweep_successes == 1
    assert result.stop_sweep == 1
    assert result.stop_pair == 1
    assert result.stop_direction == "right"
    assert result.stop_local_step == 0
    assert result.center == 1
    assert result.best_sweep == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, reference.get_parameter(name), **CLOSE)
        if name == "fc2.weight":
            assert torch.equal(result.best_state[name].cpu(), initial[name].cpu())


def test_sweep_zero_is_the_initial_best_candidate():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(model, train_x, train_y, val_x, val_y, max_sweeps=0),
    )
    assert result.best_sweep == 0
    assert result.completed_sweeps == 0
    assert result.history[0]["sweep"] == 0
    assert result.history[0]["is_best"] is True
    assert result.best_state["fc2.weight"].data_ptr() != model.fc2.weight.data_ptr()
    assert torch.equal(result.best_state["fc2.weight"], model.fc2.weight)


def test_early_stopping_delta_patience_and_independent_best(monkeypatch):
    sequence = [1.0, 0.97, 0.5, 0.49, 0.49]
    calls = {"n": 0}

    def fake_evaluate(model, loader, criterion, device):
        if len(loader.dataset) == 4:
            value = sequence[calls["n"]]
            calls["n"] += 1
            return value, 0.2
        return 0.3, 0.4

    monkeypatch.setattr(
        "nn_compression.training.tt_linear_full_batch_sweep.evaluate",
        fake_evaluate,
    )
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 7),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            max_sweeps=6,
            early_stopping_patience=2,
            early_stopping_min_delta=0.1,
        ),
    )
    assert result.stop_reason == "early_stopping"
    assert result.completed_sweeps == 4
    assert [row["validation_loss"] for row in result.history] == sequence
    assert [row["stall_count"] for row in result.history] == [0, 1, 0, 1, 2]
    assert [row["early_stop_baseline"] for row in result.history] == [1.0, 1.0, 0.5, 0.5, 0.5]
    # 0.97 は min_delta 未満なので基準は動かないが、best は更新される。
    assert result.best_sweep == 3
    assert result.best_validation_loss == 0.49


def test_advance_stopping_treats_small_improvement_as_best_only():
    baseline, best, stall, improved, best_updated = _advance_stopping(1.0, 1.0, 0, 0.97, 0.1)
    assert improved is False
    assert best_updated is True
    assert baseline == 1.0
    assert best == 0.97
    assert stall == 1
    baseline, best, stall, improved, best_updated = _advance_stopping(baseline, best, stall, 0.8, 0.1)
    assert improved is True
    assert stall == 0
    assert baseline == 0.8
    assert best == 0.8


def test_save_and_reload_reproduces_forward_and_validation(tmp_path):
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    val_loader = _loader(val_x, val_y, 4)
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        val_loader,
        **_train_kwargs(model, train_x, train_y, val_x, val_y, max_sweeps=1),
    )
    criterion = nn.CrossEntropyLoss()
    loss_before, acc_before = evaluate(model, val_loader, criterion, torch.device("cpu"))
    logits_before = model(val_x)
    config = {
        "layer_name": "fc1",
        "out_modes": OUT_MODES,
        "in_modes": IN_MODES,
        "rank_cap": 4,
        "tt_pair_learning_rate": 1e-2,
        "dense_parameter_learning_rate": 1e-3,
        "batch_size": 3,
        "local_steps_per_pair": 1,
        "max_sweeps": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "seed": 0,
        "dtype": "float64",
        "best_sweep": result.best_sweep,
        "best_validation_loss": result.best_validation_loss,
        "completed_sweeps": result.completed_sweeps,
        "successful_local_updates": result.successful_local_updates,
    }
    save_tt_linear_full_batch_sweep_run(tmp_path, model, result, config)
    assert (tmp_path / "initial_model.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    local_header = (tmp_path / "local_updates.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "tt_relative_update" in local_header.split(",")
    assert "dense_relative_update" in local_header.split(",")
    assert (tmp_path / "local_updates.csv").is_file()
    assert (tmp_path / "run_state.json").is_file()

    # 最終モデルを保存したあと、別インスタンスへ best を読む。
    reloaded = _build_model(seed=1)
    load_tt_linear_sweep_checkpoint(tmp_path / "best_model.pt", reloaded)
    logits_after = reloaded(val_x)
    loss_after, acc_after = evaluate(reloaded, val_loader, criterion, torch.device("cpu"))
    torch.testing.assert_close(logits_after, logits_before if result.best_sweep == 1 else reloaded(val_x), **CLOSE)
    if result.best_sweep == 1:
        torch.testing.assert_close(logits_after, logits_before, **CLOSE)
        assert loss_after == pytest.approx(loss_before, rel=1e-8, abs=1e-8)
        assert acc_after == pytest.approx(acc_before)
    else:
        initial = _build_model()
        load_tt_linear_sweep_checkpoint(tmp_path / "initial_model.pt", initial)
        torch.testing.assert_close(reloaded(val_x), initial(val_x), **CLOSE)
        loss_initial, _ = evaluate(initial, val_loader, criterion, torch.device("cpu"))
        assert loss_after == pytest.approx(loss_initial, rel=1e-8, abs=1e-8)

    final_model = _build_model(seed=2)
    load_tt_linear_sweep_checkpoint(tmp_path / "final_model.pt", final_model)
    torch.testing.assert_close(final_model(val_x), logits_before, **CLOSE)


def _fake_validation_loss(monkeypatch, sequence):
    calls = {"n": 0}

    def fake_evaluate(model, loader, criterion, device):
        if len(loader.dataset) == 4:
            value = sequence[calls["n"]]
            calls["n"] += 1
            return value, 0.2
        return 0.3, 0.4

    monkeypatch.setattr(
        "nn_compression.training.tt_linear_full_batch_sweep.evaluate",
        fake_evaluate,
    )
    return calls


def test_disabling_early_stopping_reaches_max_sweeps_and_keeps_best(monkeypatch):
    sequence = [1.0, 0.5, 0.5, 0.5]
    calls = _fake_validation_loss(monkeypatch, sequence)
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    disabled = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 7),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            max_sweeps=3,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            early_stopping_enabled=False,
        ),
    )
    assert disabled.stop_reason == "max_sweeps"
    assert disabled.completed_sweeps == 3
    assert disabled.best_sweep == 1
    assert disabled.best_validation_loss == 0.5
    assert "early stopping は無効" in disabled.stop_detail

    calls["n"] = 0
    enabled_model = _build_model()
    enabled = train_tt_linear_full_batch_sweeps(
        enabled_model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 7),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            enabled_model,
            train_x,
            train_y,
            val_x,
            val_y,
            max_sweeps=3,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            early_stopping_enabled=True,
        ),
    )
    assert enabled.stop_reason == "early_stopping"
    assert enabled.completed_sweeps == 2
    assert enabled.best_sweep == 1
    assert enabled.best_validation_loss == 0.5


def test_early_stopping_enabled_rejects_non_bool():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    with pytest.raises(TypeError, match="early_stopping_enabled"):
        train_tt_linear_full_batch_sweeps(
            model,
            _loader(train_x, train_y, 3),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 4),
            **_train_kwargs(
                model,
                train_x,
                train_y,
                val_x,
                val_y,
                early_stopping_enabled=1,
            ),
        )


class SixCoreMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 3, dtype=torch.float64)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


def test_local_steps_repeat_on_updated_cores_and_six_cores_take_10l_updates():
    out_modes = (2, 2, 2, 2, 2, 2)
    in_modes = (2, 2, 2, 2, 2, 2)
    rank_cap = 3
    local_steps = 4
    generator = torch.Generator().manual_seed(0)
    train_x = torch.randn(5, 64, generator=generator, dtype=torch.float64)
    train_y = torch.randint(0, 3, (5,), generator=generator)
    val_x = torch.randn(4, 64, generator=generator, dtype=torch.float64)
    val_y = torch.randint(0, 3, (4,), generator=generator)
    torch.manual_seed(0)
    model = factorize_named_tt_linear(SixCoreMLP(), "fc1", out_modes, in_modes, max_rank=rank_cap)
    canonical = canonicalize_tt_matrix_cores(list(model.fc1.cores), out_modes, in_modes, 0)
    assign_tt_linear_cores(model.fc1, canonical, requires_grad=False)
    seen = []

    def observer(event):
        cores = [core.detach() for core in model.fc1.cores]
        errors = tt_matrix_canonicality_errors(cores, out_modes, in_modes, event["center"])
        ortho = [float(value) for value in errors["left"]] + [float(value) for value in errors["right"]]
        seen.append(
            {
                "pair_index": event["pair_index"],
                "direction": event["direction"],
                "local_step": event["local_step"],
                "center": event["center"],
                "left_core": event["left_core"].clone(),
                "dense": model.fc2.weight.detach().clone(),
                "bias": model.fc1.bias.detach().clone(),
                "max_ortho": max(ortho) if ortho else 0.0,
                "ranks": tt_matrix_ranks(cores),
            }
        )

    images, labels = train_x[:2].clone(), train_y[:2].clone()
    result = train_tt_linear_full_batch_sweeps(
        model,
        _loader(train_x, train_y, 2),
        _loader(train_x, train_y, 2),
        _loader(val_x, val_y, 4),
        layer_name="fc1",
        rank_cap=rank_cap,
        tt_pair_learning_rate=1e-2,
        dense_parameter_learning_rate=1e-2,
        max_sweeps=1,
        local_steps_per_pair=local_steps,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        early_stopping_enabled=False,
        diagnostic_images=images,
        diagnostic_labels=labels,
        device=torch.device("cpu"),
        center=0,
        observer=observer,
    )
    assert result.stop_reason == "max_sweeps"
    assert result.completed_sweeps == 1
    assert result.successful_local_updates == 10 * local_steps
    assert result.center == 0
    assert len(seen) == 10 * local_steps
    for event in seen:
        pair_index = event["pair_index"]
        assert event["center"] in {pair_index, pair_index + 1}
        if event["local_step"] == 0:
            expected = pair_index if event["direction"] == "right" else pair_index + 1
        else:
            expected = pair_index + 1 if event["direction"] == "right" else pair_index
        assert event["center"] == expected
        assert event["max_ortho"] < 1e-8
        assert event["ranks"][0] == 1 and event["ranks"][-1] == 1
        assert all(rank <= rank_cap for rank in event["ranks"])
    for previous, current in zip(seen, seen[1:]):
        same_pair = (
            previous["pair_index"] == current["pair_index"]
            and previous["direction"] == current["direction"]
        )
        if same_pair:
            assert current["local_step"] == previous["local_step"] + 1
            assert not torch.equal(current["left_core"], previous["left_core"])
            assert not torch.equal(current["dense"], previous["dense"])
            assert not torch.equal(current["bias"], previous["bias"])
    for row in result.local_updates:
        if row["status"] != "ok":
            continue
        if row["direction"] == "right":
            assert row["center_after"] == row["pair_index"] + 1
        else:
            assert row["center_after"] == row["pair_index"]
        assert row["retained_rank"] <= rank_cap


def test_each_condition_reloads_the_same_initial_state(tmp_path):
    train_x, train_y, val_x, val_y = _synthetic()
    source = _build_model()
    path = tmp_path / "shared_init.pt"
    torch.save(
        {
            "layer_name": "fc1",
            "out_modes": list(OUT_MODES),
            "in_modes": list(IN_MODES),
            "state_dict": {key: value.detach().cpu() for key, value in source.state_dict().items()},
        },
        path,
    )
    digest = path.read_bytes()

    def fresh():
        model = _build_model(seed=7)
        load_tt_linear_sweep_checkpoint(path, model)
        return model

    first = fresh()
    second = fresh()
    for key, value in source.state_dict().items():
        assert torch.equal(first.state_dict()[key], value)
        assert torch.equal(second.state_dict()[key], value)

    result_first = train_tt_linear_full_batch_sweeps(
        first,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            first,
            train_x,
            train_y,
            val_x,
            val_y,
            tt_pair_learning_rate=1e-2,
            dense_parameter_learning_rate=1e-2,
            max_sweeps=1,
            early_stopping_enabled=False,
        ),
    )
    result_second = train_tt_linear_full_batch_sweeps(
        second,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 4),
        **_train_kwargs(
            second,
            train_x,
            train_y,
            val_x,
            val_y,
            tt_pair_learning_rate=1e-1,
            dense_parameter_learning_rate=1e-1,
            max_sweeps=1,
            early_stopping_enabled=False,
        ),
    )
    assert torch.equal(result_first.initial_state["fc2.weight"], source.fc2.weight)
    assert torch.equal(result_second.initial_state["fc2.weight"], source.fc2.weight)
    assert not torch.equal(first.fc2.weight, second.fc2.weight)
    assert not torch.equal(result_second.initial_state["fc2.weight"], first.fc2.weight)
    assert path.read_bytes() == digest


_GRID_FIELDS = (
    "condition_id",
    "joint_learning_rate_multiplier",
    "max_sweeps",
    "local_steps_per_pair",
    "status",
)
_GRID_SETTINGS = (
    "joint_learning_rate_multiplier",
    "max_sweeps",
    "local_steps_per_pair",
)
_GRID_ARTIFACTS = (
    "initial_model.pt",
    "final_model.pt",
    "best_model.pt",
    "history.csv",
    "local_updates.csv",
    "run_state.json",
)


def _grid_spec(condition_id, max_sweeps=5):
    return {
        "condition_id": condition_id,
        "joint_learning_rate_multiplier": 1.0,
        "max_sweeps": max_sweeps,
        "local_steps_per_pair": 1,
    }


def _finish_condition(models_dir, spec, status="正常完了"):
    directory = Path(models_dir) / spec["condition_id"]
    directory.mkdir(parents=True)
    for name in _GRID_ARTIFACTS:
        if name == "run_state.json":
            continue
        (directory / name).write_text("saved", encoding="utf-8")
    (directory / "run_state.json").write_text(
        json.dumps({"config": spec}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**spec, "status": status}


def test_condition_summary_is_saved_after_each_condition(tmp_path):
    summary = tmp_path / "results" / "condition_summary.csv"
    models = tmp_path / "models"
    calls = []

    def run_one(spec):
        calls.append(spec["condition_id"])
        return _finish_condition(models, spec)

    rows = run_resumable_condition_grid(
        [_grid_spec("jointx1_s5_l1")],
        summary_path=summary,
        models_dir=models,
        fieldnames=_GRID_FIELDS,
        setting_fields=_GRID_SETTINGS,
        run_one=run_one,
    )
    assert calls == ["jointx1_s5_l1"]
    assert [row["condition_id"] for row in rows] == ["jointx1_s5_l1"]
    assert len(summary.read_text(encoding="utf-8").splitlines()) == 2
    assert not (tmp_path / "results" / "condition_summary.csv.tmp").exists()


def test_recorded_condition_is_verified_and_skipped(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = _grid_spec("jointx1_s5_l1")
    calls = []

    def run_one(item):
        calls.append(item["condition_id"])
        return _finish_condition(models, item)

    run_resumable_condition_grid(
        [spec],
        summary_path=summary,
        models_dir=models,
        fieldnames=_GRID_FIELDS,
        setting_fields=_GRID_SETTINGS,
        run_one=run_one,
    )
    run_resumable_condition_grid(
        [spec],
        summary_path=summary,
        models_dir=models,
        fieldnames=_GRID_FIELDS,
        setting_fields=_GRID_SETTINGS,
        run_one=run_one,
    )
    assert calls == ["jointx1_s5_l1"]
    text = summary.read_text(encoding="utf-8")
    text = text.replace(",5,", ",9,", 1)
    summary.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="max_sweeps"):
        run_resumable_condition_grid(
            [spec],
            summary_path=summary,
            models_dir=models,
            fieldnames=_GRID_FIELDS,
            setting_fields=_GRID_SETTINGS,
            run_one=run_one,
        )
    assert calls == ["jointx1_s5_l1"]


def test_resume_runs_only_missing_conditions(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    calls = []

    def run_one(spec):
        calls.append(spec["condition_id"])
        return _finish_condition(models, spec)

    first = _grid_spec("jointx1_s5_l1")
    second = _grid_spec("jointx1_s20_l1", max_sweeps=20)
    run_resumable_condition_grid(
        [first],
        summary_path=summary,
        models_dir=models,
        fieldnames=_GRID_FIELDS,
        setting_fields=_GRID_SETTINGS,
        run_one=run_one,
    )
    rows = run_resumable_condition_grid(
        [first, second],
        summary_path=summary,
        models_dir=models,
        fieldnames=_GRID_FIELDS,
        setting_fields=_GRID_SETTINGS,
        run_one=run_one,
    )
    assert calls == ["jointx1_s5_l1", "jointx1_s20_l1"]
    assert [row["condition_id"] for row in rows] == ["jointx1_s5_l1", "jointx1_s20_l1"]


def test_orphan_condition_directory_without_summary_row_is_rejected(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    directory = models / "jointx1_s5_l1"
    directory.mkdir(parents=True)
    (directory / "history.csv").write_text("partial", encoding="utf-8")

    def run_one(spec):
        raise AssertionError("不完全なディレクトリを実行してはいけません。")

    with pytest.raises(RuntimeError, match="集計CSVに記録がありません"):
        run_resumable_condition_grid(
            [_grid_spec("jointx1_s5_l1")],
            summary_path=summary,
            models_dir=models,
            fieldnames=_GRID_FIELDS,
            setting_fields=_GRID_SETTINGS,
            run_one=run_one,
        )
    assert (directory / "history.csv").read_text(encoding="utf-8") == "partial"
    assert not summary.exists()


def test_duplicate_condition_id_is_rejected(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    summary.write_text(
        "condition_id,max_sweeps\nsame,5\nsame,5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重複"):
        load_condition_summary(summary)
    with pytest.raises(ValueError, match="重複"):
        run_resumable_condition_grid(
            [_grid_spec("same"), _grid_spec("same")],
            summary_path=tmp_path / "fresh.csv",
            models_dir=tmp_path / "models",
            fieldnames=_GRID_FIELDS,
            setting_fields=_GRID_SETTINGS,
            run_one=lambda spec: (_ for _ in ()).throw(AssertionError("実行してはいけません。")),
        )


def test_unexpected_exception_keeps_earlier_summary_rows(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"

    def run_one(spec):
        if spec["condition_id"] == "second":
            raise RuntimeError("想定外の実装例外")
        return _finish_condition(models, spec)

    with pytest.raises(RuntimeError, match="想定外の実装例外"):
        run_resumable_condition_grid(
            [_grid_spec("first"), _grid_spec("second")],
            summary_path=summary,
            models_dir=models,
            fieldnames=_GRID_FIELDS,
            setting_fields=_GRID_SETTINGS,
            run_one=run_one,
        )
    rows = load_condition_summary(summary)
    assert [row["condition_id"] for row in rows] == ["first"]


def _file_snapshot(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in Path(path).rglob("*")
        if item.is_file()
    }


def _sha_spec(condition_id, sha="sha-original", max_sweeps=5):
    spec = _grid_spec(condition_id, max_sweeps)
    spec["checkpoint_sha256"] = sha
    return spec


_SHA_FIELDS = _GRID_FIELDS + ("checkpoint_sha256",)
_SHA_SETTINGS = _GRID_SETTINGS + ("checkpoint_sha256",)


def test_matching_checkpoint_sha256_is_skipped(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = _sha_spec("jointx1_s5_l1")
    calls = []

    def run_one(item):
        calls.append(item["condition_id"])
        return _finish_condition(models, item)

    kwargs = {
        "summary_path": summary,
        "models_dir": models,
        "fieldnames": _SHA_FIELDS,
        "setting_fields": _SHA_SETTINGS,
        "run_one": run_one,
    }
    run_resumable_condition_grid([spec], **kwargs)
    before = _file_snapshot(tmp_path)
    run_resumable_condition_grid([spec], **kwargs)
    assert calls == ["jointx1_s5_l1"]
    assert _file_snapshot(tmp_path) == before


def test_different_summary_checkpoint_sha256_is_rejected_without_changes(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = _sha_spec("jointx1_s5_l1")
    run_resumable_condition_grid(
        [spec],
        summary_path=summary,
        models_dir=models,
        fieldnames=_SHA_FIELDS,
        setting_fields=_SHA_SETTINGS,
        run_one=lambda item: _finish_condition(models, item),
    )
    text = summary.read_text(encoding="utf-8")
    summary.write_text(text.replace("sha-original", "sha-changed", 1), encoding="utf-8")
    before = _file_snapshot(tmp_path)

    def run_one(item):
        raise AssertionError("SHA-256 が違う条件を実行してはいけません。")

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_resumable_condition_grid(
            [spec],
            summary_path=summary,
            models_dir=models,
            fieldnames=_SHA_FIELDS,
            setting_fields=_SHA_SETTINGS,
            run_one=run_one,
        )
    assert _file_snapshot(tmp_path) == before


def test_different_run_state_checkpoint_sha256_is_rejected_without_changes(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = _sha_spec("jointx1_s5_l1")
    run_resumable_condition_grid(
        [spec],
        summary_path=summary,
        models_dir=models,
        fieldnames=_SHA_FIELDS,
        setting_fields=_SHA_SETTINGS,
        run_one=lambda item: _finish_condition(models, item),
    )
    state_path = models / "jointx1_s5_l1" / "run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["config"]["checkpoint_sha256"] = "sha-changed"
    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    before = _file_snapshot(tmp_path)

    def run_one(item):
        raise AssertionError("SHA-256 が違う条件を実行してはいけません。")

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_resumable_condition_grid(
            [spec],
            summary_path=summary,
            models_dir=models,
            fieldnames=_SHA_FIELDS,
            setting_fields=_SHA_SETTINGS,
            run_one=run_one,
        )
    assert _file_snapshot(tmp_path) == before


def test_exception_during_resume_leaves_existing_artifacts_and_summary(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    first = _sha_spec("first")
    run_resumable_condition_grid(
        [first],
        summary_path=summary,
        models_dir=models,
        fieldnames=_SHA_FIELDS,
        setting_fields=_SHA_SETTINGS,
        run_one=lambda item: _finish_condition(models, item),
    )
    before = _file_snapshot(tmp_path)

    def run_one(item):
        if item["condition_id"] == "second":
            raise RuntimeError("想定外の実装例外")
        return _finish_condition(models, item)

    with pytest.raises(RuntimeError, match="想定外の実装例外"):
        run_resumable_condition_grid(
            [first, _sha_spec("second")],
            summary_path=summary,
            models_dir=models,
            fieldnames=_SHA_FIELDS,
            setting_fields=_SHA_SETTINGS,
            run_one=run_one,
        )
    assert _file_snapshot(tmp_path) == before
