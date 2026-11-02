"""更新 scope と GPU 常駐 batch 供給の契約を、小さい合成データで検証する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import factorize_named_tt_linear
from nn_compression.training import (
    GpuResidentTensorBatches,
    TTUpdateScope,
    accumulate_scoped_full_batch_gradients,
    accumulate_tt_linear_pair_full_batch_gradients,
    assert_scope_budget_completed,
    incomplete_scope_names,
    load_tt_linear_sweep_checkpoint,
    reject_incomplete_scope_rows,
    require_recorded_smoke_gate,
    run_resumable_condition_grid,
    save_scoped_full_batch_run,
    train_scoped_full_batch_budget,
)
from nn_compression.training import tt_linear_scoped_update as scoped
from nn_compression.tt_mps import merge_tt_matrix_cores, tt_matrix_to_dense

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)
CLOSE = {"rtol": 1e-8, "atol": 1e-8}


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


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


def _budget_kwargs(train_x, train_y, val_x, val_y, **overrides):
    kwargs = {
        "layer_name": "fc1",
        "batch_size": 3,
        "total_updates": 4,
        "evaluation_interval": 4,
        "rank_cap": 4,
        "tt_learning_rate": 1e-2,
        "dense_learning_rate": 1e-3,
        "diagnostic_images": train_x[:2].clone(),
        "diagnostic_labels": train_y[:2].clone(),
        "device": torch.device("cpu"),
        "local_steps_per_pair": 1,
        "max_sweeps": 1,
        "center": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _dense_state(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("fc1.cores.")
    }


def _core_state(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("fc1.cores.")
    }


def test_dataloader_joint_gradients_match_existing_accumulator():
    """既存の DataLoader 集計と、scope=joint の集計が一致する。"""
    train_x, train_y, _, _ = _synthetic()
    model = _build_model()
    loader = _loader(train_x, train_y, batch_size=3)
    merged = merge_tt_matrix_cores(model.fc1.cores[0], model.fc1.cores[1])
    merged = merged.detach().clone().requires_grad_(True)
    old_grad, old_dense = accumulate_tt_linear_pair_full_batch_gradients(
        model,
        loader,
        "fc1",
        0,
        merged,
        device=torch.device("cpu"),
    )
    new_grad, new_dense, _ = accumulate_scoped_full_batch_gradients(
        model,
        loader,
        scope=TTUpdateScope("joint"),
        layer_name="fc1",
        batch_size=3,
        device=torch.device("cpu"),
        pair_index=0,
    )
    torch.testing.assert_close(new_grad, old_grad, **CLOSE)
    assert set(new_dense) == set(old_dense)
    for name in old_dense:
        torch.testing.assert_close(new_dense[name], old_dense[name], **CLOSE)


def test_gpu_resident_batches_keep_order_and_can_be_scanned_again():
    images = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    labels = torch.arange(5)
    source = GpuResidentTensorBatches(images, labels)
    first = [(batch[0].clone(), batch[1].clone()) for batch in source.batches(2)]
    second = [(batch[0].clone(), batch[1].clone()) for batch in source.batches(2)]
    assert [int(batch[1][0]) for batch in first] == [0, 2, 4]
    assert [int(item.shape[0]) for item, _ in first] == [2, 2, 1]
    assert all(torch.equal(left[0], right[0]) and torch.equal(left[1], right[1]) for left, right in zip(first, second))
    seen = torch.cat([labels for _, labels in second])
    assert torch.equal(seen, labels)


def test_resident_gradients_match_loader_including_short_last_batch():
    train_x, train_y, _, _ = _synthetic(n_train=5)
    model = _build_model()
    loader = _loader(train_x, train_y, batch_size=2)
    resident = GpuResidentTensorBatches(train_x, train_y)
    loader_grad, loader_dense, loader_loss = accumulate_scoped_full_batch_gradients(
        model,
        loader,
        scope=TTUpdateScope("joint"),
        layer_name="fc1",
        batch_size=2,
        device=torch.device("cpu"),
        pair_index=0,
    )
    resident_grad, resident_dense, resident_loss = accumulate_scoped_full_batch_gradients(
        model,
        resident,
        scope=TTUpdateScope("joint"),
        layer_name="fc1",
        batch_size=2,
        device=torch.device("cpu"),
        pair_index=0,
    )
    torch.testing.assert_close(resident_grad, loader_grad, **CLOSE)
    for name in loader_dense:
        torch.testing.assert_close(resident_dense[name], loader_dense[name], **CLOSE)
    assert resident_loss == pytest.approx(loader_loss)


def test_weighted_loss_uses_sample_counts_not_batch_means():
    """最後の小さい batch を、batch 数ではなく sample 数で重み付ける。"""
    train_x, train_y, _, _ = _synthetic(n_train=5)
    model = _build_model()
    source = GpuResidentTensorBatches(train_x, train_y)
    _, _, loss = accumulate_scoped_full_batch_gradients(
        model,
        source,
        scope=TTUpdateScope("dense_only"),
        layer_name="fc1",
        batch_size=2,
        device=torch.device("cpu"),
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    manual = 0.0
    for start, end in ((0, 2), (2, 4), (4, 5)):
        logits = model(train_x[start:end])
        manual += (end - start) / 5 * float(criterion(logits, train_y[start:end]).detach())
    assert loss == pytest.approx(manual)


def test_joint_updates_tt_and_dense_parameters():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    before_dense = _dense_state(model)
    before_cores = _core_state(model)
    result = train_scoped_full_batch_budget(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 2),
        scope=TTUpdateScope("joint"),
        **_budget_kwargs(train_x, train_y, val_x, val_y),
    )
    assert result.stop_reason == "max_updates"
    assert result.completed_updates == 4
    assert result.svd_call_count == 4
    assert result.dense_candidate_count == 4
    assert result.core_writeback_count == 4
    assert result.center == 0
    assert any(not torch.equal(before_cores[name], value) for name, value in _core_state(model).items())
    assert any(not torch.equal(before_dense[name], value) for name, value in _dense_state(model).items())
    assert [row["pair_index"] for row in result.local_updates] == [0, 1, 1, 0]
    assert [row["direction"] for row in result.local_updates] == ["right", "right", "left", "left"]
    assert [row["center_after"] for row in result.local_updates] == [1, 2, 1, 0]
    assert result.history[0]["cumulative_updates"] == 0
    assert result.history[0]["is_best"] is True or result.history[-1]["is_best"] is True


def test_tt_only_freezes_dense_parameters_and_builds_no_dense_candidates():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    before_dense = _dense_state(model)
    before_cores = _core_state(model)

    def reject_dense_candidates(*args, **kwargs):
        raise AssertionError("tt_only は通常 parameter の更新候補を作りません。")

    original_candidates = scoped._dense_update_candidates
    scoped._dense_update_candidates = reject_dense_candidates
    try:
        result = train_scoped_full_batch_budget(
            model,
            GpuResidentTensorBatches(train_x, train_y),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 2),
            scope=TTUpdateScope("tt_only"),
            **_budget_kwargs(train_x, train_y, val_x, val_y, dense_learning_rate=None),
        )
    finally:
        scoped._dense_update_candidates = original_candidates
    assert result.stop_reason == "max_updates"
    assert result.dense_candidate_count == 0
    assert result.svd_call_count == 4
    assert all(torch.equal(before_dense[name], value) for name, value in _dense_state(model).items())
    assert any(not torch.equal(before_cores[name], value) for name, value in _core_state(model).items())
    assert all(row["dense_gradient_norm"] is None for row in result.local_updates)
    assert all(row["dense_relative_update"] is None for row in result.local_updates)
    assert all(row["retained_rank"] is not None for row in result.local_updates)
    assert all(row["fixed_state_ok"] is True for row in result.history)


def test_dense_only_never_calls_svd_or_core_writeback():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    before_cores = _core_state(model)
    before_dense = _dense_state(model)
    before_weight = tt_matrix_to_dense(list(model.fc1.cores), OUT_MODES, IN_MODES).detach().clone()

    def reject(*args, **kwargs):
        raise AssertionError("dense_only は TT の結合・SVD・書き戻しを呼びません。")

    originals = {
        "split_merged_tt_matrix": scoped.split_merged_tt_matrix,
        "merge_tt_matrix_cores": scoped.merge_tt_matrix_cores,
        "replace_tt_matrix_core_pair": scoped.replace_tt_matrix_core_pair,
        "assign_tt_linear_cores": scoped.assign_tt_linear_cores,
    }
    for name, _function in originals.items():
        setattr(scoped, name, reject)
    try:
        result = train_scoped_full_batch_budget(
            model,
            GpuResidentTensorBatches(train_x, train_y),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 2),
            scope=TTUpdateScope("dense_only"),
            **_budget_kwargs(
                train_x,
                train_y,
                val_x,
                val_y,
                tt_learning_rate=None,
                local_steps_per_pair=None,
                max_sweeps=None,
            ),
        )
    finally:
        for name, function in originals.items():
            setattr(scoped, name, function)
    assert result.stop_reason == "max_updates"
    assert result.svd_call_count == 0
    assert result.core_writeback_count == 0
    assert result.center is None
    assert all(torch.equal(before_cores[name], value) for name, value in _core_state(model).items())
    after_weight = tt_matrix_to_dense(list(model.fc1.cores), OUT_MODES, IN_MODES)
    torch.testing.assert_close(after_weight, before_weight)
    assert any(not torch.equal(before_dense[name], value) for name, value in _dense_state(model).items())
    for row in result.local_updates:
        assert row["pair_index"] is None
        assert row["direction"] is None
        assert row["center_before"] is None
        assert row["center_after"] is None
        assert row["retained_rank"] is None
        assert row["discarded_squared_norm"] is None
        assert row["tt_gradient_norm"] is None
        assert row["tt_relative_update"] is None
        assert row["dense_gradient_norm"] is not None


def test_initial_state_is_best_and_equal_loss_keeps_earlier_checkpoint():
    assert scoped._prefers_validation_checkpoint(0.2, 80, 0.2, 0) is False
    assert scoped._prefers_validation_checkpoint(0.2, 0, 0.2, 80) is True
    assert scoped._prefers_validation_checkpoint(0.1, 80, 0.2, 0) is True

    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    calls = {"validation": 0}

    def fake_evaluate(model, loader, criterion, device):
        if len(loader.dataset) == len(train_x):
            return 1.5, 0.25
        calls["validation"] += 1
        if calls["validation"] == 1:
            return 0.4, 0.2
        return 0.4, 0.9

    original = scoped.evaluate
    scoped.evaluate = fake_evaluate
    try:
        result = train_scoped_full_batch_budget(
            model,
            _loader(train_x, train_y, 3),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 2),
            scope=TTUpdateScope("joint"),
            **_budget_kwargs(train_x, train_y, val_x, val_y),
        )
    finally:
        scoped.evaluate = original
    assert result.best_update_count == 0
    assert result.best_validation_loss == pytest.approx(0.4)
    assert result.history[0]["is_best"] is True
    assert result.history[-1]["is_best"] is False
    assert result.history[-1]["early_stopping_validation_accuracy"] == pytest.approx(0.9)


def test_non_finite_stop_is_distinct_from_implementation_exceptions():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def raise_non_finite(*args, **kwargs):
        raise scoped.NonFiniteLocalUpdate("合成した非有限値")

    original = scoped.accumulate_scoped_full_batch_gradients
    scoped.accumulate_scoped_full_batch_gradients = raise_non_finite
    try:
        result = train_scoped_full_batch_budget(
            model,
            _loader(train_x, train_y, 3),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 2),
            scope=TTUpdateScope("joint"),
            **_budget_kwargs(train_x, train_y, val_x, val_y),
        )
    finally:
        scoped.accumulate_scoped_full_batch_gradients = original
    assert result.stop_reason == "non_finite"
    assert result.completed_updates == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())

    def raise_bug(*args, **kwargs):
        raise KeyError("実装例外")

    scoped.accumulate_scoped_full_batch_gradients = raise_bug
    try:
        with pytest.raises(KeyError, match="実装例外"):
            train_scoped_full_batch_budget(
                model,
                _loader(train_x, train_y, 3),
                _loader(train_x, train_y, 3),
                _loader(val_x, val_y, 2),
                scope=TTUpdateScope("joint"),
                **_budget_kwargs(train_x, train_y, val_x, val_y),
            )
    finally:
        scoped.accumulate_scoped_full_batch_gradients = original
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_out_of_memory_is_a_separate_stop_reason():
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()

    def raise_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("合成したメモリ不足")

    original = scoped.accumulate_scoped_full_batch_gradients
    scoped.accumulate_scoped_full_batch_gradients = raise_oom
    try:
        result = train_scoped_full_batch_budget(
            model,
            _loader(train_x, train_y, 3),
            _loader(train_x, train_y, 3),
            _loader(val_x, val_y, 2),
            scope=TTUpdateScope("dense_only"),
            **_budget_kwargs(
                train_x,
                train_y,
                val_x,
                val_y,
                tt_learning_rate=None,
                local_steps_per_pair=None,
                max_sweeps=None,
            ),
        )
    finally:
        scoped.accumulate_scoped_full_batch_gradients = original
    assert result.stop_reason == "out_of_memory"
    assert result.completed_updates == 0


def test_save_and_reload_round_trip(tmp_path):
    train_x, train_y, val_x, val_y = _synthetic()
    model = _build_model()
    result = train_scoped_full_batch_budget(
        model,
        _loader(train_x, train_y, 3),
        _loader(train_x, train_y, 3),
        _loader(val_x, val_y, 2),
        scope=TTUpdateScope("dense_only"),
        **_budget_kwargs(
            train_x,
            train_y,
            val_x,
            val_y,
            tt_learning_rate=None,
            local_steps_per_pair=None,
            max_sweeps=None,
        ),
    )
    config = {
        "layer_name": "fc1",
        "out_modes": OUT_MODES,
        "in_modes": IN_MODES,
        "scope": "dense_only",
    }
    save_scoped_full_batch_run(tmp_path, model, result, config)
    reloaded = _build_model(seed=1)
    load_tt_linear_sweep_checkpoint(tmp_path / "final_model.pt", reloaded)
    for name, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name].cpu(), value.cpu())
    with (tmp_path / "local_updates.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pair_index"] == ""
    assert rows[0]["discarded_squared_norm"] == ""
    assert rows[0]["tt_gradient_norm"] == ""
    assert rows[0]["dense_gradient_norm"] != ""
    assert rows[0]["dense_gradient_norm"] != "0"
    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["config"]["scope"] == "dense_only"
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "initial_model.pt").is_file()
    assert (tmp_path / "best_model.pt").is_file()


def test_mismatched_settings_do_not_skip_or_overwrite(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = {
        "condition_id": "joint",
        "scope": "joint",
        "checkpoint_sha256": "sha-original",
    }
    fields = ("condition_id", "scope", "checkpoint_sha256")

    def run_one(item):
        directory = models / item["condition_id"]
        directory.mkdir(parents=True)
        for name in (
            "initial_model.pt",
            "final_model.pt",
            "best_model.pt",
            "history.csv",
            "local_updates.csv",
        ):
            (directory / name).write_text("kept", encoding="utf-8")
        (directory / "run_state.json").write_text(
            json.dumps({"config": item}),
            encoding="utf-8",
        )
        return item

    run_resumable_condition_grid(
        [spec],
        summary_path=summary,
        models_dir=models,
        fieldnames=fields,
        setting_fields=("scope", "checkpoint_sha256"),
        run_one=run_one,
    )
    changed = dict(spec)
    changed["checkpoint_sha256"] = "sha-changed"
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    def reject(item):
        raise AssertionError("設定が違う結果を skip せず、実行してもいけません。")

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_resumable_condition_grid(
            [changed],
            summary_path=summary,
            models_dir=models,
            fieldnames=fields,
            setting_fields=("scope", "checkpoint_sha256"),
            run_one=reject,
        )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_smoke_gate_rejects_mismatch_without_starting():
    expected = {"checkpoint_sha256": "abc", "old_new_joint_passed": True}
    with pytest.raises(RuntimeError, match="未完了"):
        require_recorded_smoke_gate({}, expected, list(expected))
    with pytest.raises(RuntimeError, match="主実験は開始しません"):
        require_recorded_smoke_gate(
            {"checkpoint_sha256": "other", "old_new_joint_passed": False},
            expected,
            list(expected),
        )
    require_recorded_smoke_gate(expected, expected, list(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA がないため GPU 常駐 batch は確認しません。")
def test_cuda_resident_batches_stay_on_the_specified_device():
    images = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
    labels = torch.arange(3, device="cuda")
    source = GpuResidentTensorBatches(images, labels)
    batches = list(source.batches(2))
    assert [int(batch.shape[0]) for batch, _ in batches] == [2, 1]
    assert all(batch.device.type == "cuda" and label.device.type == "cuda" for batch, label in batches)
    again = list(source.batches(2))
    assert torch.equal(again[0][1], batches[0][1])


_BUDGET_FIELDS = ("condition_id", "scope", "stop_reason", "completed_updates")
_BUDGET_ARTIFACTS = (
    "initial_model.pt",
    "final_model.pt",
    "best_model.pt",
    "history.csv",
    "local_updates.csv",
    "run_state.json",
)


def _write_budget_artifacts(directory, spec):
    directory.mkdir(parents=True, exist_ok=True)
    for name in _BUDGET_ARTIFACTS:
        if name == "run_state.json":
            (directory / name).write_text(json.dumps({"config": spec}), encoding="utf-8")
        else:
            (directory / name).write_text("kept", encoding="utf-8")


def _snapshot(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_only_completed_budget_is_skipped_on_resume(tmp_path):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = {"condition_id": "joint", "scope": "joint"}
    calls = []

    def run_one(item):
        calls.append(item["condition_id"])
        _write_budget_artifacts(models / item["condition_id"], item)
        assert_scope_budget_completed("max_updates", 3200, 3200)
        return {**item, "stop_reason": "max_updates", "completed_updates": 3200}

    kwargs = {
        "summary_path": summary,
        "models_dir": models,
        "fieldnames": _BUDGET_FIELDS,
        "setting_fields": ("scope",),
        "run_one": run_one,
    }
    run_resumable_condition_grid([spec], **kwargs)
    run_resumable_condition_grid([spec], **kwargs)
    assert calls == ["joint"]


@pytest.mark.parametrize(
    ("stop_reason", "completed_updates"),
    [
        ("non_finite", 10),
        ("out_of_memory", 80),
        ("max_updates", 80),
    ],
)
def test_abnormal_stop_is_not_recorded_or_skipped(tmp_path, stop_reason, completed_updates):
    summary = tmp_path / "condition_summary.csv"
    models = tmp_path / "models"
    spec = {"condition_id": "joint", "scope": "joint"}

    def run_one(item):
        _write_budget_artifacts(models / item["condition_id"], item)
        assert_scope_budget_completed(stop_reason, completed_updates, 3200)
        raise AssertionError("正常完了していない行を返してはいけません。")

    with pytest.raises(RuntimeError, match="記録しません"):
        run_resumable_condition_grid(
            [spec],
            summary_path=summary,
            models_dir=models,
            fieldnames=_BUDGET_FIELDS,
            setting_fields=("scope",),
            run_one=run_one,
        )
    assert not summary.exists()
    assert (models / "joint" / "final_model.pt").read_text(encoding="utf-8") == "kept"
    before = _snapshot(tmp_path)

    def should_not_run(item):
        raise AssertionError("異常停止を完了済みとしてスキップまたは再実行してはいけません。")

    with pytest.raises(RuntimeError, match="スキップはしません"):
        run_resumable_condition_grid(
            [spec],
            summary_path=summary,
            models_dir=models,
            fieldnames=_BUDGET_FIELDS,
            setting_fields=("scope",),
            run_one=should_not_run,
        )
    assert _snapshot(tmp_path) == before


def test_recorded_abnormal_row_is_rejected_before_resume_skip():
    rows = [
        {"condition_id": "joint", "stop_reason": "non_finite", "completed_updates": "10"},
        {"condition_id": "tt_only", "stop_reason": "out_of_memory", "completed_updates": "80"},
        {"condition_id": "dense_only", "stop_reason": "max_updates", "completed_updates": "100"},
    ]
    for row in rows:
        with pytest.raises(RuntimeError, match="記録しません"):
            reject_incomplete_scope_rows([row], ("joint", "tt_only", "dense_only"), 3200)
    reject_incomplete_scope_rows(
        [{"condition_id": "joint", "stop_reason": "max_updates", "completed_updates": "3200"}],
        ("joint", "tt_only", "dense_only"),
        3200,
    )


def _ready_observation():
    return {
        "settings_match": True,
        "history_exists": True,
        "stop_reason": "max_updates",
        "completed_updates": 3200,
        "final_updates": 3200,
    }


def test_partial_completion_skips_comparison_and_invalid_records_raise():
    scopes = ("joint", "tt_only", "dense_only")
    ready = _ready_observation()
    assert incomplete_scope_names({"joint": ready, "tt_only": None, "dense_only": None}, scopes, 3200) == [
        "tt_only",
        "dense_only",
    ]
    assert incomplete_scope_names({"joint": ready, "tt_only": ready, "dense_only": None}, scopes, 3200) == [
        "dense_only",
    ]
    assert incomplete_scope_names({name: ready for name in scopes}, scopes, 3200) == []
    assert incomplete_scope_names({name: None for name in scopes}, scopes, 3200) == list(scopes)

    mismatched = dict(ready)
    mismatched["settings_match"] = False
    with pytest.raises(ValueError, match="設定"):
        incomplete_scope_names({"joint": mismatched, "tt_only": ready, "dense_only": ready}, scopes, 3200)

    missing_history = dict(ready)
    missing_history["history_exists"] = False
    with pytest.raises(ValueError, match="history.csv"):
        incomplete_scope_names({"joint": ready, "tt_only": missing_history, "dense_only": ready}, scopes, 3200)

    short = dict(ready)
    short["final_updates"] = 80
    with pytest.raises(ValueError, match="最終更新数"):
        incomplete_scope_names({"joint": ready, "tt_only": ready, "dense_only": short}, scopes, 3200)

    stopped = dict(ready)
    stopped["stop_reason"] = "non_finite"
    with pytest.raises(ValueError, match="正常完了"):
        incomplete_scope_names({"joint": stopped, "tt_only": ready, "dense_only": ready}, scopes, 3200)
