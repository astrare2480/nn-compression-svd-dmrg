"""utils.experiment_generation の世代保存・検証・判定。"""

from __future__ import annotations

import hashlib
import json

import pytest

from nn_compression.utils import (
    ActiveGeneration,
    GenerationDecision,
    compare_generation_config,
    decide_generation_action,
    load_active_generation,
    publish_generation,
    validate_generation_state,
)

STATE_KEYS = ("completed_sweeps", "total_local_updates", "stop_reason")
PINNED = {"initial_model.pt": "initial_model_sha256"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish(directory, *, config, run_state, initial=b"initial", extra=None):
    def prepare(root):
        target = root / "initial_model.pt"
        if not target.exists():
            target.write_bytes(initial)

    def write(partial):
        (partial / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (partial / "run_state.json").write_text(json.dumps(run_state), encoding="utf-8")
        (partial / "model.bin").write_bytes(b"weights")
        return ["model.bin", "config.json", "run_state.json"]

    fields = {key: run_state[key] for key in STATE_KEYS}
    fields["initial_model_sha256"] = _sha(initial)
    fields.update(extra or {})
    return publish_generation(
        directory,
        write,
        manifest_fields=fields,
        pointer_fields=STATE_KEYS,
        prepare=prepare,
    )


def _config(max_sweeps=10, **overrides):
    config = {"lr": 0.3, "max_sweeps": max_sweeps, "notebook_code_sha256": "a"}
    config.update(overrides)
    return config


def _state(stop_reason="running", sweeps=2):
    return {"completed_sweeps": sweeps, "total_local_updates": sweeps * 80, "stop_reason": stop_reason}


def test_publish_and_load_round_trip(tmp_path):
    directory = tmp_path / "condition"
    final = _publish(directory, config=_config(), run_state=_state())
    assert final == directory / "generation_000001"
    active = load_active_generation(directory, pinned_files=PINNED, required_manifest_keys=("initial_model_sha256",))
    assert isinstance(active, ActiveGeneration)
    assert active.path == final
    assert active.generation == 1
    assert active.pointer == {
        "generation": 1,
        "directory": "generation_000001",
        "completed_sweeps": 2,
        "total_local_updates": 160,
        "stop_reason": "running",
    }
    assert set(active.manifest["files"]) == {"model.bin", "config.json", "run_state.json"}
    assert active.manifest["files"]["model.bin"] == _sha(b"weights")
    assert sorted(path.name for path in directory.iterdir()) == [
        "active_generation.json",
        "generation_000001",
        "initial_model.pt",
    ]

    second = _publish(directory, config=_config(), run_state=_state(sweeps=3))
    assert second.name == "generation_000002"
    assert load_active_generation(directory).generation == 2
    assert (directory / "generation_000001").is_dir()


def test_reads_layout_written_by_existing_notebooks(tmp_path):
    directory = tmp_path / "rank_04" / "dmrg_like"
    artifact = directory / "generation_000003"
    artifact.mkdir(parents=True)
    (directory / "initial_model.pt").write_bytes(b"init")
    (artifact / "history.csv").write_text("a\r\n1\r\n", encoding="utf-8", newline="")
    manifest = {
        "generation": 3,
        "completed_epochs": None,
        "completed_sweeps": 4,
        "optimizer_steps": 320,
        "total_local_updates": 320,
        "history_rows": 5,
        "center": 0,
        "stop_reason": "max_sweeps",
        "initial_model_sha256": _sha(b"init"),
        "rank_initial_sha256": _sha(b"init"),
        "files": {"history.csv": _sha("a\r\n1\r\n".encode("utf-8"))},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (directory / "active_generation.json").write_text(
        json.dumps({"generation": 3, "directory": "generation_000003", "stop_reason": "max_sweeps"}),
        encoding="utf-8",
    )
    active = load_active_generation(
        directory,
        legacy_names=("history.csv",),
        pinned_files=PINNED,
        required_manifest_keys=("rank_initial_sha256",),
    )
    assert active.path == artifact
    assert active.manifest["center"] == 0


def test_missing_pointer_is_none_unless_legacy_or_partial_exists(tmp_path):
    directory = tmp_path / "condition"
    directory.mkdir()
    assert load_active_generation(directory, legacy_names=("history.csv",)) is None
    (directory / "history.csv").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy"):
        load_active_generation(directory, legacy_names=("history.csv",))
    (directory / "history.csv").unlink()
    (directory / ".partial_generation_000001_1_2").mkdir()
    with pytest.raises(RuntimeError, match="partial"):
        load_active_generation(directory)


def test_tampered_or_inconsistent_generation_is_rejected(tmp_path):
    directory = tmp_path / "condition"
    final = _publish(directory, config=_config(), run_state=_state())
    (final / "model.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="manifest と一致しない"):
        load_active_generation(directory)

    other = tmp_path / "other"
    _publish(other, config=_config(), run_state=_state())
    (other / "initial_model.pt").write_bytes(b"changed")
    assert load_active_generation(other) is not None
    with pytest.raises(RuntimeError, match="initial_model.pt"):
        load_active_generation(other, pinned_files=PINNED)
    with pytest.raises(RuntimeError, match="必要な key"):
        load_active_generation(other, required_manifest_keys=("rank_initial_sha256",))

    pointer = json.loads((other / "active_generation.json").read_text(encoding="utf-8"))
    pointer["generation"] = 9
    (other / "active_generation.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(RuntimeError, match="世代が一致しません"):
        load_active_generation(other)
    pointer["directory"] = "../outside"
    (other / "active_generation.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(RuntimeError, match="世代ディレクトリではありません"):
        load_active_generation(other)


def test_publish_failure_keeps_previous_generation_and_cleans_up(tmp_path):
    directory = tmp_path / "condition"
    _publish(directory, config=_config(), run_state=_state())
    before = (directory / "active_generation.json").read_bytes()

    def broken(partial):
        (partial / "half.bin").write_bytes(b"x")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        publish_generation(directory, broken)
    assert (directory / "active_generation.json").read_bytes() == before
    assert sorted(path.name for path in directory.iterdir()) == [
        "active_generation.json",
        "generation_000001",
        "initial_model.pt",
    ]


@pytest.mark.parametrize(
    ("names", "error"),
    [
        ([], ValueError),
        (["a.bin", "a.bin"], ValueError),
        (["../a.bin"], ValueError),
        (["manifest.json"], ValueError),
        (["missing.bin"], RuntimeError),
        ("a.bin", TypeError),
    ],
)
def test_publish_rejects_bad_file_lists(tmp_path, names, error):
    directory = tmp_path / "condition"

    def write(partial):
        (partial / "a.bin").write_bytes(b"a")
        (partial / "manifest.json").write_text("{}", encoding="utf-8")
        return names

    with pytest.raises(error):
        publish_generation(directory, write)
    assert sorted(path.name for path in directory.iterdir()) == []


def test_publish_rejects_reserved_fields_and_missing_pointer_fields(tmp_path):
    def write(partial):
        (partial / "a.bin").write_bytes(b"a")
        return ["a.bin"]

    with pytest.raises(ValueError, match="予約"):
        publish_generation(tmp_path / "c1", write, manifest_fields={"files": {}})
    with pytest.raises(ValueError, match="pointer_fields"):
        publish_generation(tmp_path / "c2", write, pointer_fields=("stop_reason",))
    assert not (tmp_path / "c1" / "active_generation.json").exists()


def test_manifest_fields_callable_is_evaluated_after_writing(tmp_path):
    seen = {}

    def write(partial):
        (partial / "a.bin").write_bytes(b"a")
        seen["seconds"] = 1.5
        return ["a.bin"]

    final = publish_generation(tmp_path / "c", write, manifest_fields=lambda: {"seconds": seen["seconds"]})
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seconds"] == 1.5


def test_publish_lock_rejects_concurrent_writer_and_is_released(tmp_path):
    directory = tmp_path / "condition"
    directory.mkdir()
    (directory / ".publish_lock").write_text("123", encoding="ascii")

    def write(partial):
        (partial / "a.bin").write_bytes(b"a")
        return ["a.bin"]

    with pytest.raises(RuntimeError, match="並行保存"):
        publish_generation(directory, write)
    assert (directory / ".publish_lock").read_text(encoding="ascii") == "123"
    (directory / ".publish_lock").unlink()
    publish_generation(directory, write)
    assert not (directory / ".publish_lock").exists()


def test_generation_numbers_skip_existing_directories(tmp_path):
    directory = tmp_path / "condition"
    (directory / "generation_000004").mkdir(parents=True)
    (directory / ".partial_generation_000007_1_2").mkdir()

    def write(partial):
        (partial / "a.bin").write_bytes(b"a")
        return ["a.bin"]

    assert publish_generation(directory, write).name == "generation_000008"


def test_validate_generation_state():
    validate_generation_state({"stop_reason": "running"}, {"stop_reason": "running"}, ("stop_reason", "center"))
    with pytest.raises(RuntimeError, match="stop_reason"):
        validate_generation_state({"stop_reason": "running"}, {"stop_reason": "max_sweeps"}, ("stop_reason",))


def test_compare_generation_config():
    saved = _config(10)
    assert compare_generation_config(saved, _config(10, notebook_code_sha256="b"), volatile_keys=("notebook_code_sha256",), cap_fields=("max_sweeps",)) == ()
    assert compare_generation_config(saved, _config(20), volatile_keys=("notebook_code_sha256",), cap_fields=("max_sweeps",)) == ("max_sweeps",)
    with pytest.raises(ValueError, match="上限以外"):
        compare_generation_config(saved, _config(10, lr=0.1), volatile_keys=("notebook_code_sha256",), cap_fields=("max_sweeps",))
    with pytest.raises(ValueError, match="上限以外"):
        compare_generation_config(saved, _config(10, notebook_code_sha256="b"), cap_fields=("max_sweeps",))
    with pytest.raises(ValueError, match="項目"):
        compare_generation_config(saved, {**_config(10), "new": 1}, cap_fields=("max_sweeps",))
    with pytest.raises(ValueError, match="上限 field"):
        compare_generation_config(saved, saved, cap_fields=("max_epochs",))


NOTEBOOK07_RULES = {
    "cap_fields": ("max_sweeps",),
    "finished_stops": ("early_stopping", "max_epochs", "max_sweeps"),
    "resumable_stops": ("running", "max_epochs", "max_sweeps"),
    "blocked_stops": ("non_finite", "out_of_memory"),
    "volatile_keys": ("notebook_code_sha256",),
    "state_keys": STATE_KEYS,
}
NOTEBOOK08_RULES = {
    "cap_fields": ("max_sweeps",),
    "finished_stops": ("early_stopping", "max_sweeps"),
    "resumable_stops": ("running", "max_sweeps"),
    "blocked_stops": ("non_finite", "out_of_memory", "invalid_singular_values"),
    "cap_independent_stops": ("early_stopping",),
    "volatile_keys": ("notebook_code_sha256",),
    "state_keys": STATE_KEYS,
}


def _decide(directory, config, rules):
    return decide_generation_action(directory, config, pinned_files=PINNED, **rules)


def test_decision_start_skip_resume(tmp_path):
    missing = tmp_path / "missing"
    decision = _decide(missing, _config(), NOTEBOOK07_RULES)
    assert decision == GenerationDecision("start", None, None, None, ())

    running = tmp_path / "running"
    _publish(running, config=_config(), run_state=_state("running"))
    decision = _decide(running, _config(10, notebook_code_sha256="new"), NOTEBOOK07_RULES)
    assert decision.action == "resume"
    assert decision.run_state == _state("running")
    assert decision.saved_config == _config()
    assert decision.generation.generation == 1

    capped = tmp_path / "capped"
    _publish(capped, config=_config(), run_state=_state("max_sweeps", sweeps=10))
    assert _decide(capped, _config(10), NOTEBOOK07_RULES).action == "skip"
    extended = _decide(capped, _config(20), NOTEBOOK07_RULES)
    assert extended.action == "resume"
    assert extended.changed_caps == ("max_sweeps",)


def test_early_stopping_with_raised_cap_follows_rules(tmp_path):
    directory = tmp_path / "early"
    _publish(directory, config=_config(), run_state=_state("early_stopping"))
    assert _decide(directory, _config(10), NOTEBOOK07_RULES).action == "skip"
    with pytest.raises(RuntimeError, match="再開できない"):
        _decide(directory, _config(20), NOTEBOOK07_RULES)
    assert _decide(directory, _config(20), NOTEBOOK08_RULES).action == "skip"


@pytest.mark.parametrize("stop_reason", ["non_finite", "out_of_memory"])
def test_blocked_stops_are_refused(tmp_path, stop_reason):
    directory = tmp_path / stop_reason
    _publish(directory, config=_config(), run_state=_state(stop_reason))
    with pytest.raises(RuntimeError, match="再開も上書きもしません"):
        _decide(directory, _config(), NOTEBOOK07_RULES)


def test_decision_rejects_lowered_cap_changed_config_and_unknown_state(tmp_path):
    directory = tmp_path / "condition"
    _publish(directory, config=_config(), run_state=_state("max_sweeps", sweeps=10))
    with pytest.raises(ValueError, match="下げる"):
        _decide(directory, _config(5), NOTEBOOK07_RULES)
    with pytest.raises(ValueError, match="上限以外"):
        _decide(directory, _config(10, lr=0.1), NOTEBOOK07_RULES)

    odd = tmp_path / "odd"
    _publish(odd, config=_config(), run_state=_state("paused"))
    with pytest.raises(RuntimeError, match="再開できない"):
        _decide(odd, _config(), NOTEBOOK07_RULES)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "initial_model.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="世代情報の無い"):
        _decide(empty, _config(), NOTEBOOK07_RULES)


def test_decision_detects_manifest_run_state_mismatch(tmp_path):
    directory = tmp_path / "condition"
    final = _publish(directory, config=_config(), run_state=_state("running"))
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    manifest["stop_reason"] = "max_sweeps"
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stop_reason"):
        _decide(directory, _config(), NOTEBOOK07_RULES)


def test_decision_rejects_overlapping_stop_sets(tmp_path):
    rules = dict(NOTEBOOK07_RULES, blocked_stops=("running",))
    with pytest.raises(ValueError, match="重なって"):
        _decide(tmp_path / "x", _config(), rules)
