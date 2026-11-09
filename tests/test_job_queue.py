"""ロックによる仕事の分け合いと、プロセスの生死の判定。

複数のプロセスを使うテストは、子プロセスに ``src`` を見せて同じ関数を呼ばせる。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import nn_compression
from nn_compression.utils.job_queue import (
    GUARD_NAME,
    JobClaim,
    claim_next_job,
    clear_stale_locks,
    dump_jobs,
    exclusive_lock,
    job_lock_path,
    load_jobs,
    pid_alive,
    read_lock,
    release_claim,
    try_claim,
)

SRC_DIR = Path(nn_compression.__file__).resolve().parents[1]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
    return env


def _python(code: str, *args: str, **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )


def _finished_pid(exit_code: int = 0) -> int:
    child = subprocess.Popen([sys.executable, "-c", f"import os; os._exit({exit_code})"])
    child.wait()
    return child.pid


def test_pid_alive_tells_running_from_finished_processes():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(-1)
    # 親が子の handle を持ったままでも、終わったプロセスは生きていない（Windows で os.kill(pid, 0) は誤る）。
    assert not pid_alive(_finished_pid())


def test_pid_alive_is_not_fooled_by_exit_code_259_or_huge_pids():
    # 259 は Windows の STILL_ACTIVE と同じ値。終了コードで判定すると「動いている」と誤る。
    child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(259)"])
    child.wait()
    assert not pid_alive(child.pid)
    # 32 bit に切り詰めると自分の pid になる値。
    assert not pid_alive(2**32 + os.getpid())
    with pytest.raises(TypeError):
        pid_alive("123")


def test_claim_writes_pid_owner_and_token(tmp_path):
    lock = tmp_path / "a.lock"
    token = try_claim(lock, "worker-A")
    assert token is not None
    assert try_claim(lock, "worker-B") is None
    info = read_lock(lock)
    assert info == {"pid": os.getpid(), "owner": "worker-A", "token": token}
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == token


def test_claims_do_not_overlap_and_skip_finished_jobs(tmp_path):
    jobs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    first = claim_next_job(jobs, tmp_path, {"a"}, "w1")
    assert isinstance(first, JobClaim) and first.job == {"id": "b"}
    assert first.lock_path == tmp_path / "b.lock"
    assert claim_next_job(jobs, tmp_path, {"a"}, "w2").job["id"] == "c"
    assert claim_next_job(jobs, tmp_path, {"a"}, "w3") is None
    assert first.release()
    assert not first.release()
    assert claim_next_job(jobs, tmp_path, {"a"}).job["id"] == "b"


def test_non_numeric_owner_does_not_make_a_live_lock_stale(tmp_path):
    # owner は記録用の名前。持ち主の生死は別に書いた pid で決まる。
    assert claim_next_job([{"id": "x"}], tmp_path, set(), "worker-A") is not None
    assert clear_stale_locks(tmp_path) == []
    assert (tmp_path / "x.lock").exists()


@pytest.mark.parametrize("job_id", ["", ".", "..", "a/b", "../x", "a\\b", ".hidden", "c:x", 3])
def test_job_ids_must_be_plain_file_names(tmp_path, job_id):
    lock_dir = tmp_path / "locks"
    with pytest.raises(ValueError):
        claim_next_job([{"id": "ok"}, {"id": job_id}], lock_dir, set())
    # 1 つでも不正な id があれば、何も取らない。
    assert not lock_dir.exists() or not any(lock_dir.glob("*.lock"))
    assert not (tmp_path / "x.lock").exists()
    assert job_lock_path(tmp_path, "12_q_proj_25.5") == tmp_path / "12_q_proj_25.5.lock"


def test_stale_locks_are_removed_but_live_ones_stay(tmp_path):
    try_claim(tmp_path / "live.lock", "me")
    (tmp_path / "unreadable.lock").write_text("not a pid", encoding="utf-8")
    (tmp_path / "empty.lock").write_text("", encoding="utf-8")
    (tmp_path / "dead.lock").write_text(json.dumps({"pid": _finished_pid(), "owner": "", "token": "t"}), encoding="utf-8")
    # 04 の書き方（pid だけ）も読める。
    (tmp_path / "legacy_dead.lock").write_text(str(_finished_pid()), encoding="utf-8")
    (tmp_path / "legacy_live.lock").write_text(str(os.getpid()), encoding="utf-8")
    assert clear_stale_locks(tmp_path) == ["dead.lock", "empty.lock", "legacy_dead.lock", "unreadable.lock"]
    assert (tmp_path / "live.lock").exists() and (tmp_path / "legacy_live.lock").exists()
    assert (tmp_path / GUARD_NAME).exists()
    assert clear_stale_locks(tmp_path / "missing") == []


def test_release_needs_the_matching_token(tmp_path):
    lock = tmp_path / "a.lock"
    token = try_claim(lock)
    assert not release_claim(lock, "someone-else")
    assert lock.exists()
    assert release_claim(lock, token)
    assert not lock.exists()
    assert not release_claim(tmp_path / "missing" / "a.lock", token)


def test_stale_clearing_waits_for_the_guard_and_rereads_the_lock(tmp_path):
    """順序を固定した競合: 残骸の判断と、別プロセスの取り直しが重なっても、生きたロックは消えない。"""
    lock = tmp_path / "job.lock"
    lock.write_text(json.dumps({"pid": _finished_pid(), "owner": "old", "token": "old"}), encoding="utf-8")
    with exclusive_lock(tmp_path / GUARD_NAME):
        child = _python(
            """
            import sys
            from nn_compression.utils.job_queue import clear_stale_locks
            print("start", flush=True)
            print(clear_stale_locks(sys.argv[1]))
            """,
            str(tmp_path),
        )
        assert child.stdout.readline().strip() == "start"
        time.sleep(0.5)
        assert child.poll() is None, "見張り役のロックを待たずに消しにいった"
        # 待っているあいだに、別の持ち主（自分）が取り直したことにする。
        lock.write_text(json.dumps({"pid": os.getpid(), "owner": "new", "token": "new"}), encoding="utf-8")
    out, err = child.communicate(timeout=30)
    assert child.returncode == 0, err
    assert out.strip() == "[]"  # "start" の行は先に読んだ
    assert read_lock(lock)["token"] == "new"


def test_claims_across_processes_never_overlap(tmp_path):
    jobs = [{"id": f"job{i:02d}"} for i in range(24)]
    dump_jobs(tmp_path / "jobs.json", jobs)
    code = """
        import sys
        from nn_compression.utils.job_queue import claim_next_job, load_jobs
        jobs = load_jobs(sys.argv[1])
        while True:
            claim = claim_next_job(jobs, sys.argv[2], set(), sys.argv[3])
            if claim is None:
                break
            print(claim.job["id"], flush=True)
        """
    children = [_python(code, str(tmp_path / "jobs.json"), str(tmp_path / "locks"), f"w{i}") for i in range(4)]
    claimed: list[str] = []
    for child in children:
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        claimed.extend(out.split())
    assert sorted(claimed) == [job["id"] for job in jobs]


def test_exclusive_lock_serializes_processes(tmp_path):
    counter = tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    code = """
        import sys, time
        from pathlib import Path
        from nn_compression.utils.job_queue import exclusive_lock
        path = Path(sys.argv[1])
        for _ in range(20):
            with exclusive_lock(sys.argv[2], poll_seconds=0.001):
                value = int(path.read_text(encoding="utf-8"))
                time.sleep(0.002)
                path.write_text(str(value + 1), encoding="utf-8")
        """
    children = [_python(code, str(counter), str(tmp_path / "counter.lock")) for _ in range(4)]
    for child in children:
        _out, err = child.communicate(timeout=120)
        assert child.returncode == 0, err
    assert counter.read_text(encoding="utf-8") == "80"


def _enters_lock_in_thread(path: Path) -> threading.Event:
    entered = threading.Event()

    def enter():
        with exclusive_lock(path, poll_seconds=0.01):
            entered.set()

    threading.Thread(target=enter, daemon=True).start()
    return entered


def test_exclusive_lock_waits_for_threads_and_keeps_the_file(tmp_path):
    path = tmp_path / "shared.lock"
    with exclusive_lock(path):
        entered = _enters_lock_in_thread(path)
        assert not entered.wait(0.3)
    assert entered.wait(5)
    assert path.exists()
    with pytest.raises(ValueError):
        with exclusive_lock(path, poll_seconds=0):
            pass


def test_exclusive_lock_is_released_when_the_holder_dies(tmp_path):
    path = tmp_path / "shared.lock"
    holder = _python(
        """
        import sys, time
        from nn_compression.utils.job_queue import exclusive_lock
        with exclusive_lock(sys.argv[1]):
            print("held", flush=True)
            time.sleep(120)
        """,
        str(path),
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        entered = _enters_lock_in_thread(path)
        assert not entered.wait(0.5)
        holder.kill()
        holder.wait(timeout=30)
        assert entered.wait(15), "落ちた持ち主のロックが外れない"
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.communicate(timeout=30)


def test_jobs_round_trip(tmp_path):
    jobs = [{"id": "x", "budget": 25}, {"id": "y", "budget": 50}]
    dump_jobs(tmp_path / "jobs.json", jobs)
    assert load_jobs(tmp_path / "jobs.json") == jobs
