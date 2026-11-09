"""ファイルのロックで、複数のプロセスが仕事の一覧を重ならずに分け合う。

2 種類のロックがある。

- ``exclusive_lock``: 共有ファイルを書き換えるあいだの短いロック。OS のファイルロック
  （Windows は ``msvcrt.locking``、ほかは ``fcntl.flock``）を使うので、持ち主のプロセスが
  落ちると OS が外す。ロックのファイルそのものは消さずに残す。
- 仕事のロック: 1 つの仕事は ``<lock_dir>/<仕事の id>.lock`` を作れたプロセスが受け持つ。
  中には JSON で ``{"pid": 持ち主の pid, "owner": 任意の名前, "token": 取るたびに変わる合言葉}``
  を書く。持ち主が終わったロックは残骸として消せる。

仕事のロックを作る・残骸を消す・解放するの 3 つは、``<lock_dir>/_queue.guard`` の
``exclusive_lock`` の内側で行う。こうすると、書きかけのロックを読むことも、
「残骸だと判断したあとに、別のプロセスが取り直したロック」を消すこともない。
解放は、取ったときの合言葉が一致するときだけ消す。``lock_dir`` は仕事のロック専用にする。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json

GUARD_NAME = "_queue.guard"
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# pid の上限。Windows は DWORD（32 bit 符号なし）、ほかは pid_t（32 bit 符号付き）。
_MAX_PID = 0xFFFFFFFF if sys.platform == "win32" else 2**31 - 1

_WINDOWS_SYNCHRONIZE = 0x00100000
_WINDOWS_WAIT_OBJECT_0 = 0
_WINDOWS_ERROR_ACCESS_DENIED = 5


def pid_alive(pid: int) -> bool:
    """pid のプロセスが動いていれば True。0 以下と、OS の pid の範囲を超える値は False。

    Windows では ``os.kill(pid, 0)`` が「生きているか」の確認にならない
    （0 は Ctrl+C の送信になり、終わったプロセスでも成功することがある）。
    終了コードも、259 で終わったプロセスを「動いている」と取り違える。
    そこでプロセスの handle が「終わった」状態かを ``WaitForSingleObject`` で待たずに調べる。
    判定できないときは、動いているとみなす（使用中のロックを消さない側に倒す）。
    """
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise TypeError(f"pid は整数です: {pid!r}")
    if pid <= 0 or pid > _MAX_PID:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(_WINDOWS_SYNCHRONIZE, False, pid)
    if not handle:
        # 開けない: 存在しない（ERROR_INVALID_PARAMETER）か、権限がない（ERROR_ACCESS_DENIED）。
        return ctypes.get_last_error() == _WINDOWS_ERROR_ACCESS_DENIED
    try:
        # WAIT_OBJECT_0 なら終わっている。WAIT_TIMEOUT は動いている。WAIT_FAILED は動いているとみなす。
        return kernel32.WaitForSingleObject(handle, 0) != _WINDOWS_WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


def _try_os_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _os_unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(lock_path: Path | str, *, poll_seconds: float = 0.05) -> Iterator[None]:
    """共有ファイルを書き換えるあいだ、ほかのプロセスとスレッドを待たせるロック。

    OS のファイルロックなので、持ち主が落ちれば OS が外す（残骸を消す処理はいらない）。
    ``lock_path`` のファイルは作るが、抜けるときに消さない（消すと、別のプロセスが
    開いたままの古いファイルと新しいファイルで、ロックが 2 つに分かれる）。
    """
    if not (poll_seconds > 0):
        raise ValueError(f"poll_seconds は正の値です: {poll_seconds}")
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while not _try_os_lock(fd):
            time.sleep(poll_seconds)
        try:
            yield
        finally:
            _os_unlock(fd)
    finally:
        os.close(fd)


def _guard(lock_dir: Path) -> Any:
    return exclusive_lock(lock_dir / GUARD_NAME, poll_seconds=0.01)


def read_lock(lock_path: Path | str) -> dict[str, Any] | None:
    """仕事のロックの中身（``pid``・``owner``・``token``）。読めなければ ``None``。

    中身が整数だけのロック（04 の書き方）は、``{"pid": その整数, "owner": "", "token": None}`` と読む。
    """
    try:
        text = Path(lock_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if isinstance(payload, int) and not isinstance(payload, bool):
        return {"pid": payload, "owner": "", "token": None}
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    return {"pid": pid, "owner": str(payload.get("owner", "")), "token": payload.get("token")}


def try_claim(lock_path: Path | str, owner: str = "") -> str | None:
    """仕事のロックを作れたら、その合言葉（token）を返す。すでにあれば ``None``。

    中には自分の pid、``owner``（記録用の任意の名前）、新しい合言葉を書く。
    作るのは ``lock_path`` と同じフォルダの見張り役のロックの内側なので、
    ほかのプロセスが書きかけの中身を読むことはない。
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    content = json.dumps({"pid": os.getpid(), "owner": str(owner), "token": token})
    with _guard(lock_path.parent):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    return token


def release_claim(lock_path: Path | str, token: str) -> bool:
    """合言葉が一致するときだけ仕事のロックを消し、消したら True。

    一致しない（残骸として消されたあと、別のプロセスが取り直した）ロックは消さない。
    """
    lock_path = Path(lock_path)
    if not lock_path.parent.is_dir():
        return False
    with _guard(lock_path.parent):
        info = read_lock(lock_path)
        if info is None or info["token"] != token:
            return False
        lock_path.unlink(missing_ok=True)
    return True


def clear_stale_locks(lock_dir: Path | str) -> list[str]:
    """持ち主が終わった ``*.lock`` を消し、消したファイル名を返す。

    中身が読めないロックも残骸とみなす（見張り役のロックの内側で書くので、
    読めないのは書いている途中でプロセスが落ちたときだけ）。消せないファイルは飛ばす。
    """
    lock_dir = Path(lock_dir)
    removed: list[str] = []
    if not lock_dir.is_dir():
        return removed
    with _guard(lock_dir):
        for path in sorted(lock_dir.glob("*.lock")):
            info = read_lock(path)
            if info is not None and pid_alive(info["pid"]):
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            removed.append(path.name)
    return removed


def job_lock_path(lock_dir: Path | str, job_id: str) -> Path:
    """仕事の id のロックファイルの場所 ``<lock_dir>/<job_id>.lock``。

    id は英数字で始まり、英数字・``.``・``_``・``-`` だけからなる文字列に限る
    （パスの区切りや ``..`` で ``lock_dir`` の外やサブフォルダに作らないため）。
    """
    if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError(f"仕事の id はファイル名に使える文字だけにしてください: {job_id!r}")
    return Path(lock_dir) / f"{job_id}.lock"


@dataclass(frozen=True)
class JobClaim:
    """``claim_next_job`` で取った仕事。終わったら ``release()`` でロックを外す。"""

    job: dict[str, Any]
    lock_path: Path
    token: str

    def release(self) -> bool:
        """自分のロックなら消して True。"""
        return release_claim(self.lock_path, self.token)


def claim_next_job(
    jobs: Sequence[Mapping[str, Any]],
    lock_dir: Path | str,
    done_ids: set[str],
    owner: str = "",
) -> JobClaim | None:
    """一覧の順に見て、終わっておらず、まだ誰も取っていない最初の仕事を取る。

    仕事は ``"id"`` を持つ辞書（id の制約は ``job_lock_path``）。取る前に全部の id を確かめる。
    取れる仕事がなければ ``None``。
    """
    lock_paths = [job_lock_path(lock_dir, job["id"]) for job in jobs]
    for job, lock_path in zip(jobs, lock_paths, strict=True):
        if job["id"] in done_ids:
            continue
        token = try_claim(lock_path, owner)
        if token is not None:
            return JobClaim(job=dict(job), lock_path=lock_path, token=token)
    return None


def dump_jobs(path: Path | str, jobs: Sequence[Mapping[str, Any]]) -> None:
    """仕事の一覧を ``{"jobs": [...]}`` の JSON として書く。"""
    atomic_write_json(Path(path), {"jobs": list(jobs)})


def load_jobs(path: Path | str) -> list[dict[str, Any]]:
    """``dump_jobs`` で書いた仕事の一覧を読む。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["jobs"])
