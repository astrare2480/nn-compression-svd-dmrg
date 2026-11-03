"""実験条件ディレクトリの保存世代（generation）を読む・検証する・公開する。

条件ディレクトリの構成::

    <condition>/
        active_generation.json      # 有効世代への pointer
        generation_000001/          # 公開済み世代（manifest.json と成果物）
        .partial_generation_...     # 公開途中の一時ディレクトリ
        .publish_lock               # 公開中だけ存在する lock

新しい世代は一時ディレクトリへ書き、manifest の SHA-256 を検証してから
``os.replace`` で公開し、最後に pointer を atomic に切り替える。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_sha256

POINTER_NAME = "active_generation.json"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".publish_lock"
GENERATION_PREFIX = "generation_"
PARTIAL_PREFIX = ".partial_generation_"
_RESERVED_MANIFEST_KEYS = frozenset({"generation", "files"})


@dataclass(frozen=True)
class ActiveGeneration:
    """検証済みの有効世代。``path`` は世代ディレクトリ。"""

    path: Path
    generation: int
    manifest: dict[str, Any]
    pointer: dict[str, Any]


@dataclass(frozen=True)
class GenerationDecision:
    """保存済み世代に対する判定結果。

    ``action`` は ``"start"``（世代なし）・``"skip"``（完成済み）・``"resume"``（再開可能）。
    ``changed_caps`` は保存時から値が変わった上限 field。
    """

    action: str
    generation: ActiveGeneration | None
    saved_config: dict[str, Any] | None
    run_state: dict[str, Any] | None
    changed_caps: tuple[str, ...]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _names(values: Collection[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} は文字列の列である必要があります。")
    names = tuple(values)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} の要素は空でない文字列である必要があります: {name!r}")
    return names


def _generation_number(name: str) -> int | None:
    for prefix in (GENERATION_PREFIX, PARTIAL_PREFIX):
        if name.startswith(prefix):
            token = name.removeprefix(prefix).split("_", 1)[0]
            return int(token) if token.isdigit() else None
    return None


def load_active_generation(
    directory: str | os.PathLike,
    *,
    legacy_names: Collection[str] = (),
    pinned_files: Mapping[str, str] | None = None,
    required_manifest_keys: Collection[str] = (),
) -> ActiveGeneration | None:
    """pointer が指す世代を検証して返す。pointer が無ければ ``None``。

    pointer が無いのに ``legacy_names`` のファイルや公開途中の一時ディレクトリが
    あれば、旧形式・不完全な保存として ``RuntimeError``。
    manifest の ``files`` に記録された各ファイルの SHA-256 を照合する。
    ``pinned_files`` は「条件ディレクトリ直下のファイル名 → その SHA-256 を持つ
    manifest の key」で、世代の外に置いたファイルも照合する。
    """
    root = Path(directory)
    legacy = _names(legacy_names, "legacy_names")
    pinned = dict(pinned_files or {})
    required = _names(required_manifest_keys, "required_manifest_keys")
    pointer_path = root / POINTER_NAME
    if not pointer_path.is_file():
        partial = sorted(path.name for path in root.glob(PARTIAL_PREFIX + "*"))
        found = [name for name in legacy if (root / name).is_file()]
        if found or partial:
            raise RuntimeError(
                "保存世代の pointer がありません。旧形式または不完全な保存は自動採用せず、上書きもしません: "
                f"{root} legacy={found} partial={partial}"
            )
        return None
    pointer = _read_json(pointer_path)
    for key in ("generation", "directory"):
        if key not in pointer:
            raise RuntimeError(f"pointer に {key} がありません: {pointer_path}")
    artifact = root / str(pointer["directory"])
    if artifact.parent != root or not artifact.name.startswith(GENERATION_PREFIX):
        raise RuntimeError(f"pointer の directory が世代ディレクトリではありません: {pointer['directory']!r}")
    manifest_path = artifact / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"有効世代の manifest がありません: {artifact}")
    manifest = _read_json(manifest_path)
    if int(manifest.get("generation", -1)) != int(pointer["generation"]):
        raise RuntimeError("active_generation と manifest の世代が一致しません。")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError(f"manifest に files がありません: {manifest_path}")
    for name, digest in files.items():
        path = artifact / name
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"世代 {pointer['generation']} の {name} が manifest と一致しないため採用しません。")
    for name, key in pinned.items():
        path = root / name
        if key not in manifest:
            raise RuntimeError(f"manifest に {name} の hash（{key}）がありません。自動採用しません。")
        if not path.is_file() or file_sha256(path) != manifest[key]:
            raise RuntimeError(f"条件の {name} が世代 manifest と一致しません。")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise RuntimeError(f"manifest に必要な key がありません。自動採用しません: {missing}")
    return ActiveGeneration(
        path=artifact,
        generation=int(pointer["generation"]),
        manifest=manifest,
        pointer=pointer,
    )


def validate_generation_state(
    manifest: Mapping[str, Any],
    run_state: Mapping[str, Any],
    keys: Collection[str],
) -> None:
    """manifest に記録された ``keys`` が run_state と一致することを確かめる。

    manifest に無い key は照合しない（旧世代で記録していない項目のため）。
    """
    for key in _names(keys, "keys"):
        if key in manifest and manifest[key] != run_state.get(key):
            raise RuntimeError(f"世代 manifest と run_state の {key} が一致しません。")


def compare_generation_config(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    volatile_keys: Collection[str] = (),
    cap_fields: Collection[str] = (),
) -> tuple[str, ...]:
    """``volatile_keys`` を除いて config を比べ、値が変わった上限 field 名を返す。

    key の集合が違う場合と、上限以外の値が違う場合は ``ValueError``。
    """
    volatile = set(_names(volatile_keys, "volatile_keys"))
    caps = _names(cap_fields, "cap_fields")
    saved_keys = {key for key in saved if key not in volatile}
    current_keys = {key for key in current if key not in volatile}
    if saved_keys != current_keys:
        raise ValueError(
            "config の項目が一致しません: "
            f"saved_only={sorted(saved_keys - current_keys)} current_only={sorted(current_keys - saved_keys)}"
        )
    missing_caps = [field for field in caps if field not in current_keys]
    if missing_caps:
        raise ValueError(f"上限 field が config にありません: {missing_caps}")
    changed = sorted(key for key in current_keys if key not in caps and saved[key] != current[key])
    if changed:
        raise ValueError(f"上限以外の設定が一致しません: {changed}")
    return tuple(field for field in caps if saved[field] != current[field])


def decide_generation_action(
    directory: str | os.PathLike,
    config: Mapping[str, Any],
    *,
    cap_fields: Collection[str],
    finished_stops: Collection[str],
    resumable_stops: Collection[str],
    blocked_stops: Collection[str] = (),
    cap_independent_stops: Collection[str] = (),
    volatile_keys: Collection[str] = (),
    state_keys: Collection[str] = (),
    config_name: str = "config.json",
    run_state_name: str = "run_state.json",
    legacy_names: Collection[str] = (),
    pinned_files: Mapping[str, str] | None = None,
    required_manifest_keys: Collection[str] = (),
) -> GenerationDecision:
    """保存済み世代を読み、新規開始・完成済み・再開可能のどれかを判定する。

    判定順:

    1. 条件ディレクトリが無ければ ``"start"``。あるのに有効世代が無ければ ``RuntimeError``。
    2. config を比較し（``compare_generation_config``）、上限を下げていれば ``ValueError``。
    3. ``blocked_stops`` の停止は再開も再実行もしないため ``RuntimeError``。
    4. 上限が同じで ``finished_stops`` の停止なら ``"skip"``。
       ``cap_independent_stops`` の停止は上限を上げても ``"skip"``。
    5. ``resumable_stops`` の停止なら ``"resume"``。それ以外は ``RuntimeError``。
    """
    root = Path(directory)
    caps = _names(cap_fields, "cap_fields")
    finished = set(_names(finished_stops, "finished_stops"))
    resumable = set(_names(resumable_stops, "resumable_stops"))
    blocked = set(_names(blocked_stops, "blocked_stops"))
    independent = set(_names(cap_independent_stops, "cap_independent_stops"))
    overlap = blocked & (finished | resumable | independent)
    if overlap:
        raise ValueError(f"blocked_stops が他の停止分類と重なっています: {sorted(overlap)}")
    if not root.exists():
        return GenerationDecision("start", None, None, None, ())
    active = load_active_generation(
        root,
        legacy_names=legacy_names,
        pinned_files=pinned_files,
        required_manifest_keys=required_manifest_keys,
    )
    if active is None:
        contents = sorted(path.name for path in root.iterdir())
        raise RuntimeError(f"空、または世代情報の無いディレクトリは上書きしません: {root} contents={contents}")
    saved = _read_json(active.path / config_name)
    run_state = _read_json(active.path / run_state_name)
    validate_generation_state(active.manifest, run_state, state_keys)
    changed = compare_generation_config(saved, config, volatile_keys=volatile_keys, cap_fields=caps)
    lowered = [field for field in changed if int(config[field]) < int(saved[field])]
    if lowered:
        raise ValueError(f"安全上限を下げる再開はしません: {root} {lowered}")
    status = run_state.get("stop_reason")
    if status in blocked:
        raise RuntimeError(f"{status} で停止した条件は再開も上書きもしません: {root}")
    if (not changed and status in finished) or status in independent:
        action = "skip"
    elif status in resumable:
        action = "resume"
    else:
        raise RuntimeError(f"再開できない stop_reason です: {root} {status} changed_caps={list(changed)}")
    return GenerationDecision(action, active, saved, run_state, changed)


def _acquire_lock(root: Path) -> Path:
    lock_path = root / LOCK_NAME
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"同じ条件への並行保存には対応していません。{LOCK_NAME} があるため保存を拒否します: {root}"
        ) from error
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
    finally:
        os.close(handle)
    return lock_path


def _next_generation(root: Path) -> int:
    used: set[int] = set()
    pointer_path = root / POINTER_NAME
    if pointer_path.is_file():
        used.add(int(_read_json(pointer_path)["generation"]))
    for path in root.iterdir():
        number = _generation_number(path.name)
        if number is not None:
            used.add(number)
    return max(used, default=0) + 1


def _written_names(names: Sequence[str], partial: Path) -> list[str]:
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise TypeError("write_files はファイル名の列を返す必要があります。")
    result = list(names)
    if not result:
        raise ValueError("公開するファイルがありません。")
    if len(set(result)) != len(result):
        raise ValueError(f"公開するファイル名に重複があります: {result}")
    for name in result:
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"世代直下のファイル名である必要があります: {name!r}")
        if name == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} は publish_generation が書きます。")
        if not (partial / name).is_file():
            raise RuntimeError(f"write_files が {name} を作成していません。")
    return result


def publish_generation(
    directory: str | os.PathLike,
    write_files: Callable[[Path], Sequence[str]],
    *,
    manifest_fields: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    pointer_fields: Collection[str] = (),
    prepare: Callable[[Path], Any] | None = None,
) -> Path:
    """新しい世代を一時ディレクトリへ書き、検証してから公開して pointer を切り替える。

    ``write_files(partial_dir)`` は一時ディレクトリへ成果物を書き、そのファイル名の列を返す。
    ``manifest_fields`` は manifest へ加える項目（``write_files`` の後に評価する callable も可）。
    ``pointer_fields`` は manifest から pointer へ写す key。``prepare(directory)`` は
    lock 取得後・世代番号の決定前に呼ぶ（条件直下の共有ファイルの配置など）。
    lock は ``O_EXCL`` で取り、既にあれば ``RuntimeError``。公開前に失敗した場合は
    一時ディレクトリを消し、有効世代と pointer は変更しない。戻り値は公開した世代ディレクトリ。
    """
    if not callable(write_files):
        raise TypeError("write_files は呼び出し可能である必要があります。")
    if prepare is not None and not callable(prepare):
        raise TypeError("prepare は呼び出し可能である必要があります。")
    pointer_keys = _names(pointer_fields, "pointer_fields")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(root)
    partial: Path | None = None
    try:
        if prepare is not None:
            prepare(root)
        generation = _next_generation(root)
        final = root / f"{GENERATION_PREFIX}{generation:06d}"
        if final.exists():
            raise RuntimeError(f"未使用であるはずの世代ディレクトリが既にあるため保存しません: {final}")
        candidate = root / f"{PARTIAL_PREFIX}{generation:06d}_{os.getpid()}_{time.time_ns()}"
        if candidate.exists():
            raise RuntimeError(f"一時ディレクトリ名が衝突したため停止します: {candidate}")
        candidate.mkdir()
        partial = candidate
        names = _written_names(write_files(partial), partial)
        fields = manifest_fields() if callable(manifest_fields) else (manifest_fields or {})
        reserved = sorted(_RESERVED_MANIFEST_KEYS & set(fields))
        if reserved:
            raise ValueError(f"manifest_fields に予約 key は使えません: {reserved}")
        manifest = {
            "generation": generation,
            **dict(fields),
            "files": {name: file_sha256(partial / name) for name in names},
        }
        missing = [key for key in pointer_keys if key not in manifest]
        if missing:
            raise ValueError(f"pointer_fields が manifest にありません: {missing}")
        atomic_write_json(partial / MANIFEST_NAME, manifest)
        for name, digest in manifest["files"].items():
            if file_sha256(partial / name) != digest:
                raise RuntimeError(f"公開前の manifest 検証に失敗したため pointer は切り替えません: {name}")
        pointer = {
            "generation": generation,
            "directory": final.name,
            **{key: manifest[key] for key in pointer_keys},
        }
        os.replace(partial, final)
        partial = None
        atomic_write_json(root / POINTER_NAME, pointer)
        return final
    finally:
        if partial is not None:
            shutil.rmtree(partial, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
