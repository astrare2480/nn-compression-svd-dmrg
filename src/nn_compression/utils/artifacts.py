"""実験成果物の atomic 保存とファイル hash。

保存はすべて「同じディレクトリの一時ファイルへ書く → ``os.replace`` で差し替える」
で行う。書き込み途中で例外が起きても既存ファイルは元のまま残り、一時ファイルは消す。

読む側の小さな補助（CSV を行の辞書で読む、JSON に書ける値へ直す）もここに置く。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_REPLACE_ATTEMPTS = 12
_REPLACE_BASE_DELAY_SECONDS = 0.05


def _temporary_path(path: Path) -> Path:
    """同じディレクトリに、拡張子を保った衝突しにくい一時ファイル名を作る。"""
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    return path.with_name(f".{path.stem}.{token}.tmp{path.suffix}")


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Windows でウイルス対策などが一時的に掴んだときだけ再試行する。"""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BASE_DELAY_SECONDS * (attempt + 1))


def atomic_write_bytes(path: str | os.PathLike, writer: Callable[[Path], Any]) -> Path:
    """``writer(temporary_path)`` に一時ファイルを書かせ、成功後に ``path`` へ差し替える。

    一時ファイルは ``path`` と同じディレクトリに、同じ拡張子で作る。
    ``writer`` が例外を出した場合、または一時ファイルを作らなかった場合は、
    一時ファイルを消して例外を送出し、既存の ``path`` は変更しない。
    親ディレクトリが無ければ作る。戻り値は保存先の ``Path``。
    """
    if not callable(writer):
        raise TypeError(f"writer は呼び出し可能である必要があります: {writer!r}")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"保存先がディレクトリです: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise RuntimeError(f"writer が一時ファイルを作成しませんでした: {temporary}")
        _fsync_file(temporary)
        _replace_with_retry(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | os.PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> Path:
    """文字列を atomic に保存する。既定では OS によらず改行を ``\\n`` にする。"""
    if not isinstance(text, str):
        raise TypeError(f"text は str である必要があります: {type(text)!r}")

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding=encoding, newline=newline) as handle:
            handle.write(text)

    return atomic_write_bytes(path, write)


def atomic_write_json(path: str | os.PathLike, payload: Any) -> Path:
    """``indent=2``・key 整列・非 ASCII そのままの JSON を末尾改行付きで保存する。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path, text)


def atomic_write_csv(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    extrasaction: str = "raise",
) -> Path:
    """dict の列をヘッダー付き CSV として保存する。

    値の文字列化は ``csv`` module に任せる（``None`` は空欄）。
    ``fieldnames`` に無い key は ``extrasaction="raise"`` なら ``ValueError``、
    ``"ignore"`` なら無視する。``rows`` は保存前にすべて読み出す。
    """
    if isinstance(fieldnames, str):
        raise TypeError("fieldnames は列名の列である必要があります。")
    names = list(fieldnames)
    if not names:
        raise ValueError("fieldnames が空です。")
    if len(set(names)) != len(names):
        raise ValueError(f"fieldnames に重複があります: {names}")
    if extrasaction not in {"raise", "ignore"}:
        raise ValueError(f'extrasaction は "raise" または "ignore" です: {extrasaction!r}')
    materialized = list(rows)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction=extrasaction)
            writer.writeheader()
            writer.writerows(materialized)

    return atomic_write_bytes(path, write)


def atomic_torch_save(path: str | os.PathLike, payload: Any) -> Path:
    """``torch.save`` の結果を atomic に保存する。"""
    return atomic_write_bytes(path, lambda temporary: torch.save(payload, temporary))


def file_sha256(path: str | os.PathLike) -> str:
    """ファイル内容の SHA-256 を 1 MiB ずつ読んで返す。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: str | os.PathLike, *, missing_ok: bool = False) -> list[dict[str, str]]:
    """CSV を、見出しを key にした辞書の列で読む。値は文字列のまま。

    ``missing_ok`` が真でファイルがなければ空の列を返す。偽なら ``FileNotFoundError``。
    """
    path = Path(path)
    if missing_ok and not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_jsonable(value: Any) -> Any:
    """``json.dumps`` で標準の JSON になる値へ再帰的に直す。

    dict の key は文字列にする。tuple は list、``Path`` は文字列、NumPy の整数・浮動小数点は
    Python の数、0 次元の ``torch.Tensor`` は Python の数にする。
    NaN・正の無限大・負の無限大は、それぞれ文字列 ``"nan"``・``"inf"``・``"-inf"`` にする
    （標準の JSON はこれらを持てない）。直せない型は ``TypeError``。
    文字列にした key が重なる dict（``{1: ..., "1": ...}`` など）は、値を黙って失わないよう ``ValueError``。
    """
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key in converted:
                raise ValueError(f"文字列にすると重なる key があります: {text_key!r}")
            converted[text_key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(f"0 次元でない Tensor は JSON にしません: shape={tuple(value.shape)}")
        return to_jsonable(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(value)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy は torch と一緒に入る
        np = None
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return to_jsonable(float(value))
    raise TypeError(f"JSON にできない型です: {type(value)!r}")
