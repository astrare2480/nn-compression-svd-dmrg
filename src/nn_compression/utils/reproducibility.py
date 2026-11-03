"""再現性の確認に使う hash と乱数状態の保存・復元。"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

_RNG_REQUIRED_KEYS = ("python", "numpy", "torch")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """dtype・shape・値をまとめた SHA-256。

    値は CPU 上の連続メモリの生 bytes（実行環境の byte order）で読む。
    入力 Tensor は変更しない。疎 Tensor は扱わない。
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor は torch.Tensor である必要があります: {type(tensor)!r}")
    if tensor.layout != torch.strided:
        raise ValueError(f"strided layout の Tensor だけを扱います: {tensor.layout}")
    values = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(int(size)) for size in tensor.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _integer_indices(indices: Any) -> list[int]:
    source = indices.indices if isinstance(indices, Subset) else indices
    if torch.is_tensor(source):
        if source.ndim != 1:
            raise ValueError(f"index Tensor は 1 階である必要があります: shape={tuple(source.shape)}")
        if source.dtype == torch.bool or source.is_floating_point() or source.is_complex():
            raise TypeError(f"index Tensor は整数 dtype である必要があります: {source.dtype}")
        return [int(value) for value in source.tolist()]
    if isinstance(source, (str, bytes)) or not isinstance(source, Iterable):
        raise TypeError(f"indices は整数の列、または Subset である必要があります: {type(indices)!r}")
    values: list[int] = []
    for position, value in enumerate(source):
        if isinstance(value, bool) or torch.is_tensor(value):
            raise TypeError(f"indices[{position}] は整数である必要があります: {value!r}")
        try:
            values.append(operator.index(value))
        except TypeError as exc:
            raise TypeError(f"indices[{position}] は整数である必要があります: {value!r}") from exc
    return values


def index_sha256(indices: Any) -> str:
    """index 列を int64 little-endian の bytes にした SHA-256。

    ``torch.utils.data.Subset`` を渡した場合は ``subset.indices`` を使う。
    dtype・shape は含めない（保存済みの split hash と同じ値になるようにするため）。
    """
    values = np.asarray(_integer_indices(indices), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _assignment_pattern(names: Sequence[str]) -> re.Pattern[str] | None:
    if isinstance(names, str):
        raise TypeError("excluded_assignments は変数名の列である必要があります。")
    cleaned = []
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"excluded_assignments の要素は識別子である必要があります: {name!r}")
        cleaned.append(re.escape(name))
    if not cleaned:
        return None
    return re.compile(r"^(?:" + "|".join(cleaned) + r")\s*(?::[^=]*)?=(?!=)")


def _normalized_code_cell(source: Any, pattern: re.Pattern[str] | None) -> str:
    if isinstance(source, list):
        source = "".join(source)
    if not isinstance(source, str):
        raise TypeError(f"code cell の source は str または list[str] です: {type(source)!r}")
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    kept = []
    for line in text.split("\n"):
        if pattern is not None and pattern.match(line.strip()):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip("\n") + "\n"


def notebook_code_sha256(
    notebook: Mapping[str, Any] | str | os.PathLike,
    *,
    excluded_assignments: Sequence[str] = (),
) -> str:
    """Notebook の code cell の source だけから SHA-256 を作る。

    output・execution count・cell ID・metadata・Markdown は含めない。
    改行は ``\\n`` へそろえ、行末の空白と cell 前後の空行は無視する。
    ``excluded_assignments`` に挙げた変数への代入行（例: 実行フラグや上限値）は
    行頭のインデントに関係なく除外する。``notebook`` は読み込み済みの dict か
    ``.ipynb`` のパス。入力の dict は変更しない。
    """
    if isinstance(notebook, (str, os.PathLike)):
        notebook = json.loads(Path(notebook).read_text(encoding="utf-8"))
    if not isinstance(notebook, Mapping) or not isinstance(notebook.get("cells"), list):
        raise ValueError("notebook は cells を持つ Notebook JSON である必要があります。")
    pattern = _assignment_pattern(excluded_assignments)
    digest = hashlib.sha256()
    for cell in notebook["cells"]:
        if not isinstance(cell, Mapping):
            raise ValueError("Notebook の cell は dict である必要があります。")
        if cell.get("cell_type") != "code":
            continue
        digest.update(_normalized_code_cell(cell.get("source", ""), pattern).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_numpy_item(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _copy_numpy_item(item) for key, item in value.items()}
    return value


def _copy_numpy_state(state: Any) -> Any:
    if isinstance(state, tuple):
        return tuple(_copy_numpy_item(item) for item in state)
    if isinstance(state, dict):
        return _copy_numpy_item(state)
    raise TypeError(f"NumPy の乱数状態の形式が不正です: {type(state)!r}")


def _byte_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(value)!r}")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous().clone()


def capture_rng_state(*, include_cuda: bool | None = None) -> dict[str, Any]:
    """Python・NumPy・PyTorch CPU と（必要なら）全 CUDA device の乱数状態を複製して返す。

    ``include_cuda=None`` のときは CUDA が利用できる場合だけ ``"cuda"`` を含める。
    ``True`` で CUDA が無い場合は ``RuntimeError``。
    """
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _copy_numpy_state(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できないため CUDA の乱数状態を保存できません。")
        state["cuda"] = [tensor.clone() for tensor in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """``capture_rng_state`` の結果から乱数状態を戻す。``state`` は変更しない。

    ``"cuda"`` があり CUDA が利用できる場合は全 device を戻す。保存時と device 数が
    違えば ``ValueError``。CUDA が利用できない環境では CUDA 分は戻さない。
    """
    if not isinstance(state, Mapping):
        raise TypeError(f"state は dict である必要があります: {type(state)!r}")
    missing = [key for key in _RNG_REQUIRED_KEYS if key not in state]
    if missing:
        raise KeyError(f"乱数状態に必要な key がありません: {missing}")
    torch_state = _byte_tensor(state["torch"], "state['torch']")
    numpy_state = _copy_numpy_state(state["numpy"])
    cuda_states = None
    if "cuda" in state and torch.cuda.is_available():
        saved = state["cuda"]
        if isinstance(saved, (str, bytes)) or not isinstance(saved, Sequence):
            raise TypeError("state['cuda'] は Tensor の列である必要があります。")
        if len(saved) != torch.cuda.device_count():
            raise ValueError(
                "保存時と CUDA device 数が違います: "
                f"saved={len(saved)} current={torch.cuda.device_count()}"
            )
        cuda_states = [_byte_tensor(value, f"state['cuda'][{index}]") for index, value in enumerate(saved)]
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
