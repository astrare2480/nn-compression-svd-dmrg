"""``nn.Linear`` に入る直前の活性化を、forward pre-hook で取り出す。"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def capture_forward_pre_inputs(module: nn.Module) -> Iterator[list[torch.Tensor]]:
    """forward pre-hook で入力 Tensor をリストへ追加する。

    抜けたとき、正常終了でも例外でも hook を外す。
    リストの中身は呼び出し側が消してよい。hook は入力を変更しない。
    """
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()


def collect_sequence_inputs(
    model: nn.Module,
    module: nn.Module,
    sequences: Sequence[torch.Tensor],
    document_ids: Sequence[str],
    *,
    device: torch.device | str,
) -> list[dict[str, object]]:
    """系列ごとに順伝播し、``module`` へ入った活性化を float64 CPU で返す。

    sequences:
        各要素は 1 系列の token id。shape ``(length,)`` の整数 Tensor。
    document_ids:
        系列と同じ長さの文書 id。

    戻り値の各辞書は ``document_id`` と ``activation``。
    ``activation`` は shape ``(length, n)`` の float64 CPU Tensor。
    1 回の順伝播で hook が 1 回以外なら ``RuntimeError``。
    この関数は ``model.eval()`` と ``torch.no_grad()`` の中で順伝播する。
    呼び出し前の ``model.training`` は戻す。
    """
    if len(sequences) != len(document_ids):
        raise ValueError(
            f"sequences ({len(sequences)}) と document_ids ({len(document_ids)}) の長さが違います。"
        )
    was_training = model.training
    records: list[dict[str, object]] = []
    model.eval()
    try:
        with torch.no_grad(), capture_forward_pre_inputs(module) as captured:
            for token_ids, document_id in zip(sequences, document_ids):
                captured.clear()
                model(token_ids.reshape(1, -1).to(device), use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError(f"pre-hook が {len(captured)} 回呼ばれました。1 回である必要があります。")
                activation = captured[0].detach()
                if activation.ndim == 3:
                    if activation.shape[0] != 1:
                        raise RuntimeError(f"batch 次元が 1 ではありません: shape={tuple(activation.shape)}")
                    activation = activation[0]
                if activation.ndim != 2:
                    raise RuntimeError(f"活性化は 2 次元である必要があります: shape={tuple(activation.shape)}")
                records.append(
                    {
                        "document_id": document_id,
                        "activation": activation.to(dtype=torch.float64, device="cpu").contiguous(),
                    }
                )
    finally:
        model.train(was_training)
    return records


def _token_feature_matrix(activation: torch.Tensor, name: str) -> torch.Tensor:
    """pre-hook の入力を ``(tokens, features)`` にする。活性化そのものは保存しない。"""
    if activation.ndim == 3:
        if activation.shape[0] != 1:
            raise RuntimeError(f"{name} の batch 次元が 1 ではありません: shape={tuple(activation.shape)}")
        activation = activation[0]
    if activation.ndim != 2:
        raise RuntimeError(f"{name} の入力は 2 次元である必要があります: shape={tuple(activation.shape)}")
    return activation


def accumulate_input_grams(
    model: nn.Module,
    modules: Mapping[str, nn.Module],
    sequences: Sequence[torch.Tensor],
    *,
    device: torch.device | str,
) -> dict[str, dict[str, object]]:
    """1 回の順伝播で、複数モジュールの入力 Gram を float64 で足す。

    活性化は系列のあいだ保存しない。各名前について ``total``（中心化しない二次モーメント）と
    ``n_tokens`` を返す。``total`` は CPU の float64 で、``gram_matrix`` と同じ対称化をする。
    ある系列で hook が 1 回も呼ばれなければ ``RuntimeError``。
    呼び出し前の ``model.training`` は戻す。
    """
    if not modules:
        raise ValueError("modules が空です。")
    names = tuple(modules)
    if len(names) != len(set(names)):
        raise ValueError(f"modules の名前が重複しています: {names}")
    totals: dict[str, torch.Tensor | None] = {name: None for name in names}
    counts = {name: 0 for name in names}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            features = _token_feature_matrix(inputs[0].detach(), name).to(dtype=torch.float64)
            gram = features.transpose(0, 1) @ features
            current = totals[name]
            totals[name] = gram if current is None else current + gram
            counts[name] += int(features.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    forward_parameters = inspect.signature(model.forward).parameters
    accepts_cache = "use_cache" in forward_parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in forward_parameters.values()
    )
    forward_kwargs = {"use_cache": False} if accepts_cache else {}
    try:
        with torch.no_grad():
            for index, token_ids in enumerate(sequences):
                before = dict(counts)
                model(token_ids.reshape(1, -1).to(device), **forward_kwargs)
                for name in names:
                    if counts[name] == before[name]:
                        raise RuntimeError(f"{name} の pre-hook が系列 {index} で呼ばれませんでした。")
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    result: dict[str, dict[str, object]] = {}
    for name in names:
        total = totals[name]
        if total is None:
            raise RuntimeError(f"{name} の Gram がありません。")
        symmetric = (total + total.transpose(0, 1)) * 0.5
        result[name] = {
            "total": symmetric.detach().to(dtype=torch.float64, device="cpu").contiguous(),
            "n_tokens": counts[name],
        }
    return result
