"""固定した滑り窓で、因果言語モデルの perplexity を計算する。

``sliding_window_perplexity`` はモデル全体を 1 度に評価する。窓を自分で回したいとき
（途中の層から先だけを計算し直すときなど）は、同じ窓の切り方を
``sliding_windows``・``window_target_labels``・``shifted_nll_sum`` で組み立てる。
``sliding_window_perplexity`` は、それらと照らし合わせる独立な参照として、窓の計算を自分で持つ。
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def temporary_linear_weight(module: nn.Linear, weight: torch.Tensor) -> Iterator[None]:
    """``module.weight`` を一時的に差し替え、抜けたら元の値に戻す。

    ``weight`` は ``module.weight`` と同じ shape。dtype と device は
    モジュール側に合わせる。パラメータオブジェクトは差し替えず、
    ``data`` をコピーする。例外が出ても元に戻す。
    bias は変えない。
    """
    if weight.shape != module.weight.shape:
        raise ValueError(
            f"weight の shape {tuple(weight.shape)} が {tuple(module.weight.shape)} と一致しません。"
        )
    original = module.weight.data.detach().clone()
    module.weight.data.copy_(weight.to(dtype=module.weight.dtype, device=module.weight.device))
    try:
        yield
    finally:
        module.weight.data.copy_(original)


def sliding_window_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_length: int = 1024,
    stride: int = 512,
    use_cache: bool = False,
) -> dict[str, float | int | bool]:
    """滑り窓のトークン平均 NLL から perplexity を返す。

    input_ids:
        shape ``(1, L)`` または ``(L,)`` の整数 Tensor。
        評価の前に、モデルのパラメータと同じ device へ移す。
    各窓では、新しく入ったトークンだけを loss の対象にする
    （``labels`` の前方を -100 にする）。Hugging Face の因果 LM は
    内部で 1 トークンずらすので、数えるトークン数はそのずらしの後に
    -100 でないラベルの数である。
    perplexity は ``exp(合計 NLL / 評価トークン数)``。窓の loss を
    窓の本数で平均するのではない。

    戻り値のキー:
        ``perplexity``, ``mean_nll``, ``n_tokens``, ``n_windows``,
        ``max_length``, ``stride``, ``use_cache``。
    """
    if input_ids.ndim == 1:
        tokens = input_ids.reshape(1, -1)
    elif input_ids.ndim == 2 and input_ids.shape[0] == 1:
        tokens = input_ids
    else:
        raise ValueError(f"input_ids は (L,) または (1, L) である必要があります: shape={tuple(input_ids.shape)}")
    if max_length < 2 or stride < 1:
        raise ValueError(f"max_length>=2, stride>=1 である必要があります: {max_length}, {stride}")
    tokens = tokens.to(next(model.parameters()).device)
    sequence_length = int(tokens.shape[1])
    nll_sum = 0.0
    n_tokens = 0
    n_windows = 0
    previous_end = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for begin in range(0, sequence_length, stride):
                end = min(begin + max_length, sequence_length)
                target_length = end - previous_end
                chunk = tokens[:, begin:end]
                labels = chunk.clone()
                if target_length < chunk.shape[1]:
                    labels[:, :-target_length] = -100
                output = model(chunk, labels=labels, use_cache=use_cache)
                scored = int((labels[:, 1:] != -100).sum().item())
                if scored == 0:
                    raise RuntimeError("評価対象のトークンが 0 です。")
                nll_sum += float(output.loss.detach().item()) * scored
                n_tokens += scored
                n_windows += 1
                previous_end = end
                if end == sequence_length:
                    break
    finally:
        model.train(was_training)
    mean_nll = nll_sum / n_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
        "max_length": max_length,
        "stride": stride,
        "use_cache": use_cache,
    }


def sliding_windows(n_tokens: int, *, max_length: int, stride: int) -> list[tuple[int, int, int]]:
    """``sliding_window_perplexity`` と同じ窓を ``(begin, end, 評価する新しいトークンの数)`` で返す。

    窓は ``begin`` を ``stride`` ずつ進め、``end = min(begin + max_length, n_tokens)``。
    評価するのは、前の窓の ``end`` より後ろのトークンだけ。最後の窓が列の終わりに届いたら止める。
    すべてのトークンをちょうど 1 回ずつ評価するため、``1 <= stride <= max_length`` と
    ``n_tokens >= 2`` を求める（``stride > max_length`` だと窓の間に評価しないトークンが残る）。
    """
    if max_length < 2 or not 1 <= stride <= max_length:
        raise ValueError(f"max_length>=2, 1<=stride<=max_length である必要があります: {max_length}, {stride}")
    if n_tokens < 2:
        raise ValueError(f"n_tokens は 2 以上である必要があります: {n_tokens}")
    windows = []
    previous_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_length, n_tokens)
        windows.append((begin, end, end - previous_end))
        previous_end = end
        if end == n_tokens:
            break
    return windows


def window_target_labels(chunk: torch.Tensor, target_length: int) -> torch.Tensor:
    """窓のトークン ``chunk``（shape ``(1, L)``）から、末尾 ``target_length`` 個だけを数えるラベルを作る。

    数えない位置は -100。``1 <= target_length <= L`` でなければ ``ValueError``。
    """
    if chunk.ndim != 2 or chunk.shape[0] != 1:
        raise ValueError(f"chunk は (1, L) である必要があります: shape={tuple(chunk.shape)}")
    if not 1 <= target_length <= chunk.shape[1]:
        raise ValueError(f"target_length は 1 以上 {chunk.shape[1]} 以下です: {target_length}")
    labels = chunk.clone()
    if target_length < chunk.shape[1]:
        labels[:, :-target_length] = -100
    return labels


def shifted_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """1 トークンずらした交差エントロピーの合計（float64 で足す）と、数えたトークン数。

    Hugging Face の因果 LM の loss と同じずらし方。-100 のラベルは数えない。
    数えるトークンが 0 なら ``RuntimeError``。
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    scored = int((shift_labels != -100).sum())
    if scored == 0:
        raise RuntimeError("評価対象のトークンが 0 です。")
    losses = nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    )
    return float(losses.to(torch.float64).sum()), scored
