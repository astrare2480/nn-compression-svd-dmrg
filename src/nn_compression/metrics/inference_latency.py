"""同期付き wall-clock で、eager 推論の反復時間を測る。"""

from __future__ import annotations

import time

import torch
from torch import nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_synchronized_inference_latency(
    model: nn.Module,
    images: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    """入力を移動せず、反復ごとに同期した推論時間の中央値と四分位を返す。

    ``images`` はすでにモデルと同じ device にあること。この関数は
    ``.to()`` を呼ばない。呼び出し前に ``model.modules()`` の各 training
    フラグを保存し、``torch.inference_mode`` と ``eval`` で実行する。
    成功時も例外時も、各 module のフラグを個別に戻す。
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup は 0 以上の整数である必要があります: {warmup!r}")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError(
            f"repetitions は 1 以上の整数である必要があります: {repetitions!r}"
        )
    if not torch.is_tensor(images) or images.ndim < 1 or images.shape[0] < 1:
        raise ValueError("images は batch 次元を持つ Tensor である必要があります。")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("parameter を持たないモデルは測定できません。")
    if images.device != parameter.device:
        raise ValueError(
            "images はモデルと同じ device に事前配置してください: "
            f"images={images.device} model={parameter.device}"
        )
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    samples: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(images)
            _synchronize(images.device)
            for _ in range(repetitions):
                _synchronize(images.device)
                started = time.perf_counter()
                model(images)
                _synchronize(images.device)
                samples.append(time.perf_counter() - started)
    finally:
        for module, was_training in training_flags:
            module.training = was_training
    ordered = sorted(samples)
    def quartile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "median_seconds": quartile(0.5),
        "q1_seconds": quartile(0.25),
        "q3_seconds": quartile(0.75),
        "repetitions": repetitions,
        "warmup": warmup,
        "batch_size": int(images.shape[0]),
    }
