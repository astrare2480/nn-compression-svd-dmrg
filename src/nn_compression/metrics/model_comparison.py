"""圧縮前後のモデル比較に使う汎用指標。

移植元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
    「精度の低下量」「推論速度比較」「予測一致率」「logitsのRMSE」
    「パラメータ数を数える」の各セル
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from ..training.loops import evaluate


def count_parameters(model):
    """学習可能・非学習可能を問わず、モデルが保持する全要素数を数える。"""
    return sum(parameter.numel() for parameter in model.parameters())


def parameters_reduction(baseline_model, compressed_model):
    """圧縮モデルで削減できたパラメータ数の割合を返す。

    0.0 は削減なし、0.9 はパラメータ数を90%削減したことを表す。
    """
    return 1.0 - (
        count_parameters(compressed_model) / count_parameters(baseline_model)
    )


def accuracy_drop(baseline_accuracy, compressed_accuracy, verbose=True):
    """圧縮により減った accuracy を返す（正なら精度低下）。"""
    drop = baseline_accuracy - compressed_accuracy
    if verbose:
        print("Baseline accuracy:", baseline_accuracy)
        print("Compressed accuracy:", compressed_accuracy)
        print("Accuracy drop:", round(drop, 5))
    return drop


def agreement(baseline_model, compressed_model, loader, device):
    """2モデルが同じ予測クラスを出す割合を返す。

    正解ラベルに対する accuracy とは異なり、圧縮前のモデルの判断を
    圧縮後のモデルがどの程度保っているかを測る。
    """
    baseline_model.eval()
    compressed_model.eval()

    agreement_count = 0
    total = 0
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            baseline_prediction = baseline_model(images).argmax(dim=1)
            compressed_prediction = compressed_model(images).argmax(dim=1)
            agreement_count += (
                baseline_prediction == compressed_prediction
            ).sum().item()
            total += images.size(0)

    return agreement_count / total


def logits_rmse(baseline_model, compressed_model, loader, device):
    """圧縮前後の出力（logits）の RMSE を返す。

    agreement が最終クラスだけを見るのに対して、この指標は出力要素全体の
    ずれを評価する。小さいほど圧縮前モデルに近い。クラス数は仮定しない。
    """
    baseline_model.eval()
    compressed_model.eval()

    squared_error_sum = 0.0
    element_count = 0
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            difference = baseline_model(images) - compressed_model(images)
            squared_error_sum += difference.square().sum().item()
            element_count += difference.numel()

    return (squared_error_sum / element_count) ** 0.5


def _synchronize(device: torch.device) -> None:
    """CUDA 非同期 kernel を、測定している device 上で完了させてから時刻を取る。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_input_batch(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if torch.is_tensor(batch):
        return batch
    raise TypeError(
        "input_batch は Tensor、または (images, labels) である必要があります。"
    )


def take_inference_batch(data_loader: DataLoader) -> torch.Tensor:
    """比較用の 1 バッチを取る。shuffle Generator は消費しない。

    baseline / compressed は同じ Tensor を ``input_batch`` として共有する。
    loader から都度取ると入力条件差が latency に混ざる。
    """
    sampler = getattr(data_loader, "sampler", None)
    if isinstance(sampler, RandomSampler):
        raise ValueError(
            "shuffle 付き DataLoader は Generator を消費します。"
            " input_batch を渡すか、shuffle=False の loader を使ってください。"
        )
    try:
        batch = next(iter(data_loader))
    except StopIteration as exc:
        raise ValueError("空の DataLoader では推論時間を測れません。") from exc
    return _as_input_batch(batch)


def benchmark_inference(
    model,
    data_loader=None,
    device=None,
    warmup=10,
    repeats=5000,
    *,
    input_batch=None,
    return_details: bool = False,
):
    """同一入力バッチの平均推論時間を秒単位で測定する。

    ``input_batch`` があれば loader を走査しない。
    baseline と compressed は同じ ``input_batch`` / warmup / repeats で測る。
    条件を揃えないとモデル差以外が混入する。
    """
    if device is None:
        raise TypeError("device が必要です。")
    if warmup < 0:
        raise ValueError(f"warmup は 0 以上にしてください: {warmup}")
    if repeats <= 0:
        raise ValueError(f"repeats は 1 以上にしてください: {repeats}")

    if input_batch is None:
        if data_loader is None:
            raise TypeError("data_loader または input_batch が必要です。")
        images = take_inference_batch(data_loader)
    else:
        images = _as_input_batch(input_batch)

    model.eval()
    images = images.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(images)

    _synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(images)

    _synchronize(device)
    time_s = (time.perf_counter() - start) / repeats

    if not return_details:
        return time_s

    return {
        "time_s": time_s,
        "time_ms": time_s * 1000,
        "batch_size": int(images.size(0)),
        "input_shape": tuple(images.shape),
        "warmup": int(warmup),
        "repeats": int(repeats),
    }


def collect_compression_metrics(
    baseline_model,
    compressed_model,
    loader,
    criterion,
    device,
    *,
    baseline_acc: float,
    baseline_time_s: float,
    warmup: int = 5,
    repeats: int = 200,
    compressed_macs: int | None = None,
    compute_reduction: float | None = None,
    baseline_macs: int | None = None,
    include_model: bool = False,
    input_batch=None,
) -> dict:
    """圧縮モデルを validation loader で評価し、共通指標の dict を返す。

    層名・rank・CSV 名は含めない。呼び出し側が実験固有の列を足す。
    既定 ``include_model=False`` は rank sweep 向け。全候補の model を
    表に残さず、メモリと CSV を軽くし、必要な候補は rank から再構築する。
    Fine-tuning 済みなど、保持が必要なときだけ ``include_model=True``。
    """
    validation_loss, validation_acc = evaluate(
        compressed_model,
        loader,
        criterion,
        device,
    )
    if input_batch is None:
        input_batch = take_inference_batch(loader)
    details = benchmark_inference(
        compressed_model,
        device=device,
        warmup=warmup,
        repeats=repeats,
        input_batch=input_batch,
        return_details=True,
    )
    row = {
        "parameters": count_parameters(compressed_model),
        "parameters_reduction": parameters_reduction(
            baseline_model,
            compressed_model,
        ),
        "validation_loss": validation_loss,
        "validation_acc": validation_acc,
        "accuracy_drop": accuracy_drop(
            baseline_acc,
            validation_acc,
            verbose=False,
        ),
        "baseline_time_ms": baseline_time_s * 1000,
        "compressed_time_ms": details["time_ms"],
        "benchmark_batch_size": details["batch_size"],
        "benchmark_input_shape": details["input_shape"],
        "benchmark_warmup": details["warmup"],
        "benchmark_repeats": details["repeats"],
        "agreement": agreement(
            baseline_model,
            compressed_model,
            loader,
            device,
        ),
        "logits_rmse": logits_rmse(
            baseline_model,
            compressed_model,
            loader,
            device,
        ),
    }
    if include_model:
        row["model"] = compressed_model
    if baseline_macs is not None:
        row["baseline_macs"] = baseline_macs
    if compressed_macs is not None:
        row["compressed_macs"] = compressed_macs
    if compute_reduction is not None:
        row["compute_reduction"] = compute_reduction
    return row


def benchmark_inference_print(
    baseline_model,
    compressed_model,
    data_loader,
    device,
    verbose=True,
    *,
    warmup=10,
    repeats=5000,
    input_batch=None,
):
    """圧縮前後の平均推論時間を、同じ入力バッチで測定して返す。"""
    if input_batch is None:
        input_batch = take_inference_batch(data_loader)
    baseline_details = benchmark_inference(
        baseline_model,
        device=device,
        warmup=warmup,
        repeats=repeats,
        input_batch=input_batch,
        return_details=True,
    )
    compressed_details = benchmark_inference(
        compressed_model,
        device=device,
        warmup=warmup,
        repeats=repeats,
        input_batch=input_batch,
        return_details=True,
    )
    if verbose:
        print(
            "batch_size:",
            baseline_details["batch_size"],
            "input_shape:",
            baseline_details["input_shape"],
            "warmup:",
            warmup,
            "repeats:",
            repeats,
        )
        print(f"Baseline: {baseline_details['time_ms']:.3f} ms/batch")
        print(f"Compressed: {compressed_details['time_ms']:.3f} ms/batch")
    return baseline_details["time_s"], compressed_details["time_s"]
