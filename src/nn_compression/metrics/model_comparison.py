"""圧縮前後のモデル比較に使う汎用指標。

移植元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
    「精度の低下量」「推論速度比較」「予測一致率」「logitsのRMSE」
    「パラメータ数を数える」の各セル
"""

import time

import torch


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
    """圧縮前後の10クラス logits の RMSE を返す。

    agreement が最終クラスだけを見るのに対して、この指標は各クラスの
    生スコア全体のずれを評価する。小さいほど圧縮前モデルに近い。
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


def benchmark_inference(model, data_loader, device, warmup=10, repeats=5000):
    """同一バッチの平均推論時間を秒単位で測定する。

    GPU は非同期実行のため、計時の前後で同期する。小さな MLP では
    実測時間の差が小さく揺れやすいので、理論MACsと併用して解釈する。
    """
    model.eval()
    images, _ = next(iter(data_loader))
    images = images.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def benchmark_inference_print(
    baseline_model,
    compressed_model,
    data_loader,
    device,
    verbose=True,
):
    """圧縮前後の平均推論時間を測定して返す。"""
    baseline_time = benchmark_inference(baseline_model, data_loader, device)
    compressed_time = benchmark_inference(compressed_model, data_loader, device)
    if verbose:
        print(f"Baseline: {baseline_time * 1000:.3f} ms/batch")
        print(f"TwolayerSVD: {compressed_time * 1000:.3f} ms/batch")
    return baseline_time, compressed_time
