r"""分類モデルの logits を集めて、損失・正解率・logits 差を計算する。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/08_pretrained_mlp_fc1_tt_rank_sweep.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb

TT 圧縮の rank sweep では、候補ごとに「全サンプルの logits を順に集める → 損失 /
正解率を出す → ベースラインとの agreement / RMSE / 最大誤差を出す」を繰り返す。
その共通部分だけをここに置く。どの層を圧縮するか、どの候補を評価するか、
結果をどう表にするかは呼び出し側（Notebook）の責任。

評価関数なので autograd は使わず、返す Tensor は通常の Tensor（inference tensor
ではない）にして、呼び出し側が後で自由に演算できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..training.loops import _preserve_module_training_modes
from .tensor_comparison import (
    _error_summary,
    _scaled_float64_pair,
    _validate_comparable,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """logits とラベルから求めた分類指標。

    - ``loss``: 全サンプル平均のクロスエントロピー（float64 で計算）。
    - ``accuracy``: ``argmax`` が正解ラベルと一致した割合（0〜1）。
    - ``num_samples``: サンプル数。
    """

    loss: float
    accuracy: float
    num_samples: int


@dataclass(frozen=True)
class LogitsComparison:
    """2つの logits の比較結果。

    - ``prediction_agreement``: 行ごとの ``argmax`` が一致した割合（0〜1）。
    - ``logits_rmse``: 全要素平均の二乗誤差の平方根。
    - ``logits_max_abs_error``: 最大絶対誤差。
    """

    prediction_agreement: float
    logits_rmse: float
    logits_max_abs_error: float


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """``(inputs, labels)`` の batch を取り出す。それ以外の形式は ``TypeError``。"""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(
            "DataLoader の各 batch は (inputs, labels) の2要素である必要があります: "
            f"{type(batch)!r}"
        )
    inputs, labels = batch
    if not torch.is_tensor(inputs) or not torch.is_tensor(labels):
        raise TypeError(
            "batch の inputs と labels は torch.Tensor である必要があります: "
            f"inputs={type(inputs)!r}, labels={type(labels)!r}"
        )
    return inputs, labels


def collect_model_logits(
    model: nn.Module,
    data_loader,
    device: torch.device | str,
    *,
    output_device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``data_loader`` の全サンプルについて ``(logits, labels)`` を順番通りに集める。

    model:
        評価するモデル。出力は ``(batch, num_classes)`` の logits。
    data_loader:
        ``(inputs, labels)`` を返す iterable。shuffle する loader を渡すと、
        呼び出しごとに順序が変わり、2つのモデルの logits を行で比べられなくなる。
        比較に使うときは ``non_shuffling_loader`` などで順序を固定する。
    device:
        ``inputs`` を送って forward する device。
    output_device:
        集めた ``logits`` / ``labels`` を置く device（既定は CPU）。

    戻り値:
        ``(logits, labels)``。``logits`` は ``(N, C)``、``labels`` は ``(N, ...)``。
        行は loader の走査順で、バッチ内の順序も保たれる。どちらも ``output_device`` に
        置かれ、inference tensor ではない通常の Tensor。

    Note:
        model と全 submodule の ``training`` を個別に保存し、内部では ``eval()`` +
        ``torch.inference_mode()`` で forward して、終了後（例外時も）に元へ戻す。
        一部だけ ``eval()`` にしていた submodule（凍結 BatchNorm など）の状態も壊さない。
        ``labels`` は model へ渡さず、dtype も変えない。
        空の loader、``(inputs, labels)`` 以外の batch、出力が2階でない場合や batch
        サイズがラベルと一致しない場合は ``ValueError`` / ``TypeError``。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    device = torch.device(device)
    output_device = torch.device(output_device)

    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    with _preserve_module_training_modes(model):
        model.eval()
        with torch.inference_mode():
            for batch in data_loader:
                inputs, labels = _unpack_batch(batch)
                outputs = model(inputs.to(device))
                if not torch.is_tensor(outputs) or outputs.ndim != 2:
                    raise ValueError(
                        "model の出力は (batch, num_classes) の Tensor である必要があります: "
                        f"{getattr(outputs, 'shape', type(outputs))!r}"
                    )
                if labels.ndim == 0 or labels.shape[0] != outputs.shape[0]:
                    raise ValueError(
                        "batch 内の labels の数が model の出力行数と一致しません: "
                        f"labels={tuple(labels.shape)}, logits={tuple(outputs.shape)}"
                    )
                logits_batches.append(outputs.to(output_device))
                label_batches.append(labels.to(output_device))

    if not logits_batches:
        raise ValueError("data_loader が空なので logits を集められません。")

    # inference_mode の外で連結する。cat は常に新しいメモリへ書くので、
    # 戻り値は inference tensor ではなく通常の Tensor になる。
    logits = torch.cat(logits_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    return logits, labels_all


def _validate_logits(logits: torch.Tensor, *, name: str) -> None:
    """``(N, C)`` の有限な float32 / float64 の非空 logits であることを確かめる。"""
    if not torch.is_tensor(logits):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(logits)!r}")
    if logits.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name}.dtype={logits.dtype} は未対応です（torch.float32 / torch.float64 のみ）。"
        )
    if logits.ndim != 2:
        raise ValueError(
            f"{name} は (サンプル数, クラス数) の2階である必要があります: "
            f"shape={tuple(logits.shape)}"
        )
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError(f"{name} は空にできません: shape={tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} に NaN / inf が含まれています。")


def classification_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> ClassificationMetrics:
    """logits と整数ラベルから、平均クロスエントロピーと正解率を返す。

    logits:
        ``(N, C)`` の有限な float32 / float64。
    labels:
        ``(N,)`` の整数 Tensor（bool は不可）。``0 <= label < C`` かつ ``logits`` と
        同じ device。dtype は変えずに内部で ``int64`` として扱う（引数は書き換えない）。

    戻り値:
        ``ClassificationMetrics``。損失は float64 で計算し、全サンプルで平均する
        （バッチごとの平均を平均するのではないので、バッチサイズに依存しない）。

    例外:
        型・dtype の誤りは ``TypeError``、shape / device / ラベル範囲 / NaN・inf /
        空入力は ``ValueError``。

    Note:
        評価専用で、autograd のグラフは作らない。
    """
    _validate_logits(logits, name="logits")
    if not torch.is_tensor(labels):
        raise TypeError(f"labels は torch.Tensor である必要があります: {type(labels)!r}")
    if labels.dtype == torch.bool or labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError(f"labels.dtype={labels.dtype} は整数である必要があります。")
    if labels.shape != (logits.shape[0],):
        raise ValueError(
            f"labels は (N,) である必要があります: labels={tuple(labels.shape)}, "
            f"logits={tuple(logits.shape)}"
        )
    if labels.device != logits.device:
        raise ValueError(
            f"labels.device={labels.device} は logits.device={logits.device} と一致する必要があります。"
        )
    num_samples, num_classes = logits.shape
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError(
            f"labels は 0 以上 {num_classes} 未満である必要があります: "
            f"min={int(labels.min())}, max={int(labels.max())}"
        )

    with torch.no_grad():
        logits64 = logits.detach().to(torch.float64)
        targets = labels.detach().to(torch.int64)
        loss = float(F.cross_entropy(logits64, targets))
        correct = int((logits.detach().argmax(dim=1) == targets).sum())

    return ClassificationMetrics(
        loss=loss,
        accuracy=correct / num_samples,
        num_samples=int(num_samples),
    )


def compare_logits(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> LogitsComparison:
    """2つの logits について、予測一致率・RMSE・最大絶対誤差を返す。

    reference_logits / candidate_logits:
        ``(N, C)`` の有限な Tensor。shape・dtype・device が一致している必要がある。
        行はサンプルに対応する（同じ順序で集めたもの）。変更しない。

    戻り値:
        ``LogitsComparison``。RMSE と最大誤差は float64 で、2 のべき乗スケールを使って
        計算するため、大きな有限値でも不要な overflow を起こさない。
        ``reference_logits`` が零でも計算できる（相対誤差は返さない）。

    Note:
        予測一致率の ``argmax`` は同点のとき最初の最大要素を選ぶ（``torch.argmax`` の
        既定動作）。同点は両方の logits に同じ規則で適用される。
    """
    _validate_comparable(
        reference_logits,
        candidate_logits,
        names=("reference_logits", "candidate_logits"),
        ndim=2,
    )
    with torch.no_grad():
        agreement = float(
            (reference_logits.argmax(dim=1) == candidate_logits.argmax(dim=1))
            .to(torch.float64)
            .mean()
        )
        reference_scaled, candidate_scaled, scale = _scaled_float64_pair(
            reference_logits, candidate_logits
        )
        _, rmse, maximum = _error_summary(reference_scaled, candidate_scaled, scale)

    return LogitsComparison(
        prediction_agreement=agreement,
        logits_rmse=rmse,
        logits_max_abs_error=maximum,
    )
