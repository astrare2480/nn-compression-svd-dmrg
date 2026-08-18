"""任意の named 層に対する rank sweep。

層名・rank 候補・空間サイズは呼び出し側が渡す。
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..metrics.macs import estimate_conv2d_macs
from ..metrics.model_comparison import collect_compression_metrics
from ..utils.modules import get_named_module, set_named_module
from .conv_svd import factorize_conv2d_layer
from .svd import retained_energy


def sweep_layer_ranks(
    model: nn.Module,
    layer_name: str,
    ranks: Sequence[int],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    factorize: Callable[[nn.Module, int], nn.Module],
    baseline_acc: float,
    baseline_time_s: float,
    macs_fn: Callable[[nn.Module, int], tuple[int, float]] | None = None,
    warmup: int = 5,
    repeats: int = 200,
    verbose: bool = True,
) -> list[dict]:
    """1つの named 層だけを各 rank で置き換え、validation 指標を集める。

    ``factorize(original_layer, rank)`` が置換モジュールを返す。
    他層は触らない。元の ``model`` も変更しない。
    """
    original_layer = get_named_module(model, layer_name)
    rank_results: list[dict] = []

    for rank in ranks:
        compressed_model = copy.deepcopy(model)
        set_named_module(
            compressed_model,
            layer_name,
            factorize(original_layer, rank),
        )

        compressed_macs = None
        compute_reduction = None
        if macs_fn is not None:
            compressed_macs, compute_reduction = macs_fn(original_layer, rank)

        row = collect_compression_metrics(
            model,
            compressed_model,
            loader,
            criterion,
            device,
            baseline_acc=baseline_acc,
            baseline_time_s=baseline_time_s,
            warmup=warmup,
            repeats=repeats,
            compressed_macs=compressed_macs,
            compute_reduction=compute_reduction,
        )
        row["layer"] = layer_name
        row["rank"] = int(rank)
        row["retained_energy"] = retained_energy(original_layer, rank)
        rank_results.append(row)

        if verbose:
            macs_text = (
                f" MACs={compressed_macs}" if compressed_macs is not None else ""
            )
            print(
                f"{layer_name} r={int(rank):3d}| "
                f"Acc={row['validation_acc']:.4f} "
                f"loss={row['validation_loss']:.4f} | "
                f"params={row['parameters']}"
                f"{macs_text} | "
                f"agreement={row['agreement']:.4f}"
            )

    if verbose:
        print(f"集計完了: {layer_name} {len(rank_results)} 件")
    return rank_results


def sweep_conv2d_ranks(
    model: nn.Module,
    layer_name: str,
    ranks: Sequence[int],
    out_hw: tuple[int, int],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    baseline_acc: float,
    baseline_time_s: float,
    warmup: int = 5,
    repeats: int = 200,
    verbose: bool = True,
) -> list[dict]:
    """Conv2d 1層の SVD rank sweep。空間サイズ ``out_hw`` は呼び出し側が渡す。"""

    def macs_fn(layer: nn.Module, rank: int) -> tuple[int, float]:
        _, compressed_macs, compute_reduction = estimate_conv2d_macs(
            layer,
            rank,
            out_hw,
            verbose=False,
        )
        return compressed_macs, compute_reduction

    return sweep_layer_ranks(
        model,
        layer_name,
        ranks,
        loader,
        criterion,
        device,
        factorize=factorize_conv2d_layer,
        baseline_acc=baseline_acc,
        baseline_time_s=baseline_time_s,
        macs_fn=macs_fn,
        warmup=warmup,
        repeats=repeats,
        verbose=verbose,
    )


def sweep_conv_svd_ranks(
    model: nn.Module,
    layer_name: str,
    rank_list: list[int],
    out_hw: tuple[int, int],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    baseline_acc: float,
    baseline_time_s: float,
    warmup: int = 5,
    repeats: int = 200,
    verbose: bool = True,
) -> list[dict]:
    """Notebook 互換の引数名。本体は ``sweep_conv2d_ranks``。"""
    return sweep_conv2d_ranks(
        model,
        layer_name,
        rank_list,
        out_hw,
        loader,
        criterion,
        device,
        baseline_acc=baseline_acc,
        baseline_time_s=baseline_time_s,
        warmup=warmup,
        repeats=repeats,
        verbose=verbose,
    )
