"""TTLinear を含むモデルの rank・parameter 数・理論 MACs をまとめて数える。

metrics は compression に依存しないため、TTLinear は型ではなく
``cores``（``nn.ParameterList``）・``out_modes``・``in_modes`` を持つ module として判定する。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..tt_mps import tt_matrix_ranks
from ..utils.modules import get_named_module
from .macs import linear_macs, tt_linear_direct_contraction_macs
from .model_comparison import count_parameters


@dataclass(frozen=True)
class TTLinearStructureMetrics:
    """1 つの TTLinear 層と、それを含むモデル全体の構造量。

    ``layer_macs`` は対象層を左から右へ直接縮約したときの 1 sample あたり MACs。
    ``total_macs`` は TTLinear 層の直接縮約 MACs と ``nn.Linear`` 層の
    ``in_features * out_features`` の合計。bias・活性化は数えない。
    """

    layer_name: str
    actual_ranks: tuple[int, ...]
    tt_core_elements: int
    layer_parameters: int
    total_parameters: int
    layer_macs: int
    total_macs: int


def _is_tt_linear(module: nn.Module) -> bool:
    return (
        isinstance(getattr(module, "cores", None), nn.ParameterList)
        and hasattr(module, "out_modes")
        and hasattr(module, "in_modes")
    )


def _tt_linear_macs(module: nn.Module) -> int:
    ranks = tt_matrix_ranks(list(module.cores))
    return int(sum(tt_linear_direct_contraction_macs(module.out_modes, module.in_modes, ranks)))


def _model_macs(model: nn.Module) -> int:
    total = 0
    tt_prefixes: list[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in tt_prefixes):
            continue
        if _is_tt_linear(module):
            tt_prefixes.append(name)
            total += _tt_linear_macs(module)
        elif isinstance(module, nn.Linear):
            total += int(linear_macs(module))
        elif any(True for _ in module.parameters(recurse=False)):
            label = name or "<root>"
            raise TypeError(
                f"理論 MACs を数えられない parameter 付き module があります: {label} ({type(module).__name__})"
            )
    return total


def tt_linear_structure_metrics(model: nn.Module, layer_name: str) -> TTLinearStructureMetrics:
    """``layer_name`` の TTLinear について、実 rank・要素数・parameter 数・理論 MACs を返す。

    実 rank は core の shape から読む。モデル全体の MACs は TTLinear と ``nn.Linear``
    だけで構成されるモデルを対象とし、それ以外の parameter 付き module があれば
    ``TypeError``。モデルは変更しない。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError(f"layer_name は空でない文字列である必要があります: {layer_name!r}")
    layer = get_named_module(model, layer_name)
    if not _is_tt_linear(layer):
        raise TypeError(f"{layer_name} は TTLinear ではありません: {type(layer).__name__}")
    ranks = tuple(int(rank) for rank in tt_matrix_ranks(list(layer.cores)))
    return TTLinearStructureMetrics(
        layer_name=layer_name,
        actual_ranks=ranks,
        tt_core_elements=int(sum(core.numel() for core in layer.cores)),
        layer_parameters=int(sum(parameter.numel() for parameter in layer.parameters())),
        total_parameters=int(count_parameters(model)),
        layer_macs=_tt_linear_macs(layer),
        total_macs=_model_macs(model),
    )
