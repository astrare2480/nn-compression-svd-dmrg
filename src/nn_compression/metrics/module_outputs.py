"""名前で指定した module の forward 出力と、その段階の変化量。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from ..utils.module_state import _preserve_module_training_modes


@dataclass(frozen=True)
class CapturedModuleOutputs:
    """1 回の forward で得た、モデル出力と指定 module の出力。

    Tensor は forward 中の値を ``detach().clone()`` したものである。
    入力モデルは変更しない。
    """

    output: torch.Tensor
    modules: tuple[tuple[str, torch.Tensor], ...]

    def module(self, name: str) -> torch.Tensor:
        for module_name, tensor in self.modules:
            if module_name == name:
                return tensor
        raise KeyError(name)


@dataclass(frozen=True)
class StageChangeSummary:
    """候補出力と baseline 出力の、サンプル別 L2 とデータ全体の relative L2。"""

    stage_name: str
    dataset_relative_l2: float
    diff_l2: torch.Tensor
    baseline_l2: torch.Tensor
    diff_l2_mean: float
    diff_l2_median: float
    diff_l2_p90: float
    diff_l2_p95: float
    diff_l2_max: float


def _module_names(module_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(module_names, str):
        raise TypeError("module_names は module 名の列である必要があります。")
    names = tuple(module_names)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"module 名は空でない文字列である必要があります: {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"module 名が重複しています: {names}")
    return names


def _tensor_outside_inference(value: torch.Tensor) -> torch.Tensor:
    """hook 内で複製済みの値を、inference tensor ではない Tensor へ写す。

    device を変える場合も、この複製のあとで行う。
    """
    copied = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    copied.copy_(value)
    return copied


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} に非有限値があります。0 へは置き換えません。")


def capture_internal_outputs(
    model: nn.Module,
    inputs: torch.Tensor,
    module_names: Sequence[str],
) -> CapturedModuleOutputs:
    """``module_names`` の各 module 出力と、モデル forward の戻り値を取る。

    hook は forward が成功しても例外でも必ず外す。``model`` と全 submodule の
    ``training`` は、一部だけ eval だった状態も含めて元へ戻す。
    ``inplace=True`` の module は、前段の出力を上書きしうるため拒否する。
    同じ module が 1 回の forward で複数回呼ばれた場合も拒否する。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not torch.is_tensor(inputs):
        raise TypeError(f"inputs は torch.Tensor である必要があります: {type(inputs)!r}")
    names = _module_names(module_names)
    targets: list[tuple[str, nn.Module]] = []
    for name in names:
        module = model.get_submodule(name)
        if bool(getattr(module, "inplace", False)):
            raise ValueError(f"{name} は inplace のため、前段の出力を別に保持できません。")
        targets.append((name, module))
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def save(name: str):
        def hook(module: nn.Module, hook_inputs, output: torch.Tensor) -> None:
            if name in captured:
                raise RuntimeError(f"{name} が 1 回の forward で複数回呼ばれました。")
            if not torch.is_tensor(output):
                raise TypeError(f"{name} の出力は Tensor である必要があります: {type(output)!r}")
            captured[name] = output.detach().clone()

        return hook

    try:
        for name, module in targets:
            handles.append(module.register_forward_hook(save(name)))
        with _preserve_module_training_modes(model):
            model.eval()
            with torch.inference_mode():
                output = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.is_tensor(output):
        raise TypeError(f"model の出力は Tensor である必要があります: {type(output)!r}")
    missing = [name for name in names if name not in captured]
    if missing:
        raise RuntimeError(f"指定 module の出力を取得できませんでした: {missing}")
    _finite_tensor(output, "model 出力")
    copied = []
    for name in names:
        _finite_tensor(captured[name], name)
        copied.append((name, _tensor_outside_inference(captured[name])))
    return CapturedModuleOutputs(output=output.detach().clone(), modules=tuple(copied))


def _quantile(values: torch.Tensor, quantile: float) -> float:
    result = torch.quantile(values, float(quantile))
    if not torch.isfinite(result):
        raise ValueError("分位点が有限ではありません。")
    return float(result)


def stage_change_summary(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    stage_name: str,
) -> StageChangeSummary:
    """サンプルごとの L2 と、全サンプルをまとめた relative L2 を返す。

    relative L2 の分母は baseline 全体の L2 である。分母が 0 のときは
    epsilon に置き換えず ``ValueError`` にする。
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise ValueError(f"stage_name は空でない文字列である必要があります: {stage_name!r}")
    if not torch.is_tensor(candidate) or not torch.is_tensor(baseline):
        raise TypeError("candidate と baseline は torch.Tensor である必要があります。")
    if candidate.shape != baseline.shape or candidate.ndim != 2:
        raise ValueError(
            f"{stage_name} の出力形状が baseline と一致しません: "
            f"{tuple(candidate.shape)} vs {tuple(baseline.shape)}"
        )
    if candidate.shape[0] < 1:
        raise ValueError(f"{stage_name} の対象サンプルが空です。")
    difference = candidate.detach().to(dtype=torch.float64) - baseline.detach().to(dtype=torch.float64)
    baseline64 = baseline.detach().to(dtype=torch.float64)
    _finite_tensor(difference, stage_name)
    _finite_tensor(baseline64, stage_name)
    diff_l2 = torch.linalg.vector_norm(difference, dim=1)
    baseline_l2 = torch.linalg.vector_norm(baseline64, dim=1)
    denominator = torch.linalg.vector_norm(baseline64)
    if not torch.isfinite(denominator) or float(denominator) == 0.0:
        raise ValueError(
            f"{stage_name} の dataset-level relative L2 の分母がゼロです。"
            "epsilon には置き換えず停止します。"
        )
    numerator = torch.linalg.vector_norm(difference)
    if not torch.isfinite(numerator) or not torch.isfinite(diff_l2).all() or not torch.isfinite(baseline_l2).all():
        raise ValueError(f"{stage_name} の L2 norm が有限ではありません。")
    relative = numerator / denominator
    if not torch.isfinite(relative):
        raise ValueError(f"{stage_name} の dataset-level relative L2 が有限ではありません。")
    mean = diff_l2.mean()
    if not torch.isfinite(mean):
        raise ValueError(f"{stage_name} の差分 L2 の平均が有限ではありません。")
    return StageChangeSummary(
        stage_name=stage_name,
        dataset_relative_l2=float(relative),
        diff_l2=diff_l2,
        baseline_l2=baseline_l2,
        diff_l2_mean=float(mean),
        diff_l2_median=_quantile(diff_l2, 0.5),
        diff_l2_p90=_quantile(diff_l2, 0.9),
        diff_l2_p95=_quantile(diff_l2, 0.95),
        diff_l2_max=float(diff_l2.max()),
    )
