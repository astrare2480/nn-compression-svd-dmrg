r"""TTLinear の 2-core 局所更新で使う forward と core の書き戻し。

結合・分割・正準化そのものは ``nn_compression.tt_mps.matrix_two_site``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from ..tt_mps.matrix_two_site import _pair_index, _validate_same_dtype_device
from ..tt_mps.matrix_validation import validate_tt_matrix_cores, validate_tt_matrix_modes
from ..utils.modules import get_named_module
from .tt_linear import TTLinear, tt_linear_forward
from .tt_linear_validation import validate_tt_linear_bias, validate_tt_linear_input


def _contract_one_core(state: torch.Tensor, core: torch.Tensor, site_index: int) -> torch.Tensor:
    """``tt_linear_forward`` と同じ1 site 縮約。``state`` の軸順も同じ。"""
    left_bond_axis = 1 + site_index
    in_mode_axis = left_bond_axis + 1
    state = torch.tensordot(state, core, dims=([left_bond_axis, in_mode_axis], [0, 2]))
    return state.movedim([-2, -1], [left_bond_axis, left_bond_axis + 1])


def _contract_merged_pair(
    state: torch.Tensor,
    merged: torch.Tensor,
    site_index: int,
) -> torch.Tensor:
    """site ``k`` と ``k+1`` を、6階 ``T`` で一度に縮約する。

    縮約前の ``state`` は ``tt_linear_forward`` が site ``k`` に入るときの軸

        (batch, i_1, ..., i_{k-1}, α, j_k, j_{k+1}, ..., j_d)

    を持つ。``T`` の軸は ``(α, i_k, j_k, i_{k+1}, j_{k+1}, γ)``。
    ``α, j_k, j_{k+1}`` を縮約し、残る ``(i_k, i_{k+1}, γ)`` を ``α`` があった
    位置へ戻す。結果は、通常の forward が site ``k+1`` を終えた直後の軸

        (batch, i_1, ..., i_{k+1}, γ, j_{k+2}, ..., j_d)

    と一致する。
    """
    left_bond_axis = 1 + site_index
    j_k_axis = left_bond_axis + 1
    j_next_axis = left_bond_axis + 2
    state = torch.tensordot(
        state,
        merged,
        dims=([left_bond_axis, j_k_axis, j_next_axis], [0, 2, 4]),
    )
    return state.movedim(
        [-3, -2, -1],
        [left_bond_axis, left_bond_axis + 1, left_bond_axis + 2],
    )


def tt_linear_forward_with_merged_pair(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    pair_index: int,
    merged: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""隣接 pair の2 core の代わりに結合テンソル ``T`` を使い、``Y = X W^T + b`` を計算する。

    x:
        shape ``(batch, in_features)``。``tt_linear_forward`` と同じ。
    cores:
        層が持つ TT-matrix core 列。pair 以外の site の**値**を縮約に使う。
        ``cores[pair_index]`` と ``cores[pair_index + 1]`` は、左右の bond と
        mode が ``merged`` と合うかの検証にだけ使い、値は読まない。
    pair_index:
        結合した左 core の site index。``0 <= pair_index <= d - 2``。
    merged:
        ``merge_tt_matrix_cores`` と同じ軸順の6階テンソル。
        shape ``(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})``。
        中間 bond ``r_k`` は既に縮約済みなので、この引数には現れない。
    out_modes, in_modes, bias:
        ``tt_linear_forward`` と同じ。

    戻り値:
        shape ``(batch, out_features)``。``x`` / ``merged`` / pair 以外の core /
        ``bias`` の autograd graph をつなぐ。dense な ``W`` は作らない。

    成立条件:
        ``merged`` がちょうど ``cores[k]`` と ``cores[k+1]`` の縮約なら、
        戻り値は ``tt_linear_forward`` と一致する（浮動小数点の丸めを除く）。
        学習で ``merged`` を更新したあとも、同じ縮約経路で ``Y`` を計算する。
    """
    validate_tt_matrix_cores(cores, out_modes, in_modes)
    validate_tt_linear_input(x, cores, in_modes)
    validate_tt_linear_bias(bias, cores, out_modes)
    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    index = _pair_index(pair_index, len(cores))

    if not torch.is_tensor(merged):
        raise TypeError(f"merged は torch.Tensor である必要があります: {type(merged)!r}")
    _validate_same_dtype_device((("cores[0]", cores[0]), ("merged", merged)))
    expected_shape = (
        cores[index].shape[0],
        out_modes_t[index],
        in_modes_t[index],
        out_modes_t[index + 1],
        in_modes_t[index + 1],
        cores[index + 1].shape[3],
    )
    if tuple(merged.shape) != expected_shape:
        raise ValueError(
            f"merged.shape={tuple(merged.shape)} は pair={index} の結合 shape "
            f"{expected_shape} と一致する必要があります。"
        )

    batch = x.shape[0]
    state = x.reshape(batch, *in_modes_t).unsqueeze(1)
    site = 0
    while site < len(cores):
        if site == index:
            state = _contract_merged_pair(state, merged, site)
            site += 2
        else:
            state = _contract_one_core(state, cores[site], site)
            site += 1

    # 最終軸は右端 bond α_d = 1。出力 mode (i_1, ..., i_d) だけを残す。
    y = state.squeeze(-1).reshape(batch, math.prod(out_modes_t))
    if bias is not None:
        y = y + bias
    return y


def assign_tt_linear_cores(
    layer: TTLinear,
    cores: Sequence[torch.Tensor],
    *,
    requires_grad: bool | None = None,
) -> None:
    """``TTLinear`` の core Parameter へ、別の core 列の値を書き込む。

    同じ shape の core は既存 Parameter へ ``copy_`` する。bond 次元が変わり
    shape が違う core は、その位置の Parameter を差し替える。

    layer:
        書き込み先。``out_modes`` / ``in_modes`` は変えない。
    cores:
        ``layer`` の mode と一致する TT-matrix core 列。
    requires_grad:
        ``None`` なら各 core の現在の ``requires_grad`` を保つ。
        bool ならすべての core をその値にする。

    この関数は gauge 変換や SVD 後の core をモジュールへ戻すための値の代入であり、
    学習グラフは保持しない。shape が変わるときは ``TTLinear`` の初期化と同じく
    ``detach().clone()`` した leaf を登録する。同じ shape の ``copy_`` は
    ``torch.no_grad`` の中で行う。

    bias と、この層以外の parameter には触らない。

    shape が変わる core は、既存の ``nn.Parameter`` を新しい Parameter に差し替える。
    差し替え前の Parameter を optimizer の ``param_groups`` が参照している場合、
    その optimizer は古い Parameter を見続け、新しい Parameter 用の state は作られない。
    この関数は optimizer state を移行しない。shape が変わる書き戻しのあとで同じ
    optimizer を使うときは、呼び出し側で optimizer を作り直す。同じ shape の
    ``copy_`` は Parameter の identity を保つので、既存 optimizer の参照はそのまま有効である。
    """
    if not isinstance(layer, TTLinear):
        raise TypeError(f"layer は TTLinear である必要があります: {type(layer).__name__}")
    if requires_grad is not None and not isinstance(requires_grad, bool):
        raise TypeError(
            f"requires_grad は bool または None である必要があります: {type(requires_grad)!r}"
        )
    validate_tt_matrix_cores(cores, layer.out_modes, layer.in_modes)
    if len(cores) != len(layer.cores):
        raise ValueError(
            f"len(cores)={len(cores)} は len(layer.cores)={len(layer.cores)} と一致する必要があります。"
        )
    _validate_same_dtype_device(
        [("layer.cores[0]", layer.cores[0]), *((f"cores[{i}]", core) for i, core in enumerate(cores))]
    )

    for index, core in enumerate(cores):
        keep_grad = layer.cores[index].requires_grad if requires_grad is None else requires_grad
        if tuple(layer.cores[index].shape) == tuple(core.shape):
            with torch.no_grad():
                layer.cores[index].copy_(core)
            layer.cores[index].requires_grad_(keep_grad)
        else:
            layer.cores[index] = nn.Parameter(
                core.detach().clone(memory_format=torch.contiguous_format),
                requires_grad=keep_grad,
            )


def _module_forward_replacing_tt_linear(model: nn.Module, inputs: torch.Tensor, layer_name: str, forward):
    """名前付き ``TTLinear`` の ``forward`` だけを一時的に差し替えて ``model`` を評価する。"""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model).__name__}")
    if not isinstance(layer_name, str) or layer_name == "":
        raise ValueError(f"layer_name は空でない str である必要があります: {layer_name!r}")
    layer = get_named_module(model, layer_name)
    if not isinstance(layer, TTLinear):
        raise TypeError(
            f"{layer_name} は TTLinear である必要があります: {type(layer).__name__}"
        )
    original = layer.forward
    layer.forward = forward
    try:
        return model(inputs)
    finally:
        layer.forward = original


def module_forward_with_merged_tt_linear_pair(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    pair_index: int,
    merged: torch.Tensor,
) -> torch.Tensor:
    """モデル全体の forward のうち、指定 TTLinear の隣接 pair だけを ``merged`` で計算する。

    対象層以外（活性化、後続層、対象層の bias と pair 以外の core）は
    ``model`` が持つ実装のまま動く。dense 重みは再構成しない。
    ``torch.no_grad`` では囲まないので、``merged`` から出力までのグラフは残る。

    戻り値は ``model(inputs)`` と同じ shape。例外時も対象層の ``forward`` は戻す。
    """
    layer = get_named_module(model, layer_name)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return tt_linear_forward_with_merged_pair(
            x,
            list(layer.cores),
            pair_index,
            merged,
            layer.out_modes,
            layer.in_modes,
            layer.bias,
        )

    return _module_forward_replacing_tt_linear(model, inputs, layer_name, forward)


def module_forward_with_tt_linear_cores(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    cores: Sequence[torch.Tensor],
) -> torch.Tensor:
    """モデル全体の forward のうち、指定 TTLinear だけを明示した core 列で計算する。

    SVD で分け直した core を、モジュールの Parameter へ書き戻す前に
    出力比較へ使う。``tt_linear_forward`` を呼ぶので dense 重みは作らない。
    ``cores`` を ``detach`` しない。``torch.no_grad`` では囲まない。

    戻り値は ``model(inputs)`` と同じ shape。例外時も対象層の ``forward`` は戻す。
    """
    layer = get_named_module(model, layer_name)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return tt_linear_forward(
            x,
            cores,
            layer.out_modes,
            layer.in_modes,
            layer.bias,
        )

    return _module_forward_replacing_tt_linear(model, inputs, layer_name, forward)

