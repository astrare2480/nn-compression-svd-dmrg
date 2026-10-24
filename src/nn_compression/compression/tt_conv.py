r"""TT-matrix core で重みを保持する Conv2d（TT-Conv）と、dense ``nn.Conv2d`` からの変換。

参照元:
    notebooks/30_tt_mps/20_fashion_mnist_cnn/00_ttconv_representation_and_dense_kernel_reconstruction.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/01_ttconv_direct_forward_and_backward_equivalence.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/04_pretrained_cnn_conv2_tt_cores_only_finetuning.ipynb

Conv2d の重み ``(out, in, kH, kW)`` を ``(out, in * kH * kW)`` の行列とみなすと、
畳み込みは「入力を ``F.unfold`` で patch 行列にして、行列積（Linear）を掛ける」
計算になる。その行列を TT-matrix で持てば、既存の ``tt_linear_forward`` が
dense kernel を作らずに forward を担当する。

    patch 行列: (batch * H_out * W_out, in * kH * kW)   # 各行が1つの patch
    出力      : (batch * H_out * W_out, out) -> (batch, out, H_out, W_out)

``unfold`` の patch 軸は ``(in, kH, kW)`` の順（channel が最も外側）で、
``weight.reshape(out, -1)`` の列順と同じ。そのため、行列化・TT-SVD・patch 行列の
どこでも並べ替えは不要。

この module が持つもの:

- ``tt_conv2d_forward``: dense kernel を作らない forward（関数版）。
- ``TTConv2d``: core と bias を ``nn.Parameter`` として持つ ``nn.Module``。
- ``build_tt_conv2d``: dense ``nn.Conv2d`` から ``TTConv2d`` を作る。
- ``factorize_named_tt_conv2d``: 名前で指した ``nn.Conv2d`` だけを置換したコピーを返す。
- ``tt_conv2d_to_dense_weight``: 検証・診断・dense 層への再変換用に dense kernel を復元する。
- ``sweep_tt_conv2d_weight_ranks``: ``Conv2d`` の weight に対する rank 上限ごとの誤差と parameter 数。

TT-SVD・dense 復元・TT 縮約は既存の ``dense_to_tt_matrix_cores`` /
``tt_matrix_to_dense`` / ``tt_linear_forward`` に任せ、ここでは再実装しない。

対応する設定: ``groups = 1``、zero padding、``int`` または2要素の
``stride`` / ``padding`` / ``dilation``。未対応: ``groups != 1``、
``padding_mode != "zeros"``、文字列 padding（``"same"`` / ``"valid"``）、
転置畳み込み。

Notebook に残すもの: checkpoint・CSV・plot のパス、固定 modes / rank 候補、
モデル全体の accuracy 評価、Notebook 内の ``assert`` による検証表示。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.modules import set_named_module
from ._named_layer import _get_named_layer
from .conv_svd import conv2d_weight_matrix
from .svd import coerce_integer_scalar
from .tt_matrix import (
    dense_to_tt_matrix_cores,
    tt_linear_forward,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from .tt_matrix_validation import (
    validate_dense_tt_matrix_weight,
    validate_tt_linear_bias,
    validate_tt_matrix_cores,
    validate_tt_matrix_modes,
)
from .tt_modes import _positive_int
from .tt_rank_sweep import sweep_tt_matrix_weight_ranks


# ---------------------------------------------------------------------------
# 非公開 helper（設定の正規化と検証）
# ---------------------------------------------------------------------------


def _as_pair(value, *, name: str, minimum: int) -> tuple[int, int]:
    """``int`` または2要素の列を、``(高さ, 幅)`` の ``tuple[int, int]`` へ正規化する。

    bool・float・文字列・Tensor・ndarray は ``TypeError``。要素数が2でない列と
    ``minimum`` 未満の値は ``ValueError``。numpy 整数 scalar は許す。
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} は int または2要素の整数列である必要があります: {value!r}")

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} は2要素（高さ, 幅）である必要があります: {value!r}")
        pair = tuple(
            coerce_integer_scalar(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    else:
        scalar = coerce_integer_scalar(value, name=name)
        pair = (scalar, scalar)

    for index, item in enumerate(pair):
        if item < minimum:
            raise ValueError(f"{name}[{index}] は {minimum} 以上である必要があります: {item}")
    return pair


def _validate_supported_settings(groups: int, padding_mode: str) -> None:
    """``groups = 1`` と zero padding 以外を ``NotImplementedError`` にする。"""
    if groups != 1:
        raise NotImplementedError(f"groups=1 のみ対応しています: groups={groups}")
    if not isinstance(padding_mode, str):
        raise TypeError(f"padding_mode は str である必要があります: {padding_mode!r}")
    if padding_mode != "zeros":
        raise NotImplementedError(
            f"padding_mode='zeros'（zero padding）のみ対応しています: {padding_mode!r}"
        )


def _validate_conv_modes(
    out_modes: tuple[int, ...],
    in_modes: tuple[int, ...],
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
) -> None:
    """mode の積が channel 数と受容野の幅に一致することを検証する。"""
    out_product = math.prod(out_modes)
    if out_product != out_channels:
        raise ValueError(
            f"prod(out_modes)={out_product} は out_channels={out_channels} と"
            "一致する必要があります。"
        )
    patch_width = in_channels * kernel_size[0] * kernel_size[1]
    in_product = math.prod(in_modes)
    if in_product != patch_width:
        raise ValueError(
            f"prod(in_modes)={in_product} は in_channels * kernel_height * kernel_width"
            f"={in_channels} * {kernel_size[0]} * {kernel_size[1]}={patch_width} と"
            "一致する必要があります。"
        )


def _conv_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    """1 軸分の出力サイズ（``nn.Conv2d`` と同じ式）。0 以下は不正な設定。"""
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _validate_conv_for_tt(
    conv: nn.Conv2d,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], torch.Tensor]:
    """``build_tt_conv2d`` と ``sweep_tt_conv2d_weight_ranks`` が共有する入力検証。

    ``conv`` が TT 化できる ``nn.Conv2d`` か、modes が weight 行列へ合うか、
    weight が有限かを確かめる。TT-SVD は NaN / inf に対して backend 依存の
    ``_LinAlgError`` を出すため、その前に ``ValueError`` で拒否する。

    戻り値:
        ``(out_modes, in_modes, weight_matrix)``。``weight_matrix`` は
        ``conv.weight.detach()`` を ``(out, in * kH * kW)`` へ並べた行列（view の
        ことがあるので、呼び出し側は書き込まない）。
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"conv は nn.Conv2d である必要があります: {type(conv).__name__}")
    if isinstance(conv.padding, str):
        raise NotImplementedError(
            f"文字列 padding は未対応です: padding={conv.padding!r}"
            "（int または2要素の padding のみ）"
        )
    _validate_supported_settings(conv.groups, conv.padding_mode)

    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    _validate_conv_modes(
        out_modes_t,
        in_modes_t,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=tuple(conv.kernel_size),
    )

    weight_matrix = conv2d_weight_matrix(conv.weight.detach())
    validate_dense_tt_matrix_weight(
        weight_matrix, out_modes_t, in_modes_t, name="conv.weight（行列化後）"
    )
    if not bool(torch.isfinite(weight_matrix).all()):
        raise ValueError("conv.weight に NaN / inf が含まれています。")
    return out_modes_t, in_modes_t, weight_matrix


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def tt_conv2d_forward(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    bias: torch.Tensor | None = None,
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> torch.Tensor:
    r"""TT-matrix core を直接縮約して、``Conv2d`` 相当の forward を計算する。

    dense kernel は作らない。次の4段で計算する。

    1. ``F.unfold`` で patch を取り出す: ``(B, C*kH*kW, L)``（``L = H_out * W_out``）。
    2. 各 patch を1行に並べる: ``(B * L, C*kH*kW)``（行順は画像0の全位置、画像1の全位置、…）。
    3. 既存の ``tt_linear_forward`` を適用する: ``(B * L, out_channels)``。
    4. ``(B, out_channels, H_out, W_out)`` へ戻す（連続メモリ）。

    x:
        入力。shape ``(batch, in_channels, height, width)``。変更しない。
        dtype / device は ``cores[0]`` と一致する必要がある。
    cores:
        TT-matrix core 列（``validate_tt_matrix_cores`` の contract）。変更しない。
    out_modes, in_modes:
        ``prod(out_modes) = out_channels``、
        ``prod(in_modes) = in_channels * kernel_height * kernel_width``。
        ``d = 1``（core が1個）も使える。
    bias:
        ``None`` または shape ``(out_channels,)``。dtype / device は ``cores[0]`` と一致。
    kernel_size, stride, padding, dilation:
        ``int`` または ``(高さ, 幅)`` の2要素。zero padding のみ。
        ``kernel_size`` は必須（keyword-only）。

    戻り値:
        shape ``(batch, out_channels, H_out, W_out)``。dtype / device / autograd
        は ``x`` / ``cores`` / ``bias`` を引き継ぐ。出力サイズは ``nn.Conv2d`` と同じ式
        ``(size + 2*padding - dilation*(kernel - 1) - 1) // stride + 1``。

    例外:
        型の誤りは ``TypeError``。``x`` が4階でない、channel 数が mode の積と合わない、
        dtype 以外の ``device`` 不一致、出力サイズが 0 以下になる設定は ``ValueError``
        （dtype 不一致は ``TypeError``）。

    Note:
        NaN / inf の検査はしない（``nn.Conv2d`` や ``tt_linear_forward`` と同じ）。
        入力の NaN / inf は IEEE の規則で出力へ伝わる。入力・core・bias を
        in-place に変更せず、``detach`` や CPU 転送もしないので autograd は切れない。
    """
    kernel = _as_pair(kernel_size, name="kernel_size", minimum=1)
    stride_pair = _as_pair(stride, name="stride", minimum=1)
    padding_pair = _as_pair(padding, name="padding", minimum=0)
    dilation_pair = _as_pair(dilation, name="dilation", minimum=1)

    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    validate_tt_matrix_cores(cores, out_modes_t, in_modes_t)
    validate_tt_linear_bias(bias, cores, out_modes_t)

    if not torch.is_tensor(x):
        raise TypeError(f"x は torch.Tensor である必要があります: {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(
            f"x は4階（batch, channels, height, width）である必要があります: "
            f"ndim={x.ndim}, shape={tuple(x.shape)}"
        )
    if x.dtype != cores[0].dtype:
        raise TypeError(
            f"x.dtype={x.dtype} は cores[0].dtype={cores[0].dtype} と一致する必要があります。"
        )
    if x.device != cores[0].device:
        raise ValueError(
            f"x.device={x.device} は cores[0].device={cores[0].device} と"
            "一致する必要があります。"
        )

    batch, in_channels, height, width = x.shape
    out_channels = math.prod(out_modes_t)
    _validate_conv_modes(
        out_modes_t,
        in_modes_t,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel,
    )

    out_height = _conv_output_size(
        height, kernel[0], stride_pair[0], padding_pair[0], dilation_pair[0]
    )
    out_width = _conv_output_size(
        width, kernel[1], stride_pair[1], padding_pair[1], dilation_pair[1]
    )
    if out_height <= 0 or out_width <= 0:
        raise ValueError(
            "この設定では出力の空間サイズが 0 以下になります: "
            f"入力=({height}, {width}), kernel_size={kernel}, stride={stride_pair}, "
            f"padding={padding_pair}, dilation={dilation_pair}"
        )

    unfolded = F.unfold(
        x,
        kernel_size=kernel,
        dilation=dilation_pair,
        padding=padding_pair,
        stride=stride_pair,
    )
    patch_width = in_channels * kernel[0] * kernel[1]
    patch_matrix = unfolded.permute(0, 2, 1).reshape(batch * out_height * out_width, patch_width)

    patch_output = tt_linear_forward(patch_matrix, cores, out_modes_t, in_modes_t, bias)

    # (B*H*W, C_out) -> (B, H, W, C_out) -> (B, C_out, H, W)。nn.Conv2d と同じく連続メモリで返す。
    return (
        patch_output.reshape(batch, out_height, out_width, out_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# 層
# ---------------------------------------------------------------------------


class TTConv2d(nn.Module):
    """TT-matrix core 列で重みを保持し、dense kernel を作らずに畳み込む層。

    cores:
        TT-matrix core 列（list / tuple）。各 core の shape は
        ``(r_{k-1}, m_k, n_k, r_k)``、``r_0 = r_d = 1``。contract は
        ``validate_tt_matrix_cores`` と同じ。
    out_modes, in_modes:
        ``prod(out_modes) = out_channels``、
        ``prod(in_modes) = in_channels * kernel_height * kernel_width``。
    in_channels, out_channels:
        bool ではない正の整数。``out_channels`` は ``prod(out_modes)`` と一致が必要。
    kernel_size, stride, padding, dilation:
        ``int`` または ``(高さ, 幅)``。属性は ``nn.Conv2d`` と同じく ``tuple[int, int]``。
        ``padding`` は 0 以上の整数のみ（文字列は未対応）。
    groups:
        1 のみ。それ以外は ``NotImplementedError``。
    padding_mode:
        ``"zeros"`` のみ。それ以外は ``NotImplementedError``。
    bias:
        ``None`` または shape ``(out_channels,)``。dtype / device は core と一致。

    Parameter 構成:
        - ``self.cores``: ``nn.ParameterList``。``cores.{k}`` は
          ``(r_{k-1}, m_k, n_k, r_k)`` の leaf ``nn.Parameter``。
        - ``self.bias``: leaf ``nn.Parameter`` または ``None``。
        - dense な ``weight`` parameter は持たない。

        いずれも渡された Tensor の ``detach().clone()`` で、この層が独立して所有する
        （元の Tensor と storage を共有せず、勾配も流れない）。既定で
        ``requires_grad=True``。凍結は通常の PyTorch の方法で行う。

    入力:
        ``forward`` は ``(batch, in_channels, height, width)`` の4階 Tensor。
        dtype / device は core と一致が必要（``tt_conv2d_forward`` の contract）。

    Note:
        ``.to(device)`` / ``.to(dtype)`` は通常の ``nn.Module`` として動く。
        float32 / float64 以外へ変換すると、forward は dtype 検証で ``TypeError``。
        ``forward`` は dense kernel を復元しない（復元は ``tt_conv2d_to_dense_weight``）。
    """

    def __init__(
        self,
        cores: Sequence[torch.Tensor],
        out_modes: Sequence[int],
        in_modes: Sequence[int],
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        in_channels_int = _positive_int(in_channels, name="in_channels")
        out_channels_int = _positive_int(out_channels, name="out_channels")
        groups_int = _positive_int(groups, name="groups")
        _validate_supported_settings(groups_int, padding_mode)

        kernel = _as_pair(kernel_size, name="kernel_size", minimum=1)
        stride_pair = _as_pair(stride, name="stride", minimum=1)
        padding_pair = _as_pair(padding, name="padding", minimum=0)
        dilation_pair = _as_pair(dilation, name="dilation", minimum=1)

        out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
        validate_tt_matrix_cores(cores, out_modes_t, in_modes_t)
        validate_tt_linear_bias(bias, cores, out_modes_t)
        _validate_conv_modes(
            out_modes_t,
            in_modes_t,
            in_channels=in_channels_int,
            out_channels=out_channels_int,
            kernel_size=kernel,
        )

        self.cores = nn.ParameterList(
            [
                nn.Parameter(core.detach().clone(memory_format=torch.contiguous_format))
                for core in cores
            ]
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_parameter(
                "bias",
                nn.Parameter(bias.detach().clone(memory_format=torch.contiguous_format)),
            )

        self.in_channels: int = in_channels_int
        self.out_channels: int = out_channels_int
        self.kernel_size: tuple[int, int] = kernel
        self.stride: tuple[int, int] = stride_pair
        self.padding: tuple[int, int] = padding_pair
        self.dilation: tuple[int, int] = dilation_pair
        self.groups: int = groups_int
        self.padding_mode: str = padding_mode
        self.out_modes: tuple[int, ...] = out_modes_t
        self.in_modes: tuple[int, ...] = in_modes_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tt_conv2d_forward(
            x,
            list(self.cores),
            self.out_modes,
            self.in_modes,
            self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def extra_repr(self) -> str:
        ranks = tt_matrix_ranks(list(self.cores))
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"padding_mode={self.padding_mode!r}, "
            f"in_modes={self.in_modes}, out_modes={self.out_modes}, "
            f"ranks={ranks}, bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# dense Conv2d からの変換
# ---------------------------------------------------------------------------


def build_tt_conv2d(
    conv: nn.Conv2d,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> TTConv2d:
    """dense ``nn.Conv2d`` から ``TTConv2d`` を作る。

    ``conv.weight`` ``(out, in, kH, kW)`` を ``(out, in * kH * kW)`` へ並べ、
    既存の ``dense_to_tt_matrix_cores`` で TT-matrix core にする。

    conv:
        変換元。変更しない。dtype は float32 / float64。``groups = 1``、
        ``padding_mode = "zeros"``、int / 2要素の padding のみ。
        weight に NaN / inf があれば ``ValueError``（SVD の ``_LinAlgError`` を出さない）。
    out_modes, in_modes:
        ``prod(out_modes) = conv.out_channels``、
        ``prod(in_modes) = conv.in_channels * kernel_height * kernel_width``。
    max_rank:
        ``None``（既定）なら打ち切りなしの exact TT-SVD。正の整数ならその値を
        bond rank の上限とする。``d = 1`` では結果を変えない。不正値は
        ``dense_to_tt_matrix_cores`` と同じ contract で拒否する。

    戻り値:
        ``TTConv2d``。

        - dtype / device は ``conv.weight`` を維持する。bias の有無も維持する。
        - Parameter は ``conv`` と storage を共有しない（``d = 1`` でも）。
        - ``requires_grad`` は weight → 全 core、bias → bias に、それぞれ引き継ぐ。
        - ``training`` フラグは ``conv.training`` に合わせる。
        - stride / padding / dilation / kernel_size は ``conv`` と同じ。

    例外:
        ``conv`` が ``nn.Conv2d`` でなければ ``TypeError``。``groups != 1``、非 zero padding、
        文字列 padding は ``NotImplementedError``。
    """
    out_modes_t, in_modes_t, weight_matrix = _validate_conv_for_tt(conv, out_modes, in_modes)

    cores = dense_to_tt_matrix_cores(weight_matrix, out_modes_t, in_modes_t, max_rank=max_rank)
    layer = TTConv2d(
        cores,
        out_modes_t,
        in_modes_t,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        padding_mode=conv.padding_mode,
        bias=conv.bias,
    )

    for core in layer.cores:
        core.requires_grad_(conv.weight.requires_grad)
    if layer.bias is not None:
        layer.bias.requires_grad_(conv.bias.requires_grad)

    layer.train(conv.training)
    return layer


def factorize_named_tt_conv2d(
    model: nn.Module,
    layer_name: str,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> nn.Module:
    """指定した名前の ``nn.Conv2d`` だけを ``TTConv2d`` へ置換したコピーを返す。

    ``factorize_named_tt_linear`` と同じ方針: ``model`` は変更せず ``deepcopy`` した
    コピー側だけを置換する。コピーを fine-tune しても元の model の Parameter は動かない。

    layer_name:
        ``"conv2"`` や ``"features.3"`` のようなドット区切りの名前（ネストした名前と
        ``nn.Sequential`` 内の番号付き名前を含む）。空文字は不可。存在しない名前は
        ``get_submodule`` と同じ ``AttributeError``。
    out_modes, in_modes, max_rank:
        ``build_tt_conv2d`` と同じ。

    例外:
        ``layer_name`` が ``str`` でなければ ``TypeError``、空なら ``ValueError``。
        指す module が ``nn.Conv2d`` でなければ ``TypeError``。変換できない設定は
        ``build_tt_conv2d`` と同じ。

    Note:
        model 全体の dtype / device / 各 module の ``training`` 状態は ``deepcopy`` が
        そのまま保持する。置換される層の ``training`` は元の ``nn.Conv2d`` に揃える。
    """
    layer = _get_named_layer(model, layer_name, nn.Conv2d)
    tt_layer = build_tt_conv2d(layer, out_modes, in_modes, max_rank=max_rank)

    compressed_model = copy.deepcopy(model)
    set_named_module(compressed_model, layer_name, tt_layer)
    return compressed_model


def tt_conv2d_to_dense_weight(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    in_channels: int,
    kernel_size: int | Sequence[int],
) -> torch.Tensor:
    """TT-matrix core を、診断・検証用の dense Conv2d weight へ戻す。

    ``tt_matrix_to_dense`` で ``(out, in * kH * kW)`` を復元し、
    ``(out_channels, in_channels, kernel_height, kernel_width)`` へ reshape する。
    等価性の検証・デバッグ・dense 層への再変換用で、``TTConv2d.forward`` は
    この関数を使わない。

    cores:
        TT-matrix core 列。変更しない。
    out_modes, in_modes:
        ``out_channels = prod(out_modes)``、
        ``prod(in_modes) = in_channels * kernel_height * kernel_width``。
    in_channels:
        bool ではない正の整数。
    kernel_size:
        ``int`` または ``(高さ, 幅)``。

    戻り値:
        shape ``(out_channels, in_channels, kernel_height, kernel_width)``。
        dtype / device / autograd は ``cores`` を引き継ぐ。``cores`` と storage を
        共有しない（``d = 1`` でも書き込みで core を壊さない）。
    """
    kernel = _as_pair(kernel_size, name="kernel_size", minimum=1)
    in_channels_int = _positive_int(in_channels, name="in_channels")
    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    validate_tt_matrix_cores(cores, out_modes_t, in_modes_t)
    out_channels = math.prod(out_modes_t)
    _validate_conv_modes(
        out_modes_t,
        in_modes_t,
        in_channels=in_channels_int,
        out_channels=out_channels,
        kernel_size=kernel,
    )

    weight_matrix = tt_matrix_to_dense(cores, out_modes_t, in_modes_t)
    weight = weight_matrix.reshape(out_channels, in_channels_int, kernel[0], kernel[1])
    if len(cores) == 1:
        # d = 1 の復元結果は core の view になりうる。書き込みが core を壊さないよう複製する。
        weight = weight.clone()
    return weight


# ---------------------------------------------------------------------------
# rank sweep
# ---------------------------------------------------------------------------


def sweep_tt_conv2d_weight_ranks(
    conv: nn.Conv2d,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
) -> list[dict]:
    """rank 上限の候補ごとに、``Conv2d`` の weight を TT 近似した結果を記録する。

    ``conv.weight`` を ``(out, in * kH * kW)`` へ並べ、既存の
    ``sweep_tt_matrix_weight_ranks`` に渡す。TT-SVD・誤差計算・スケール処理は
    そちらの contract に従う（再実装しない）。parameter 数だけ、bias を含む
    層全体の値を加える。

    conv:
        対象の ``nn.Conv2d``。変更しない。条件は ``build_tt_conv2d`` と同じ
        （零 weight と NaN / inf は ``ValueError``）。
    out_modes, in_modes:
        ``build_tt_conv2d`` と同じ。
    ranks:
        rank 上限の候補列（空でない、各要素は bool ではない 1 以上の整数）。
        重複や非昇順も許し、与えた順にそのまま記録する。

    戻り値:
        ``ranks`` と同じ順序の ``list[dict]``。各 record のキーと値の型:

        - ``requested_rank`` (int): 渡された rank 上限。
        - ``actual_ranks`` (tuple[int, ...]): 境界を含む実際の TT-rank。
        - ``tt_core_parameters`` (int): core の総要素数（行列 sweep の ``tt_parameters``）。
        - ``dense_weight_parameters`` (int): ``conv.weight.numel()``
          （行列 sweep の ``dense_parameters``）。
        - ``bias_parameters`` (int): bias の要素数。bias なしなら 0。
        - ``dense_layer_parameters`` (int): ``dense_weight_parameters + bias_parameters``。
        - ``tt_layer_parameters`` (int): ``tt_core_parameters + bias_parameters``。
        - ``weight_compression_ratio`` (float): ``dense_weight_parameters / tt_core_parameters``
          （行列 sweep の ``compression_ratio``）。
        - ``layer_compression_ratio`` (float): ``dense_layer_parameters / tt_layer_parameters``。
        - ``absolute_error`` (float): ``||W - W_hat||_F``（行列 sweep と同じ値）。
        - ``relative_error`` (float): ``||W - W_hat||_F / ||W||_F``（同上）。
        - ``is_weight_compressed`` (bool): ``tt_core_parameters < dense_weight_parameters``
          （行列 sweep の ``is_compressed``）。
        - ``is_layer_compressed`` (bool): ``tt_layer_parameters < dense_layer_parameters``。

    Note:
        autograd の計算グラフは作らない。数値はすべて Python の ``int`` / ``float`` /
        ``bool`` / ``tuple``。誤差が rank とともに単調に減ることは保証しない。
    """
    out_modes_t, in_modes_t, weight_matrix = _validate_conv_for_tt(conv, out_modes, in_modes)
    matrix_records = sweep_tt_matrix_weight_ranks(weight_matrix, out_modes_t, in_modes_t, ranks)

    bias_parameters = 0 if conv.bias is None else int(conv.bias.numel())
    records: list[dict] = []
    for record in matrix_records:
        tt_core_parameters = record["tt_parameters"]
        dense_weight_parameters = record["dense_parameters"]
        dense_layer_parameters = dense_weight_parameters + bias_parameters
        tt_layer_parameters = tt_core_parameters + bias_parameters
        records.append(
            {
                "requested_rank": record["requested_rank"],
                "actual_ranks": record["actual_ranks"],
                "tt_core_parameters": tt_core_parameters,
                "dense_weight_parameters": dense_weight_parameters,
                "bias_parameters": bias_parameters,
                "dense_layer_parameters": dense_layer_parameters,
                "tt_layer_parameters": tt_layer_parameters,
                "weight_compression_ratio": record["compression_ratio"],
                "layer_compression_ratio": dense_layer_parameters / tt_layer_parameters,
                "absolute_error": record["absolute_error"],
                "relative_error": record["relative_error"],
                "is_weight_compressed": record["is_compressed"],
                "is_layer_compressed": tt_layer_parameters < dense_layer_parameters,
            }
        )
    return records
