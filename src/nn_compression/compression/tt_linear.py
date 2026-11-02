"""学習可能な TT-Linear 層（``nn.Module``）と、dense ``nn.Linear`` からの変換。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb
    notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb
    notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb

Notebook 00 の教材用 ``TTLinear`` は、core を ``requires_grad=False`` にし、
bias を buffer にしていたため、fine-tuning には使えなかった。ここでは
core と bias を通常の ``nn.Parameter`` として保持し、optimizer で更新できる
production 向けの層にする。

forward は既存の ``tt_linear_forward`` を呼ぶだけで、dense weight を復元しない。
変換と層置換は、既存の ``dense_to_tt_matrix_cores`` と
``utils.get_named_module`` / ``utils.set_named_module`` を再利用する
（Notebook 内の ``get_module_by_path`` / ``set_module_by_path`` は移さない）。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
from torch import nn

from ..utils.modules import set_named_module
from ._named_layer import _get_named_layer
from ..tt_mps.matrix import dense_to_tt_matrix_cores, tt_matrix_ranks
from ..tt_mps.matrix_validation import validate_tt_matrix_cores, validate_tt_matrix_modes
from .tt_linear_validation import validate_tt_linear_bias, validate_tt_linear_input


def tt_linear_forward(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""dense weight を作らずに、TT-matrix core 列から ``Y = X W^T + b`` を計算する。

    x:
        入力。shape ``(batch, n)``（``n = product(in_modes)``）の2階 Tensor
        のみ正式対応。変更しない。
    cores:
        ``validate_tt_matrix_cores`` の contract を満たす TT-matrix core 列。
        変更しない。
    out_modes:
        出力側 physical modes ``(m_1, ..., m_d)``。``m = product(out_modes)``。
    in_modes:
        入力側 physical modes ``(n_1, ..., n_d)``。``n = product(in_modes)``。
    bias:
        ``None``（既定）または shape ``(m,)`` の1階 Tensor。変更しない。

    戻り値:
        shape ``(batch, m)`` の Tensor。``x`` / ``cores`` の dtype / device /
        autograd を維持する。

    アルゴリズム（dense化しない縮約）:
        ``x`` を ``(batch, n_1, ..., n_d)`` へ reshape したのち、site
        ``k = 1, ..., d`` の順に、現在の状態テンソルが持つ

            (batch, i_1, ..., i_{k-1}, alpha_{k-1}, j_k, j_{k+1}, ..., j_d)

        という軸配置から、``alpha_{k-1}``（左 bond）と ``j_k``（入力側
        physical軸）を ``core_k``（shape ``(alpha_{k-1}, i_k, j_k,
        alpha_k)``）と ``torch.tensordot`` で直接縮約し、新しくできる
        ``(i_k, alpha_k)`` を ``torch.movedim`` で正しい位置（直前の
        ``i_{k-1}`` の後ろ）へ戻す。この手順を ``d`` 回繰り返すと、最終的に

            (batch, i_1, ..., i_d, alpha_d)

        になる（``alpha_d`` は境界rankで常にサイズ1）。これを ``squeeze``
        してから ``(batch, m)`` へ ``reshape`` し、``bias`` を加える。

        この手順は ``tt_matrix_to_dense`` を一度も呼ばず、``(m, n)`` 相当の
        dense Tensor を一切構築しない。``torch.einsum`` の固定文字列
        （アルファベット26文字の上限）には依存しないため、``d`` の大きさに
        制約を課さない。

    Note:
        ``tensordot`` / ``movedim`` / ``reshape`` / ``squeeze`` はいずれも
        非破壊的な演算であり、``x`` / ``cores`` / ``bias`` を in-place に
        変更しない。``.item()`` / ``.detach()`` / 明示的な CPU 転送は内部で
        行わないため、autograd のグラフは切れない。
    """
    validate_tt_matrix_cores(cores, out_modes, in_modes)
    validate_tt_linear_input(x, cores, in_modes)
    validate_tt_linear_bias(bias, cores, out_modes)

    out_modes_t = tuple(int(m) for m in out_modes)
    in_modes_t = tuple(int(n) for n in in_modes)
    d = len(cores)
    batch = x.shape[0]

    # (batch, n) -> (batch, n_1, ..., n_d) -> (batch, alpha_0=1, n_1, ..., n_d)
    state = x.reshape(batch, *in_modes_t).unsqueeze(1)

    for k in range(d):
        core = cores[k]  # (alpha_{k-1}, m_k, n_k, alpha_k)
        # state の軸: (batch, i_1,...,i_{k-1}, alpha_{k-1}, j_k,...,j_d)
        # alpha_{k-1} は軸 (1 + k) 番目、j_k はその次。
        left_bond_axis = 1 + k
        in_mode_axis = left_bond_axis + 1
        state = torch.tensordot(
            state,
            core,
            dims=([left_bond_axis, in_mode_axis], [0, 2]),
        )
        # tensordot後、新しい (i_k, alpha_k) は末尾2軸に付く。
        # これを i_{k-1} の直後（left_bond_axis, left_bond_axis+1）へ戻す。
        state = state.movedim([-2, -1], [left_bond_axis, left_bond_axis + 1])

    # state の軸: (batch, i_1, ..., i_d, alpha_d=1)
    state = state.squeeze(-1)
    m = math.prod(out_modes_t)
    y = state.reshape(batch, m)

    if bias is not None:
        y = y + bias
    return y



class TTLinear(nn.Module):
    """TT-matrix core 列で重みを保持し、``Y = X W^T + b`` を計算する線形層。

    cores:
        TT-matrix core 列（list / tuple）。各 core の shape は
        ``(r_{k-1}, m_k, n_k, r_k)``、``r_0 = r_d = 1``。
        contract は ``tt_matrix_validation.validate_tt_matrix_cores`` と同じ
        （4階、bond 一致、dtype は float32 / float64、core 間で dtype / device 一致）。
    out_modes:
        出力側 mode ``(m_1, ..., m_d)``。``out_features = prod(out_modes)``。
    in_modes:
        入力側 mode ``(n_1, ..., n_d)``。``in_features = prod(in_modes)``。
    bias:
        ``None`` または shape ``(out_features,)`` の Tensor。dtype / device は
        core と一致している必要がある。

    Parameter 構成:
        - ``self.cores``: ``nn.ParameterList``。要素 ``cores.{k}`` は
          ``(r_{k-1}, m_k, n_k, r_k)`` の leaf ``nn.Parameter``。
        - ``self.bias``: leaf ``nn.Parameter``、または ``None``。

        いずれも渡された Tensor を ``detach().clone()`` したコピーで、
        このモジュールが所有する。元の Tensor は変更しない。既定で
        ``requires_grad=True``。凍結したい場合は、呼び出し側が通常の PyTorch の方法
        （``param.requires_grad_(False)``）で設定する。

    入力:
        ``forward`` は ``(batch, in_features)`` の2階 Tensor のみ受け付ける
        （``tt_linear_forward`` の contract）。dtype / device は core と一致が必要。

    Note:
        ``.to(device)`` / ``.to(dtype)`` は通常の ``nn.Module`` として core と bias を
        移す。ただし ``torch.float32`` / ``torch.float64`` 以外へ変換した場合、
        forward は dtype 検証で ``TypeError`` になる。
    """

    def __init__(
        self,
        cores: Sequence[torch.Tensor],
        out_modes: Sequence[int],
        in_modes: Sequence[int],
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
        validate_tt_matrix_cores(cores, out_modes_t, in_modes_t)
        validate_tt_linear_bias(bias, cores, out_modes_t)

        self.out_modes: tuple[int, ...] = out_modes_t
        self.in_modes: tuple[int, ...] = in_modes_t

        self.cores = nn.ParameterList(
            [
                nn.Parameter(
                    core.detach().clone(memory_format=torch.contiguous_format)
                )
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

    @property
    def in_features(self) -> int:
        return math.prod(self.in_modes)

    @property
    def out_features(self) -> int:
        return math.prod(self.out_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tt_linear_forward(
            x,
            list(self.cores),
            self.out_modes,
            self.in_modes,
            self.bias,
        )

    def extra_repr(self) -> str:
        ranks = tt_matrix_ranks(list(self.cores))
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_modes={self.in_modes}, out_modes={self.out_modes}, "
            f"ranks={ranks}, bias={self.bias is not None}"
        )


def build_tt_linear(
    linear: nn.Linear,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> TTLinear:
    """dense ``nn.Linear`` から ``TTLinear`` を作る。

    linear:
        変換元。変更しない。dtype は float32 / float64 のみ。
    out_modes, in_modes:
        ``prod(out_modes) == linear.out_features``、
        ``prod(in_modes) == linear.in_features`` を満たす mode 列。
    max_rank:
        ``None``（既定）なら打ち切りなし（``tt_svd_exact``）。
        正の整数なら、その値を bond rank の上限とする ``tt_svd``。
        ``d = 1``（1-site）は内部 bond がないため、``max_rank`` に依らず 1 core。
        不正値は ``dense_to_tt_matrix_cores`` と同じ contract で拒否する。

    戻り値:
        ``TTLinear``。

        - dtype / device は ``linear.weight`` を維持する。
        - bias の有無を維持する。
        - Parameter は ``linear`` から独立（storage を共有しない）。
        - ``requires_grad`` は ``linear.weight`` / ``linear.bias`` の状態を引き継ぐ
          （凍結した層から学習可能な層を作らない）。
        - ``training`` フラグは ``linear.training`` に合わせる。
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"linear は nn.Linear である必要があります: {type(linear).__name__}"
        )

    cores = dense_to_tt_matrix_cores(
        linear.weight.detach(), out_modes, in_modes, max_rank=max_rank
    )
    layer = TTLinear(cores, out_modes, in_modes, bias=linear.bias)

    for core in layer.cores:
        core.requires_grad_(linear.weight.requires_grad)
    if layer.bias is not None:
        layer.bias.requires_grad_(linear.bias.requires_grad)

    layer.train(linear.training)
    return layer


def factorize_named_tt_linear(
    model: nn.Module,
    layer_name: str,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
) -> nn.Module:
    """指定した名前の ``nn.Linear`` だけを ``TTLinear`` へ置換したコピーを返す。

    ``factorize_named_linear`` と同じ方針: ``model`` は変更せず ``deepcopy`` した
    コピー側だけを置換する。そのため、コピーを fine-tune しても元の model の
    Parameter は動かない。

    layer_name:
        ``"fc1"`` や ``"encoder.0"`` のようなドット区切りの名前。空文字は不可。
        存在しない名前は ``get_submodule`` と同じ ``AttributeError``。
    out_modes, in_modes, max_rank:
        ``build_tt_linear`` と同じ。

    例外:
        ``layer_name`` が ``str`` でなければ ``TypeError``、空なら ``ValueError``。
        指す module が ``nn.Linear`` でなければ ``TypeError``。

    Note:
        model 全体の dtype / device / 各 module の ``training`` 状態は ``deepcopy`` が
        そのまま保持する。置換される層の ``training`` は元の ``nn.Linear`` に揃える。
    """
    layer = _get_named_layer(model, layer_name, nn.Linear)

    tt_layer = build_tt_linear(layer, out_modes, in_modes, max_rank=max_rank)

    compressed_model = copy.deepcopy(model)
    set_named_module(compressed_model, layer_name, tt_layer)
    return compressed_model


def tt_linear_initialization_stages(
    model: nn.Module,
    layer_name: str,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    max_rank: int | None = None,
    canonical_center: int = 0,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """同じ dense 初期値から、TT 化の 3 段階を独立した model として返す。

    A:
        ``model`` の deepcopy。``layer_name`` は ``nn.Linear`` のまま。
    B:
        A を ``factorize_named_tt_linear`` した直後。TT-SVD と ``max_rank``
        による rank 上限は、この既存関数の中で一度に行われる。正準化はしない。
    C:
        B の deepcopy に、site ``canonical_center`` を中心とする正準 core を
        書き戻したもの。B の core は上書きしない。

    ``model`` は変更しない。A、B、C は互いに storage を共有しない。
    ``layer_name`` 以外の parameter と buffer、および TT 化した層の bias は、
    A からコピーした同じ値である。
    """
    if isinstance(canonical_center, bool) or not isinstance(canonical_center, int):
        raise TypeError(
            f"canonical_center は int である必要があります: {canonical_center!r}"
        )
    if canonical_center < 0:
        raise ValueError(
            f"canonical_center は 0 以上である必要があります: {canonical_center}"
        )

    from ..tt_mps.matrix_two_site import canonicalize_tt_matrix_cores

    from .tt_linear_two_site import assign_tt_linear_cores

    stage_a = copy.deepcopy(model)
    stage_b = factorize_named_tt_linear(
        stage_a,
        layer_name,
        out_modes,
        in_modes,
        max_rank=max_rank,
    )
    stage_c = copy.deepcopy(stage_b)
    layer_c = stage_c.get_submodule(layer_name)
    if not isinstance(layer_c, TTLinear):
        raise TypeError(f"{layer_name} は TTLinear である必要があります。")
    if canonical_center >= len(layer_c.cores):
        raise ValueError(
            f"canonical_center={canonical_center} は core 数 {len(layer_c.cores)} の範囲外です。"
        )
    canonical_cores = canonicalize_tt_matrix_cores(
        list(layer_c.cores),
        out_modes,
        in_modes,
        canonical_center,
    )
    assign_tt_linear_cores(layer_c, canonical_cores, requires_grad=False)
    return stage_a, stage_b, stage_c
