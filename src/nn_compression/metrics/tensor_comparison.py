r"""任意の2つの同 shape Tensor を比べる近似誤差の指標。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/08_pretrained_mlp_fc1_tt_rank_sweep.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb

重み・特徴マップ・logits など、分解手法やモデルに依存しない形で
「reference と approximation がどれだけ違うか」を Python の数値で返す。
``tensor_approximation.relative_frobenius_error`` は相対誤差だけを Tensor で返すが、
ここでは絶対誤差・RMSE・最大絶対誤差もまとめ、DataFrame の1行にそのまま入れられる
dataclass にする。評価用であり、autograd は契約に含めない。

数値の扱い:
    差のノルムは二乗和を使うため、float32 の ``1e30`` 程度の値でも二乗が溢れて
    ``inf`` になる。そこで次の手順で計算する。

    1. 入力を float64 へ昇格する（float32 → float64 は厳密）。
    2. ``2`` のべき乗の ``scale``（最大絶対値以下で最大のもの）で割る。
       2 のべき乗での除算は丸め誤差を作らないので、通常の範囲では結果は
       スケールしない計算とビットまで一致する。
    3. ``[-2, 2]`` に収まった値で norm を取り、最後に ``scale`` を掛けて戻す。

    戻した値が float64 の最大有限値を超えるときだけ ``inf`` になる（NaN にはしない）。
    これは数学的な値が表現範囲を超えていることを意味する。

    相対誤差は比なので、reference と approximation の桁が大きく離れると、
    共通の scale で小さい側が 0 に潰れる。そのときは分子と分母の norm を
    別々の 2 のべき乗で評価する。元の reference が非零なら零 Tensor とは扱わず、
    比が float64 の最大有限値を超えるときだけ ``inf`` にする。厳密な零 Tensor
    のときだけ ``ValueError``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .tensor_approximation import relative_frobenius_error

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class TensorApproximationMetrics:
    """``reference`` に対する ``approximation`` の誤差（すべて Python の ``float``）。

    - ``absolute_frobenius_error``: ``||reference - approximation||_F``。
    - ``relative_frobenius_error``: ``||reference - approximation||_F / ||reference||_F``。
    - ``rmse``: ``sqrt(mean((reference - approximation)**2))``（全要素平均）。
    - ``max_abs_error``: ``max |reference - approximation|``。
    """

    absolute_frobenius_error: float
    relative_frobenius_error: float
    rmse: float
    max_abs_error: float


def _validate_comparable(
    reference: torch.Tensor,
    other: torch.Tensor,
    *,
    names: tuple[str, str],
    ndim: int | None = None,
) -> None:
    """同 shape・同 dtype・同 device の有限な float Tensor 2つであることを検証する。

    ``ndim`` を与えると、その階数であることも確かめる。空 Tensor は拒否する。
    型の誤りと dtype の不一致は ``TypeError``、それ以外は ``ValueError``。
    """
    ref_name, other_name = names
    for name, value in ((ref_name, reference), (other_name, other)):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} は torch.Tensor である必要があります: {type(value)!r}")
        if value.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{name}.dtype={value.dtype} は未対応です（torch.float32 / torch.float64 のみ）。"
            )
    if other.dtype != reference.dtype:
        raise TypeError(
            f"{other_name}.dtype={other.dtype} は {ref_name}.dtype={reference.dtype} と"
            "一致する必要があります。"
        )
    if other.device != reference.device:
        raise ValueError(
            f"{other_name}.device={other.device} は {ref_name}.device={reference.device} と"
            "一致する必要があります。"
        )
    if ndim is not None:
        for name, value in ((ref_name, reference), (other_name, other)):
            if value.ndim != ndim:
                raise ValueError(
                    f"{name} は{ndim}階である必要があります: "
                    f"ndim={value.ndim}, shape={tuple(value.shape)}"
                )
    if reference.shape != other.shape:
        raise ValueError(
            f"shape が一致しません: {ref_name}={tuple(reference.shape)}, "
            f"{other_name}={tuple(other.shape)}"
        )
    if reference.numel() == 0:
        raise ValueError(f"{ref_name} と {other_name} は空にできません: shape={tuple(reference.shape)}")
    for name, value in ((ref_name, reference), (other_name, other)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} に NaN / inf が含まれています。")


def _scaled_float64_pair(
    reference: torch.Tensor,
    other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """float64 へ昇格し、2 のべき乗 ``scale`` で割った2つの Tensor と ``scale`` を返す。

    ``scale`` は最大絶対値以下で最大の 2 のべき乗（両方が零なら ``1.0``）。
    除算は厳密なので、戻り値の組から ``scale`` 倍すれば元の値になる。
    入力は有限であることを前提にする（``_validate_comparable`` の後に呼ぶ）。
    """
    reference64 = reference.detach().to(torch.float64)
    other64 = other.detach().to(torch.float64)
    largest = max(float(reference64.abs().amax()), float(other64.abs().amax()))
    if largest == 0.0:
        return reference64, other64, 1.0
    _, exponent = math.frexp(largest)
    scale = math.ldexp(1.0, exponent - 1)  # 最大絶対値 / scale は [1, 2)
    return reference64 / scale, other64 / scale, scale


def _norm_mantissa_exponent(values: torch.Tensor) -> tuple[float, int]:
    """``||values|| = mantissa * 2**exponent`` となる組を返す。

    ``values`` は有限な float64。``mantissa == 0`` なのは全要素が厳密に 0 のときだけ。
    最大絶対値の指数だけずらしてから norm を取るので、subnormal の二乗和が
    0 に潰れて「零 Tensor」と誤認されることはない。
    """
    largest = float(values.abs().amax())
    if largest == 0.0:
        return 0.0, 0
    _, exponent = math.frexp(largest)
    scale_exponent = exponent - 1
    shift = torch.tensor(-scale_exponent, dtype=torch.int32, device=values.device)
    mantissa = float(torch.linalg.vector_norm(torch.ldexp(values, shift)))
    return mantissa, scale_exponent


def _relative_error_with_independent_scales(
    reference64: torch.Tensor,
    reference_scaled: torch.Tensor,
    approximation_scaled: torch.Tensor,
    scale: float,
) -> float:
    """分子と分母を別の 2 のべき乗で評価した相対誤差を返す。

    共通 scale では reference 側の norm が 0 になったときだけ使う。
    元の reference が非零なら、ここで零 reference の ``ValueError`` にはしない。
    比が float64 の最大有限値を超えるときは ``inf``（``NaN`` にはしない）。
    """
    denominator_mantissa, denominator_exponent = _norm_mantissa_exponent(reference64)
    if denominator_mantissa == 0.0:
        raise ValueError("reference が零 Tensor なので相対誤差を定義できません。")
    numerator_mantissa = float(torch.linalg.vector_norm(reference_scaled - approximation_scaled))
    _, scale_exponent = math.frexp(scale)
    try:
        relative = math.ldexp(
            numerator_mantissa / denominator_mantissa,
            (scale_exponent - 1) - denominator_exponent,
        )
    except OverflowError:
        return math.inf
    # 有限な入力から NaN は返さない。桁が溢れた比は inf。
    return math.inf if math.isnan(relative) else relative


def _error_summary(
    reference_scaled: torch.Tensor,
    other_scaled: torch.Tensor,
    scale: float,
) -> tuple[float, float, float]:
    """``(絶対 Frobenius 誤差, RMSE, 最大絶対誤差)`` を元の単位で返す。

    スケール後の値で計算してから ``scale`` を掛ける。元の単位で表せない
    ほど大きいときは ``inf``。
    """
    difference = reference_scaled - other_scaled
    absolute = float(torch.linalg.vector_norm(difference))
    rmse = float(torch.linalg.vector_norm(difference) / math.sqrt(difference.numel()))
    maximum = float(difference.abs().amax())
    return scale * absolute, scale * rmse, scale * maximum


def tensor_approximation_metrics(
    reference: torch.Tensor,
    approximation: torch.Tensor,
) -> TensorApproximationMetrics:
    """``reference`` と ``approximation`` の誤差をまとめて返す。

    reference:
        基準となる Tensor。任意の階数。変更しない。
    approximation:
        比較対象。``reference`` と shape・dtype・device が一致している必要がある。
        変更しない。

    戻り値:
        ``TensorApproximationMetrics``（4つの Python ``float``）。
        DataFrame へは ``dataclasses.asdict`` で渡せる。

    相対誤差の契約:
        ``reference`` が厳密な零 Tensor のときだけ、相対誤差は定義できないので
        ``relative_frobenius_error`` と同じく ``ValueError`` にする（``inf`` や
        ``NaN`` を返さない）。スケール差で内部表現が 0 になっただけの非零
        reference は零 Tensor と扱わない。その比が float64 の最大有限値を
        超えるときは ``inf``。絶対誤差だけが欲しいときは、零でない基準を使うか、
        呼び出し側で別に差の norm を取る。

    例外:
        Tensor でない・float32 / float64 以外・dtype 不一致は ``TypeError``。
        shape / device の不一致、空 Tensor、NaN / inf（どの引数かを示す）、零 reference は
        ``ValueError``。

    Note:
        計算は float64 で行い、2 のべき乗でスケールするため、``1e30`` のような
        大きな有限値でも不要な overflow を起こさない（通常の範囲ではスケールしない
        計算と同じ値）。数学的な誤差が float64 の範囲を超えるときだけ ``inf``。
        評価用の関数で、autograd のグラフは作らず、勾配は流れない。
    """
    _validate_comparable(reference, approximation, names=("reference", "approximation"))
    if not bool((reference != 0).any()):
        raise ValueError("reference が零 Tensor なので相対誤差を定義できません。")

    reference_scaled, approximation_scaled, scale = _scaled_float64_pair(reference, approximation)
    absolute, rmse, maximum = _error_summary(reference_scaled, approximation_scaled, scale)
    # 通常の桁では、共通 scale のあとで既存の相対誤差を使う（結果を変えない）。
    # reference だけが 0 に潰れたときは、分子と分母を別々に評価する。
    if float(torch.linalg.vector_norm(reference_scaled)) > 0.0:
        relative = float(relative_frobenius_error(reference_scaled, approximation_scaled))
    else:
        relative = _relative_error_with_independent_scales(
            reference.detach().to(torch.float64),
            reference_scaled,
            approximation_scaled,
            scale,
        )

    return TensorApproximationMetrics(
        absolute_frobenius_error=absolute,
        relative_frobenius_error=relative,
        rmse=rmse,
        max_abs_error=maximum,
    )
