"""logits の誤差上界から argmax（予測クラス）の不変性を保証する certificate。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/04_classification_margin_argmax_stability.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/06_minimum_margin_batchwide_stability.ipynb

sample ``n`` の dense logits を ``l_n``、予測クラスを ``y*_n = argmax_k l_{n,k}`` とする。
classification margin を

    gamma_n = l_{n,y*_n} - max_{k != y*_n} l_{n,k}   (= top-1 logit - top-2 logit)

と定め、圧縮後の logits の誤差ベクトル ``delta l_n`` が
``||delta l_n||_inf <= U_n`` を満たすなら、

    U_n < gamma_n / 2

のとき圧縮後の argmax は ``y*_n`` のまま変わらない（十分条件）。
``||.||_inf <= ||.||_2 <= ||.||_F`` なので、``U_n`` には ℓ2 / Frobenius 型の上界
（``error_bounds`` の関数など）も使える。

この条件は十分条件であり、満たさなくても argmax が変わるとは限らない
（「保証できない」だけ）。判定は厳密な不等号で、``U_n = gamma_n / 2`` は不成立。
top-1 logit が tie の sample は ``gamma_n = 0`` となり、``U_n >= 0`` のため
certificate は成立しない。

margin の計算・最小 margin・certificate 判定を小さな関数に分けてある。表示、
DataFrame 化、実測 argmax との照合は呼び出し側に置く。

値の検証（NaN / inf・誤差上界の非負性）のために ``torch.isfinite(...).all()`` を
bool 評価する。これは CUDA では同期を伴う。検証以外の計算には ``.item()`` /
``.detach()`` / ``.cpu()`` を使わない。
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class MinimumMargin:
    """batch 内の最小 classification margin と、その bottleneck sample。

    value:
        最小 margin（0 階 Tensor、margins と同じ dtype / device）。
    sample_index:
        最小 margin を持つ sample の index（0 階 ``torch.int64``）。同値が複数
        あれば最初の index。
    """

    value: torch.Tensor
    sample_index: torch.Tensor


@dataclass(frozen=True)
class BatchStabilityCertificate:
    """batch 全体の prediction stability certificate の計算結果。

    margins:
        sample ごとの classification margin。shape ``(batch,)``。
    minimum_margin:
        最小 margin（0 階 Tensor）。
    bottleneck_index:
        最小 margin の sample index（0 階 ``torch.int64``）。
    sample_certified:
        sample ごとに ``U_n < gamma_n / 2`` が成立するか。shape ``(batch,)``、bool。
    all_certified:
        全 sample で成立するか（0 階 bool Tensor）。
    """

    margins: torch.Tensor
    minimum_margin: torch.Tensor
    bottleneck_index: torch.Tensor
    sample_certified: torch.Tensor
    all_certified: torch.Tensor


def _check_float_tensor(value, *, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(value)!r}")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name}.dtype={value.dtype} は未対応です"
            "（torch.float32 / torch.float64 のみ）。"
        )


def _check_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} に NaN / inf が含まれています。")


def _finite_lower_logit_margin(top: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """``top - second`` の有限な下界を返す。

    ``top >= second`` かつ両方有限であることを仮定する。戻り値は真の差以下で、
    dtype の有限範囲に収まる。

    - 差が正確、または下向きに丸まっているときは、通常の ``top - second``。
      勾配もそれを通る。
    - 上向きに丸まっているときは、1 ulp 小さい値（真値を超えない側）。
      この行は定数補正を選ぶので、勾配は 0。
    - 差が dtype の最大有限値を超えて ``inf`` になるときは、その最大有限値。
      オーバーフローは真の差が最大有限値より大きいときに起きるので、この
      cap も下界である。定数 ``cap`` を選ぶので、この行の勾配も 0。

    tie（差が正確に 0）はそのまま 0。
    """
    diff = top - second
    finite = torch.isfinite(diff)
    cap = torch.tensor(torch.finfo(top.dtype).max, dtype=top.dtype, device=top.device)

    # inf 行を 0 に置き換えてから誤差を見ると、TwoSum が NaN を出さない。
    safe_top = torch.where(finite, top, torch.zeros_like(top))
    safe_second = torch.where(finite, second, torch.zeros_like(second))
    safe_diff = safe_top - safe_second
    negated = -safe_second
    a_hat = safe_diff - negated
    b_hat = safe_diff - a_hat
    # a + (-b) = safe_diff + error。error < 0 なら safe_diff は真値より大きい。
    error = (safe_top - a_hat) + (negated - b_hat)
    # nextafter に逆伝播は無い。補正値はグラフから外す。
    # 勾配が 0 になる行は次の2つ。
    # - needs_step: 上向き丸めを 1 ulp 下げた行（detach した定数）
    # - finite でない行: overflow した差を定数 cap に置き換えた行
    # 正確な差と下向き丸めの行は diff = top - second の勾配を保つ。
    stepped = torch.nextafter(safe_diff.detach(), torch.zeros_like(safe_diff))
    needs_step = finite & (error < 0)
    capped = torch.where(finite, diff, cap)
    return torch.where(needs_step, stepped, capped)


def classification_margins(logits: torch.Tensor) -> torch.Tensor:
    """batch logits から sample ごとの classification margin を返す。

    ``gamma_n = (top-1 logit) - (top-2 logit)``。top-1 が tie なら 0。
    常に ``gamma_n >= 0``。

    logits:
        shape ``(batch, num_classes)``。``batch >= 1``、``num_classes >= 2``、
        dtype は float32 / float64、全要素が有限。変更しない。

    戻り値:
        shape ``(batch,)``。dtype / device は ``logits`` を維持する。
        ``logits`` が計算グラフに載っていれば、その勾配経路を保つ。

        有限な logits でも ``top - second`` が dtype の最大値を超えると、同じ
        dtype の減算は ``inf`` になる。そのときは最大有限値へ cap し、上向き
        丸めで真値を超えるときも 1 ulp 下げる。戻り値は真の margin 以下の有限値
        なので、certificate を危険側（成立しやすくする側）へずらさない。
        正確な差と下向き丸めの行は ``top - second`` の勾配を保つ。次の行は
        定数を選ぶため勾配が 0 になる。
        - 上向き丸めを 1 ulp 補正した行
        - overflow した差を最大有限値へ cap した行
    """
    _check_float_tensor(logits, name="logits")
    if logits.ndim != 2:
        raise ValueError(
            f"logits は2階（batch, num_classes）である必要があります: "
            f"ndim={logits.ndim}, shape={tuple(logits.shape)}"
        )
    batch, num_classes = logits.shape
    if batch < 1:
        raise ValueError("logits の batch サイズは1以上である必要があります。")
    if num_classes < 2:
        raise ValueError(
            f"class 数は2以上である必要があります: num_classes={num_classes}"
        )
    _check_finite(logits, name="logits")

    top2 = torch.topk(logits, k=2, dim=1).values
    return _finite_lower_logit_margin(top2[:, 0], top2[:, 1])


def minimum_classification_margin(margins: torch.Tensor) -> MinimumMargin:
    """sample ごとの margin から、最小 margin と bottleneck sample index を返す。

    margins:
        shape ``(batch,)``、``batch >= 1``、float32 / float64、全要素が有限。
        通常は ``classification_margins`` の戻り値。

    戻り値:
        ``MinimumMargin``。``value`` は ``margins`` の dtype / device を維持する。
    """
    _check_float_tensor(margins, name="margins")
    if margins.ndim != 1:
        raise ValueError(
            f"margins は1階（batch,）である必要があります: "
            f"ndim={margins.ndim}, shape={tuple(margins.shape)}"
        )
    if margins.shape[0] < 1:
        raise ValueError("margins は空にできません。")
    _check_finite(margins, name="margins")

    index = torch.argmin(margins)
    return MinimumMargin(value=margins[index], sample_index=index)


def sample_stability_certificates(
    margins: torch.Tensor,
    error_bound: torch.Tensor | float,
) -> torch.Tensor:
    """sample ごとに ``U_n < gamma_n / 2`` を判定する。

    margins:
        shape ``(batch,)`` の margin ``gamma_n``（``classification_margins`` の戻り値など）。
    error_bound:
        誤差上界 ``U``（``||delta l_n||_inf`` の上界）。次のいずれか。

        - 0 階 Tensor: 全 sample に共通の scalar 上界（batch 全体の上界など）。
          dtype は float32 / float64。``margins`` より広くてもよい。比較は
          両者を promote した dtype で行い、上界を狭い dtype へ切り捨てない。
        - shape ``(batch,)`` の Tensor: sample ごとの上界 ``U_n``。
          dtype と device は ``margins`` と一致している必要がある。
        - Python の実数: scalar 上界。float64 として扱い、``margins`` が
          float32 でもそこへ下向き丸めしない。非常に小さい正数を 0 にして
          certificate を誤って成立させないため。

        Tensor の device は ``margins`` と一致している必要がある。
        全要素が有限かつ非負（誤差上界は負にならない）。

    戻り値:
        shape ``(batch,)`` の bool Tensor（``margins`` と同じ device）。
        厳密な不等号で判定する: ``U_n = gamma_n / 2`` は不成立。判定は
        ``2 * U_n < gamma_n``（``U_n < gamma_n / 2`` と同値で、``/ 2`` の
        丸めが入らない形）で行う。margin が 0（tie）の sample は常に不成立。
    """
    _check_float_tensor(margins, name="margins")
    if margins.ndim != 1:
        raise ValueError(
            f"margins は1階（batch,）である必要があります: "
            f"ndim={margins.ndim}, shape={tuple(margins.shape)}"
        )
    _check_finite(margins, name="margins")

    if torch.is_tensor(error_bound):
        _check_float_tensor(error_bound, name="error_bound")
        if error_bound.device != margins.device:
            raise ValueError(
                f"error_bound.device={error_bound.device} は margins.device={margins.device} と"
                "一致する必要があります。"
            )
        if error_bound.ndim == 1:
            # sample ごとの上界は並べ替えできないので、dtype も一致させる。
            if error_bound.dtype != margins.dtype:
                raise TypeError(
                    f"error_bound.dtype={error_bound.dtype} は margins.dtype={margins.dtype} と"
                    "一致する必要があります。"
                )
            if error_bound.shape != margins.shape:
                raise ValueError(
                    f"error_bound.shape={tuple(error_bound.shape)} は "
                    f"margins.shape={tuple(margins.shape)} と一致する必要があります。"
                )
        elif error_bound.ndim != 0:
            raise ValueError(
                "error_bound は0階（scalar）または1階（sample ごと）である必要があります: "
                f"shape={tuple(error_bound.shape)}"
            )
        bound = error_bound
    elif isinstance(error_bound, numbers.Real) and not isinstance(error_bound, bool):
        # Python float は float64。margin が float32 でも最近接で 0 に落とさない。
        bound = torch.tensor(float(error_bound), dtype=torch.float64, device=margins.device)
    else:
        raise TypeError(
            "error_bound は torch.Tensor または Python の実数（bool 不可）である必要があります: "
            f"{type(error_bound)!r}"
        )

    _check_finite(bound, name="error_bound")
    if bool((bound < 0).any()):
        raise ValueError("error_bound は非負である必要があります。")

    # 上界を狭い dtype へ切ると小さくなることがあり、certificate の false positive になる。
    # promote は float32→float64 の拡張なので、上界も margin も値を小さくしない。
    compare_dtype = torch.promote_types(bound.dtype, margins.dtype)
    promoted_bound = bound.to(dtype=compare_dtype)
    promoted_margins = margins.to(dtype=compare_dtype)
    return (promoted_bound * 2) < promoted_margins


def batch_stability_certificate(
    logits: torch.Tensor,
    error_bound: torch.Tensor | float,
) -> BatchStabilityCertificate:
    """batch logits と誤差上界から、batch 内全 sample の certificate を計算する。

    ``classification_margins`` → ``minimum_classification_margin`` →
    ``sample_stability_certificates`` を順に呼んで結果をまとめるだけの合成関数。

    logits:
        dense（圧縮前）model の logits。shape ``(batch, num_classes)``。
    error_bound:
        scalar または sample ごとの誤差上界。
        ``sample_stability_certificates`` と同じ契約。

    戻り値:
        ``BatchStabilityCertificate``。
    """
    margins = classification_margins(logits)
    minimum = minimum_classification_margin(margins)
    sample_certified = sample_stability_certificates(margins, error_bound)
    return BatchStabilityCertificate(
        margins=margins,
        minimum_margin=minimum.value,
        bottleneck_index=minimum.sample_index,
        sample_certified=sample_certified,
        all_certified=torch.all(sample_certified),
    )
