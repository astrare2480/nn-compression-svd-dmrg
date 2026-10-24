r"""学習対象 parameter の範囲を名前 prefix で管理する。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/13_*.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/14_*.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/04_pretrained_cnn_conv2_tt_cores_only_finetuning.ipynb

TT cores だけを fine-tune する実験では、``fc1.cores.`` や ``conv2.cores.`` に一致する
parameter だけを学習し、残りを凍結する。このとき次の3つがほぼ毎回必要になる。

1. prefix に一致する parameter だけ ``requires_grad=True`` にする。
2. optimizer に渡す parameter 集合を、検証用の名前つきで同じ結果から作る。
3. 学習前の state と比べ、凍結した parameter が本当に変わっていないか確かめる。

ここにはその3つだけを置く。どのモデルをどの層から作るか、どの prefix を使うかは
呼び出し側の責任。parameter の値は一切書き換えない（``requires_grad`` だけを変える）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def _normalize_prefixes(prefixes) -> tuple[str, ...]:
    """prefix の列を検証して tuple にする（重複は最初の出現だけ残す）。

    文字列そのものを渡すと1文字ずつの prefix になってしまうため拒否する。
    """
    if isinstance(prefixes, (str, bytes)) or not isinstance(prefixes, Sequence):
        raise TypeError(
            "prefixes は文字列の列（list / tuple）である必要があります: "
            f"{type(prefixes)!r}"
        )
    if len(prefixes) == 0:
        raise ValueError("prefixes は空にできません。")
    for prefix in prefixes:
        if not isinstance(prefix, str):
            raise TypeError(f"prefix は str である必要があります: {prefix!r} ({type(prefix)!r})")
        if prefix == "":
            raise ValueError("prefix に空文字列は使えません（全 parameter に一致してしまいます）。")
    return tuple(dict.fromkeys(prefixes))


def _name_matches(name: str, prefix: str) -> bool:
    """``name`` が module 階層の境界を守って ``prefix`` に一致するかを返す。

    - ``prefix`` が ``"."`` で終わる（``"fc1.cores."``）なら単純な前方一致。
    - そうでなければ、``name == prefix`` か ``name`` が ``prefix + "."`` で始まるときだけ一致。
      したがって ``"fc1"`` は ``fc1.weight`` に一致するが ``fc10.weight`` には一致しない。
    """
    if prefix.endswith("."):
        return name.startswith(prefix)
    return name == prefix or name.startswith(prefix + ".")


def set_trainable_parameter_prefixes(
    model: nn.Module,
    prefixes,
    *,
    freeze_others: bool = True,
) -> tuple[str, ...]:
    """名前が ``prefixes`` のどれかに一致する parameter だけを学習対象にする。

    model:
        対象のモデル。``requires_grad`` だけを in-place で変更する（値・構造・
        training 状態・buffer は変更しない）。
    prefixes:
        ``"fc1.cores."`` のような parameter 名 prefix の列（list / tuple、非空）。
        各要素は空でない ``str``。文字列1つを直接渡すことはできない。
        一致は module 階層の境界を守る: ``"."`` で終わらない prefix（``"fc1"``）は
        ``fc1.weight`` などには一致するが、``fc10.weight`` のような別 module には
        一致しない。``"."`` で終わる prefix は単純な前方一致。
    freeze_others:
        ``True``（既定）なら一致しなかった parameter をすべて凍結する。
        ``False`` なら一致しなかった parameter の ``requires_grad`` は変更しない。

    戻り値:
        一致した parameter 名を ``model.named_parameters()`` の登録順で並べた tuple。
        一致した parameter はすべて ``requires_grad=True`` になる。

    例外:
        model が ``nn.Module`` でない、prefixes が列でない・要素が ``str`` でない、
        ``freeze_others`` が ``bool`` でない場合は ``TypeError``。
        prefixes が空・空文字列を含む、または1件も一致しない場合は ``ValueError``。
        これらは **model を変更する前** に検出するので、失敗時に model は元のまま。

    Note:
        共有 parameter（同じ ``nn.Parameter`` を複数の名前で登録したもの）は
        ``named_parameters()`` と同様に最初の名前だけで扱う。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not isinstance(freeze_others, bool):
        raise TypeError(f"freeze_others は bool である必要があります: {type(freeze_others)!r}")
    normalized = _normalize_prefixes(prefixes)

    named = list(model.named_parameters())
    matched_names = tuple(
        name for name, _ in named if any(_name_matches(name, prefix) for prefix in normalized)
    )
    if not matched_names:
        available = ", ".join(name for name, _ in named) or "(parameter なし)"
        raise ValueError(
            f"prefixes={list(normalized)} に一致する parameter がありません。"
            f" 利用可能な名前: {available}"
        )

    matched = set(matched_names)
    for name, parameter in named:
        if name in matched:
            parameter.requires_grad_(True)
        elif freeze_others:
            parameter.requires_grad_(False)
    return matched_names


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """``requires_grad=True`` の parameter を ``(名前, parameter)`` の登録順 list で返す。

    optimizer に渡す集合と検証用の名前を同じ結果から作れる::

        trainable = trainable_named_parameters(model)
        optimizer = torch.optim.Adam([p for _, p in trainable])
        trainable_names = [n for n, _ in trainable]

    model・parameter の値や ``requires_grad`` は変更しない。戻り値の parameter は
    model が持つ実体そのもの（コピーではない）。共有 parameter は最初の名前で1回だけ現れる。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)


def _python_int(value: torch.Tensor) -> int:
    """0次元の整数 Tensor を Python int にする（uint64 の上位ビットも落とさない）。"""
    return int(value.detach().cpu().numpy().item())


def _real_difference(current64: torch.Tensor, before64: torch.Tensor) -> torch.Tensor:
    """実数の差。同じ位置の同符号 inf は変化量 0 とし、``inf - inf`` の NaN を作らない。"""
    same_infinity = (torch.isposinf(current64) & torch.isposinf(before64)) | (
        torch.isneginf(current64) & torch.isneginf(before64)
    )
    current64 = torch.where(same_infinity, torch.zeros_like(current64), current64)
    before64 = torch.where(same_infinity, torch.zeros_like(before64), before64)
    return current64 - before64


def _max_abs_change_real(current: torch.Tensor, before: torch.Tensor) -> float:
    """float parameter の最大絶対変化。NaN を含むときは ``nan``（拒否はしない）。"""
    current64 = current.detach().to(torch.float64)
    before64 = before.detach().to(torch.float64)
    if bool(torch.isnan(current64).any() or torch.isnan(before64).any()):
        return float("nan")
    return float(_real_difference(current64, before64).abs().amax())


def _max_abs_change_complex(current: torch.Tensor, before: torch.Tensor) -> float:
    """複素差の絶対値の最大。実部へキャストして虚部を落とさない。"""
    current128 = current.detach().to(torch.complex128)
    before128 = before.detach().to(torch.complex128)
    has_nan = torch.isnan(current128.real) | torch.isnan(current128.imag)
    has_nan = has_nan | torch.isnan(before128.real) | torch.isnan(before128.imag)
    if bool(has_nan.any()):
        return float("nan")
    real_change = _real_difference(current128.real, before128.real)
    imag_change = _real_difference(current128.imag, before128.imag)
    return float(torch.hypot(real_change, imag_change).amax())


def _max_abs_int64(current: torch.Tensor, before: torch.Tensor) -> int:
    """int64 の要素ごとの絶対差の最大を、減算 overflow なしの Python int で返す。

    同符号の差は int64 に収まる。異符号の ``|a| + |b|`` は最大 ``2**64 - 1`` なので
    uint64 で足す。``int64`` の最小値だけは絶対値が int64 に収まらないので分ける。
    """
    same_sign = ((current >= 0) & (before >= 0)) | ((current < 0) & (before < 0))
    current_same = torch.where(same_sign, current, torch.zeros_like(current))
    before_same = torch.where(same_sign, before, torch.zeros_like(before))
    same_span = torch.maximum(current_same, before_same) - torch.minimum(current_same, before_same)
    max_same = _python_int(same_span.amax())

    opposite = ~same_sign
    if not bool(opposite.any()):
        return max_same

    positive = torch.where(opposite, torch.where(current >= 0, current, before), torch.zeros_like(current))
    negative = torch.where(opposite, torch.where(current < 0, current, before), torch.zeros_like(current))
    is_lowest = negative == torch.iinfo(torch.int64).min
    # 異符号の |a|+|b| は最大 2**64-1。この環境の uint64 加算は使えないので、
    # 上下 32bit に分けて int64 のまま足し、最大値だけ Python int に戻す。
    low_mask = (1 << 32) - 1
    positive_low = positive & low_mask
    positive_high = positive >> 32
    negative_for_abs = torch.where(is_lowest, torch.zeros_like(negative), negative)
    absolute_negative = -negative_for_abs
    negative_low = absolute_negative & low_mask
    negative_high = absolute_negative >> 32
    negative_low = torch.where(is_lowest, torch.zeros_like(negative_low), negative_low)
    negative_high = torch.where(is_lowest, torch.full_like(negative_high, 1 << 31), negative_high)
    sum_low = positive_low + negative_low
    carry = sum_low >> 32
    sum_low = sum_low & low_mask
    sum_high = positive_high + negative_high + carry
    max_high = int(sum_high.amax().item())
    low_at_max_high = torch.where(sum_high == max_high, sum_low, torch.zeros_like(sum_low))
    max_low = int(low_at_max_high.amax().item())
    return max(max_same, max_high * (1 << 32) + max_low)


def _max_abs_change_integer(current: torch.Tensor, before: torch.Tensor) -> float:
    """整数 parameter の最大絶対変化。float へ変換してから引かない。"""
    current = current.detach()
    before = before.detach()
    if current.dtype == torch.uint64:
        span = torch.maximum(current, before) - torch.minimum(current, before)
        return float(_python_int(span.amax()))
    if current.dtype != torch.int64:
        current = current.to(torch.int64)
        before = before.to(torch.int64)
    return float(_max_abs_int64(current, before))


def _max_abs_change(current: torch.Tensor, before: torch.Tensor) -> float:
    """要素ごとの変化の最大絶対値。完全一致のときは差分を計算せず ``0.0``。"""
    if torch.equal(current, before):
        return 0.0
    if current.is_floating_point():
        return _max_abs_change_real(current, before)
    if current.is_complex():
        return _max_abs_change_complex(current, before)
    if current.dtype in _INTEGER_DTYPES:
        return _max_abs_change_integer(current, before)
    difference = current.detach().to(torch.float64) - before.detach().to(torch.float64)
    return float(difference.abs().amax())


def parameter_change_summary(
    before_state: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> list[dict]:
    """学習前に保存した state と現在の model を、parameter ごとに比較する。

    before_state:
        ``copy.deepcopy(model.state_dict())`` で作った ``Mapping[str, Tensor]``
        （parameter だけを集めた dict でもよい）。``model`` のすべての parameter を
        含み、shape・dtype・device が現在の parameter と一致する必要がある。
        parameter 以外のキーは、``model.state_dict()`` に存在する buffer 名だけ許す。
    model:
        現在のモデル。変更しない。

    戻り値:
        ``model.named_parameters()`` の登録順に、parameter ごとの dict の list。
        ``pandas.DataFrame(戻り値)`` へそのまま渡せる（pandas には依存しない）。
        各 dict のキー:

        - ``name``: parameter 名。
        - ``requires_grad``: 現在の ``requires_grad``（``bool``）。
        - ``numel``: 要素数（``int``）。
        - ``exactly_equal``: 値が完全一致か（``bool``）。
        - ``max_abs_change``: 要素ごとの変化の最大絶対値（``float``）。

        凍結 parameter が変わっていないかは
        ``all(r["exactly_equal"] for r in rows if not r["requires_grad"])`` で判定できる。
        ここで ``requires_grad`` は現在の値なので、学習前後で凍結設定を変えない前提で使う。

    例外:
        Mapping でない・値が Tensor でない・dtype 不一致は ``TypeError``。
        parameter の不足、parameter でも buffer でもない余分なキー、shape / device の不一致は
        ``ValueError``（どの名前かをメッセージに含める）。

    Note:
        ``exactly_equal`` が True なら ``max_abs_change`` は必ず ``0.0`` で、そのときは
        差分を計算しない。float32 / float64 の有限値は float64 で差を取るので、
        float32 の 1 ulp も隠れない。

        complex64 / complex128 は実部へキャストせず、複素差の絶対値を取る。
        例えば ``1+2j`` から ``1+5j`` は ``3.0``。

        整数は float64 へ変換してから引かない（大きな int64 の 1 差が消えるため）。
        int64 の最小値・最大値付近でも減算は overflow させない。
        ``2**62`` から ``2**62 + 1`` は ``1.0``。差が Python float の桁を超えても
        有限なら、その整数差を Python float にしたもの（例えば最小値と最大値の差）を返す。

        同じ位置の ``+inf`` 同士、``-inf`` 同士は変化量 0。finite と inf の入れ替わり、
        ``+inf`` から ``-inf`` は ``inf``。NaN を含む入力は拒否しない。NaN が一つでも
        あると一致せず、``max_abs_change`` は ``nan``。NaN を含まない入力から
        ``max_abs_change=nan`` は作らない。要素数 0 の parameter は
        ``exactly_equal=True``・``max_abs_change=0.0``。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model は nn.Module である必要があります: {type(model)!r}")
    if not isinstance(before_state, Mapping):
        raise TypeError(f"before_state は Mapping である必要があります: {type(before_state)!r}")

    named = list(model.named_parameters())
    parameter_names = {name for name, _ in named}
    allowed_extra = set(model.state_dict().keys())

    missing = [name for name, _ in named if name not in before_state]
    if missing:
        raise ValueError(f"before_state に parameter が不足しています: {missing}")
    extra = [key for key in before_state if key not in parameter_names and key not in allowed_extra]
    if extra:
        raise ValueError(f"before_state に model に存在しないキーがあります: {extra}")

    rows: list[dict] = []
    with torch.no_grad():
        for name, parameter in named:
            before = before_state[name]
            if not torch.is_tensor(before):
                raise TypeError(f"before_state[{name!r}] は torch.Tensor である必要があります: {type(before)!r}")
            if before.shape != parameter.shape:
                raise ValueError(
                    f"before_state[{name!r}] の shape が一致しません: "
                    f"state={tuple(before.shape)}, model={tuple(parameter.shape)}"
                )
            if before.dtype != parameter.dtype:
                raise TypeError(
                    f"before_state[{name!r}].dtype={before.dtype} は "
                    f"model の dtype={parameter.dtype} と一致する必要があります。"
                )
            if before.device != parameter.device:
                raise ValueError(
                    f"before_state[{name!r}].device={before.device} は "
                    f"model の device={parameter.device} と一致する必要があります。"
                )

            current = parameter.detach()
            if current.numel() == 0:
                exactly_equal, max_abs_change = True, 0.0
            else:
                exactly_equal = bool(torch.equal(current, before))
                # 完全一致（同じ inf を含む）なら差分を計算しない。inf - inf の NaN を避ける。
                max_abs_change = 0.0 if exactly_equal else _max_abs_change(current, before)
            rows.append(
                {
                    "name": name,
                    "requires_grad": bool(parameter.requires_grad),
                    "numel": int(parameter.numel()),
                    "exactly_equal": exactly_equal,
                    "max_abs_change": max_abs_change,
                }
            )
    return rows
