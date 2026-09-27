"""単一bondのSVD打ち切りとTT-rounding。

参照元:
    notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb
    notebooks/30_tt_mps/00_fundamentals/09_truncated_svd_and_bond_rank_truncation.ipynb
    notebooks/30_tt_mps/00_fundamentals/10_eckart_young_mirsky_and_single_bond_optimality.ipynb
    notebooks/30_tt_mps/00_fundamentals/12_tt_rounding_right_canonicalization_and_truncated_svd.ipynb

Notebook 12 の ``choose_rank`` / ``right_canonicalize`` /
``left_to_right_truncate`` / ``tt_round`` は、演習用に ``assert`` で
public validationを行い、``local_erorr`` という変数名のtypoがあった。
ここではアルゴリズム自体（末尾から特異値を積み上げて予算内で最大
truncationするロジック、right-canonicalization → left-to-right truncated
SVD sweepという2段構成）は変更せず、validationを ``TypeError`` /
``ValueError`` に置き換え、typoを修正した production 実装にする。

``right_canonicalize`` は、center=0 の ``tt_canonicalize`` と同じ操作
（対象が全 core になるだけ）なので、重複実装せず ``tt_canonicalize`` を
再利用する。
"""

from __future__ import annotations

import math

import torch

from .svd import coerce_integer_scalar
from .tt import tt_physical_shape, tt_ranks
from .tt_canonical import tt_canonicalize
from .tt_validation import validate_tt_cores, validate_tt_dtype


def _coerce_eps_rel(eps_rel: object) -> float:
    """``eps_rel`` を有限・非負の ``float`` へ変換する（``tt_round`` 専用）。

    bool、Tensor scalar 以外の非数値、NaN、inf、負値を拒否する。
    """
    if isinstance(eps_rel, bool):
        raise TypeError(f"eps_rel は bool 以外の実数である必要があります: {eps_rel!r}")

    if torch.is_tensor(eps_rel):
        if eps_rel.dtype == torch.bool:
            raise TypeError(f"eps_rel は bool Tensor を受け付けません: {eps_rel!r}")
        if eps_rel.numel() != 1:
            raise TypeError(f"eps_rel はスカラーである必要があります: shape={tuple(eps_rel.shape)}")
        value = float(eps_rel.item())
    elif isinstance(eps_rel, (int, float)):
        value = float(eps_rel)
    else:
        raise TypeError(f"eps_rel は実数である必要があります: {eps_rel!r}")

    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"eps_rel は有限である必要があります: {value!r}")
    if value < 0:
        raise ValueError(f"eps_rel は0以上である必要があります: {value!r}")
    return value


def _is_exact_zero(x: torch.Tensor) -> bool:
    """``x`` の全要素が厳密に0かどうか（bitwise一致、丸め誤差の許容なし）。

    x:
        任意shapeの ``Tensor``。変更しない。

    戻り値:
        Python ``bool``。ノルムを計算してから0と比較するのではなく、
        要素同士の比較 ``x == 0`` を使う。``NaN == 0`` は常に ``False`` な
        ので、``NaN`` を含む場合はここで ``False`` になる（``NaN`` を
        「零テンソル」として誤って握り潰さない）。

    Note:
        autogradには関与しない（``bool`` を返す判定関数であり、勾配を
        流す対象ではない）。dtype/deviceは問わない（入力のdtype上での
        比較演算のみ使用する）。
    """
    return bool(torch.all(x == 0).item())


def _scale_safe_norm(x: torch.Tensor) -> torch.Tensor:
    r"""``x`` 全要素の2-normを、素朴な二乗和ではなくscale-safeに計算する。

    ``sqrt(sum(x**2))`` は、``x`` の最大絶対値がdtypeの「二乗しても
    underflowしない」範囲を下回ると、二乗した時点で0になり、真の非零
    ノルムを0と誤判定する（例: float32で ``|x|~1e-30`` は正常に表現できる
    が、``x**2~1e-60`` はfloat32のsubnormal最小値 ``~1.4e-45`` を下回り、
    厳密に0になる）。

    ここでは ``scale = max(|x|)`` で正規化してから2-normを取るので、
    正規化後の要素は ``O(1)`` のオーダーに収まり、二乗のunderflow・
    overflowが起きない（``scale`` 自身がoverflow・underflowしていない
    限り）。

    $$
    \|x\|_2 = s \cdot \left\|\frac{x}{s}\right\|_2,\quad s=\max_i|x_i|
    $$

    x:
        任意shapeの実数値 ``Tensor``。変更しない。

    戻り値:
        shape ``()`` のスカラー ``Tensor``（``x.dtype`` / ``x.device`` を
        維持する）。

    Note:
        - **戻り値の型**: ``Tensor``（``.item()`` は呼び出し側の責任）。
        - **autograd**: ``x.requires_grad=True`` の場合、``amax`` /
          除算 / ``vector_norm`` / 乗算はすべて微分可能なので、通常の
          backwardが通る（``amax`` の勾配はPyTorchの標準的な
          subgradientで、最大要素にのみ流れる）。
        - **dtype/device**: 入力のものをそのまま維持する。
        - **zero**: ``x`` が厳密に全要素0の場合、``scale=0`` になるので、
          0除算（``0/0`` のNaN）を避けて厳密な0を返す。
        - **NaN**: ``x`` にNaNが含まれる場合、``scale=max(|x|)`` もNaNに
          なり（PyTorchの ``amax`` はNaNを伝播する）、戻り値もNaNになる。
          NaNを0や有限値へ黙って変換しない。
        - **inf**: ``x`` に ``inf`` が含まれる場合、``scale=inf`` になり、
          有限要素は ``x/scale=0``、無限要素は ``inf/inf=NaN`` になるため、
          戻り値はNaNになりやすい（``inf`` を含む入力のノルムを厳密な
          ``inf`` として保証するものではない）。
        - ``scale`` 自身がoverflow・underflowするほど極端な値
          （``|x|`` が dtypeの表現限界に近い、例えば float64で
          ``1e300`` を超える、または ``1e-300`` 未満など）では、
          ``scale`` の計算自体が丸め誤差の影響を受け、このscale-safe化
          でも救えない場合がある。
    """
    flat = x.reshape(-1)
    scale = flat.abs().amax()
    is_zero_scale = scale == 0
    safe_scale = torch.where(is_zero_scale, torch.ones_like(scale), scale)
    normalized = flat / safe_scale
    norm = torch.linalg.vector_norm(normalized)
    return torch.where(is_zero_scale, torch.zeros_like(norm), norm * scale)


def _tail_norm(previous_tail_norm: float, singular_value: float) -> float:
    r"""打ち切り候補に特異値を1つ追加したときの、tail normをscale-safeに更新する。

    $$
    \text{new} = \sqrt{\text{previous}^2 + \text{singular\_value}^2}
    $$

    を、``math.hypot`` を使って計算する。``math.hypot`` はscale-safeな
    アルゴリズム（大きい方の引数で正規化してから2乗する）で計算するため、
    ``singular_value`` がdtypeの「二乗するとunderflowする」範囲
    （例: float32の ``1e-30``）にあっても、Python ``float``
    （常にIEEE754 binary64 = float64）へ変換した時点でそれ自体は
    underflowしていないので、素朴な二乗和よりはるかに小さい値まで
    正確に積み上げられる。

    previous_tail_norm, singular_value:
        非負の Python ``float``。呼び出し側で ``float(tensor.item())`` の
        ように変換してから渡すこと。

    戻り値:
        Python ``float``。

    Note:
        - **戻り値の型**: Python ``float``（``Tensor`` ではない）。
        - **autograd**: 関与しない。rank選択はもともと離散判定であり、
          ``_choose_truncation_rank`` は元から ``singular_values[k]`` を
          ``.item()`` してから使っていた（勾配グラフを断ち切る点）は
          この変更の前後で変わらない。この関数はPython floatだけを扱う。
        - **NaN/inf**: Python標準の ``math.hypot`` の挙動に従う
          （NaN入力はNaNを返し、無限入力を含む場合は ``inf`` を返す）。
        - **zero**: ``previous_tail_norm=0.0`` かつ
          ``singular_value=0.0`` なら厳密に ``0.0``。
        - 個々の ``singular_value`` は正確にfloat64表現できる前提だが、
          最終的な ``tail_norm`` を二乗して ``tail_sq`` を作る場面
          （``_choose_truncation_rank`` / ``tt_truncate_bond``）では、
          その二乗自体がfloat64の表現範囲（絶対値 ``~1e-154`` 未満の
          二乗はunderflowする）を下回ることがある。この関数自体は
          二乗しない（``hypot`` の内部でスケール補正された二乗を使う）ため、
          この制約は「戻り値の使い方」側の問題である。
    """
    return math.hypot(previous_tail_norm, singular_value)


def _choose_truncation_rank(
    singular_values: torch.Tensor,
    budget: float,
    *,
    min_rank: int = 1,
    slack_factor: float = 100.0,
) -> dict[str, float | int]:
    """特異値と局所誤差予算（norm、非二乗）から、保持 rank を決める（``tt_round`` 専用）。

    singular_values:
        降順の特異値ベクトル ``(ρ,)``。
    budget:
        局所誤差予算 ``δ``（**二乗していない** norm空間の値）。非負の
        有限 Python ``float``。呼び出し側で ``delta**2`` のような二乗を
        行ってから渡してはならない（``delta`` が非常に大きい場合に
        overflowし、非常に小さい場合にunderflowして0になるため）。
    min_rank:
        最低保持 rank（既定 1）。
    slack_factor:
        丸め余裕の係数。各打ち切り候補ごとに
        ``slack = slack_factor * eps * max(budget, candidate_norm)`` を計算する
        （``candidate_norm`` はその時点で捨てる予定の特異値のtail norm）。

    戻り値:
        ``new_rank``, ``tail_sq``, ``local_error``, ``discard_count``,
        ``slack`` を含む dict。``tail_sq`` / ``local_error`` は slack を
        含めない実際の値。``slack`` は最後に採用した候補で使った値
        （何も打ち切らなかった場合は0.0）。

    Note:
        slackは``budget``と、いま判定している``candidate_norm``自身の
        スケールにのみ依存させる。``singular_values``全体のエネルギー
        （``total``）を基準にすると、大きい特異値を多く残す一方で
        ごく小さい``budget``を要求した場合に、
        ``slack_factor * eps * total``が``budget``より
        桁違いに大きくなり、要求精度を大幅に超えて打ち切ってしまう
        （例: ``singular_values=[1, 1e-8]``、``budget`` が
        ``1e-8``よりずっと小さいのに、``total≈1``を基準にしたslackが
        その特異値を丸め誤差の範囲内と誤判定し、丸ごと切り捨てて
        しまう）。

        tail（捨てる候補の集合）の大きさは、``singular_values[k].square()``
        のような ``singular_values`` のdtype（float32等）上での二乗では
        なく、``_tail_norm``（Python float、内部は ``math.hypot``）で
        積み上げる。``singular_values`` の各値は ``.item()`` でPython
        floatへ変換したあと使う（rank選択はもともと離散判定であり、
        ここでautogradのグラフを切ることは既存の契約と同じ。打ち切り後に
        残す特異値・特異ベクトル自体は、呼び出し側で元の ``Tensor`` から
        スライスするので勾配は保たれる）。判定は最初から最後まで
        norm空間（``candidate_norm <= budget + slack``）で行い、
        ``budget`` 自身を二乗する場面は一度も現れない
        （例: float32の ``singular_value=1e-30`` は、二乗すれば
        ``1e-60`` がfloat32でunderflowするが、Python floatとしての
        ``1e-30`` 自体はfloat64で正確に表現できる。同様に、非常に大きい
        ``budget``（例: ``3e155``程度）を二乗すればfloat64でも
        overflowしてinfになるが、``budget`` を二乗しなければ問題にならない）。

        戻り値の ``tail_sq`` は、報告用に最後の ``tail_norm`` を1度だけ
        二乗した値である。``tail_norm`` の絶対値がおおよそ ``1e-154``
        を下回るような極端な場合は、この最後の二乗自体がfloat64の表現
        範囲を下回ってunderflowし、``tail_sq==0.0`` でも
        ``local_error>0.0`` になり得る（``local_error`` の方を信頼する
        こと）。同様に、``tail_norm`` が ``1e155`` 程度を超える極端な
        場合は、この最後の二乗がfloat64でoverflowし ``tail_sq==inf`` に
        なり得る。rank判定そのものはこの二乗の前に完了しているため、
        影響を受けるのは診断用の ``tail_sq`` のみである。
    """
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(f"budget は非負の有限floatである必要があります: {budget!r}")

    eps = torch.finfo(singular_values.dtype).eps

    tail_norm = 0.0
    new_rank = singular_values.shape[0]
    slack = 0.0
    # 末尾（最小の特異値）から順に予算内で最大限捨てる。
    for k in range(singular_values.shape[0] - 1, min_rank - 1, -1):
        value = float(singular_values[k].item())
        candidate_norm = _tail_norm(tail_norm, value)
        candidate_slack = slack_factor * eps * max(budget, candidate_norm)
        if candidate_norm <= budget + candidate_slack:
            tail_norm = candidate_norm
            new_rank = k
            slack = candidate_slack
        else:
            break

    tail_sq = tail_norm * tail_norm
    return {
        "new_rank": new_rank,
        "tail_sq": tail_sq,
        "local_error": tail_norm,
        "discard_count": singular_values.shape[0] - new_rank,
        "slack": slack,
    }


def _left_unfolding_svd(
    core: torch.Tensor,
) -> tuple[tuple[int, int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """3階coreの左展開行列をreduced SVDし、元shapeと3因子を返す。"""
    r_left, n_k, r_right = core.shape
    mat = core.reshape(r_left * n_k, r_right)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    return (r_left, n_k, r_right), U, S, Vh


def _absorb_truncated_svd_right(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    next_core: torch.Tensor,
    *,
    left_shape: tuple[int, int],
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上位SVD成分を左coreへ戻し、``S @ Vh``を右隣coreへ吸収する。"""
    r_left, n_k = left_shape
    new_left_core = U[:, :rank].reshape(r_left, n_k, rank)

    # diag(S)を作らず、行ごとのbroadcast乗算でS @ Vhを構成する。
    transfer = S[:rank].unsqueeze(1) * Vh[:rank, :]
    new_right_core = torch.tensordot(transfer, next_core, dims=([1], [0]))
    return new_left_core, new_right_core


def tt_bond_singular_values(
    cores: list[torch.Tensor],
    bond: int,
) -> torch.Tensor:
    r"""指定bondのSchmidt特異値を、mixed-canonical環境で計算する。

    ``bond`` は ``cores[bond]`` と ``cores[bond + 1]`` の間を表す。
    ``center=bond`` へcanonicalizeした後、center coreの左展開行列を
    reduced SVDし、降順の特異値ベクトルだけを返す。左右環境が等長写像
    なので、これは元TTが表すTensorの指定cutにおける特異値と一致する。

    入力core列は変更しない。戻り値は入力と同じdtype/deviceを持ち、
    SVDのautograd graphを維持する。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    d = len(cores)
    bond = coerce_integer_scalar(bond, name="bond")
    if not 0 <= bond < d - 1:
        raise ValueError(f"bond={bond} は 0 <= bond < {d - 1} を満たす必要があります。")

    canon = tt_canonicalize(cores, center=bond)
    _, _, singular_values, _ = _left_unfolding_svd(canon[bond])
    return singular_values


def tt_truncate_bond(
    cores: list[torch.Tensor],
    bond: int,
    max_rank: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """単一bondをSVDで rank ``max_rank`` 以下へ打ち切る。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。この
        リストおよび各 Tensor は変更しない。
    bond:
        打ち切り対象の bond index（0始まり）。``cores[bond]`` と
        ``cores[bond + 1]`` の間の bond を表す。``0 <= bond < d - 1``。
    max_rank:
        打ち切り後の bond rank の上限。1以上の整数。bool は拒否する。

    戻り値:
        ``(new_cores, diagnostics)``。

        ``new_cores`` は、``bond`` 周辺を ``center=bond`` の
        mixed-canonical form へ変換したうえで、``cores[bond]`` の左展開
        行列 ``(r_left * n_bond, r_bond)`` を
        ``torch.linalg.svd(..., full_matrices=False)`` で分解し、上位
        ``min(max_rank, r_bond)`` 成分だけを残した TT core 列。

        ``diagnostics`` は次のキーを持つ dict:

        - ``bond``: 対象 bond index（int）
        - ``rank_before``: 打ち切り前の bond rank（int）
        - ``rank_after``: 打ち切り後の bond rank（int）
        - ``singular_values``: 打ち切り前の特異値全体（``torch.Tensor``）
        - ``discarded_tail_sq``: 捨てた特異値の二乗和（float、scale-safe）
        - ``local_error``: 局所誤差 ``sqrt(discarded_tail_sq)``（float、
          scale-safe。``discarded_tail_sq`` を二乗して得るのではなく、
          こちらを先に計算する）

        mixed-canonical 環境（``center=bond`` により左右が等長写像）の
        もとでは、この局所打ち切り誤差がそのまま全体の Frobenius 誤差
        （``tt_reconstruct`` した dense tensor の誤差）に一致する
        （Eckart–Young–Mirsky の単一bond最適性）。

    Note:
        ``singular_values`` は配列なので ``torch.Tensor`` のまま返す
        （dtype/device・勾配情報を保持できる）。一方 ``rank_before`` /
        ``rank_after`` はTT-rankの整数contract（``validate_tt_cores`` が
        期待する ``int``）に合わせて Python ``int`` を、
        ``discarded_tail_sq`` / ``local_error`` はテスト側での比較や
        print 用途を優先して Python ``float`` を返す。

        ``discarded_tail_sq`` / ``local_error`` は、``S`` のdtype
        （float32等）のまま ``.square().sum()`` するのではなく、
        捨てた特異値をPython floatへ変換したうえで ``_tail_norm``
        （``math.hypot`` ベース）でscale-safeに積み上げてから
        ``local_error`` を求め、``discarded_tail_sq = local_error**2``
        をPython floatの二乗として最後に1度だけ計算する（推奨順序:
        1. 捨てた特異値をPython float/float64へ変換、2. scale-safeに
        ``local_error`` を計算、3. Python float上で二乗）。
        個々の特異値は正確にfloat64表現できる前提だが、最終的な
        ``local_error`` の絶対値がおおよそ ``1e-154`` を下回るような
        極端な場合は、この最後の二乗自体がPython float（float64）の
        表現範囲を下回ってunderflowし、``discarded_tail_sq==0.0`` でも
        ``local_error>0.0`` になり得る。``local_error`` の方を信頼する
        こと。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    d = len(cores)
    bond = coerce_integer_scalar(bond, name="bond")
    if not 0 <= bond < d - 1:
        raise ValueError(f"bond={bond} は 0 <= bond < {d - 1} を満たす必要があります。")

    max_rank = coerce_integer_scalar(max_rank, name="max_rank")
    if max_rank < 1:
        raise ValueError(f"max_rank は1以上である必要があります: {max_rank}")

    # bond 周辺を等長写像にするため、center=bond の mixed-canonical form へ変換する。
    canon = tt_canonicalize(cores, center=bond)

    (r_left, n_bond, r_bond), U, S, Vh = _left_unfolding_svd(canon[bond])

    rank_before = r_bond
    rank_after = min(max_rank, S.shape[0])

    # 捨てた特異値をPython float(float64)へ変換したうえで、
    # math.hypotによりscale-safeに積み上げる（先にnormを求め、
    # tail_sqはその二乗としてあとから計算する）。
    discarded_values = [float(v) for v in S[rank_after:].detach().tolist()]
    local_error = math.hypot(*discarded_values) if discarded_values else 0.0
    discarded_tail_sq = local_error * local_error

    new_cores = [core.clone() for core in canon]
    new_cores[bond], new_cores[bond + 1] = _absorb_truncated_svd_right(
        U,
        S,
        Vh,
        canon[bond + 1],
        left_shape=(r_left, n_bond),
        rank=rank_after,
    )

    diagnostics: dict[str, object] = {
        "bond": bond,
        "rank_before": int(rank_before),
        "rank_after": int(rank_after),
        "singular_values": S,
        "discarded_tail_sq": discarded_tail_sq,
        "local_error": local_error,
    }
    return new_cores, diagnostics


def tt_round(
    cores: list[torch.Tensor],
    eps_rel: float,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """既存TT core列を dense化せずに、指定相対誤差予算内で再圧縮する。

    アルゴリズムは right-canonicalization（``tt_canonicalize(cores,
    center=0)`` による right-to-left QR sweep）に続けて、
    left-to-right truncated SVD sweep を行う。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。この
        リストおよび各 Tensor は変更しない。
    eps_rel:
        全体相対誤差予算 ``ε``。有限かつ0以上の実数。bool、NaN、inf、
        負値、非数値型は拒否する。

    戻り値:
        ``(rounded_cores, diagnostics)``。

        ``rounded_cores`` は rank を落とした新しい TT core 列。物理
        shape（``tt_physical_shape``）は変換前後で変わらない。

        ``diagnostics`` は次のキーを持つ dict:

        - ``input_ranks``: 入力の境界含みTT-rank列（``tt_ranks`` の戻り値）
        - ``output_ranks``: 出力の境界含みTT-rank列
        - ``norm_x``: ``||X||_F``（right-canonical化後の第1coreから計算）
        - ``eps_requested``: 実際に使った ``eps_rel``（float）
        - ``delta``: 1 bondあたりの局所絶対誤差予算 ``δ``
        - ``local_errors``: 各bondの局所誤差 ``e_k`` のlist（float）
        - ``local_tail_sq``: 各bondで捨てた特異値の二乗和のlist（float）
        - ``global_error_estimate``: ``sqrt(sum(local_tail_sq))``
        - ``relative_error_estimate``: ``global_error_estimate / norm_x``

    Note:
        dense reconstruction はアルゴリズム内部で使用しない。
        ``d == 1``（内部bondなし）または ``eps_rel == 0``
        （意図的な打ち切りなし）の場合は、right-canonicalize 済みの
        core列をそのまま返す（QRはlosslessなgauge変換なので、rankは
        増えない。ただしreduced QRの経済サイズにより、冗長なbondは
        rankが減ることがある）。零テンソル（``rc_cores[0]`` の全要素が
        厳密に0）の場合は、物理shapeを保った rank-1 の零TTを返す。

        ``norm_x`` は ``torch.linalg.vector_norm``（内部で先に二乗する
        素朴な実装）ではなく ``_scale_safe_norm`` で計算する。float32の
        ``diag(1e-30, 1e-30)`` のように、値自体は表現できるが二乗すると
        underflowして厳密な0になるケースで、``norm_x`` を誤って0と
        判定しないようにするため。零テンソル分岐へ入るかどうかの判定は
        ``norm_x == 0.0`` という計算結果ではなく、``_is_exact_zero``
        （全要素がbitwiseに厳密な0かどうか）で行う。これにより、
        「本当に零テンソルである」ことと「計算した norm が0になった」
        ことを分離している。

        bondあたりの局所絶対誤差予算 ``delta`` は、``delta**2`` のように
        二乗してから ``_choose_truncation_rank`` へ渡すことはしない。
        ``delta`` が非常に大きい（例: ``3e155``程度）場合、``delta**2``
        はfloat64の表現上限（``~1.8e308``）を超えてoverflowし、
        非常に小さい（例: ``1e-190``程度）場合、``delta**2``はfloat64の
        最小非正規化数（``~5e-324``）を下回りunderflowして0になる。
        いずれの場合も、二乗前の``delta``自体は有限かつ正しい値なので、
        ``_choose_truncation_rank`` は二乗していない``budget``を受け取り、
        norm空間だけで判定する。

        既知の制約（今回のsrc化では対応しない）: ``tt_canonicalize``
        （``center=0``のright-to-left QR sweep）は、入力の dtype のまま
        QRとRの吸収を行う。coreごとのスケールが極端に不均衡なgauge
        （例: float32のrank-1 TTでcore scaleが``1e30``, ``1e-30``,
        ``1e-30``）では、``rc_cores[0]``を得る前の中間演算自体が
        overflow・underflowし得る（``1e-30 * 1e-30 -> 1e-60 -> 0``）。
        この場合、本関数内の``_scale_safe_norm``・``_choose_truncation_rank``
        のscale-safe化には到達する前に情報が失われるため、これらの
        scale-safe化では救えない。この関数は、通常の正準化済み・
        scale-balancedなTTを対象とし、任意のdynamic rangeに対する
        scale-invariant性は保証しない。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")
    eps_value = _coerce_eps_rel(eps_rel)

    d = len(cores)
    input_ranks = tt_ranks(cores)

    # Step 1: right-to-left QR sweep（lossless gauge変換）。
    rc_cores = tt_canonicalize(cores, center=0)
    is_exact_zero_x = _is_exact_zero(rc_cores[0])
    norm_x = 0.0 if is_exact_zero_x else float(_scale_safe_norm(rc_cores[0]).item())

    if d == 1 or eps_value == 0.0:
        output_cores = [core.clone() for core in rc_cores]
        return output_cores, {
            "input_ranks": input_ranks,
            "output_ranks": tt_ranks(output_cores),
            "norm_x": norm_x,
            "eps_requested": eps_value,
            "delta": 0.0,
            "local_errors": [],
            "local_tail_sq": [],
            "global_error_estimate": 0.0,
            "relative_error_estimate": 0.0,
        }

    if is_exact_zero_x:
        physical_shape = tt_physical_shape(rc_cores)
        output_cores = [
            torch.zeros(1, n_k, 1, dtype=rc_cores[0].dtype, device=rc_cores[0].device)
            for n_k in physical_shape
        ]
        return output_cores, {
            "input_ranks": input_ranks,
            "output_ranks": tt_ranks(output_cores),
            "norm_x": 0.0,
            "eps_requested": eps_value,
            "delta": 0.0,
            "local_errors": [],
            "local_tail_sq": [],
            "global_error_estimate": 0.0,
            "relative_error_estimate": 0.0,
        }

    # Step 2: 全体相対誤差 eps から、bondあたりの局所絶対誤差予算 delta を作る。
    # right-canonical後は ||X||_F == ||G^(1)||_F なので dense tensorは不要。
    # delta は二乗せずそのまま _choose_truncation_rank へ渡す（delta**2は
    # deltaが極端に大きい/小さい場合にoverflow/underflowし、正しい予算を
    # rank選択へ渡せなくなるため）。
    delta = eps_value * norm_x / math.sqrt(d - 1)

    work = [core.clone() for core in rc_cores]
    local_errors: list[float] = []
    local_tail_sq: list[float] = []

    for k in range(d - 1):
        (r_left, n_k, _), U, S, Vh = _left_unfolding_svd(work[k])

        info = _choose_truncation_rank(S, delta)
        r = info["new_rank"]

        work[k], work[k + 1] = _absorb_truncated_svd_right(
            U,
            S,
            Vh,
            work[k + 1],
            left_shape=(r_left, n_k),
            rank=r,
        )

        local_errors.append(info["local_error"])
        local_tail_sq.append(info["tail_sq"])

    # local_errorsは既にnorm空間の値なので、二乗和のsqrtではなく
    # math.hypotでscale-safeに合成する（極小値が並ぶ場合の
    # 二重のunderflowを避ける）。
    global_error_estimate = math.hypot(*local_errors) if local_errors else 0.0
    relative_error_estimate = global_error_estimate / norm_x if norm_x > 0 else 0.0

    diagnostics: dict[str, object] = {
        "input_ranks": input_ranks,
        "output_ranks": tt_ranks(work),
        "norm_x": norm_x,
        "eps_requested": eps_value,
        "delta": delta,
        "local_errors": local_errors,
        "local_tail_sq": local_tail_sq,
        "global_error_estimate": global_error_estimate,
        "relative_error_estimate": relative_error_estimate,
    }
    return work, diagnostics
