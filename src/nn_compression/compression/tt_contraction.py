"""dense化しないTT間のcontraction（内積・Frobeniusノルム）。

参照元:
    notebooks/30_tt_mps/00_fundamentals/13_tt_inner_frobenius_norm_and_distance.ipynb

Notebook側の ``tt_inner`` / ``tt_fro_norm`` / ``tt_distance_sq`` /
``tt_distance`` は演習用に ``assert`` で public validation を行い、
検証専用の ``tt_to_dense`` をモジュール内に再実装していた。ここでは
``tt_inner`` の environment 更新のアルゴリズム自体
（``torch.einsum("ab,aic,bid->cd", env, core_a, core_b)``）は変更せず、
validation を ``TypeError`` / ``ValueError`` に置き換え、dense化には
既存の ``tt_reconstruct`` を使う（呼び出し側のテストでのみ使用し、
実装本体では使わない）。

``tt_distance_sq`` / ``tt_distance`` はこのモジュールに実装していない。
Bugbotの4回にわたるレビューで、次の設計が順番にすべて反例を持つことが
確認された。

    - 偏極恒等式 ``<A,A>+<B,B>-2<A,B>``（近接大規模TTで桁落ち）
    - direct sum + QR正規化（QRが差分を桁落ちさせ、rank落ちでbackwardがNaN）
    - equal-rank telescoping和（複数core変化時に非対称）
    - different-rankの部分空間公式（outside_sqの桁落ちで非対称）
    - Dekkerのdouble-double（分割がoverflow/underflowし、1サイト差高速経路が
      underflowし、bitwise一致の特殊分岐が誤った勾配 ``+1/-1`` を返し、
      1サイト差高速経路がもう一方の計算グラフを切って誤った勾配を返し、
      site contractionが ``O(n r^4)`` の一時テンソルを作る）

これらはいずれも「桁落ちを直す」「特殊分岐を直す」の繰り返しで収束しなかった。
production APIとして距離を再導入するには、共通の直交基底・TTの加減算・
正準化を数値的に安定な形で先に設計する必要がある。それまでは
``tt_distance_sq`` / ``tt_distance`` を公開せず、``tt_inner`` /
``tt_fro_norm`` だけをこのモジュールの public APIとする。
"""

from __future__ import annotations

import torch

from .tt_validation import validate_tt_cores, validate_tt_dtype, validate_tt_pair


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """非負スカラーの平方根。``x>0`` では ``torch.sqrt(x)`` と一致し、``x==0`` でも有限勾配。

    ``torch.sqrt`` は ``x=0`` で勾配が発散し、backwardが ``NaN`` になる。
    ``sqrt(x + eps) - eps`` のような平滑化は ``x=0`` の勾配を有限にする一方、
    正の微小値を ``sqrt(x)`` より大幅に小さくしてしまう。

    distance APIを削除した現在、``_safe_sqrt`` は ``tt_fro_norm`` の
    ``sqrt(<A,A>)`` にしか使わない。自己内積は理論上非負であり、負値
    （``-0.0`` を除く）が現れるのは丸め誤差ではなく上流の計算失敗
    （overflow・cancellationなど）である。そのため、有意・微小を問わず
    負の有限値と ``-inf`` はすべて ``ValueError`` とし、黙って0へ丸める
    ことはしない。

    - ``x>0``（``+inf`` を含む）: forwardは ``torch.sqrt(x)`` と一致し、
      勾配は ``1/(2*sqrt(x))``（``x=+inf`` では0）。
    - ``x==0`` または ``x==-0.0``: forwardは厳密に0。勾配は0（有限）。
    - ``x`` が有限の負値、または ``-inf``: ``ValueError``。
    - ``x`` が ``NaN``: forwardは ``NaN`` のまま（数値計算の失敗を偽の0で
      隠さない）。``NaN`` を明示的な例外にはしていない
      （呼び出し側で ``torch.isnan`` を確認すること）。``nan_to_num`` の
      ような変換は使わない。

    Note:
        負値チェック（``torch.any(is_negative).item()``）はPython側の
        ``bool`` へ変換するため、``x`` がCUDA上のTensorの場合はこの
        チェックでCPU/GPU間の同期（デバイス側の演算完了待ち）が発生する。
    """
    is_nan = torch.isnan(x)
    # -0.0 < 0 は False（IEEE754の比較規則）なので、-0.0 は負値判定に
    # 引っかからず、下の positive=False 分岐でそのまま 0 になる。
    is_negative = (x < 0) & ~is_nan
    if bool(torch.any(is_negative).item()):
        bad_value = x[is_negative].reshape(-1)[0].item()
        raise ValueError(
            "_safe_sqrt は負の入力を受け付けません"
            "（自己内積の丸め誤差ではなく、上流計算の失敗として扱う）: "
            f"最初に見つかった負値={bad_value!r}"
        )

    positive = x > 0
    safe_input = torch.where(positive, x, torch.ones_like(x))
    sqrt_value = torch.sqrt(safe_input)
    result = torch.where(positive, sqrt_value, torch.zeros_like(sqrt_value))
    # NaNだった要素だけ、0扱いの枝を打ち消してNaNへ戻す。
    return torch.where(is_nan, x, result)


def tt_inner(cores_a: list[torch.Tensor], cores_b: list[torch.Tensor]) -> torch.Tensor:
    r"""2つの実数値TTの内積 ``<A, B>`` を、denseへ戻さずsiteごとに縮約して計算する。

    cores_a, cores_b:
        ``validate_tt_cores`` の contract を満たすTT core列。それぞれ
        ``A^{(k)} in R^{r_{k-1}^A x n_k x r_k^A}``、
        ``B^{(k)} in R^{r_{k-1}^B x n_k x r_k^B}`` で、境界rankは1。
        order（サイト数 ``d``）・物理次元 ``n_k``・dtype・deviceは
        A/Bで一致する必要があるが、TT-rank（``r_k^A`` と ``r_k^B``）は
        異なっていてよい。両リストおよび各Tensorは変更しない。

    戻り値:
        shape ``()`` のスカラー ``torch.Tensor``。

        $$
        \langle\mathcal A,\mathcal B\rangle
        =
        \sum_{i_1,\ldots,i_d}
        \mathcal A_{i_1,\ldots,i_d}
        \mathcal B_{i_1,\ldots,i_d}
        $$

        を、left environment ``E_k in R^{r_k^A x r_k^B}`` を
        ``E_0 = [1]`` から

        $$
        E_k(\alpha_k,\beta_k)
        =
        \sum_{\alpha_{k-1},\beta_{k-1},i_k}
        E_{k-1}(\alpha_{k-1},\beta_{k-1})
        A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
        B^{(k)}_{\beta_{k-1},i_k,\beta_k}
        $$

        で更新して計算する（``tt_reconstruct`` 等によるdense化はしない）。

    計算量:
        orderについて ``O(d)``（サイトを1回だけ走査）。一様なTT-rank ``r``、
        物理次元 ``n`` の場合、サイト ``k`` の ``einsum("ab,aic,bid->cd", ...)``
        は ``O(n r^3)`` の乗算で、``env`` は ``(r^A, r^B)`` の1テンソルだけを
        保持する。4本のbond軸を同時に持つ ``O(n r^4)`` の一時テンソルは作らない。

    Note:
        ``.item()`` や ``.detach()`` は使わない。TT coreが
        ``requires_grad=True`` の場合、戻り値からbackwardできる。
        実数の ``torch.float32`` / ``torch.float64`` のみ正式対応。

        AとBのスケールが極端に近い（例: ``A`` と ``A`` に僅かな摂動を
        加えたもの）かつ大きい場合、``<A,A>`` のような自己内積自体は
        単一の（符号が揃った）大きな和であり桁落ちの心配はないが、
        ``<A,B>`` を独立に計算した値を他の計算（偏極恒等式など）で
        引き合わせると桁落ちしうる。この関数自体はそのような引き合わせを
        行わないため、単独では桁落ちの問題はない。

        極端に大きい／小さいcore（``1e300`` 程度）を含むTTでは、
        ``einsum`` の中間積がoverflow・underflowし得る。この関数はスケール
        補正を行わないため、そのようなTTでは戻り値が ``inf`` や ``0`` になる
        場合がある。呼び出し側で必要ならcoreを正準化・正規化してから使うこと。
    """
    validate_tt_cores(cores_a, name="cores_a")
    validate_tt_cores(cores_b, name="cores_b")
    validate_tt_dtype(cores_a[0], name="cores_a[0]")
    validate_tt_dtype(cores_b[0], name="cores_b[0]")
    validate_tt_pair(cores_a, cores_b, name_a="cores_a", name_b="cores_b")

    env = torch.ones((1, 1), dtype=cores_a[0].dtype, device=cores_a[0].device)
    for core_a, core_b in zip(cores_a, cores_b):
        # a: A左bond, b: B左bond, i: 共通physical index, c: A右bond, d: B右bond
        env = torch.einsum("ab,aic,bid->cd", env, core_a, core_b)

    # 境界rankが1なので env は最終的に (1, 1)。scalarへ落とす。
    return env.reshape(())


def tt_fro_norm(cores: list[torch.Tensor]) -> torch.Tensor:
    r"""実数値TTのFrobeniusノルム ``||A||_F`` をdense化せずに計算する。

    cores:
        ``validate_tt_cores`` の contract を満たすTT core列。変更しない。

    戻り値:
        shape ``()`` のスカラー ``torch.Tensor``。

        $$
        \|\mathcal A\|_F=\sqrt{\langle\mathcal A,\mathcal A\rangle}
        $$

        自己内積 ``tt_inner(cores, cores)`` は理論上非負である。丸め誤差で
        微小な負値になったとしても、``_safe_sqrt`` はそれを0として扱わず
        ``ValueError`` を送出する（後述）。

    計算量:
        ``tt_inner`` に同じ（orderについて ``O(d)``、一様rank ``r`` で
        サイトあたり ``O(n r^3)``）。

    Note:
        平方根は ``_safe_sqrt`` を使う。``x>0`` では ``torch.sqrt(x)`` と
        一致し、``x=0``（零TTなど）でもbackwardの勾配は有限である。

        ``tt_inner`` と同じく、極端に大きい／小さいcoreではoverflow・
        underflowし得る。この関数はスケール補正を行わない。

        ``tt_inner(cores, cores)`` がoverflow・cancellationにより ``NaN``
        になった場合、``tt_fro_norm`` は ``NaN`` をそのまま返す
        （``_safe_sqrt`` はNaNを0へ変換しない）。

        ``tt_inner(cores, cores)`` が有限の負値または ``-inf`` になった
        場合（理論上あり得ないが、桁落ちにより発生し得る）は、
        ``_safe_sqrt`` が ``ValueError`` を送出する。数値計算の失敗を、
        誤った零ノルムとして黙って隠すことはしない。
    """
    norm_sq = tt_inner(cores, cores)
    return _safe_sqrt(norm_sq)
