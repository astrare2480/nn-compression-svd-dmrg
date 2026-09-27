---
title: TT-roundingのPyTorch実装設計と検証
tags:
  - TensorTrain
  - MPS
  - TT-rounding
  - PyTorch
  - QR
  - SVD
aliases:
  - TT-roundingのPyTorch実装
---

# TT-roundingのPyTorch実装設計と検証

## このノートの位置づけ

このノートは、TT-roundingをPyTorchへ落とすときのshape、添字、関数分割、境界条件、検証項目を整理する。

理論は次の3ノートに分けている。

- 2方向sweepと各行列のshape：[[00_基礎理論/01_数学基礎/02_テンソル代数/48_TT-roundingの定義と正準化sweep]]
- 環境行列と誤差の直交分解：[[00_基礎理論/01_数学基礎/02_テンソル代数/49_TT-roundingの環境行列と誤差直交分解]]
- 誤差予算とrank選択：[[00_基礎理論/01_数学基礎/02_テンソル代数/50_TT-roundingの誤差予算とrank選択]]

ここに示すコードは実装設計と小さい確認例である。既存の学習NotebookのTODOを埋めたことや、Notebookを実行して数値結果を確認したことを意味しない。

---

## 1. 実装する処理の全体像

入力TTを

$$
\mathcal X
=
\llbracket
G^{(1)},\ldots,G^{(d)}
\rrbracket,
\qquad
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

とする。境界rankは

$$
r_0=r_d=1
$$

である。

標準的な処理順は

$$
\boxed{
\text{入力検証}
\longrightarrow
\text{右から左へのreduced QR}
\longrightarrow
\text{局所誤差予算の計算}
\longrightarrow
\text{左から右へのtruncated SVD}
\longrightarrow
\text{検証とログ}
}
$$

である。

1回のTT-roundingでは、QRとSVDの2方向sweepを行った後に終了する。DMRGのように局所最適化と環境更新を収束まで往復させる処理ではない。

---

## 2. PyTorchの基本設定

小さい例で数式と数値を照合するときは、まず再現性と丸め誤差の観察を優先する。

```python
import math

import torch


# 教材内の数値比較を安定させるため、CPU・float64を使う。
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

device = torch.device("cpu")
```

主に使うAPIは次である。

```python
# reduced QR
Q, R = torch.linalg.qr(A, mode="reduced")

# economy-size SVD
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

# 2次元行列の転置
At = A.mT
```

`Tensor.T` は高階テンソルでは全軸を逆順にするため、2次元行列の転置であることを明示したい箇所では `.mT` を使う。

---

## 3. 入力TTの検証

最低限、次を確かめる。

1. コア列が空でない。
2. 全コアが3階テンソルである。
3. 左端rankと右端rankが1である。
4. 隣接コアのbond次元が一致する。
5. dtypeとdeviceが全コアで一致する。
6. 実数の浮動小数点テンソルである。

```python
def validate_tt_cores(cores: list[torch.Tensor]) -> None:
    """TTコア列のshape・dtype・deviceを検証する。"""
    if not cores:
        raise ValueError("cores must contain at least one TT core.")

    first = cores[0]
    if not first.is_floating_point() or first.is_complex():
        raise TypeError("TT cores must be real floating-point tensors.")

    dtype = first.dtype
    device = first.device

    for k, core in enumerate(cores):
        if core.ndim != 3:
            raise ValueError(
                f"cores[{k}] must be 3-dimensional, got {core.ndim}."
            )
        if core.dtype != dtype:
            raise TypeError("All TT cores must have the same dtype.")
        if core.device != device:
            raise ValueError("All TT cores must be on the same device.")
        if min(core.shape) <= 0:
            raise ValueError("All TT core dimensions must be positive.")
        if not torch.isfinite(core).all():
            raise ValueError("TT cores must not contain NaN or Inf values.")

    if cores[0].shape[0] != 1:
        raise ValueError("The left boundary rank must be 1.")
    if cores[-1].shape[2] != 1:
        raise ValueError("The right boundary rank must be 1.")

    for k in range(len(cores) - 1):
        if cores[k].shape[2] != cores[k + 1].shape[0]:
            raise ValueError(
                f"Bond mismatch between cores[{k}] and cores[{k + 1}]."
            )
```

入力を直接変更しないため、作業用コアはcloneする。

```python
def clone_cores(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """autograd履歴を保ったまま、コアTensorを別領域へ複製する。"""
    return [core.clone() for core in cores]


def tt_ranks(cores: list[torch.Tensor]) -> list[int]:
    """内部TT-rankを左から順に返す。"""
    return [int(core.shape[2]) for core in cores[:-1]]


def tt_physical_shape(cores: list[torch.Tensor]) -> tuple[int, ...]:
    """物理shape (n_1, ..., n_d) を返す。"""
    return tuple(int(core.shape[1]) for core in cores)
```

---

## 4. 小規模検証用のdense復元

小さいテンソルでは、各コアを順に縮約してdenseテンソルと比較できる。

```python
def tt_to_dense(cores: list[torch.Tensor]) -> torch.Tensor:
    """検証用にTTをdenseテンソルへ復元する。"""
    validate_tt_cores(cores)

    # 左端bond r_0=1を除き、shapeを (n_1, r_1) にする。
    value = cores[0].squeeze(0)

    for core in cores[1:]:
        # 末尾bond aと次コアの左bond aを縮約する。
        value = torch.einsum("...a,aib->...ib", value, core)

    # 右端bond r_d=1を除く。
    return value.squeeze(-1)


def fro_norm(x: torch.Tensor) -> torch.Tensor:
    """全要素に対するFrobeniusノルムを返す。"""
    return torch.linalg.vector_norm(x)
```

dense復元は要素数

$$
\prod_{k=1}^{d}n_k
$$

のテンソルを作るため、大規模TTの通常処理には使わない。

---

## 5. 右から左へのQR sweep

### 5.1 右展開

第 $k$ コアを

$$
H_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1},n_kr_k)
\right)
$$

とする。right-canonical条件は行直交性

$$
H_kH_k^T=I_{r_{k-1}}
$$

である。

QRは $Q$ の列を直交化するので、$H_k$ ではなく $H_k^T$ に適用する。

$$
H_k^T=Q_kR_k.
$$

reduced QRでは

$$
q_k
=
\min(n_kr_k,r_{k-1}),
$$

$$
Q_k
\in
\mathbb R^{(n_kr_k)\times q_k},
\qquad
R_k
\in
\mathbb R^{q_k\times r_{k-1}}.
$$

したがって、新しい第 $k$ コアは

$$
\widehat G^{(k)}
=
\operatorname{reshape}
\left(
Q_k^T,
(q_k,n_k,r_k)
\right)
$$

である。

### 5.2 1ステップの実装

```python
r_left, n_k, r_right = cores_work[k].shape

H = cores_work[k].reshape(r_left, n_k * r_right)
Q, R = torch.linalg.qr(H.mT, mode="reduced")

new_left_rank = Q.shape[1]

# Qそのものではなく、Q.mTを3階コアへ戻す。
cores_work[k] = Q.mT.reshape(
    new_left_rank,
    n_k,
    r_right,
)

# R.mTを左隣コアの右bondへ吸収する。
cores_work[k - 1] = torch.einsum(
    "aib,bc->aic",
    cores_work[k - 1],
    R.mT,
)
```

`einsum` の各添字は

$$
\widehat G^{(k-1)}(a,i,c)
=
\sum_b
G^{(k-1)}(a,i,b)
R_k^T(b,c)
$$

に対応する。

- `a`：第 $k-2$ bond
- `i`：第 $k-1$ 物理添字
- `b`：更新前の第 $k-1$ bond
- `c`：更新後の第 $k-1$ bond

### 5.3 sweep全体

```python
def right_canonicalize(
    cores: list[torch.Tensor],
) -> list[torch.Tensor]:
    """表すテンソルを変えず、右から左へright-canonical化する。"""
    validate_tt_cores(cores)
    work = clone_cores(cores)

    # Python添字では d-1, d-2, ..., 1 を処理する。
    # cores[0] は最後にorthogonality centerとして残る。
    for k in range(len(work) - 1, 0, -1):
        r_left, n_k, r_right = work[k].shape
        H = work[k].reshape(r_left, n_k * r_right)

        Q, R = torch.linalg.qr(H.mT, mode="reduced")
        new_left_rank = Q.shape[1]

        work[k] = Q.mT.reshape(
            new_left_rank,
            n_k,
            r_right,
        )

        work[k - 1] = torch.einsum(
            "aib,bc->aic",
            work[k - 1],
            R.mT,
        )

    return work
```

### 5.4 間違えやすい代入

次は誤りである。

```python
cores_work[k] = Q
cores_work[k + 1] = cores[k + 1]
```

理由は次の通りである。

- `Q` は2階行列であり、3階TTコアではない。
- right-canonicalコアにするのは `Q.mT` である。
- QRの残り `R.mT` を渡す相手は右隣 `k + 1` ではなく左隣 `k - 1` である。
- `k + 1` はすでに1つ前の反復で処理済みなので、元の値へ戻してはいけない。

### 5.5 reduced QRでbondが縮む例

$$
G^{(k)}
\in
\mathbb R^{10\times2\times3}
$$

なら

$$
H_k
\in
\mathbb R^{10\times6},
\qquad
H_k^T
\in
\mathbb R^{6\times10}.
$$

よって

$$
Q_k
\in
\mathbb R^{6\times6},
\qquad
R_k
\in
\mathbb R^{6\times10},
$$

となり、新しいコアshapeは

$$
(6,2,3)
$$

である。bondは $10\to6$ へ縮むが、特異値を捨てたのではなく、行列shapeが許す冗長次元をexactに除いた結果である。reduced QRは数値rankを判定して小さい成分を打ち切る処理ではない。

---

## 6. QR phaseの検証

### 6.1 right-canonical誤差

```python
def right_orthogonality_error(core: torch.Tensor) -> torch.Tensor:
    """1コアの H H^T - I のFrobeniusノルムを返す。"""
    r_left, n_k, r_right = core.shape
    H = core.reshape(r_left, n_k * r_right)

    gram = H @ H.mT
    identity = torch.eye(
        r_left,
        dtype=core.dtype,
        device=core.device,
    )
    return fro_norm(gram - identity)


def check_right_canonical(
    cores: list[torch.Tensor],
) -> list[torch.Tensor]:
    """中心を除く第2コア以降の直交誤差を返す。"""
    return [
        right_orthogonality_error(cores[k])
        for k in range(1, len(cores))
    ]
```

### 6.2 確認する3つの関係

小さい入力では、次を別々に確認する。

$$
\mathcal X_{\mathrm{before\ QR}}
\approx
\mathcal X_{\mathrm{after\ QR}},
$$

$$
H_kH_k^T
\approx
I,
\qquad
k=2,\ldots,d,
$$

$$
\|\mathcal X\|_F
\approx
\|G^{(1)}\|_F.
$$

```python
cores_original = clone_cores(cores)
cores_rc = right_canonicalize(cores)

X_original = tt_to_dense(cores_original)
X_rc = tt_to_dense(cores_rc)

qr_reconstruction_error = fro_norm(X_original - X_rc)
right_errors = check_right_canonical(cores_rc)
norm_dense = fro_norm(X_rc)
norm_from_center = fro_norm(cores_rc[0])
```

`torch.allclose` の許容値は、dtype、テンソル規模、縮約段数に依存する。例えば $1000u\max(1,N)$ のような検証用の幅を小さい実験で採用しても、それを任意の規模・条件数のTTに対する数学的保証とは扱わない。

---

## 7. 特異値から保持rankを選ぶ

特異値を

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_q\ge0
$$

とする。rank $r$ を残したときの尾部二乗和は

$$
t(r)
=
\sum_{j=r+1}^{q}\sigma_j^2.
$$

局所予算を $\delta$ とすると、選ぶrankは

$$
\widetilde r
=
\min
\left\{
r\ge1:
t(r)\le\delta^2
\right\}.
$$

### 7.1 reverse cumulative sum

PyTorchの0始まり添字で

$$
\texttt{tail\_sq}[r]
=
\sum_{j=r}^{q-1}\texttt{S}[j]^2
$$

となる配列を作れば、各候補rankの尾部を一括して比較できる。

```python
def choose_rank(
    singular_values: torch.Tensor,
    budget_sq: float | torch.Tensor,
    *,
    min_rank: int = 1,
) -> dict[str, object]:
    """局所二乗誤差予算を満たす最小rankを選ぶ。"""
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional.")

    q = singular_values.numel()
    if not 1 <= min_rank <= q:
        raise ValueError("min_rank must satisfy 1 <= min_rank <= q.")

    squared = singular_values.square()

    # tail_sq[r] = sum_{j=r}^{q-1} squared[j]
    tail_sq = torch.zeros(
        q + 1,
        dtype=singular_values.dtype,
        device=singular_values.device,
    )
    tail_sq[:q] = torch.flip(
        torch.cumsum(torch.flip(squared, dims=(0,)), dim=0),
        dims=(0,),
    )

    budget_tensor = torch.as_tensor(
        budget_sq,
        dtype=singular_values.dtype,
        device=singular_values.device,
    )
    if budget_tensor.ndim != 0 or budget_tensor < 0:
        raise ValueError("budget_sq must be a nonnegative scalar.")

    candidates = torch.arange(
        min_rank,
        q + 1,
        device=singular_values.device,
    )
    feasible = candidates[tail_sq[min_rank:] <= budget_tensor]

    # r=qなら尾部は0なので、通常は少なくとも1候補が存在する。
    new_rank = int(feasible[0].item())
    discarded_sq = tail_sq[new_rank]

    return {
        "new_rank": new_rank,
        "tail_sq": float(discarded_sq.item()),
        "local_error": float(torch.sqrt(discarded_sq).item()),
        "discard_count": q - new_rank,
    }
```

ここでは `tail_sq[q] = 0` を配列の末尾に残している。逆累積和の**先頭**に0を追加すると、`tail_sq[r]` と「上位 $r$ 個を保持したときの尾部」の対応が1つずれる。

### 7.2 数値例

```python
S = torch.tensor([5.0, 2.0, 0.6, 0.3])
delta = 0.7

rank_info = choose_rank(S, delta**2)

assert rank_info["new_rank"] == 2
assert math.isclose(
    rank_info["tail_sq"],
    0.6**2 + 0.3**2,
    rel_tol=0.0,
    abs_tol=1e-12,
)
```

このとき

$$
e
=
\sqrt{0.6^2+0.3^2}
=
\sqrt{0.45}
\approx
0.6708
<
0.7
$$

である。rank $1$ では

$$
\sqrt{2^2+0.6^2+0.3^2}
>
0.7
$$

なので、予算を満たす最小rankは $2$ である。

---

## 8. 固定の加算slackをrank選択へ入れない理由

例えば

```python
slack = 100 * torch.finfo(S.dtype).eps * max(
    budget_sq,
    torch.sum(S.square()).item(),
)
```

を無条件に足すと、非常に小さい相対誤差目標ではslackが本来の予算を上回る。

例として

$$
\varepsilon=10^{-8},
\qquad
d=3
$$

なら、均等予算は

$$
\delta^2
=
\frac{\varepsilon^2N^2}{d-1}
=
5\times10^{-17}N^2.
$$

一方、float64のmachine epsilonを

$$
u\approx2.22\times10^{-16}
$$

とすると

$$
100uN^2
\approx
2.22\times10^{-14}N^2
$$

であり、予算の約444倍になる。これは単なる等号判定の補助ではなく、許容誤差を実質的に緩める。

したがって、基本実装では次を分ける。

- rank選択：$t(r)\le\delta^2$ を保守的に判定する。
- 数値検証：丸め誤差を考慮した `allclose` 等を別に使う。
- $\varepsilon=0$：通常のrank選択へ流さず、独立に分岐する。

境界付近で比較が不確かな場合は、小さいrankへ攻めるより1つ大きいrankを残す方が、誤差上限の契約には保守的である。

float32では

$$
u
\approx
1.19\times10^{-7}
$$

であり、float64より丸め誤差が大きい。小さい相対誤差目標では、QRの直交性、特異値尾部、誤差境界の区別自体が見えにくくなる。そのため学習時の検証はfloat64を基本とする。入力をfloat64へ昇格してroundingし、最後にfloat32へ戻す設計も可能だが、最後のcastが追加誤差を生むため、その誤差も別途測る。

---

## 9. 左から右へのtruncated SVD

### 9.1 1 bondの処理

第 $k$ コアを

$$
V_k
=
\operatorname{reshape}
\left(
G^{(k)},
(r_{k-1}n_k,r_k)
\right)
$$

とし、記号衝突を避けて

$$
V_k
=
U_k\Sigma_kW_k^T
$$

と書く。PyTorchの `Vh` は $W_k^T$ に対応する。

```python
r_left, n_k, r_right = work[k].shape
V = work[k].reshape(r_left * n_k, r_right)

U, S, Vh = torch.linalg.svd(V, full_matrices=False)

rank_info = choose_rank(S, budget_sq)
new_rank = rank_info["new_rank"]

# 保持した左特異ベクトルを第kコアにする。
work[k] = U[:, :new_rank].reshape(
    r_left,
    n_k,
    new_rank,
)

# B = Sigma_keep W_keep^T
B = S[:new_rank, None] * Vh[:new_rank, :]

# Bの旧右bondを、右隣コアの左bondへ吸収する。
work[k + 1] = torch.einsum(
    "ab,bcd->acd",
    B,
    work[k + 1],
)
```

shapeは

$$
U_{k,\mathrm{keep}}
\in
\mathbb R^{(r_{k-1}n_k)\times\widetilde r_k},
$$

$$
B_k
=
\Sigma_{k,\mathrm{keep}}W_{k,\mathrm{keep}}^T
\in
\mathbb R^{\widetilde r_k\times r_k},
$$

$$
G^{(k+1)}
\in
\mathbb R^{r_k\times n_{k+1}\times r_{k+1}},
$$

$$
\widehat G^{(k+1)}
\in
\mathbb R^{\widetilde r_k\times n_{k+1}\times r_{k+1}}
$$

である。

`einsum` は

$$
\widehat G^{(k+1)}(a,i,c)
=
\sum_b
B_k(a,b)
G^{(k+1)}(b,i,c)
$$

に対応する。

### 9.2 局所予算

right-canonical化後に

$$
N
=
\|\mathcal X\|_F
=
\|G^{(1)}\|_F
$$

を求め、均等配分なら

$$
\delta
=
\frac{\varepsilon N}{\sqrt{d-1}}
$$

とする。

実局所誤差は

$$
e_k
=
\left(
\sum_{j=\widetilde r_k+1}^{q_k}
\sigma_{k,j}^2
\right)^{1/2}
$$

であり、$\delta$ そのものではない。

---

## 10. 統合関数の設計

公開関数は少なくとも次を行う。

```python
def tt_round(
    cores: list[torch.Tensor],
    eps_rel: float,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """相対Frobenius誤差を目標にTT-roundingする。"""
    validate_tt_cores(cores)
    if eps_rel < 0:
        raise ValueError("eps_rel must satisfy eps_rel >= 0.")

    input_ranks = tt_ranks(cores)
    d = len(cores)

    # d=1には内部bondもSVD段階もない。
    if d == 1:
        rounded = clone_cores(cores)
        return rounded, {
            "input_ranks": input_ranks,
            "output_ranks": [],
            "eps_requested": eps_rel,
            "truncation_performed": False,
            "is_zero_tensor": bool(fro_norm(rounded[0]) == 0),
        }

    work = right_canonicalize(cores)
    norm_x = float(fro_norm(work[0]).item())

    # 零テンソルは相対誤差0/0を作らず、物理shapeを保つrank-1 TTへする。
    if norm_x == 0.0:
        zero_cores = [
            torch.zeros(
                (1, core.shape[1], 1),
                dtype=core.dtype,
                device=core.device,
            )
            for core in work
        ]
        return zero_cores, {
            "input_ranks": input_ranks,
            "output_ranks": [1] * (d - 1),
            "norm_x": 0.0,
            "eps_requested": eps_rel,
            "truncation_performed": False,
            "is_zero_tensor": True,
        }

    # eps=0では意図的なSVD打ち切りを行わない。
    if eps_rel == 0.0:
        return work, {
            "input_ranks": input_ranks,
            "output_ranks": tt_ranks(work),
            "norm_x": norm_x,
            "eps_requested": 0.0,
            "truncation_performed": False,
            "is_zero_tensor": False,
        }

    delta = eps_rel * norm_x / math.sqrt(d - 1)
    budget_sq = delta**2
    local_tail_sq: list[float] = []
    rank_details: list[dict[str, object]] = []

    for k in range(d - 1):
        r_left, n_k, r_right = work[k].shape
        V = work[k].reshape(r_left * n_k, r_right)
        U, S, Vh = torch.linalg.svd(V, full_matrices=False)

        rank_info = choose_rank(S, budget_sq)
        new_rank = int(rank_info["new_rank"])

        work[k] = U[:, :new_rank].reshape(
            r_left,
            n_k,
            new_rank,
        )
        B = S[:new_rank, None] * Vh[:new_rank, :]
        work[k + 1] = torch.einsum(
            "ab,bcd->acd",
            B,
            work[k + 1],
        )

        local_tail_sq.append(float(rank_info["tail_sq"]))
        rank_details.append({
            "bond": k + 1,
            "singular_values": S.detach().clone(),
            **rank_info,
        })

    global_error_estimate = math.sqrt(sum(local_tail_sq))
    relative_error_estimate = global_error_estimate / norm_x

    info = {
        "input_ranks": input_ranks,
        "output_ranks": tt_ranks(work),
        "norm_x": norm_x,
        "eps_requested": eps_rel,
        "local_budget": delta,
        "local_tail_sq": local_tail_sq,
        "local_errors": [math.sqrt(x) for x in local_tail_sq],
        "global_error_estimate": global_error_estimate,
        "relative_error_estimate": relative_error_estimate,
        "tolerance_met": relative_error_estimate <= eps_rel,
        "rank_capped": False,
        "truncation_performed": True,
        "is_zero_tensor": False,
        "rank_details": rank_details,
    }
    return work, info
```

これは基本設計であり、srcの公開APIとして採用する場合は、既存の型・例外・dtype・autograd契約と統合してからテストする。

---

## 11. 境界条件

### 11.1 $d=1$

内部bondがないので、QR sweepもSVD truncationもない。入力のコピーを返す。

### 11.2 零テンソル

$$
N=\|\mathcal X\|_F=0
$$

では相対誤差

$$
\frac{\|\mathcal X-\widetilde{\mathcal X}\|_F}{\|\mathcal X\|_F}
$$

が $0/0$ になる。通常の相対tolerance判定へ流さず、物理shapeを保ったrank-1零TTを返す。

### 11.3 $\varepsilon=0$

意図的なtruncationを行わない。QRによるgauge変更やexactな冗長bondの除去はあり得るが、特異値尾部を捨てる処理とは分ける。

### 11.4 rank cap

基本版ではrank capを設けない。上限 $r_{k,\max}$ を追加して

$$
r_k^{\mathrm{used}}
=
\min(\widetilde r_k,r_{k,\max})
$$

とすると、capが発動したbondでは

$$
e_k\le\delta_k
$$

を保証できない場合がある。診断情報には `rank_capped` と実際の誤差を残し、「rank制約を守った」と「toleranceを守った」を別々に報告する。

### 11.5 絶対tolerance

絶対許容誤差 $	au$ を使うなら

$$
\sum_{k=1}^{d-1}\delta_k^2
\le
\tau^2
$$

と配る。相対toleranceでは

$$
\tau=\varepsilon\|\mathcal X\|_F
$$

である。APIではどちらを受け取ったかを曖昧にしない。

---

## 12. 検証項目

### 12.1 QRだけの検証

- QR前後で物理shapeが不変。
- 入力コアを変更していない。
- dense復元が丸め誤差内で一致。
- 第2コア以降で $H_kH_k^T\approx I$。
- $\|\mathcal X\|_F\approx\|G^{(1)}\|_F$。

### 12.2 rank選択だけの検証

採用rankを $r$ としたとき

$$
t(r)\le\delta^2
$$

を確認する。さらに $r>r_{\min}$ なら

$$
t(r-1)>\delta^2
$$

も確認し、最小rankであることを検証する。

### 12.3 rounding全体の検証

小さいテンソルでは

$$
e_{\mathrm{dense}}
=
\|\mathcal X-\widetilde{\mathcal X}\|_F,
$$

$$
e_{\mathrm{logged}}
=
\left(
\sum_{k=1}^{d-1}e_k^2
\right)^{1/2}
$$

を比較する。exact arithmeticの標準sweepでは一致し、浮動小数点では近似一致になる。

さらに

$$
e_{\mathrm{logged}}
\le
\varepsilon\|\mathcal X\|_F
$$

を確かめる。

### 12.4 境界ケース

- 圧縮が起こる入力
- どのrankも縮まない入力
- $\varepsilon=0$
- 零TT
- $d=1$
- reduced QRだけでexactにbondが縮む入力
- 採用rankが予算境界付近にある入力

### 12.5 大規模TT

denseを作れない場合はTT内積を縮約して

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F^2
=
\|\mathcal X\|_F^2
+
\|\widetilde{\mathcal X}\|_F^2
-
2\langle
\mathcal X,
\widetilde{\mathcal X}
\rangle
$$

と計算できる。ただし非常に近いテンソルでは、大きな3項の差による桁落ちに注意する。

---

## 13. 学習用の小さい入力

物理shape

$$
(n_1,n_2,n_3)=(4,5,3)
$$

で、真の内部rankが

$$
(2,2)
$$

のTTを作り、零paddingで表示上のrankを

$$
(4,3)
$$

へ増やすと、同じdenseテンソルを表す冗長TTを作れる。この例なら

1. QR前後のexact性
2. reduced QRによる冗長次元の整理
3. 小さい $\varepsilon$ でのrank選択
4. 物理shape不変

を分けて観察できる。

学習用Notebookでは、次の順にTODOを進めると依存関係が見えやすい。

1. right-to-left QR
2. QR前後不変性・右直交性・中心ノルム
3. `choose_rank`
4. $(5,2,0.6,0.3)$ と $\delta=0.7$ の単体検算
5. 1 bondのtruncated SVD
6. 左から右へのSVD sweep
7. `tt_round` の統合
8. dense誤差との比較
9. $\varepsilon$ sweep
10. 零TTと $d=1$

この実装ノートはそのTODOの答えをNotebookへ書き込むものではない。Notebook側では、数式、期待shape、期待する検証結果を読んだうえで実装する余地を残す。

---

## 14. 推奨する関数分割

基本版は次の責務に分ける。

```text
validate_tt_cores
clone_cores
tt_ranks
tt_physical_shape
tt_to_dense                  # 小規模検証用
fro_norm
right_orthogonality_error
check_right_canonical
right_canonicalize
choose_rank
left_to_right_truncate
tt_round
```

実装と学習の順序は

```text
QR単独
→ rank選択単独
→ 1 bond SVD
→ SVD sweep
→ 統合関数
→ 数値検証
```

とする。一度に全処理を組むより、どの段階でshapeや誤差が崩れたかを切り分けやすい。

---

## 15. ログに残す情報

少なくとも次を返すと、圧縮結果と保証を後から確認できる。

```text
input_ranks
output_ranks
norm_x
eps_requested
local_budget
local_errors
local_tail_sq
global_error_estimate
relative_error_estimate
tolerance_met
rank_capped
is_zero_tensor
truncation_performed
rank_details
```

`tolerance_met` は実際に採用したrankと誤差から判定する。rank capや緩い比較規則を追加した場合は、その事実を別のflagで残し、理論上の保証と実測結果を混同しない。

---

## 16. 最終チェックリスト

- [ ] `H.mT` にreduced QRを適用している。
- [ ] 新コアは `Q.mT` を3階へreshapeしている。
- [ ] 新しい左bondは `Q.shape[1]` から取得している。
- [ ] `R.mT` は左隣 `k - 1` へ吸収している。
- [ ] 処理済みの右隣コアを元に戻していない。
- [ ] SVDでは `full_matrices=False` を使っている。
- [ ] $\delta_k$ と $e_k$ を別の量として記録している。
- [ ] 採用rankが予算を満たす最小rankである。
- [ ] 固定の大きな加算slackで局所予算を緩めていない。
- [ ] $d=1$、零TT、$\varepsilon=0$ を独立に扱っている。
- [ ] QR前後のdense一致と右直交性を検証している。
- [ ] dense実誤差と局所誤差二乗和を比較している。
- [ ] rank cap発動時はtolerance保証を別判定している。
- [ ] 入力コアを直接変更していない。

次の発展課題は、denseを作らずにTT同士の内積、ノルム、距離を環境縮約で計算することである。その後は、TT加算後のrounding、Hadamard積、TT-matrix / MPOによる線形演算、TT形式の線形方程式と反復法、局所変分最適化としてのDMRGへ接続できる。

---

## 17. 学習Notebookの保存出力を使った監査例

ここでは、学習Notebookに保存されている出力とコードを、上の設計に照らして監査した例を記録する。これは保存済み出力の読取り結果であり、カーネルを再起動したうえでNotebook全体を再実行した結果ではない。

### 17.1 QR正準化の保存出力

入力のphysical shapeとTT-rankは

$$
(n_1,n_2,n_3)
=
(4,5,3),
\qquad
(r_1,r_2)
=
(4,3)
$$

である。右から左へのQR後に保存されている値は次である。

- QR前後の再構成誤差：$3.395438385995374\times10^{-15}$
- 第2コアの右直交性誤差：$2.7633878174254497\times10^{-16}$
- 第3コアの右直交性誤差：$6.138747913423791\times10^{-16}$
- $\|\mathcal X\|_F$ と $\|G^{(1)}\|_F$ の差：$1.7763568394002505\times10^{-15}$

いずれもfloat64の丸め誤差程度であり、保存出力の範囲では、QRによる再構成不変性、右直交性、正準中心へのノルム集約が確認できる。

### 17.2 rank選択と1 bond更新の保存出力

特異値

$$
(5,2,0.6,0.3)
$$

と局所誤差予算

$$
\delta=0.7,
\qquad
\delta^2=0.49
$$

を使ったrank選択では、保存出力は

$$
\widetilde r=2,
\qquad
e^2=0.6^2+0.3^2=0.45,
\qquad
e=\sqrt{0.45}\simeq0.6708203932
$$

である。これは $e^2\le\delta^2$ を満たす最小保持rankの計算と一致する。

別の1 bond更新では、表示rankが

$$
4\longrightarrow2
$$

へ減った一方、記録された尾部二乗和は

$$
e^2=0
$$

である。この例は、rankが減っても、捨てた特異値が厳密に0なら近似誤差は生じないことを示す。

### 17.3 end-to-end出力の読み方

Notebookに保存された最終表示は

$$
(r_1,r_2)
:
(4,3)
\longrightarrow
(2,2),
$$

$$
N_{\mathrm{param}}
:
85
\longrightarrow
34,
$$

$$
\|\mathcal X-\widetilde{\mathcal X}\|_F
\simeq
7.12\times10^{-15}
$$

である。この入力では削除対象が零特異値だけなので、理論上の局所尾部二乗和とその集計値は0である。それでもdense再構成誤差が完全な0にならないのは、QR、SVD、行列積をfloat64で実行した際の丸め誤差による。したがって、

$$
\text{理論上のtruncation誤差}=0
$$

と

$$
\text{浮動小数点で測ったdense誤差}\simeq7.12\times10^{-15}
$$

は矛盾しない。

---

## 18. 保存コードの監査で残った確認事項

保存コードには、正しい部分と、Notebook全体の完了判定前に直すべき部分が混在している。

1. QR係数の左隣への吸収に使われている `torch.tensordot` は、縮約添字が正しいため数学的には有効である。ただし、演習で指定した添字対応を明示するなら、`torch.einsum("aib,bc->aic", ...)` の方が数式との対応を読み取りやすい。
2. 統合関数 `tt_round` の負の `eps_rel` に対する `raise ValueError(...)` は、例外メッセージまで実装されていない。`raise ValueError("eps_rel must satisfy eps_rel >= 0.")` のように具体化する必要がある。
3. `choose_rank` 内の `local_erorr` は局所変数名の誤記である。返却キーは `local_error` だが、教材コードとして変数名も統一する。
4. `choose_rank` の固定値 `slack_factor=100.0` は、小さい誤差予算を相対的に大きく緩めることがある。理論保証を実装する基本形では、実際の尾部二乗和を `budget_sq` と直接比較する。slackを導入する場合は、保証からの逸脱量を別に記録する。
5. 零テンソルをrank-1零TTへ正規化する仕様なら、零テンソル判定を `eps_rel == 0` の早期returnより前に置く。順序が逆だと、零テンソルかつ `eps_rel == 0` の入力だけ別のrank規則になる。
6. end-to-endセルには保存出力があるが、`execution_count` は `null` である。保存出力だけから、現在のコードをカーネル再起動後に先頭から実行したとは判定できない。

以上から、保存出力は数値例として利用できるが、Notebookの完了条件とは分けて扱う。完了を確認する場合は、少なくとも統合関数、rank選択、零テンソル分岐を修正し、誤差率を変えた比較と境界ケースを含めてRestart Kernel後にRun Allする必要がある。このノートでは実行結果を新たに生成せず、保存状態の監査結果だけを記録している。

TT-rounding前後の誤差をdense復元なしで測る実装は、[[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]] へ続く。
