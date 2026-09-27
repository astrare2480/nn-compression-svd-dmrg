---
title: TT内積_Frobeniusノルム_距離のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - einsum
  - inner-product
  - Frobenius
  - distance
  - validation
---

# TT内積・Frobeniusノルム・距離のPyTorch確認

## このノートの位置づけ

このノートでは、TTコアをdense tensorへ復元せずに内積・Frobeniusノルム・TT間距離を計算するPyTorch実装を整理する。

理論は次の3ノートに分けている。

- environment縮約：[[00_基礎理論/01_数学基礎/02_テンソル代数/51_TT_MPSの内積とenvironment縮約]]
- ノルムと距離：[[00_基礎理論/01_数学基礎/02_テンソル代数/52_TT_MPSのFrobeniusノルムと距離]]
- 加減算とrank増加：[[00_基礎理論/01_数学基礎/02_テンソル代数/53_TT_MPSの加減算とrank増加]]

手を動かす学習用教材は [Notebook 13](../../notebooks/30_tt_mps/00_fundamentals/13_tt_inner_frobenius_norm_and_distance.ipynb) である。本ノートはNotebookの全文を複製せず、数式とコードの対応、代表的な数値結果、検証上の注意をまとめる。

---

## 1. 固定する実行条件

小さい例で丸め誤差まで確認するため、CPUとfloat64を使う。

```python
import torch


# 学習用の数値例を再現できるように乱数を固定する。
torch.manual_seed(0)

DTYPE = torch.float64
DEVICE = torch.device("cpu")
```

TTコアのshapeは一貫して

```text
(left_rank, physical_dim, right_rank)
```

とする。入力コアをin-placeで変更せず、戻り値はPyTorch tensorのまま保つ。

---

## 2. 入力contract

内積を取る前に、各TTについて次を確認する。

1. コア列が空でない。
2. 全コアが3階tensorである。
3. 各次元が1以上である。
4. 隣接コアのright rankとleft rankが一致する。
5. 左端と右端の境界rankが1である。
6. 1本のTT内でdtypeとdeviceが統一されている。

2本のTTの間ではさらに、

1. orderが同じである。
2. physical shapeが同じである。
3. dtypeとdeviceが同じである。

ことを確認する。AとBの内部TT-rankは異なってよい。

```python
def validate_tt_cores(cores: list[torch.Tensor]) -> None:
    """TTコア列のshape・bond・dtype・deviceを検証する。"""
    assert isinstance(cores, (list, tuple))
    assert len(cores) >= 1

    dtype = cores[0].dtype
    device = cores[0].device

    for k, core in enumerate(cores):
        assert isinstance(core, torch.Tensor)
        assert core.ndim == 3
        assert all(dim >= 1 for dim in core.shape)
        assert core.dtype == dtype
        assert core.device == device

        if k > 0:
            assert cores[k - 1].shape[2] == core.shape[0]

    assert cores[0].shape[0] == 1
    assert cores[-1].shape[2] == 1
```

教材では式を追いやすくするため `assert` を使えるが、再利用APIでは `ValueError` や `TypeError` と具体的なメッセージへ置き換える方がよい。

---

## 3. left environmentの1 site更新

理論上の更新式は

$$
E_k(\alpha_k,\beta_k)
=
\sum_{\alpha_{k-1},\beta_{k-1},i_k}
E_{k-1}(\alpha_{k-1},\beta_{k-1})
A^{(k)}_{\alpha_{k-1},i_k,\alpha_k}
B^{(k)}_{\beta_{k-1},i_k,\beta_k}.
$$

PyTorchでは

```python
env = torch.einsum(
    "ab,aic,bid->cd",
    env,
    core_a,
    core_b,
)
```

と書ける。添字は

$$
a=\alpha_{k-1},
\quad
b=\beta_{k-1},
\quad
i=i_k,
\quad
c=\alpha_k,
\quad
d=\beta_k
$$

に対応する。

例えば

```text
env:    (2, 3)
core_a: (2, 7, 4)
core_b: (3, 7, 5)
```

なら、出力は

```text
new_env: (4, 5)
```

になる。`a,b,i` は出力添字にないため縮約され、`c,d` が残る。

---

## 4. `tt_inner`の参照実装

```python
def tt_physical_shape(cores: list[torch.Tensor]) -> tuple[int, ...]:
    """TTが表すphysical shapeを返す。"""
    validate_tt_cores(cores)
    return tuple(int(core.shape[1]) for core in cores)


def tt_inner(
    cores_a: list[torch.Tensor],
    cores_b: list[torch.Tensor],
) -> torch.Tensor:
    """2本の実数値TTのFrobenius内積を0次元tensorで返す。"""
    validate_tt_cores(cores_a)
    validate_tt_cores(cores_b)

    assert len(cores_a) == len(cores_b), (
        f"order mismatch: d_A={len(cores_a)}, d_B={len(cores_b)}"
    )

    physical_a = tt_physical_shape(cores_a)
    physical_b = tt_physical_shape(cores_b)
    assert physical_a == physical_b, (
        f"physical shape mismatch: A={physical_a}, B={physical_b}"
    )

    assert cores_a[0].dtype == cores_b[0].dtype, (
        f"dtype mismatch: A={cores_a[0].dtype}, B={cores_b[0].dtype}"
    )
    assert cores_a[0].device == cores_b[0].device, (
        f"device mismatch: A={cores_a[0].device}, B={cores_b[0].device}"
    )

    # E_0=[1]を先頭コアと同じdtype/deviceで作る。
    env = torch.ones(
        (1, 1),
        dtype=cores_a[0].dtype,
        device=cores_a[0].device,
    )

    # E_0 -> E_1 -> ... -> E_d と左から更新する。
    for core_a, core_b in zip(cores_a, cores_b):
        env = torch.einsum(
            "ab,aic,bid->cd",
            env,
            core_a,
            core_b,
        )

    assert tuple(env.shape) == (1, 1)

    # 1x1行列を値を保った0次元tensorへ変換する。
    return env.squeeze()
```

複素MPSへ拡張する場合はbra側を共役する。

```python
env = torch.einsum(
    "ab,aic,bid->cd",
    env,
    core_a.conj(),
    core_b,
)
```

実数TTの教材へ無条件に `.conj()` を入れても値は変わらないが、実数と複素数で内積の定義が違うことを明示して実装を分けた方が学習上は分かりやすい。

---

## 5. `env.squeeze()`はnullではない

最終environmentはshape `(1, 1)` である。例えば

```python
env = torch.tensor([[3.25]], dtype=torch.float64)
inner = env.squeeze()

print(env.shape)    # torch.Size([1, 1])
print(inner.shape)  # torch.Size([])
print(inner)        # tensor(3.2500, dtype=torch.float64)
```

shape変化は

```text
(1, 1) -> ()
```

であり、値の個数は1個のままである。`torch.Size([])` はnullや `None` ではなく、axisを持たない0次元scalar tensorを表す。

$$
\begin{bmatrix}
3.25
\end{bmatrix}
\longrightarrow
3.25
$$

である。

`squeeze()` と `.item()` の違いは次である。

| 操作 | 結果 | autograd |
| --- | --- | --- |
| `env.squeeze()` | shape `()` の `torch.Tensor` | 計算グラフを保つ |
| `env.item()` | Pythonの `float` / `complex` | tensorの計算グラフから値を取り出す |

ライブラリ関数は0次元tensorを返し、表示やログが必要な呼び出し側だけで `.item()` を使う。

無引数 `squeeze()` はsize 1の全axisを消す。今回は最終shapeが数学的に `(1, 1)` と保証されるため安全だが、batch sizeが1の一般tensorへ無条件に使うとbatch axisまで消えるので注意する。

---

## 6. ノルムと距離の実装

```python
def tt_fro_norm(cores: list[torch.Tensor]) -> torch.Tensor:
    """実数値TTのFrobeniusノルムを返す。"""
    norm_sq = tt_inner(cores, cores)

    # 理論上は非負。平方根の前に微小な負の丸め誤差だけを0へ戻す。
    safe_norm_sq = norm_sq.clamp_min(0.0)
    return torch.sqrt(safe_norm_sq)


def tt_distance_sq(
    cores_a: list[torch.Tensor],
    cores_b: list[torch.Tensor],
) -> torch.Tensor:
    """実数値TT間のFrobenius距離の二乗を返す。"""
    inner_aa = tt_inner(cores_a, cores_a)
    inner_bb = tt_inner(cores_b, cores_b)
    inner_ab = tt_inner(cores_a, cores_b)

    distance_sq = inner_aa + inner_bb - 2.0 * inner_ab
    return distance_sq.clamp_min(0.0)


def tt_distance(
    cores_a: list[torch.Tensor],
    cores_b: list[torch.Tensor],
) -> torch.Tensor:
    """実数値TT間のFrobenius距離を返す。"""
    return torch.sqrt(tt_distance_sq(cores_a, cores_b))
```

`distance_sq ** 0.5` も数値上は同じだが、Notebook内で平方根を `torch.sqrt` に統一すると意図が読みやすい。

### 6.1 `clamp_min(0.0)`がすること

$$
\operatorname{clamp\_min}(x,0)=\max(x,0).
$$

```text
4.0     -> 4.0
0.0     -> 0.0
-2e-16  -> 0.0
```

理論上0の量が丸め誤差で微小に負になり、平方根が `nan` になるのを防ぐ。

ただし大きな負値を常に0へ潰してはいけない。評価用の堅牢な実装では、値のスケールから許容負値を決め、それを超えて負なら例外にする。単純な `clamp_min` は「正しい式をfloat64で計算したときの微小誤差を処理する」基本形である。

---

## 7. 小規模検証データ

physical shapeを

$$
(n_1,n_2,n_3)=(2,3,2)
$$

とする。AとBのrankをあえて変える。

$$
(r_0^A,r_1^A,r_2^A,r_3^A)
=(1,2,3,1),
$$

$$
(r_0^B,r_1^B,r_2^B,r_3^B)
=(1,3,2,1).
$$

コアshapeは

```text
A: (1, 2, 2), (2, 3, 3), (3, 2, 1)
B: (1, 2, 3), (3, 3, 2), (2, 2, 1)
```

である。environment shapeは

$$
\boxed{
(1,1)
\longrightarrow
(2,3)
\longrightarrow
(3,2)
\longrightarrow
(1,1)
}
$$

となる。

A/Bでrankを変えることは重要である。誤って同一rankを前提にした実装でも、同一rankのテストだけなら通ってしまう場合がある。

---

## 8. dense側の検証式

dense復元は小規模検証にだけ使う。

### 8.1 `torch.sum`による内積

```python
inner_dense = torch.sum(dense_a * dense_b)
```

`dense_a * dense_b` は要素積であり、`torch.sum` が全要素を足す。

例えば

$$
A=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix},
\qquad
B=
\begin{pmatrix}
10&20\\
30&40
\end{pmatrix}
$$

なら、

$$
A\odot B
=
\begin{pmatrix}
10&40\\
90&160
\end{pmatrix}
$$

であり、

$$
\operatorname{sum}(A\odot B)
=
10+40+90+160
=
300.
$$

`dim` を指定しない `torch.sum` は全axisを縮約し、shape `()` のtensorを返す。`dim=0` は各列、`dim=1` は各行に沿って和を取るため、全内積とは異なる。

### 8.2 denseノルム

```python
norm_a_dense = torch.linalg.vector_norm(dense_a)
```

`dim=None` の既定値では、全要素のvector normを計算する。

$$
\operatorname{vector\_norm}(\mathcal A)
=
\sqrt{
\sum_{i_1,\ldots,i_d}
\mathcal A_{i_1,\ldots,i_d}^2
}
=
\|\mathcal A\|_F.
$$

次も同じ値である。

```python
norm_a_dense = torch.linalg.vector_norm(dense_a.reshape(-1))
```

`reshape(-1)` は全要素を1次元へ並べ、長さを要素数から自動計算する。例えばshape `(2, 2, 2)` のtensorは8要素なのでshape `(8,)` になる。これはFrobeniusノルムを $\|\operatorname{vec}(\mathcal A)\|_2$ と見ることをコード上で明示したい場合に有用である。

さらに定義どおり

```python
norm_a_dense = torch.sqrt(torch.sum(dense_a ** 2))
```

としてもよい。

`torch.linalg.matrix_norm` は行列として見る2軸を前提にする。一般の3階以上のTT tensor全体のノルムには `vector_norm` または全要素二乗和を使う。

### 8.3 dense距離

```python
distance_sq_dense = torch.sum((dense_a - dense_b) ** 2)
distance_dense = torch.linalg.vector_norm(dense_a - dense_b)
```

これは

$$
\|\mathcal A-\mathcal B\|_F^2
=
\sum_{\boldsymbol i}
(\mathcal A_{\boldsymbol i}-\mathcal B_{\boldsymbol i})^2
$$

およびその平方根を直接計算する。

---

## 9. `torch.testing.assert_close`

TT-only計算とdense側の計算は演算順序が異なるため、浮動小数点でbit単位に完全一致するとは限らない。`==` ではなく、

```python
torch.testing.assert_close(
    actual,
    expected,
    atol=1e-12,
    rtol=1e-12,
)
```

を使う。判定条件は

$$
\boxed{
|\mathrm{actual}-\mathrm{expected}|
\le
\mathrm{atol}
+
\mathrm{rtol}\,|\mathrm{expected}|
}
$$

である。

例えば `actual=1.0000000000001`、`expected=1.0` なら差は $10^{-13}$、許容値は

$$
10^{-12}+10^{-12}|1|=2\times10^{-12}
$$

なので通る。`actual=1.001` なら差は $10^{-3}$ なので落ちる。

`expected=0` では相対項が0になるため、自己距離の検証には `atol` が不可欠である。大きな値では `rtol` がスケールに応じた許容値を与える。

最低限、次の5量を照合する。

```python
torch.testing.assert_close(inner_tt, inner_dense, atol=1e-12, rtol=1e-12)
torch.testing.assert_close(norm_a_tt, norm_a_dense, atol=1e-12, rtol=1e-12)
torch.testing.assert_close(norm_b_tt, norm_b_dense, atol=1e-12, rtol=1e-12)
torch.testing.assert_close(distance_sq_tt, distance_sq_dense, atol=1e-12, rtol=1e-12)
torch.testing.assert_close(distance_tt, distance_dense, atol=1e-12, rtol=1e-12)
```

---

## 10. Notebook 13の保存済み数値結果

Notebook 13の現在の保存状態では、PyTorch `2.11.0+cu128`、CPU、float64、seed 0で次が記録されている。

### 10.1 environment shape

```text
[(1, 1), (2, 3), (3, 2), (1, 1)]
```

これは理論上の

$$
E_k\in\mathbb R^{r_k^A\times r_k^B}
$$

と一致する。

### 10.2 内積

```text
inner_tt:    tensor(14.2433, dtype=torch.float64)
inner_dense: tensor(14.2433, dtype=torch.float64)
absolute difference: 1.7763568394002505e-15
```

### 10.3 Frobeniusノルム

```text
A TT / dense: 7.7673103257282525 / 7.767310325728252
absolute difference: 8.881784197001252e-16

B TT / dense: 5.70851544682925 / 5.70851544682925
absolute difference: 0.0
```

### 10.4 距離

```text
distance_sq_tt:    tensor(64.4317, dtype=torch.float64)
distance_sq_dense: tensor(64.4317, dtype=torch.float64)
absolute difference: 0.0

distance_tt:    tensor(8.0269, dtype=torch.float64)
distance_dense: tensor(8.0269, dtype=torch.float64)
absolute difference: 0.0
```

自己距離も

```text
||A - A||_F: 0.0
```

と保存されている。

これらは、保存された小規模例についてTT-only計算とdense定義がfloat64の丸め誤差範囲で一致したことを示す。

---

## 11. 不正入力テスト

現在の保存出力では次が確認されている。

```text
OK [order mismatch]
OK [physical mismatch]
OK [dtype mismatch]
SKIP [device mismatch]: CUDA なし
OK [broken bond]
OK [different TT-ranks allowed]: 14.243284453983861
```

ここから次を区別できる。

- order、physical shape、dtype、device、内部bond不整合はエラーである。
- AとBのTT-rankが異なること自体はエラーではない。
- CPU固定の教材ではCUDA不在時のdevice mismatchテストをskipしてよい。

テストセルのソースを変更した後は、古い出力が残っていないか注意する。コードと保存出力が食い違う場合、出力だけを根拠に完了判定してはいけない。

---

## 12. 保存状態の監査結果

現在のNotebook 13では、以前空だった距離検証セルが実装され、次が保存されている。

- 距離二乗と距離のTT/dense照合
- dtype mismatchを含む入力検証の成功
- `tt_distance` での `torch.sqrt`
- 第1 siteの変数名 `env1`
- scalar tensorのshapeを「null」ではなく `()` とするメッセージ

対象セルに保存出力上の例外は残っていない。一方、Notebook全体の `execution_count` は上から単調増加していない。このため、現在保存されたコードをカーネル再起動後に先頭からRun Allした事実までは、ファイルの保存状態だけから証明できない。

判定を分ける。

- **数式・実装内容**：内積、ノルム、距離、入力検証は現在のソースでそろっている。
- **保存済み数値結果**：主要な照合は成功している。
- **Restart Kernel後のRun All**：本ノート作成時には新たに実行していないため未確認である。

---

## 13. 何ができるようになったか

この実装により、dense復元なしで次を評価できる。

```text
TT inner product
TT Frobenius norm
TT-to-TT Frobenius distance
TT-rounding前後の絶対・相対誤差
圧縮前後の重み差
```

内積environmentはDMRGの環境縮約へ、距離は圧縮評価へ、PyTorchの0次元tensorを保つ設計はautogradへ接続する。

次のNN接続では、TT-matrixの定義、行列tensorization、TT-Linear forwardを扱う。既存の実装ノートは [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]] を参照する。

---

## 参考

- [PyTorch `torch.einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- [PyTorch `torch.squeeze`](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html)
- [PyTorch `torch.sum`](https://docs.pytorch.org/docs/stable/generated/torch.sum.html)
- [PyTorch `torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)
- [PyTorch testing](https://docs.pytorch.org/docs/stable/testing.html)
