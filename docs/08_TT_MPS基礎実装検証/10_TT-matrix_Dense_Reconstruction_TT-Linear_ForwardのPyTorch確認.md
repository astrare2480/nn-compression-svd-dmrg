---
title: TT-matrix Dense Reconstruction TT-Linear ForwardのPyTorch確認
tags:
  - TT-matrix
  - TT-Linear
  - PyTorch
  - dense-reconstruction
  - tensorization
  - validation
---

# TT-matrix・Dense Reconstruction・TT-Linear ForwardのPyTorch確認

## このノートの位置づけ

このノートは、[Notebook 14](../../notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb) で扱ったPyTorch操作、shape contract、保存済みの数値結果、残る確認事項を整理する。

理論は次のノートへ分けている。

- 定義とKronecker積：[[00_基礎理論/01_数学基礎/02_テンソル代数/54_TT-matrixの定義とKronecker積表現]]
- dense重みのtensorization：[[00_基礎理論/01_数学基礎/02_テンソル代数/55_dense重みのTT-matrix tensorizationとTT-SVD初期化]]
- dense reconstruction：[[00_基礎理論/01_数学基礎/02_テンソル代数/56_TT-matrixのdense reconstruction]]
- 右から左へのforward縮約：[[00_基礎理論/01_数学基礎/02_テンソル代数/57_TT-Linear_forwardの縮約とshape]]

[[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]] は、全コアを一つの`einsum`へ渡す再利用可能な`nn.Module`例を扱う。本ノートはそれとは別に、右から左へ一siteずつ縮約し、途中shapeを追う学習用実装を扱う。

---

## 1. 固定した数値条件

小さい例で丸め誤差まで比較するため、Notebookでは次を使った。

```python
import torch
import torch.nn.functional as F


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

device = torch.device("cpu")
dtype = torch.float64
```

保存済み出力に記録されている環境は次である。

```text
PyTorch: 2.11.0+cu128
default dtype: torch.float64
device: cpu
```

CUDA buildのPyTorchでも、今回のtensorはCPU上で計算している。

## 2. TT-matrixコアの入力contract

各コアの軸順は

```text
(left_rank, out_mode, in_mode, right_rank)
```

へ固定する。最低限、次を検査する。

1. `cores` が空でない`list`または`tuple`である。
2. `len(cores) == len(out_modes) == len(in_modes)` である。
3. 各コアが4階tensorである。
4. `core.shape[1] == out_modes[k]` である。
5. `core.shape[2] == in_modes[k]` である。
6. 隣接コアのright rankとleft rankが一致する。
7. 左端rankと右端rankが1である。
8. 教材の数値条件としてCPU、float64である。

rank列は

```python
ranks = [
    int(cores[0].shape[0]),
    *[int(core.shape[3]) for core in cores],
]
```

で得られる。TT-matrixの保存要素数は

```python
num_parameters = sum(core.numel() for core in cores)
```

である。

Notebookのランダム3 site例は

```text
G1: (1, 2, 2, 2)
G2: (2, 3, 2, 3)
G3: (3, 2, 3, 1)
TT ranks: [1, 2, 3, 1]
TT parameters: 62
dense shape: (12, 12)
dense parameters: 144
```

であった。保存要素数だけの比は

```text
144 / 62 = 2.322580645...
```

である。この比は保存要素数の比較であり、速度や学習精度の結果ではない。

2 siteのflat index確認では、`divmod`を使って逆対応を表示できる。

```python
# i = i1 * m2 + i2 の逆変換。
i1, i2 = divmod(i, m2)

# j = j1 * n2 + j2 の逆変換。
j1, j2 = divmod(j, n2)
```

例えば $(m_1,m_2)=(2,3)$ なら、行indexは

```text
0 -> (0, 0)   1 -> (0, 1)   2 -> (0, 2)
3 -> (1, 0)   4 -> (1, 1)   5 -> (1, 2)
```

と対応する。`divmod(i, m2)`の商が外側index $i_1$、余りが最後に速く変化する内側index $i_2$ である。

## 3. dense reconstructionの実装対応

左端コアから始め、現在の最終bond軸と次コアの先頭bond軸を縮約する。

```python
# (1, m1, n1, r1) -> (m1, n1, r1)
current = cores[0].squeeze(0)

for core in cores[1:]:
    # currentの最後のbondと、coreの左bondを縮約する。
    current = torch.tensordot(
        current,
        core,
        dims=([-1], [0]),
    )

# 右端rank 1だけを落とす。
interleaved = current.squeeze(-1)
```

ここで`clone()`は入力コアを変更しないことを明示するためには使えるが、上の`torch.tensordot`、`squeeze`、`permute`、`reshape`はいずれも元コアへのin-place更新ではない。そのため、正しさのために全コアをcloneする必要はない。

3 site例の保存済みshapeは

```text
after core 0: (2, 2, 2)
after core 1: (2, 2, 3, 2, 3)
after core 2: (2, 2, 3, 2, 2, 3, 1)
interleaved:  (2, 2, 3, 2, 2, 3)
dense W:      (12, 12)
```

であり、理論上の軸順と一致した。

## 4. `permute`と`reshape`を分ける

interleavedからgroupedへの一般のpermutationは

```python
perm = (
    list(range(0, 2 * d, 2))
    + list(range(1, 2 * d, 2))
)
grouped = interleaved.permute(*perm)
W_dense = grouped.reshape(product(out_modes), product(in_modes))
```

$d=3$ では

```text
interleaved axes: (i1, j1, i2, j2, i3, j3)
perm:             [0, 2, 4, 1, 3, 5]
grouped axes:     (i1, i2, i3, j1, j2, j3)
```

である。Notebookでは

```text
interleaved.shape: (2, 2, 3, 2, 2, 3)
grouped.shape:     (2, 3, 2, 2, 2, 3)
```

を確認し、同じmulti-indexに対応する両tensorの値が一致した。

## 5. 一要素のbond縮約

選んだindexは

```text
(i1, i2, i3) = (1, 0, 1) -> flat i = 7
(j1, j2, j3) = (0, 1, 2) -> flat j = 5
```

である。4階コアを直接`"a,ab,b->"`へ渡すのではない。physical indexを固定した後のsliceを使う。

```python
A1 = G1[0, i1, j1, :]   # (r1,)     = (alpha1,)
A2 = G2[:, i2, j2, :]   # (r1, r2) = (alpha1, alpha2)
A3 = G3[:, i3, j3, 0]   # (r2,)     = (alpha2,)

value_from_cores = torch.einsum(
    "a,ab,b->",
    A1,
    A2,
    A3,
)
```

`a`と`b`は元の4階コア全体の軸ではなく、slice後に残ったbond軸を表す。保存済みshapeは

```text
A1.shape: (2,)
A2.shape: (2, 3)
A3.shape: (3,)
```

であり、保存済みrunでは

```text
W[7, 5] = -0.5731065395929638
```

として、明示的なbond縮約と一致した。

## 6. scalar tensorと`assert_close`

`W[i, j]`も、出力添字を空にした`einsum`の結果も0次元tensorである。

```text
shape: torch.Size([])
```

これは`None`や空tensorではなく、値を一つ持つscalar tensorである。比較には`.item()`でPython `float`へ変換せず、tensorのまま渡す。

```python
torch.testing.assert_close(
    W[i, j],
    value_from_cores,
    rtol=1e-12,
    atol=1e-12,
)
```

これにより値だけでなく、shape、dtype、deviceも検査できる。不一致なら`AssertionError`が発生し、後続行は実行されない。成功時は何も表示しないため、教材ではassert後に`PASS`を表示すると確認しやすい。

## 7. 右から左へのTT-Linear実装

forward内部でdense reconstructionを呼ばない。入力

```text
x: (B, n1 * ... * nd)
```

を

```text
state: (B, n1, ..., nd)
```

へreshapeし、最後のコアから処理する。

```python
# 最後のコアの右rank 1を落とす。
last_core = cores[-1].squeeze(-1)

# ndを消し、mdとr_{d-1}を作る。
state = torch.einsum(
    "...j,pij->...ip",
    state,
    last_core,
)
```

中央コアでは、Pythonの`cores[k]`が数学上のsite $k+1$ を表すという約束で、入力軸をbond直前へ移してから縮約する。

```python
for k in range(len(cores) - 2, 0, -1):
    state = state.movedim(
        k + 1,
        state.ndim - 2,
    )

    state = torch.einsum(
        "...ja,pija->...ip",
        state,
        cores[k],
    )
```

最初のコアは左端rankが1なので別処理にする。

```python
# n1を現在bondの直前へ移す。
state = state.movedim(1, state.ndim - 2)

# j1、alpha1、size 1の左端bondを消す。
state = torch.einsum(
    "...ja,pija->...i",
    state,
    cores[0],
)
```

この実装ではoutput modeが逆順に蓄積されるため、batch以外を逆順へ戻す。

```python
d = len(cores)
perm = [0] + list(range(d, 0, -1))
state = state.permute(*perm)
y = state.reshape(x.shape[0], product(out_modes))
```

`einsum`では添字文字列とoperand順を一致させる。`state`と`core`の順を交換する場合は、`"...ja,pija->...ip"`も`"pija,...ja->...ip"`へ交換する。

## 8. 3 site forwardのshape

設定

```text
out_modes = (2, 3, 2)
in_modes  = (2, 2, 3)
B = 4
```

では、理論上のshape遷移は

```text
(4, 2, 2, 3)
-> core 3: (4, 2, 2, 2, 3)
-> core 2: (4, 2, 3, 2, 2)
-> core 1: (4, 2, 3, 2)
-> flatten: (4, 12)
```

である。Notebookの現在の関数は`verbose`引数を受け取るが、中間shapeをprintする分岐は未実装である。出力shape `(4, 12)` と後続の数値一致は保存されているが、このセル出力だけから上記の全中間shapeを直接観測したとは扱わない。

## 9. dense・TT・`F.linear`の三者比較

同じrandomコア、入力、biasを使った保存済み結果は

```text
|y_tt - y_dense|_max:    5.329070518200751e-15
|y_tt - y_f_linear|_max: 5.329070518200751e-15
|y_dense - y_f_linear|_max: 3.552713678800501e-15
```

であり、すべて

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-12,
    atol=1e-12,
)
```

を通った。これは保存済みの同一run内で、

- TTコア直接縮約
- `x @ W_tt.T + bias`
- `F.linear(x, W_tt, bias)`

がfloat64の丸め誤差程度で一致したことを示す。学習済み重みの圧縮精度やRun All済みであることを示す結果ではない。

## 10. rank 1と`torch.kron`

2 site、rank 1では

```python
G1 = A.reshape(1, m1, n1, 1)
G2 = B.reshape(1, m2, n2, 1)
W_kron = torch.kron(A, B)
```

とする。Notebookでは

```text
W_tt.shape: (4, 4)
W_kron.shape: (4, 4)
OK: W_tt ≈ kron(A, B)
|y_tt - F.linear|_max: 8.881784197001252e-16
```

を確認した。これにより、dense reconstructionだけでなくforwardもKronecker積の特別ケースと整合した。

## 11. dense重みからTT-SVD入力を作る

実装の本体は

```python
grouped = W.reshape(*out_modes, *in_modes)

perm = []
for k in range(d):
    perm.append(k)
    perm.append(d + k)

interleaved = grouped.permute(*perm)

q_modes = tuple(
    m_k * n_k
    for m_k, n_k in zip(out_modes, in_modes)
)

A = interleaved.reshape(*q_modes)
```

である。`range(d)`は`0`から`d - 1`までを返す。`perm`へ入れるのはaxis sizeではなく元axis番号である。

Notebookでは

```text
out_modes = (2, 3, 2)
in_modes  = (2, 2, 3)
q_modes   = (4, 6, 6)
```

に対し、

```text
W[i, j]       = 0.8613672523333044
A[a1,a2,a3]   = 0.8613672523333044
|W - A|_abs   = 0.0
```

という要素一致が保存されている。

## 12. 現在のNotebookに残る実装上の注意

保存済みNotebookを教材の完成証明とは扱わない。現在確認できる注意点は次である。

1. `dense_to_tt_matrix_tensor`のセルは`execution_count`が`null`である一方、後続セルにはその関数を使った保存済み出力がある。カーネル状態を初期化した`Run All`の証明にはならない。
2. 同関数には重複したvalidationがあり、`assert condition, print(...)`を使う箇所がある。`print`の戻り値は`None`なので、例外メッセージとして不適切である。再利用コードでは具体的な文字列を持つ`ValueError`または`AssertionError`にする。
3. docstringにある`verbose`表示は現在の関数本体に実装されていない。
4. TT-Linear側は`bias=None`を型として許すが、validationで無条件に`bias.shape`を読むため、biasなしでは失敗する。`bias is not None`の分岐が必要である。
5. TT-Linear側のCPU/device contractは、コア検証と入力検証で一貫させる必要がある。
6. `bias`のdtypeとdeviceも入力・コアと一致することを確認する必要がある。
7. コア検証は $d=1$ を許すが、現在の右から左の関数は「最後のコア」と「最初のコア」を別々に処理するため、1 siteでは同じコアを二度処理してしまう。`d >= 2`をcontractにするか、1 site専用分岐を追加する必要がある。
8. modeについては正値だけでなく整数であることも検査した方が、`reshape`時のエラーを前段で説明できる。
9. 学習用Notebookなので、これらはユーザーが手を動かすTODOとして残し得る。docsの記録だけでNotebookを完成コードへ置き換えない。

## 13. この段階で確認できたこと

確認済みの範囲は次である。

- 4階TT-matrixコアのshapeとbond接続
- コアからのdense reconstruction
- 一要素の明示bond縮約
- interleavedからgroupedへのpermute
- denseを作らない右から左のTT-Linear forward
- randomコアに対するdense、TT、`F.linear`の数値一致
- rank 1とKronecker積の一致
- dense重みからTT-SVD入力tensorへの要素対応

未実施・未証明の範囲は次である。

- Notebookの新しいカーネルでのRestart/Run All
- backwardとgradient検証
- optimizerを用いた学習
- dense重みへのTT-SVD初期化の実行
- rank truncation後の重み誤差と層出力誤差
- 実測メモリ、速度、task accuracy

したがって、この段階はTT-matrixの添字・shape・forwardのsanity checkであり、NN圧縮実験の完了ではない。
