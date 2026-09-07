---
title: TuckerとTTの比較
aliases:
  - Tucker vs TT
  - TuckerとTensor Train
  - TuckerとMPS
tags:
  - Tucker
  - TT
  - TensorDecomposition
  - TensorNetwork
  - 比較
---

# TuckerとTTの比較

## サマリー

TuckerとTTはどちらも高階テンソルを低rank構造で表すが、「情報をどこに置くか」が大きく異なる。

- Tucker：各modeをfactor matrixで縮め、全mode間の関係を中央のdense coreへ集める。
- TT：中央の巨大coreを持たず、各modeの小さな3階coreをbondで鎖状につなぐ。

この構造差により、rankの意味、保存量、高階化したときのスケーリング、適した利用場面が変わる。

---

## 1. 式で比較

元テンソルを

$$
X_{i_1\cdots i_d}
\in
\mathbb R^{n_1\times\cdots\times n_d}
$$

とする。

### Tucker

$$
\boxed{
X_{i_1\cdots i_d}
\approx
\sum_{a_1,\ldots,a_d}
\mathcal C_{a_1\cdots a_d}
U^{(1)}_{i_1,a_1}
\cdots
U^{(d)}_{i_d,a_d}
}
$$

factorは

$$
U^{(k)}\in\mathbb R^{n_k\times R_k},
$$

中央coreは

$$
\mathcal C
\in
\mathbb R^{R_1\times\cdots\times R_d}.
$$

### TT

$$
\boxed{
X_{i_1\cdots i_d}
\approx
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,1}
}
$$

各coreは

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k},
\qquad
r_0=r_d=1.
$$

---

## 2. 構造の違い

```text
Tucker

 i1      i2                  id
 |       |                    |
U(1)    U(2)                U(d)
 \       |                  /
      dense central core C
```

Tuckerでは、全mode間の相互作用を中央core

$$
\mathcal C_{a_1\cdots a_d}
$$

がまとめて持つ。

一方TTは

```text
 i1         i2                    id
 |          |                      |
G(1) --r1-- G(2) --r2-- ... -- G(d)
```

となり、相関を

$$
\alpha_1,\ldots,\alpha_{d-1}
$$

というbondへ分散して保持する。

---

## 3. 分解手順の違い

HOSVD型のTuckerでは、各mode factorは元の同じテンソルから独立に求める。

```text
X → mode-1 unfolding → SVD → U^(1)
X → mode-2 unfolding → SVD → U^(2)
...
```

全factorを得た後、

$$
\mathcal C
=
X
\times_1U^{(1)T}
\times_2U^{(2)T}
\cdots
\times_dU^{(d)T}
$$

と中央coreへ射影する。

TT-SVDは異なる。

```text
X
→ 第1cutをSVD
→ U^(1)をcoreとして固定
→ ΣV^Tだけを次へ渡す
→ 次のphysical indexと前のbondをまとめて再SVD
→ ...
```

つまりTTは**逐次的**であり、前段で得たbond indexを次段へ引き継ぐ。

---

## 4. rankの意味が違う

### Tucker rank

Tuckerのmultilinear rankは各mode-n unfoldingのrankである。

$$
R_k
=
\operatorname{rank}(X_{(k)}).
$$

これは「第 $k$ modeだけ」と「その他すべて」を分けるrankである。

### TT-rank

TT-rankは左から順にまとめたcut unfoldingのrankである。

$$
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right).
$$

$$
(i_1,\ldots,i_k)
\mid
(i_{k+1},\ldots,i_d)
$$

というbipartitionを測る。

したがって、Tucker rankとTT-rankは同じ「rank」という名前でも、見ている行列化が異なる。

---

## 5. パラメータ数

簡単のため全mode sizeを $n$、Tucker/TT rankを概ね $r$、階数を $d$ とする。

### dense

$$
P_{\mathrm{dense}}=n^d.
$$

### Tucker

中央coreが $r^d$、factorが $d$ 個で各 $nr$ なので、

$$
\boxed{
P_{\mathrm{Tucker}}
\approx
r^d+dnr
}
$$

となる。

### TT

端のcoreは概ね $nr$、中央coreは $nr^2$ なので、

$$
\boxed{
P_{\mathrm{TT}}
\approx
2nr+(d-2)nr^2
=
O(dnr^2)
}
$$

である。

このため、rankが一定程度に抑えられるなら、TTは階数 $d$ に対してほぼ線形に増える。一方Tuckerは中央core $r^d$ が高階で大きくなりやすい。

---

## 6. 3階・4階ではTuckerも扱いやすい

TTの $O(dnr^2)$ という形は高階テンソルで特に強い。一方、3階や4階程度でmodeごとの意味が明確ならTuckerの中央coreはまだ扱いやすく、factorごとの圧縮解釈も直感的である。

例えば画像・CNN weightの

$$
C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w
$$

のような4階テンソルでは、出力channelと入力channelだけを縮めるTucker-2が自然な場合がある。

TTは、Dense weightを多数のinput/output modeへtensorizeして高階化する場合や、本来から高階の状態・関数を扱う場合に鎖構造の利点が出やすい。

---

## 7. 物理的な違い

Tuckerは一般の多線形低rank分解として使われ、中央coreが全mode間の結合をまとめて持つ。

TTは開放境界MPSと同じ構造なので、各bondが

$$
(1,\ldots,k)\mid(k+1,\ldots,d)
$$

という切断を自然に持つ。

このためTT-rankは物理的にbond dimension / Schmidt rankとして読める。canonical form、entanglement、DMRGへ直接つながるのはTT/MPS側の特徴である。

---

## 8. どちらもSVDを使うが役割が違う

Tucker/HOSVDでは、各mode unfoldingから「そのmodeを表すfactor basis」を求める。

TT-SVDでは、各bipartitionでSVDし、左側のbasisを現在coreへ取り出し、残りを右へ渡す。

したがって

```text
Tucker:
各modeの基底を同じ元Tensorから集める
→ 中央coreへ情報を集約

TT:
左から順にbasisを確定する
→ 情報をbondへ段階的に分散
```

という違いがある。

---

## 9. まとめ表

| 観点 | Tucker | TT / MPS |
| --- | --- | --- |
| 基本構造 | factor matrices + dense central core | 3階coreのchain |
| rank | mode-n unfolding rank | sequential cut unfolding rank |
| 情報の置き場所 | 中央coreへ集約 | bondへ分散 |
| 高階化 | core $r^d$ が増えやすい | rank一定なら概ね $O(dnr^2)$ |
| 分解 | 各modeを元Tensorから独立SVD | remainderを左から逐次SVD |
| 物理解釈 | 一般の多線形分解 | MPS、Schmidt、bond dimension |
| NNでの自然な例 | Conv Tucker-2 | 高階tensorized Dense / TT-matrix |

---

## 10. このプロジェクトでの位置づけ

このリポジトリでは

```text
SVD
→ Tucker / HOSVD / HOOI
→ TT / MPS
→ gauge / canonical form
→ MPO / TT-matrix
→ DMRG
```

と段階的につなぐ。

Tuckerで学んだ

- unfolding
- SVD
- rank truncation
- Frobenius誤差
- factor basis

はTTでも生きる。一方、TTでは新たに

- sequential cut
- bond index
- chained cores
- basisを右へ渡す逐次構造
- MPS / Schmidt分解

が加わる。
