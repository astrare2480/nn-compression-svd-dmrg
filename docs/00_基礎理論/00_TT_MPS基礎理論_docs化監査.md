---
title: TT・MPS基礎理論 docs化監査
aliases:
  - TT MPS資料監査
  - TT MPS Perplexity監査
  - TT docs coverage audit
tags:
  - TT
  - MPS
  - audit
  - docs
---

# TT・MPS基礎理論 docs化監査

## 目的

元資料 `TT_MPS基礎理論.md` の内容が、TT/MPS基礎docsへ抜け漏れなく整理されているかを確認する。

> 2026-09-12：本監査の12ファイルと「gauge以降は後続」という判定は、上記の元資料・監査時点についての記録である。新しく添付された左右QRの対話資料は別の [[02_TT_MPS正準形_資料対応監査]] で監査し、36〜39と操作メモ・ロードマップへ追加した。元資料の監査対象数へ別資料の追加ノートを混ぜない。

元資料はPerplexityとの長い対話ログで、同じ疑問を角度を変えて何度も確認している。監査では**会話順をそのまま複製せず、最終的に確定した理解へ論点単位で統合する**。

ここでいうcoverageは、理論論点・結論・必要な導出やindex対応がdocsに反映されているかを指す。Perplexity資料にある具体行列や全要素表示を同じ細かさで再掲しているかは**説明粒度**として別に扱い、その簡略化だけをcoverage gapとは判定しない。

監査時点の元資料は、

```text
総行数                 : 30,907
Markdown見出し総数     : 862
トップレベル会話見出し : 207
```

である。

---

## 1. docs化したファイル

### 数学・テンソル代数

```text
docs/00_基礎理論/01_数学基礎/02_テンソル代数/
├── 30_TT_MPSの定義.md
├── 31_TT-rankとunfolding.md
├── 32_TT-SVD.md
├── 33_TT-SVDの打ち切りと誤差.md
├── 34_基底変換とTT-rank不変性.md
└── 35_TT_MPSの等長写像と射影.md
```

### PyTorch / Python

```text
docs/00_基礎理論/05_PyTorch実装/
├── 27_TT_MPS基礎のPyTorch実装.md
├── 28_TT_MPS実装で使うPyTorch_Python操作メモ.md
└── 29_TT_cutとPyTorchのreshape_Kronecker順序.md
```

### 手法間・物理・ロードマップ

```text
docs/00_基礎理論/06_手法間のつながり/
├── 15_TuckerとTTの比較.md
├── 16_TT_MPSの物理解釈_縮約密度行列と局所写像.md
└── 17_TT_MPS学習ロードマップ.md
```

合計12ファイルへ整理した。

### 追加した監査対象

- 35: $U^TU=I$ と $UU^T=P$、等長写像と直交射影、TT-SVD / MPSでの意味
- 29: TT cutとTucker mode unfoldingの区別、PyTorchのreshape順、$U\otimes I$ と置換行列の対応

---

## 2. 元資料の大区分と反映先

| 元資料のおおよその範囲 | 主題 | 反映先 | 状態 |
| --- | --- | --- | --- |
| 1〜2,023 | SVDからSchmidt分解、3サイトTT、TT core、境界rank | 30, 16 | reflected / merged |
| 2,024〜4,449 | Tucker vs TT、core shapeの反復確認、TT-SVD再説明 | 15, 30, 32 | reflected / merged |
| 4,450〜11,697 | TT-rank、cut unfolding、remainder $B$、複合添字、$L_2$、Kronecker積 | 31, 34 | reflected |
| 11,698〜14,585 | exact/truncated rank、特異値、局所誤差、全体誤差、直交性の訂正 | 33 | reflected / corrected-final |
| 14,642〜17,376 | 学習順、Fashion-MNISTの位置づけ、gauge以降、DMRGまでのロードマップ | 17 | reflected / future roadmap |
| 17,377〜21,624 | 基礎Notebook設計、`numel`、shape、SVD rank、$U$のshape、物理解釈 | 27, 28, 16 | reflected |
| 21,625〜26,412 | `tensordot`、再構成、list、`squeeze`、`tt_unfold`、TT-SVD骨格、`torch.eye`、$U^TU$ | 27, 28, 16, 35 | reflected |
| 26,413〜30,907 | 第2cut、`torch.kron`、stride/contiguous、行順/permutation、rank invariance、Notebook 03レビュー | 34, 27, 28, 29 | reflected / corrected-final |

---

## 3. 30_TT_MPSの定義.md

### 反映した論点

- TT coreとは何か
- 一般core shape

$$
G^{(k)}\in\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

- $r_0=r_d=1$ の意味
- `reshape(U, 1, n_1, r_1)` が3階Tensorである理由
- coreを内部bondで縮約して元Tensorを得る式
- physical index / bond index
- TTとopen-boundary MPSの対応
- 3サイトMPS/TT
- 係数行列のSVDを状態展開へ代入してSchmidt分解を得る途中式
- 左右Schmidt vectorとSVD特異ベクトルの対応
- 特異値とSchmidt係数

### 統合した重複

元資料では、

- 「TT coreって何？」
- 「$1,n_1,r_1$ じゃないの？」
- 「reshapeすると4階では？」
- 「最初のUは何？」
- 「3サイトを最初から」

を何度も聞き直している。

これらは最終理解である

$$
G^{(1)}_{1,i_1,\alpha_1}
=
U^{(1)}_{i_1,\alpha_1}
$$

へ統合した。

---

## 4. 31_TT-rankとunfolding.md

### 反映した論点

- 行列rankの復習
- 非ゼロ特異値本数との対応
- TT cut unfolding

$$
X^{\langle k\rangle}
\in
\mathbb R^{(n_1\cdots n_k)\times(n_{k+1}\cdots n_d)}
$$

- TT-rank

$$
r_k
=
\operatorname{rank}\left(X^{\langle k\rangle}\right)
$$

- rankのshape上限
- 複合添字
- 左右を結ぶ独立チャネル数としてのrank
- TT-rank / bond dimension / Schmidt rank
- exact rankと近似bond dimensionの区別
- numerical rank
- 第2 SVDのrankが元Tensorの第2cut rankになる理由への導入

---

## 5. 32_TT-SVD.md

### 反映した論点

3サイトについて、

```text
n1 × n2 × n3
→ n1 × (n2 n3)
→ SVD
→ G1: 1 × n1 × r1
→ remainder: r1 × n2 × n3
→ (r1 n2) × n3
→ SVD
→ G2: r1 × n2 × r2
→ G3: r2 × n3 × 1
```

を途中shape付きで保存した。

また一般$d$階について、

$$
M_k
=
\operatorname{reshape}
\left(
R^{(k-1)},
r_{k-1}n_k,
n_{k+1}\cdots n_d
\right)
$$

から、

$$
G^{(k)}
=
\operatorname{reshape}
\left(U_k,r_{k-1},n_k,r_k\right)
$$

とし、$\Sigma_kV_k^T$ をremainderへ渡す流れを反映した。

### 特に保持した注意

`tt_unfold(X,k)` に対応する元Tensorのcut unfoldingと、TT-SVD内部のremainder reshapeは役割が異なる。

---

## 6. 33_TT-SVDの打ち切りと誤差.md

### 反映した論点

- 厳密rank $r_k$
- 圧縮rank $\widetilde r_k$
- 「厳密rankの先は元から0」
- 「圧縮rankの先では非ゼロ特異値も人為的に捨てる」
- truncated SVDのFrobenius誤差
- 捨てた特異値の二乗和
- TT parameter count
- rank / storage / error trade-off
- 各TT-SVD段階の局所残差
- reshapeがFrobenius normを保存すること
- 等長写像がFrobenius normを保存する途中式
- 左interfaceと直交射影
- 誤差予算
- 各cut局所最良とTT全体の大域最良を混同しない注意

### 元資料内の訂正を反映

元資料では途中で

```text
E1 ⟂ E2
```

の説明が「core同士が直交する」と誤読され得る形になり、その後に再確認・訂正している。

最終docsでは、

- core同士が直接直交するという主張ではない
- SVDが作るleft interfaceの等長性
- 段階残差を元空間へ持ち上げたときの直交構造
- 実際の局所残差と許容誤差予算を区別する

という後の整理を採用した。

---

## 7. 34_基底変換とTT-rank不変性.md

### 反映した論点

- remainder $B$ から元の第2cutへ戻す式

$$
X^{\langle2\rangle}
=
L_2B_{\mathrm{cut2}}
$$

- 成分定義

$$
(L_2)_{(i_1,i_2),(\alpha_1,j_2)}
=
U_{i_1,\alpha_1}\delta_{i_2,j_2}
$$

- Kronecker積から

$$
L_2=U\otimes I_{n_2}
$$

を得る途中式
- shape確認
- $L_2^TL_2=I$ の完全な導出
- $L_2L_2^T$ は一般にidentityではなくprojection
- 左逆を使うrank equalityの両向き証明
- 複合添字
- $U\otimes I$ と $I\otimes U$
- permutation matrix / row order
- PyTorch reshape順と `torch.kron(U, I)` の対応
- exactとtruncation時のrank不変性の違い

### 重要な最終訂正

truncation後は、元の $X$ とremainderのrankを同一視しない。

比較対象は、truncation後に表現している

$$
\widehat X
$$

と、そのtruncated remainderである。

---

## 8. 27_TT_MPS基礎のPyTorch実装.md

### 反映した論点

- 元Notebook 00〜03とsrc版Notebookの役割分担
- `tt_unfold`
- `tt_svd_exact`
- `tt_svd`
- `tt_reconstruct`
- `tt_num_parameters`
- `_tt_svd_sweep`
- `truncated_svd`
- `full_matrices=False`
- `remainder.reshape(r_left * n_mode, -1)`
- `torch.diag(S) @ Vh`
- zero Tensorでbond dimensionを最低1にするcontract
- numerical rank
- `max_rank` validation
- float32 / float64 dtype contract
- TT core validation
- `torch.tensordot`によるreconstruction
- parameter count
- Notebook 00〜03の数値検証内容
- src回帰テストで固定したcontract

### 実装履歴の扱い

Notebook中の自作実装は学習履歴として残し、src化は削除・移動ではなく抽出として整理した。

---

## 9. 28_TT_MPS実装で使うPyTorch_Python操作メモ.md

元資料で逐次確認した細かい実装知識を、理論docsから分離して保存した。

### 反映した主な項目

- `torch.Size`
- `.shape`
- `.numel()`
- `.item()`
- shape上の最大rank / numerical rank / selected rank
- `reshape`
- `reshape(..., -1)`
- contiguousな連続axisをまとめる場合の順序
- `permute → reshape` が必要な場合
- Python `list`
- `append`
- `len`
- `enumerate`
- slice `[:-1]`
- list comprehension
- dictionary access
- list比較
- `torch.tensordot(..., dims=...)`
- `squeeze(0).squeeze(-1)`
- `torch.eye`
- `torch.diag`
- `.T`
- stride
- `.contiguous()`
- `torch.kron`
- PyTorchのreshape順

---

## 10. 15_TuckerとTTの比較.md

### 反映した論点

- TuckerとTTの式
- dense central core vs chain core
- HOSVDの独立mode SVD vs TTのsequential SVD
- Tucker multilinear rank vs TT cut rank
- parameter scaling

$$
P_{\mathrm{Tucker}}
\approx
r^d+dnr
$$

$$
P_{\mathrm{TT}}
\approx
2nr+(d-2)nr^2
$$

- 低階Tensor / ConvではTuckerが自然な場合
- 高階tensorization / DenseではTTが自然な場合
- TTがMPS / Schmidt / DMRGへつながる理由

---

## 11. 16_TT_MPSの物理解釈.md

### 反映した論点

- 「サイト1の固有ベクトル」という曖昧表現の修正
- Schmidt分解から縮約密度行列を導く途中式
- SVDから

$$
\rho_1
=
X^{\langle1\rangle}
\left(X^{\langle1\rangle}\right)^\dagger
=
U\Sigma^2U^\dagger
$$

を導く式
- $U$ の列と縮約密度行列固有vector
- 特異値 / Schmidt係数 / 密度行列固有値の関係
- $r_1=\operatorname{rank}(\rho_1)$
- product stateでrank 1になる例
- MPS coreの3本のindexの物理的意味
- physical index固定時の「行列の束」
- coreを局所写像として読む見方

---

## 12. 17_TT_MPS学習ロードマップ.md

### 反映した論点

元資料後半で最終的に確定したロードマップを保存した。

```text
TT基礎
→ gauge freedom
→ QR orthogonalization
→ canonical form
→ TT-rounding
→ TT/MPS operations
→ TT-matrix / MPO
→ Fashion-MNIST
→ ALS / one-site
→ two-site DMRG
```

加えて、

- Fashion-MNISTを急がず総合実験場とする最終方針
- 理論ごとに小さな証明補助コードを書く方針
- TT-matrix実験をdense再構成版とdirect contraction版に分ける方針
- Milestone
- 数学 / 物理 / 実装の三方向で理解を判定する方針
- 時間見積りを主要KPIにしない最終方針
- 参考資料の使い分け

を反映した。

---

## 13. 重複として統合した内容

元資料では、同じ疑問を理解が進むたびに聞き直している。これらは削除ではなく、**後のより正確・詳細な説明へ統合**した。

代表例：

- 第1coreがなぜ $(1,n_1,r_1)$ か
- `reshape(U,1,n1,r1)` が何階Tensorか
- $U$ のshapeと左特異vectorの本数
- $B$ に $i_1$ がないのにrankが保存される理由
- 複合添字とは何か
- $U\otimes I$ がなぜ出るか
- $U\otimes I$ と $I\otimes U$ の違い
- permutation matrixが必要な表記と不要な実装
- $L_2^TL_2=I$ と $L_2L_2^T$ の違い
- exact rankとtruncation rank
- TT-SVD誤差が二乗和で扱える理由
- `torch.kron` とcontiguous/stride

---

## 14. 後の訂正を優先して置き換えた内容

会話ログなので途中には仮説明や誤ったコード案も含まれる。docsではそれらを併記して混乱を残さず、**後に明示的に訂正された最終理解**を採用した。

代表例：

1. `relation_error` は単なるTensor差ではなくFrobenius normで評価する。
2. truncation実験では `X_hat_cut2` を本当にtruncated TTが表す $\widehat X$ から作る。
3. truncation後のrank invarianceは $X$ ではなく $\widehat X$ とtruncated remainderの間で議論する。
4. `truncated_svd` の $U$ が必ずnon-contiguousとは限らない。
5. `U\otimes I` と `I\otimes U` は行順を固定しないまま同一視しない。
6. core同士の直交性と、left interface / residualの直交構造を混同しない。

---

## 15. 今回本文へ展開せず、ロードマップへ保留した内容

次の理論は資料内に予告・概要があるが、今回の基礎範囲ではまだ学習本体に入っていない。

- gauge freedom
- left/right QR orthogonalization
- left/right canonical form
- mixed-canonical form
- orthogonality center
- TT-roundingの完全導出
- TT basic operationsの完全実装
- TT-matrix / MPOの完全導出
- direct TT-matrix forward
- ALS
- one-site DMRG
- two-site DMRG

これらは [[17_TT_MPS学習ロードマップ]] に順序・目的だけ保存し、30〜34の「学習済み理論」と混ぜない。

---

## 16. docs本文へそのまま転記しなかったもの

### Perplexityの脚注URL列

元資料の各回答末尾に付く多数の自動脚注URLは、学習内容そのものではなくPerplexity回答のsource metadataなので、docs本文へ一括転記していない。

必要な原典・参考資料の種類は、31や17の参考資料節へ役割ベースで残した。

### 会話運用上の文

次のようなchat運用だけの内容は知識docへは転記しない。

- 「次」だけの発言
- 別chatへの引き継ぎ文そのもの
- 「このchatは1日」といった進捗会話
- Cursor/ChatGPTの役割相談そのもの

ただし、そこから確定した**学習方針・ロードマップ・Notebook方針**は17と27へ反映した。

### 後に否定された仮説・一時的コード

誤った途中案を歴史資料として複製せず、最終訂正版のみdocsへ反映する。

---

## 17. Markdown / 数式表記監査

新規TT/MPS docsでは数式表記を次へ統一する。

```text
inline  : $...$
display : $$ ... $$
```

`\[ ... \]` は使わない。

数式では、

```text
X^{\langle 2 \rangle}
U \otimes I
n_1 \times n_2
A = B
```

のように読みやすいspacingを使う。

GitHub code searchで、新規docs群に `\[` および `<br />` の残存がないことを確認した。

---

## 18. 最終判定

### reflected

今回の学習済み範囲について、

- TT/MPS定義
- Schmidt分解
- TT core shape
- TT-rank / unfolding
- exact TT-SVD
- truncation / error
- $U\otimes I$
- rank invariance
- Tucker比較
- 物理解釈
- 等長写像 / 直交射影
- TT cut / mode unfolding / reshape / Kronecker積のindex順
- PyTorch / Python実装知識
- Notebook / src contract
- 最終ロードマップ

をdocsへ反映した。

### merged

同じ疑問の再質問は、後のより詳しい最終説明へ統合した。

### deferred

gauge freedom以降はロードマップのみ反映し、学習済み理論としては展開していない。

### excluded with reason

Perplexity脚注URL列、chat運用文、後に訂正された誤説明・仮コードは、そのままの形では転記していない。

---

## 結論

`TT_MPS基礎理論.md` の**現在までに学習した実質的な理論・実装内容と、今後へ確定した学習方針はdocs化済み**である。

この判定は理論論点のcoverageに関するものであり、具体行列や全要素表示がPerplexity資料と同じ説明粒度であることまでは意味しない。

元資料の会話上の重複を除き、後の訂正を優先した最終理解として、数学・物理・PyTorch・手法比較・ロードマップへ分離した。

次の新規理論docは、ロードマップどおり **gauge freedom** から開始する。
