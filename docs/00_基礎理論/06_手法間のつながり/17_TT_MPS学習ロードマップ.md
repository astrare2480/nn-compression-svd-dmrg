---
title: TT・MPS学習ロードマップ
aliases:
  - TT MPS roadmap
  - TTからDMRGへの学習順
  - TT MPS実装ロードマップ
tags:
  - TT
  - MPS
  - DMRG
  - roadmap
  - FashionMNIST
---

# TT・MPS学習ロードマップ

> 第1・11節はTT-SVD基礎00〜03に加え、現在存在するGauge・左右QR・混合正準形・中心移動のNotebook 04〜07までの位置を示す。Notebookの存在と保存済み出力を、最新のRun Allや理解の完了と混同しない。

## サマリー

このプロジェクトでは、TT/MPSを「Fashion-MNISTの圧縮を早く動かすためだけの手法」として学ぶのではなく、

$$
\boxed{
\text{数値線形代数としてのTT}
\longleftrightarrow
\text{量子多体系としてのMPS}
\longrightarrow
\text{局所最適化・DMRG}
}
$$

という一本の流れとして理解する。

最終的に採用した方針は、

```text
理解
→ 小さいテンソルで検証
→ 次の理論
→ 演算・最適化
→ Fashion-MNISTで総合実験
```

である。

Fashion-MNISTのために理論を急いで消化するのではなく、各節を小さい「証明補助コード」で固定してから次へ進む。

---

## 1. 現在地

現在までに基礎として固定したのは、

- TTコアとbond index
- TT-rankとcut unfolding
- TT-SVDの逐次分解
- Schmidt分解との対応
- exact TT-SVDとtruncated TT-SVD
- 厳密rankと圧縮rank
- 局所SVD誤差とTT全体誤差の関係
- 第1 SVD後のremainderと元テンソルのcut rankの関係
- $L_2=U\otimes I_{n_2}$
- $X^{\langle2\rangle}=L_2B_{\mathrm{cut2}}$
- 列直交変換によるrank不変性
- 小テンソルでのPyTorch検証

である。

TT-SVDの基礎は [[30_TT_MPSの定義]] 〜 [[34_基底変換とTT-rank不変性]]、対応するPyTorch操作は [[27_TT_MPS基礎のPyTorch実装]] と [[28_TT_MPS実装で使うPyTorch_Python操作メモ]] に分けている。後続のGauge・左右QR・混合正準形・中心移動は [[36_TT_MPSのGauge自由度と左QR直交化]] 〜 [[41_TT_MPSの直交中心の移動]] と、基礎Notebook 04〜07に進んでいる。

これらの物理的意味は [[00_基礎理論/07_物理基礎/16_TT_MPSの物理解釈_縮約密度行列と局所写像]]、[[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]] で対応付ける。次の数学的な読み順は [[43_TT_MPSのSVD中心移動とSchmidt形]]、[[45_TT_MPSの単一ボンドSVD打ち切り]] であり、物理的意味は [[00_基礎理論/07_物理基礎/44_Schmidt打ち切りとボンド状態の物理的意味]] に置く。その後にTT-roundingと直接収縮へ進む。保存済みNotebook出力の確認は、Notebook全体を新たに実行したという意味ではない。

---

## 2. 最終採用ロードマップ

### 第1部：TTを数値線形代数として完成させる

1. TTの定義、TT-rank、unfolding
2. TT-SVDと厳密分解
3. truncationと誤差
4. TT-rankとSchmidt rankの対応
5. gauge freedom
6. QRによるleft/right orthogonalization
7. TT-rounding
8. TTの基本演算
   - 加算
   - 内積
   - ノルム
   - Hadamard積など
   - rank growth

この段階では、巨大なデータセットではなく

$$
2\times2\times2,
\qquad
2\times3\times2\times2
$$

程度の小テンソルで十分である。

確認したいのは、

- TT-SVDで厳密復元できる
- truncationでrankと誤差が変わる
- gauge変換でcoreは変わっても全体テンソルは不変
- QR直交化後も全体テンソルは不変
- roundingでrankを下げられる
- 内積・ノルムをdense計算と一致させられる

ことである。

---

### 第2部：MPSとして意味を与える

数値線形代数のTTを、物理のMPSへ翻訳する。

1. local physical index $i_k$ とbond index $\alpha_k$
2. Schmidt分解と各cutのbond dimension
3. left-canonical form
4. right-canonical form
5. mixed-canonical form
6. orthogonality center
7. gauge freedomの物理的意味
8. Schmidt係数、cut、entanglement
9. MPSの規格化
10. overlap・期待値の収縮

重要な対応は、

$$
\boxed{
\text{TT-rank}
=
\text{bond dimension}
=
\text{Schmidt rank}
}
$$

である。

また、後続では

$$
\boxed{
\text{QR直交化}
\longleftrightarrow
\text{canonical gaugeへの変換}
}
$$

という読み替えを固定する。

---

### 第3部：TT/MPSを演算対象にする

ここでは、分解して終わるのではなく、TT/MPSのまま計算する。

1. TT加算とrank growth
2. TTの内積とノルムの直接収縮
3. MPS overlap
4. 局所演算子と期待値
5. TT-matrix / MPO
6. MPO $\times$ MPS
7. TT-matrix $\times$ vector
8. 演算後のrank growth
9. TT-roundingによるrank制御
10. environment contraction

ここまで進むと、

```text
数値線形代数：TT-matrix × vector
物理：MPO × MPS
機械学習：圧縮Dense layerのforward
```

が同じ縮約問題として見える。

---

### 第4部：変分最適化からDMRGへ

TT/MPSを固定表現として使うだけでなく、coreを局所的に最適化する。

1. 変分MPSの考え方
2. environmentを固定した局所目的関数
3. one-site optimization / ALS
4. sweepの意味
5. one-site DMRG
6. two-site block $\Theta$
7. two-site DMRG
8. 二サイトテンソルをSVDで再分離
9. truncationによるadaptive bond dimension
10. DMRGとALSの対応
11. 数値線形代数・TT最適化としての読み替え

Two-site DMRGの基本形は、

```text
隣接2coreを結合
→ 局所問題を解く
→ SVDで2coreへ再分離
→ 小さい特異値をtruncate
→ bond dimensionを適応制御
```

である。

そのため、TT-SVDで学んだ

- reshape / unfolding
- SVD
- bond rank
- orthogonalization
- truncation

がそのまま基礎部品になる。

---

### 第5部：機械学習・Fashion-MNIST

Fashion-MNISTは、理論を急いで到達するゴールではなく、TT/MPSを圧縮・表現・最適化として検証する最初の総合実験場とする。

扱う候補は、

1. TT-matrix / MPOによるDense層表現
2. tensorizationの因数分解とindex順序
3. TT coreからのdirect forward contraction
4. post-training TT-SVD圧縮
5. TT coreのfine-tuning
6. TT-layerを最初から学習
7. MPS classifier
8. SVD / Tucker-2 / TT比較
9. 圧縮率・重み誤差・層出力誤差・accuracyの分離
10. DMRG的局所更新をNN損失へ応用できるかの検討

---

## 3. Fashion-MNISTの実験は二段階に分ける

TT-matrixを導入した後の初回実験は、正しさと効率を一度に解こうとしない。

### 段階A：denseへ戻して正しさを確認する

1. 学習済みMLPからweight $W$ を取り出す
2. tensorizeする
3. TT-SVDでTT-matrixへ分解する
4. TT coreから密行列 $\widehat W$ を再構成する
5. 通常のdense forwardで元weightと比較する

評価は少なくとも、

- weight reconstruction error
- layer output error
- classification accuracy
- TT parameter count

を分離する。

### 段階B：TT coreから直接forwardする

次に、密行列を作らずTT coreを直接縮約してforwardする。

この段階では、

- 中間Tensor shape
- contraction順序
- peak memory
- latency

も観察する。

小規模MLPでは、dense GEMMよりTT contractionが速くならない場合がある。したがって初回実験で優先するのは速度勝負ではなく、**圧縮・誤差・accuracy・fine-tuning回復量を分離して測ること**である。

---

## 4. TT-matrix実装前に固定する事項

Dense weight

$$
W\in\mathbb R^{N_{\mathrm{out}}\times N_{\mathrm{in}}}
$$

をTT-matrixへ変換するときは、まず

- どのlayerを対象にするか
- $N_{\mathrm{in}}$ をどう因数分解するか
- $N_{\mathrm{out}}$ をどう因数分解するか
- input/output modeをどの順序で組にするか
- rank列をどう選ぶか

を明示する。

TT-matrix coreは一般に

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k}
$$

の4階coreになる。

ここは通常のTT tensor core

$$
(r_{k-1},n_k,r_k)
$$

との違いを意識して学ぶ。

---

## 5. 理論ごとに小実装を挟む

コードを最後まで一切書かずに理論だけ進めるのではなく、節ごとに最小実装を置く。

| 理論 | 小実装で確認すること |
| --- | --- |
| TT-SVD | 分解・復元、unfolding shape、rank |
| gauge freedom | $G^{(k)}M$ と $M^{-1}G^{(k+1)}$ で全体不変 |
| QR直交化 | QR前後で全体不変、直交条件成立 |
| canonical form | left/right条件、center移動、norm保存 |
| TT-rounding | rank低下、捨てた特異値と誤差 |
| inner product / norm | dense計算とnetwork contractionの一致 |
| MPO作用 | dense matrix productとMPO/TT contractionの一致 |
| two-site更新 | 結合 → SVD再分離 → truncate |

現在の基礎Notebook 00〜03は、TT-SVDまでのこの方針を実行したものに相当する。

---

## 6. 基礎TT-SVDの実装境界

今回の基礎フェーズで実装した範囲は、

1. 3階Tensorのexact TT-SVD
2. 一般$d$階TT-SVD
3. TT coreからのdense reconstruction
4. sequential cut rankとの比較
5. max rankによるtruncation
6. parameter countとrelative error
7. $L_2=U\otimes I$ の数値検証
8. exact時のrank invariance
9. truncation後の $\widehat X$ とremainder間のrank invariance

である。

まだこの基礎Notebookへ入れないものは、

- gauge freedom以降の完全実装
- TT-rounding
- TT-matrix layer
- Fashion-MNIST DataLoader
- MPS classifier
- DMRG

である。

これらは後続理論と対応する別Notebookへ分ける。

---

## 7. 数学・物理・実装の三方向で到達を判定する

各トピックを「終えた」と判断するとき、暗記ではなく次を説明できるかを確認する。

### 数学

- 何をreshape / QR / SVD / contractionしているか
- rank・誤差・直交性が何を保証するか

### 物理

- Schmidt basis
- bond dimension
- canonical form
- environment

の言葉で何をしているか。

### 実装

- core shape
- どのaxisを縮約するか
- dimensionがどこへ移るか
- dense版との一致をどうテストするか

この三つがつながっていれば、後続のMPS/DMRGを別々の暗記事項にせずに済む。

---

## 8. Milestone

### Milestone 1：TT/MPS理論から演算へ

- gauge freedom
- QR orthogonalization
- canonical form
- TT-rounding
- TT-matrix / MPO
- direct contraction
- 小テンソルでの一致確認

### Milestone 2：圧縮実験

- Fashion-MNIST MLP
- post-training TT-SVD
- rank sweep
- tensorization順序比較
- compression ratio / weight error / output error / accuracy
- fine-tuning
- SVD / Tucker-2との比較

### Milestone 3：DMRG的最適化

- canonical formを実装で利用
- local optimization
- one-site / two-site sweep
- SVD再分離
- adaptive rank
- NN lossへの局所更新の適用検討

---

## 9. 時間見積りの扱い

会話途中では、Fashion-MNISTやDMRGまでの期間について複数の見積りを置いた。学習速度の前提を修正した後には、初回TT実験までを概ね1〜2週間、DMRG的TT core最適化までをさらに長い期間として見る案も出た。

ただし最終的には、**圧縮を急がない**方針へ変更したため、日数は主要な進捗指標にしない。

```text
何日経ったか
ではなく
数学・物理・実装の3方向で説明できるか
```

を区切りにする。

---

## 10. 読み物・資料の位置づけ

基礎TTでは、一冊を最初から最後まで読むより、必要な役割ごとに資料を使う。

- TTの原典・定義：TT形式、TT-rank、TT-SVD、roundingの辞書として使う
- Tensor Networkのレビュー：Tucker / TT / ML応用を横断して見る
- MPS / Schmidt分解の教材：bond indexを物理的に理解する
- PyTorch docs：SVD、reshape、stride、contiguous、kronなど実装contractを確認する

本を読むこと自体を進捗目標にせず、現在の疑問を解くための参照先として使う。

---

## 11. 現在から次へ

資料とNotebookの現在地は

```text
TT定義
→ TT-rank / unfolding
→ TT-SVD
→ truncation / error
→ Schmidt対応
→ U ⊗ I / rank invariance
→ 小実装 00〜03
→ src化・回帰テスト
→ Gauge・左右QRのNotebook 04〜05
→ Mixed-canonicalのNotebook 06
→ 中心移動のNotebook 07
```

である。04〜07には実装と保存済み出力があるが、ここでは最新のRun Allや本人の理解の完了までは判定しない。

次の学習上の接続は

$$
\boxed{
\text{直交中心の物理解釈} \longrightarrow \text{TT-rounding}
}
$$

である。Gauge自由度へ戻って最初からやり直すという意味ではない。

そこから

```text
ブロック基底・Schmidt係数・縮約密度行列
→ TT-rounding
→ TT/MPSの直接収縮・内積・ノルム
→ TT-matrix / MPO
→ Fashion-MNIST
→ ALS / one-site
→ two-site DMRG
```

---

## 12. 左右QRから混合正準形へ進む学習の接続

Gauge変換、左QR、第2コアまでの左直交化、左ブロック直交性、右端QRと右ブロックの読み順は

1. [[36_TT_MPSのGauge自由度と左QR直交化]]
2. [[37_TT_MPSの左ブロックと直交性の導出]]
3. [[38_TT_MPSの右QR直交化と右ブロック]]
4. [[39_TT_MPSの混合正準形への導入]]
5. [[40_TT_MPSの中心ノルムと内積の導出]]
6. [[41_TT_MPSの直交中心の移動]]
7. [[43_TT_MPSのSVD中心移動とSchmidt形]]
8. [[45_TT_MPSの単一ボンドSVD打ち切り]]

とする。39の混合正準形は、同じ元TTから $G_2^{[C]}=T_1G_2T_3^T$ を作る導入、40は中心のノルム・内積、41はQRによる中心移動、43はexact SVDとSchmidt形、45は単一ボンドの打ち切りを扱う。物理的解釈は [[00_基礎理論/07_物理基礎/16_TT_MPSの物理解釈_縮約密度行列と局所写像]]、[[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]]、[[00_基礎理論/07_物理基礎/44_Schmidt打ち切りとボンド状態の物理的意味]] に分ける。小行列で式を確認することと、Notebook全体を実行して検証することは区別する。

その次に、TT-rounding → TT/MPSの直接収縮 → TT-matrix/MPO → NN実験 → 局所最適化へ進む。これは学習順であって、全てを最初の重み近似の必須依存関係にするものではない。

### CNNへ適用する際の二段階

Fashion-MNIST MLPに加え、CNNのConv2d重みへのTT近似も候補になる。第5部のMLPとは別の適用先として分ける。

通常の重み順を

$$
W\in\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w}
$$

とすると、例えば $32\times16\times3\times3$ の4階テンソルをそのままTT-SVDできる。内部bondを $(r_1,r_2,r_3)=(4,4,2)$ に選ぶと、biasを除く格納要素数は

$$
\begin{aligned}
P_{\mathrm{dense}}&=32\cdot16\cdot3\cdot3=4608,\\
P_{\mathrm{TT}}&=1\cdot32\cdot4+4\cdot16\cdot4+4\cdot3\cdot2+2\cdot3\cdot1\\
&=128+256+24+6=414.
\end{aligned}
$$

これは指定rankでの**表現の格納量**の例であり、重み誤差・accuracy・速度を保証する数値ではない。mode順を変えるとcut rankや適切なrank設定も変わり得る。

段階AではTTコアから $\widehat W$ をdenseへ再構成し、元と同じConv2dで重み誤差、層出力誤差、accuracyを測る。ただし、denseの $\widehat W$ を通常の `nn.Conv2d` へ戻すだけでは、実行時の重み要素数や通常の畳み込みの理論MACsは減らない。「TTで少なく格納できる」と「圧縮層としてforwardする」は区別する。

段階BではTTコアを保持した因子化層・直接縮約を設計し、実際のparameter数、理論MACs、latency、peak memory、fine-tuningによる回復を別に測る。4階重みの任意のTT表現が、そのまま直列の小さなConv2dに置換できるとは限らない。

最初のdense重みからTTへの分解はTT-SVDであり、既存TTのrankを下げるTT-roundingとは異なる。初回の重み近似にtwo-site DMRGやTT-roundingを必須とはしない。二段階の具体的なCNN実装・実験は後続テーマとする。

### 64×32×3×3を、三つのtensorization順とcutまで追う

Conv2d重みのtensorization候補を具体的に比較する。
前の $32\times16\times3\times3$ の保存量の例とは別に、$64\times32\times3\times3$ の重みで対応を追う。
重みを

$$
W\in\mathbb R^{64\times32\times3\times3},
\qquad 64=8\cdot8,
\qquad 32=4\cdot8
$$

とする。この節だけ添字を0始まりにし、
$0\le o_1,o_2,i_2<8$、$0\le i_1<4$、$0\le h,w<3$ と置く。
元の出力・入力channelは、それぞれ

$$
o=8o_1+o_2,
\qquad i=8i_1+i_2
$$

である。最初の `reshape(8, 8, 4, 8, 3, 3)` は、
出力channelを二つ、入力channelを二つに分けるだけで、
出力と入力の脚をまだ交互には並べていない。

$$
T^A_{o_1,o_2,i_1,i_2,h,w}
=W_{8o_1+o_2,\,8i_1+i_2,\,h,w},
\qquad
\operatorname{shape}(T^A)=(8,8,4,8,3,3).
$$

残り二候補は、同じ $T^A$ の軸を並べ替えたものとして定義する。

$$
\begin{aligned}
T^B_{o_1,i_1,o_2,i_2,h,w}
&=T^A_{o_1,o_2,i_1,i_2,h,w},
&\operatorname{shape}(T^B)&=(8,4,8,8,3,3),\\
T^C_{o_1,h,o_2,w,i_1,i_2}
&=T^A_{o_1,o_2,i_1,i_2,h,w},
&\operatorname{shape}(T^C)&=(8,3,8,3,4,8).
\end{aligned}
$$

PyTorchでは、$T^B$ は `T_A.permute(0, 2, 1, 3, 4, 5)`、
$T^C$ は `T_A.permute(0, 4, 1, 5, 2, 3)` に対応する。
最初から別のshapeへ `reshape` するだけでは、この成分対応にはならない。
同じサイズ8の軸も、$o_1,o_2,i_2$ のどれなのかを区別する。

全要素数は、どの順でも

$$
\begin{aligned}
64\cdot32\cdot3\cdot3&=18432,\\
8\cdot8\cdot4\cdot8\cdot3\cdot3&=18432,\\
8\cdot4\cdot8\cdot8\cdot3\cdot3&=18432,\\
8\cdot3\cdot8\cdot3\cdot4\cdot8&=18432
\end{aligned}
$$

で変わらない。ただし、`permute` 後のstrideは変わり得て、
後続の `reshape` がコピーを必要とする場合がある。
値・要素数の保存と、物理的なstorage配置の保存は同じ主張ではない。
詳細は [[28_TT_MPS実装で使うPyTorch_Python操作メモ]] に置く。

#### 第2cutで左へまとめる元の添字が異なる

三つとも「先頭2軸でcutする」と言っても、左へまとめる物理的な添字は異なる。

$$
\begin{aligned}
T^A &: (o_1,o_2)\mid(i_1,i_2,h,w),
&T^A_{\mathrm{cut2}}&\in\mathbb R^{64\times288},\\
T^B &: (o_1,i_1)\mid(o_2,i_2,h,w),
&T^B_{\mathrm{cut2}}&\in\mathbb R^{32\times576},\\
T^C &: (o_1,h)\mid(o_2,w,i_1,i_2),
&T^C_{\mathrm{cut2}}&\in\mathbb R^{24\times768}.
\end{aligned}
$$

末尾の添字が最も速く進む順で行位置・列位置を書き下すと、
各行列の全成分は次の式で定まる。

$$
\begin{aligned}
(T^A_{\mathrm{cut2}})_{8o_1+o_2,\,72i_1+9i_2+3h+w}
&=W_{8o_1+o_2,\,8i_1+i_2,\,h,w},\\
(T^B_{\mathrm{cut2}})_{4o_1+i_1,\,72o_2+9i_2+3h+w}
&=W_{8o_1+o_2,\,8i_1+i_2,\,h,w},\\
(T^C_{\mathrm{cut2}})_{3o_1+h,\,96o_2+32w+8i_1+i_2}
&=W_{8o_1+o_2,\,8i_1+i_2,\,h,w}.
\end{aligned}
$$

例えば $(o_1,o_2,i_1,i_2,h,w)=(1,2,3,4,1,2)$ は、
元の `W[10, 28, 1, 2]` であり、三つのcut行列では

$$
(T^A_{\mathrm{cut2}})_{10,257}
=(T^B_{\mathrm{cut2}})_{7,185}
=(T^C_{\mathrm{cut2}})_{4,284}
=W_{10,28,1,2}
$$

へ対応する。
行位置だけでなく列位置も変わり、そもそも左右に分ける添字集合が変わっている。
これは [[29_TT_cutとPyTorchのreshape_Kronecker順序]] にある、
**同じcut内**の行・列の並べ替えによるrank保存とは別である。
第2cutの数学的rank上限も、それぞれ64・32・24で異なる。
実際のrankが必ず異なるという意味ではなく、各候補でcutの特異値と近似誤差を確認する。

#### 添字確認用の人工値で、行列の小さい領域を全要素表示する

ここからの値は学習済み重みではなく、上の対応を確認する人工値である。

$$
W_{o,i,h,w}=288o+9i+3h+w.
$$

これは `arange(18432).reshape(64, 32, 3, 3)` に対応する。
先ほどの成分は $288\cdot10+9\cdot28+3\cdot1+2=3137$ になる。
各cut行列の先頭2行・先頭3列だけを取り出すと、全6成分は

$$
\begin{aligned}
(T^A_{\mathrm{cut2}})_{0:2,\,0:3}
&=\begin{pmatrix}0&1&2\\288&289&290\end{pmatrix},\\
(T^B_{\mathrm{cut2}})_{0:2,\,0:3}
&=\begin{pmatrix}0&1&2\\72&73&74\end{pmatrix},\\
(T^C_{\mathrm{cut2}})_{0:2,\,0:3}
&=\begin{pmatrix}0&9&18\\3&12&21\end{pmatrix}
\end{aligned}
$$

となる。$0:2$、$0:3$ はPythonと同じ終端を含まないsliceであり、
これらの $2\times3$ は全cut行列のサイズではない。
$T^A$ の第2行は出力channelを一つ進めるため288増え、
$T^B$ の第2行は $i_1$ を一つ進めて入力channelが8増えるため $9\cdot8=72$ 増える。
$T^C$ では列を進めると $i_2$ が変わって9増え、行を進めると $h$ が変わって3増える。

元の重み順へ戻すときは、$T^B$ に `permute(0, 2, 1, 3, 4, 5)`、
$T^C$ に `permute(0, 2, 4, 5, 1, 3)` を適用して $T^A$ の順へ戻し、
その後で `reshape(64, 32, 3, 3)` を行う。
この人工値による一致は添字・逆操作の検証であり、最良tensorizationやaccuracy維持の実験結果ではない。

へ進む。

## 13. 512×256のTT-matrix候補を、3サイトの添字と格納量まで確認する

ここでは input 256からoutput 512への独立した重み行列

$$
W\in\mathbb R^{512\times256}
$$

を扱う。MLP `784 → 512 → 256 → 10` の中間層は、PyTorchの `weight=(out_features, in_features)` では $256\times512$ なので、この $512\times256$ 例をそのまま中間層のshapeとして流用しない。

因数分解を

$$
512=8\cdot8\cdot8,
\qquad
256=4\cdot8\cdot8,
$$

$$
(m_1,m_2,m_3)=(8,8,8),
\qquad
(n_1,n_2,n_3)=(4,8,8)
$$

とする。この節ではphysical添字を0始まりとし、元の行・列との対応を

$$
o=64o_1+8o_2+o_3,
\qquad
i=64i_1+8i_2+i_3,
$$

$$
W_{o,i}
=\mathcal W_{o_1,o_2,o_3,i_1,i_2,i_3}
$$

とする。例えば

$$
(o_1,o_2,o_3)=(1,2,3),
\qquad
(i_1,i_2,i_3)=(2,4,5)
$$

なら、

$$
o=64+16+3=83,
\qquad
i=128+32+5=165
$$

なので、対応要素は `W[83, 165]` である。

TT-matrixでは同じサイトの入出力脚を隣にするため、

$$
(o_1,o_2,o_3,i_1,i_2,i_3)
\longrightarrow
(o_1,i_1,o_2,i_2,o_3,i_3)
$$

へ並べ替え、サイトごとのsizeを $(32,64,64)$ としてTT-SVDへ渡す。PyTorchでは `reshape(8, 8, 8, 4, 8, 8)` の後に `permute(0, 3, 1, 4, 2, 5)` を行う。復元時は `permute(0, 2, 4, 1, 3, 5)` で元のgrouped順へ戻してから $512\times256$ へreshapeする。

境界bondを $\chi_0=\chi_3=1$ とすると、4階coreのshapeは

$$
\begin{aligned}
G^{[1]}&:(1,8,4,\chi_1),\\
G^{[2]}&:(\chi_1,8,8,\chi_2),\\
G^{[3]}&:(\chi_2,8,8,1).
\end{aligned}
$$

各要素は二つのbond和によって

$$
\begin{aligned}
&W_{64o_1+8o_2+o_3,\,64i_1+8i_2+i_3}\\
&\qquad\approx
\sum_{\alpha_1=1}^{\chi_1}
\sum_{\alpha_2=1}^{\chi_2}
G^{[1]}_{1,o_1,i_1,\alpha_1}
G^{[2]}_{\alpha_1,o_2,i_2,\alpha_2}
G^{[3]}_{\alpha_2,o_3,i_3,1}
\end{aligned}
$$

と書ける。打ち切らなければ等号、打ち切れば近似であり、biasはこの積とは別に出力へ加える。

重みだけの格納量は

$$
\begin{aligned}
P_{\mathrm{dense}}
&=512\cdot256=131072,\\
P_{\mathrm{TT}}
&=1\cdot8\cdot4\cdot\chi_1
+\chi_1\cdot8\cdot8\cdot\chi_2
+\chi_2\cdot8\cdot8\cdot1\\
&=32\chi_1+64\chi_1\chi_2+64\chi_2.
\end{aligned}
$$

$\chi_1=\chi_2=8$ なら、

$$
\begin{aligned}
P_{\mathrm{TT}}
&=256+4096+512=4864,\\
\mathrm{CR}_{\mathrm{weight}}
&=\frac{131072}{4864}
=\frac{512}{19}
\approx26.95.
\end{aligned}
$$

出力bias 512個を双方に残すなら、比較する比は $(131072+512)/(4864+512)$ である。これはshapeから求めた保存量であり、このbondでexactに表せること、accuracy維持、速度向上を示す実験結果ではない。rank候補を比べる際は、設定した最大rankだけでなく、実際の全bond・全core shape・再構成誤差を記録する。

## 14. 512×784のMPO候補を、4サイトの添字と格納量まで確認する

MPOまたはTT-matrixでは、行列の出力添字と入力添字をそれぞれ複数の添字へ分け、
同じサイトに属する入出力脚を一つのコアへ持たせる。ここではMPO層やsweepを実装せず、
後続実装で固定すべきshape、添字順、格納量、比較条件を明確にする。

### 16行16列の小行列：reshapeだけではpaired順にならない

$W\in\mathbb R^{16\times16}$ と $16=2\cdot2\cdot2\cdot2$ を考える。
0始まりの行・列を

$$
o=8o_1+4o_2+2o_3+o_4,
\qquad
i=8i_1+4i_2+2i_3+i_4,
\qquad
o_k,i_k\in\{0,1\}
$$

とすると、最初のreshapeの添字順は

$$
W(o,i)
=\mathcal W_{\mathrm{grouped}}
(o_1,o_2,o_3,o_4,i_1,i_2,i_3,i_4)
$$

である。各サイトで出力と入力を組にするには、次の並べ替えが必要になる。

$$
\mathcal W_{\mathrm{paired}}
(o_1,i_1,o_2,i_2,o_3,i_3,o_4,i_4)
=\mathcal W_{\mathrm{grouped}}
(o_1,o_2,o_3,o_4,i_1,i_2,i_3,i_4).
$$

PyTorchでは
`grouped = W.reshape(2, 2, 2, 2, 2, 2, 2, 2)` の後に
`paired = grouped.permute(0, 4, 1, 5, 2, 6, 3, 7)` とする。
例えば $o=10=(1,0,1,0)_2$、$i=5=(0,1,0,1)_2$ なら

$$
\begin{aligned}
W(10,5)
&=\mathcal W_{\mathrm{grouped}}(1,0,1,0,0,1,0,1)\\
&=\mathcal W_{\mathrm{paired}}(1,0,0,1,1,0,0,1).
\end{aligned}
$$

4個のpaired modeをそれぞれsize $2\cdot2=4$ としてTT分解した後、
各コアを $(\chi_{k-1},2,2,\chi_k)$ へ戻す。復元後は逆permuteで
grouped順へ戻してから $(16,16)$ へreshapeする。値と要素数は変わらないが、
添字の並べ方は変わり、`permute` 後の `reshape` がコピーを要する場合もある。
strideの説明は [[28_TT_MPS実装で使うPyTorch_Python操作メモ]]、
cut順の違いは [[29_TT_cutとPyTorchのreshape_Kronecker順序]] で扱う。

### 512行784列を、4サイトへそろえる

入力784から出力512への重み行列を

$$
W\in\mathbb R^{512\times784}
$$

とし、次のように分ける。

$$
512=8\cdot8\cdot8\cdot1,
\qquad
784=7\cdot7\cdot4\cdot4,
$$

$$
(m_k,n_k)=(8,7),(8,7),(8,4),(1,4).
$$

size 1の因子は、入出力のサイト数をそろえるための軸であり、要素を増やさない。
開放境界 $\chi_0=\chi_4=1$ とすると、全コアのshapeは

$$
\begin{aligned}
G^{[1]}&:(1,8,7,\chi_1),\\
G^{[2]}&:(\chi_1,8,7,\chi_2),\\
G^{[3]}&:(\chi_2,8,4,\chi_3),\\
G^{[4]}&:(\chi_3,1,4,1).
\end{aligned}
$$

従って、biasを除く格納要素数は

$$
\begin{aligned}
P_{\mathrm{MPO}}
&=\sum_{k=1}^{4}\chi_{k-1}m_kn_k\chi_k\\
&=1\cdot8\cdot7\cdot\chi_1
+\chi_1\cdot8\cdot7\cdot\chi_2
+\chi_2\cdot8\cdot4\cdot\chi_3
+\chi_3\cdot1\cdot4\cdot1\\
&=56\chi_1+56\chi_1\chi_2+32\chi_2\chi_3+4\chi_3.
\end{aligned}
$$

「bond dimension 8」とだけ記録せず、全mode、全bond、全コアshapeを記録する。
通常のbias $b\in\mathbb R^{512}$ を残すなら、総数には512を加える。
入力を $(7,7,4,4)$ にtensorizeしてコアを縮約し、$(8,8,8,1)$ の出力を
512次元へ戻してbiasを加える。この4階コアは、通常のTT tensorが持つ3階コアと異なり、
各サイトに入力脚と出力脚を一つずつ持つ。

tensorizationの候補には通常のflatten順、画像の行・列を分ける順、小ブロック、
bit分解、space-filling curveがある。どれが最良かは事前には確定しない。
同じ行列でもmodeの分け方・順序が変わると、cutの特異値や必要bondが変わり得る。

### 初期化・学習・比較を混同しない

初期化には、学習済みdense重みを分解する方法、最初からランダムなMPOコアを
学習する方法、初期dense重みを一度作って分解してからMPOとして学習する方法がある。
このロードマップでは最初に学習後圧縮を扱う。

MPOコアを `nn.Parameter` / `nn.ParameterList` へ登録し、全コアを通常の勾配降下で
fine-tuningすること自体はDMRGではない。forwardはdense重みを復元する版と、
コアを直接縮約する版を分け、後者では入力・出力添字と中間shapeを確認する。
MPOのbiasは通常の出力vectorとして残せるため、必ずMPO化する必要はない。

比較は同じparameter予算でaccuracyを見るか、同じaccuracy条件で必要parameter数を見る。
圧縮直後とfine-tuning後、fine-tuning epoch、seed、tensorization、全bond、学習時間、
推論時間を併記する。0.1または0.5 percentage pointなどの許容低下は設定値であり、
達成結果ではない。複雑な表現が単純な行列SVDを必ず上回るとは限らない。
多段構造や入力構造に合うtensorizationによって、同じparameter数でMPOがより高い
accuracyを保てる可能性は研究仮説であり、tensorization依存性とともに実験で確かめる。
また、重み格納量が少なくてもreshape、transpose、小演算、中間Tensor、kernel起動により
latencyが悪化する場合がある。この節ではDMRG、局所sweep、two-site更新の手順は扱わない。
