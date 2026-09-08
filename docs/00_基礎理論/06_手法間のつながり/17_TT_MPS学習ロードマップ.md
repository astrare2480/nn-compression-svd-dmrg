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

理論docsは [[30_TT_MPSの定義]] 〜 [[34_基底変換とTT-rank不変性]]、実装は [[27_TT_MPS基礎のPyTorch実装]] と [[28_TT_MPS実装で使うPyTorch_Python操作メモ]] に分けている。

次の理論は **gauge freedom** である。

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

現在は

```text
TT定義
→ TT-rank / unfolding
→ TT-SVD
→ truncation / error
→ Schmidt対応
→ U ⊗ I / rank invariance
→ 小実装 00〜03
→ src化・回帰テスト
```

まで完了している。

次は

$$
\boxed{
\text{gauge freedom}
}
$$

から始める。

そこから

```text
gauge freedom
→ QR orthogonalization
→ canonical form
→ TT-rounding
→ TT/MPS operations
→ TT-matrix / MPO
→ Fashion-MNIST
→ ALS / one-site
→ two-site DMRG
```

へ進む。
