---
title: TT・MPS基礎実装の確認結果
aliases:
  - TT MPS 基礎実装まとめ
  - TT fundamentals 検証結果
tags:
  - TT
  - MPS
  - PyTorch
  - 実装検証
  - src化
---

# TT・MPS基礎実装の確認結果

## このノートの目的

`notebooks/30_tt_mps/00_fundamentals/`で学習したTT/MPS基礎について、Notebookを1冊ずつ開かなくても、次を追跡できる状態にする。

- どの数式・shape・数値関係を確認したか
- 代表的な保存済み数値結果
- どこまでsrcのPublic APIへ移したか
- Notebookだけに残した学習・実験コード
- 現時点で完了と呼べる範囲と、未確認の範囲

理論導出やコード全文は複製せず、詳細はテーマ別ノートへリンクする。

## 対象範囲

基礎フェーズとして総括する対象はNotebook 00〜13である。

```text
00〜02  TT-SVD、TT-rank、rank truncation
03〜05  基底変換、gauge自由度、左右QR直交化
06〜08  mixed-canonical form、center移動、Schmidt形
09〜10  単一bond打ち切り、Eckart–Young–Mirsky最適性
11     TT-SVDの誤差上界
12     TT-rounding
13     TT内積、Frobeniusノルム、距離の学習検証
```

Notebook 14・15のTT-matrix / TT-Linearは次フェーズとして分ける。保存済み確認結果はあるが、本ノートでは基礎フェーズの完了条件へ含めない。

## 証拠レベルの区別

この章では、次の3種類を混同しない。

1. **現行srcの自動テスト**：現在の実装をpytestで再実行した結果。
2. **Notebookの保存済み結果**：Notebookファイル内に残る出力。現在のコードを新しいkernelで再実行した証明ではない。
3. **理論・実装設計ノート**：式、shape、検証方法を整理したもの。Notebook実行済みとは限らない。

特にNotebook 08と13では、保存済み出力とFresh Run Allを区別する注意が既存ノートに残っている。したがって、本ノートは「全Notebookを2026-09-29にRestart Kernel → Run Allした」という主張をしない。

## 1. TT-SVD・TT-rank・圧縮率と誤差

詳細：[[08_TT_MPS基礎実装検証/03_TT_SVDとrank_truncationのPyTorch確認]]

Notebook 00の3階Tensorでは、TT core shapeが

```text
(1,2,2), (2,3,4), (4,4,1)
```

となり、再構成誤差は約$5.96\times10^{-15}$だった。rankを切り捨てないTT-SVDがfloat64の丸め誤差水準で元Tensorを再構成することを確認した。

Notebook 01の4階Tensorでは、TT-rankと対応するcut unfolding rankがともに

```text
(2,4,2)
```

となり、relative Frobenius errorは$6.414\times10^{-16}$だった。一方、この小さい例ではdense 24要素に対してTT 48要素となり、TT表現が常に圧縮になるわけではないことも確認した。

Notebook 02では、bond rankを大きくすると誤差が減る一方、parameter数が増えた。保存済み例ではmax rank 8でrelative errorは`0.294857`まで下がったが、parameter ratioは`1.1250`となりdense表現を上回った。

## 2. 基底変換・gauge自由度・左右直交化

詳細：[[08_TT_MPS基礎実装検証/04_基底変換_Gauge自由度_左右直交化のPyTorch確認]]

Notebook 03〜05では次を確認した。

- 可逆・直交基底変換は対応するunfolding rankを変えない。
- bondへ$M$と$M^{-1}$を挿入するgauge変換は、個々のcoreを変えても全Tensorを変えない。
- 左QRでは$Q$を左直交coreとし、$R$を右隣coreへ吸収する。
- 右QRでは右展開行列の転置へQRを適用し、$Q^\mathsf{T}$を右直交core、$R^\mathsf{T}$を左隣coreへ吸収する。

右QRの保存済み例では、local absorption errorが約$2.44\times10^{-15}$、global reconstruction errorが約$6.21\times10^{-15}$であり、局所積と全Tensorが丸め誤差水準で保存された。

## 3. Mixed-canonical formとorthogonality center

詳細：[[08_TT_MPS基礎実装検証/02_混合正準形の中心摂動と等長性のPyTorch確認]]、[[08_TT_MPS基礎実装検証/01_直交中心移動のPyTorch確認]]

Notebook 06・07では次を確認した。

- centerより左のcoreは左直交、右のcoreは右直交となる。
- mixed-canonical環境では全TensorのFrobeniusノルムをcenter coreへ局所化できる。
- center coreだけを摂動したとき、等長な左右環境は局所摂動ノルムを全体ノルムへ保存する。
- QRの余りを隣接coreへ吸収することで、dense値を保ったままcenterを移動できる。

Notebook 07の保存済み例では、centerを右へ移した再構成誤差が約$9.57\times10^{-15}$、左直交性誤差が約$3.20\times10^{-16}$だった。左へ移した例でも再構成誤差は約$7.13\times10^{-15}$、右直交性誤差は約$4.84\times10^{-16}$だった。

## 4. SVD center moveとSchmidt形

詳細：[[08_TT_MPS基礎実装検証/05_SVD中心移動とSchmidt形のPyTorch確認]]

Notebook 08では、二サイト中心TensorをSVDし、特異値をbond上へ明示するSchmidt形を確認した。

```text
mixed-canonical center
→ center行列をreduced SVD
→ 左特異ベクトルを左正規coreへ戻す
→ 特異値をbondまたは右側へ保持
→ 左右Schmidt状態の直交性とnorm identityを確認
```

保存済み結果はfloat64の丸め誤差水準で再構成・直交性・norm identityを満たす。ただし既存ノートが記録しているとおり、現在のNotebookソースには小例と本実験で`r1, n2, r2`を再利用する箇所があり、この保存出力だけではFresh Run All成功を証明できない。

## 5. 単一bond打ち切りとEYM最適性

詳細：[[08_TT_MPS基礎実装検証/06_単一ボンドSVD打ち切りのPyTorch確認]]

Notebook 09では、mixed-canonical環境でcenter行列をrank 3からrank 2へ打ち切り、次を確認した。

- bond rankと隣接core shapeが実際に縮む。
- 全TensorのFrobenius誤差が捨てた特異値の2-normと一致する。
- 二乗誤差が捨てた特異値二乗和と一致する。
- 保持成分と捨てた成分の直交性からPythagoras型ノルム分解が成り立つ。

Notebook 10ではEckart–Young–Mirsky定理を単一bondへ接続した。保存済み例の局所行列は特異値`[6.0, 2.0, 0.25]`を持ち、rank 2 truncated SVDの最小誤差は`0.25`だった。randomなrank 2候補の最良誤差は約`6.0892`で、この下界を破らなかった。

mixed-canonical環境では、局所誤差`0.25000000000000006`と全Tensor誤差`0.25`の差が約$5.55\times10^{-17}$だった。実際にbond rankを2へ縮めたTTとの再構成差も約$1.44\times10^{-15}$だった。

## 6. TT-SVDの誤差上界

詳細：[[08_TT_MPS基礎実装検証/07_TT-SVDの準最適性と誤差上界のPyTorch確認]]

Notebook 11では、元Tensorの各cut unfoldingから求める教科書上の基本誤差上界を確認した。固定rank$(2,4,2)$の保存済み例では、実測誤差`13.627066726245229`に対して上界は`18.15524433133907`であり、実測誤差が上界以下となった。

ここで、Notebookの基本誤差上界と現行srcの`tt_svd_error_bound`は意味を区別する。

- Notebook 11：元Tensorの各cut unfolding tailから求める教科書上の上界。
- `tt_svd_error_bound`：`tt_svd_ranks`と同じ逐次SVDと数値rank判定を再現し、実際に捨てる局所tailを合成した実装対応上限。

## 7. TT-rounding

詳細：[[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]]

Notebook 12では、right-to-left QR sweepの後にleft-to-right truncated SVD sweepを行うTT-roundingを確認した。保存済み例では、入力rank`[4,3]`が`[2,2]`へ縮み、dense relative errorは約$4.82\times10^{-16}$、要求上限は約$1.48\times10^{-7}$だった。

零Tensorは物理shapeを保ったrank-1 TTへなり、$d=1$では内部bondを処理せずdense誤差0となる保存結果もある。

現行srcでは、局所予算を二乗せずnorm空間でrank選択し、極端な値でのoverflow / underflowが離散rank判定へ入り込まないようにしている。

## 8. TT内積とFrobeniusノルム

詳細：[[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]]

Notebook 13では、A/Bで異なる内部bond rankを持つTTを使い、left environmentを

```text
(1,1) → (2,3) → (3,2) → (1,1)
```

と更新した。保存済み結果では、TT-only内積とdense内積の差が約$1.78\times10^{-15}$、AのFrobeniusノルム差が約$8.88\times10^{-16}$、Bの差が0だった。

距離についてもNotebookの小規模例ではdense定義と一致したが、安定なPublic APIとしては公開していない。近接・大規模・gauge違いのTTで、精度、有限gradient、計算量を同時に保証できる方式が確立していないためである。

## 9. 現行srcの責務

現行の再利用APIは次へ分かれる。

```text
tt.py
→ tt_unfold
→ tt_svd_exact / tt_svd / tt_svd_ranks
→ tt_svd_error_bound
→ tt_reconstruct
→ tt_num_parameters / tt_physical_shape / tt_ranks

tt_canonical.py
→ tt_canonicalize / tt_move_center
→ tt_canonicality_errors

tt_rounding.py
→ tt_bond_singular_values
→ tt_truncate_bond / tt_round

tt_contraction.py
→ tt_inner / tt_fro_norm
```

`tt_distance_sq`と`tt_distance`はPublic APIへ含めない。詳細な延期理由は[[90_src設計/07_既知の制約と拡張方針]]を参照する。

次はsrcへ入れず、学習・検証コードとしてNotebook側に残す。

- gauge変換の実験コード
- denseな左右Schmidt状態の構築
- random TT生成
- environment shape履歴
- Kronecker積によるrank不変性確認
- `expect_fail`等の教材用表示helper

## 10. src回帰テスト

2026-09-29に現行working treeで実行した全pytestは次の結果だった。

```text
547 passed
0 failed
```

Core API v1のdocs-syncは`4 passed`、`git diff --check`にも問題はなかった。これはローカル実行結果であり、GitHub CIや全NotebookのFresh Run Allを意味しない。

## 11. 現在の到達点

Notebook 00〜13に対応するTT/MPS基礎について、次の流れを理論・小規模数値例・src回帰テストへ接続できた。

```text
TT-SVD
→ rankと圧縮誤差
→ gauge自由度
→ 左右canonical form
→ orthogonality center
→ Schmidt spectrum
→ single-bond truncation
→ TT-SVD error bound
→ TT-rounding
→ TT inner product / Frobenius norm
```

ただし、到達点は「基礎アルゴリズムとPublic API contractが揃った」という意味である。全NotebookのFresh Run All、実データ上のNN圧縮精度、TT-matrix層の学習、DMRG-like optimizationまで完了したという意味ではない。

## 12. TT-matrix基礎とNN適用章への接続

Notebook 14はTT-matrixのdense reconstructionとdirect forwardを扱い、保存済み小規模例ではdense reconstruction、direct contraction、`F.linear`がfloat64の丸め誤差水準で一致している。ただしFresh Run Allは未確認である。

Notebook 15はdense `nn.Linear`とのforward等価性を、自分でTODOを実装しながら学ぶ教材である。現在のソースにはtruncationなし2-core TT-SVD、weight再構成、direct contraction、三者比較が実装され、保存済み出力では最大絶対誤差 $1.1102230246251565\times10^{-15}$ でforwardが一致している。ただし一部セルの`execution_count`が`null`であり、現在の全ソースを新しいカーネルでFresh Run Allした証明ではない。学習済みFashion-MNIST MLPの置換結果としても扱わない。

詳細：[[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]、[[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]、[[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]]

NN適用前の一般的なtoy確認として、$(20,12)$の重みを2-core・rank 2へ打ち切り、

$$
\Delta Y=X\Delta W^{\mathsf T}
$$

と

$$
\|\Delta Y\|_F
\le
\|X\|_F\|\Delta W\|_2,
\qquad
\|\Delta Y\|_F
\le
\|X\|_2\|\Delta W\|_F
$$

を保存出力で確認した。実測$\|\Delta Y\|_F$は約4.814、上界は約8.592と11.712だった。一方、主要な中間2 cellの`execution_count`が`null`なので、現行ソースのFresh Run All成功とは扱わない。

詳細：[[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]、[[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]

続くReLUのtoy Notebookでは、4種類の符号ケースを含む $2\times4$ 行列の全8要素について

$$
|\Delta H_{ij}|
\le
|\Delta Z_{ij}|
$$

を保存出力で確認した。random Linear例の保存出力では、

$$
\|\Delta H\|_F
=
0.337250011229,
$$

$$
\|\Delta Z\|_F
=
0.519605165542,
$$

$$
\|X\|_F\|\Delta W\|_2
=
0.893184613722
$$

であり、

$$
\|\Delta H\|_F
\le
\|\Delta Z\|_F
\le
\|X\|_F\|\Delta W\|_2
$$

が成り立った。ただし、別形式の上界cellは`execution_count=null`であり、Notebook全体のFresh Run Allは未確認である。

詳細：[[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]、[[08_TT_MPS基礎実装検証/30_PyTorchのvector_norm_matrix_normとtorch_linalg]]

続く第2 Linearのtoy Notebookで使うPyTorch操作、shape、独立assert、norm APIは、[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]に整理した。実験設定、保存済み数値、scale sweepとその考察は、適用・実験側の[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]へ分ける。

1 sampleのclassification marginとargmax安定性に進むtoy Notebookでは、dense top class、strongest competitor、$\ell_\infty$ logits error、worst-case方向による $2\varepsilon$ のgap縮小、common shiftを確認する。PyTorch操作と直接条件は[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]、保存済み結果と考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]に分ける。これは手入力したtoy logits errorの検証であり、学習済みmodelを実TT-rankで圧縮した結果ではない。

続くtoy Notebookでは、固定した第1 Linearの重み摂動からbatch logits errorを作り、$\|\delta l_n\|_\infty\le\|\delta l_n\|_2\le\|\Delta L\|_F\le\|X\|_F\|\Delta W_1\|_2\|W_2\|_2$を経て、1 sampleのmargin certificateへ接続する。PyTorchのshape、norm API、scalar boolean、assertは[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、固定行列、保存済み数値、考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]に分ける。この重み摂動も手入力したproxyであり、実TT-rank打ち切り結果ではない。

さらにminimum marginへ進むtoy Notebookでは、margin vector、bottleneck sample、$U<\gamma_{\min}/2$、全sampleのsample-wise条件、dense/compressed prediction vectorの一致を確認する。`argmax`、`topk`、minimumの値とindex、訂正済みcertificateは[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、保存済み数値と考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]に分ける。これは1つのbatch内のprediction agreementであり、ground-truth correctnessやaccuracyの保証ではない。

学習済みFashion-MNIST MLPの`fc1`置換、activation比較、logits比較は基礎実装から分離し、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]に置く。

## 13. NN適用側との境界

rank sweep、test accuracy、fine-tuning、性能比較などの未完了項目は、この基礎実装章ではなく [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]] で管理する。ReLUの1-Lipschitz性は [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]、Linearからhidden activationまでの誤差伝播は [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]、第2 Linearからlogitsまでの理論は [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]、classification marginとargmax安定性は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]、batch errorからsample-wise certificateへの接続は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]、minimum marginによるbatch内全sampleの同時保証は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]、toy PyTorch確認は [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]、[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]、[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、toy実験の設定・結果・考察は [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]] に収録している。ただし、学習済みFashion-MNIST MLPの打ち切りrank、後続logits、prediction、accuracyまで実装検証済みという意味ではない。

## 関連

- [[08_TT_MPS基礎実装検証/README]]
- [[00_基礎理論/README]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
- [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]]
- [[90_src設計/README]]
- [[50_DMRG/README]]
