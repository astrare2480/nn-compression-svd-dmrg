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

## 12. 次フェーズ

Notebook 14はTT-matrixのdense reconstructionとdirect forwardを扱い、保存済み小規模例ではdense reconstruction、direct contraction、`F.linear`がfloat64の丸め誤差水準で一致している。ただしFresh Run Allは未確認である。

Notebook 15はdense `nn.Linear`とのforward等価性を、自分でTODOを実装しながら学ぶ教材であり、現在は完了済み結果として扱わない。

詳細：[[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]、[[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]]

## 関連

- [[08_TT_MPS基礎実装検証/README]]
- [[00_基礎理論/README]]
- [[90_src設計/README]]
- [[50_DMRG/README]]
