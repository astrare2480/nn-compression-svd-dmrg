# Tensor Train / MPS

このディレクトリは、SVD → Tucker / HOSVD / HOOIの次に進む **Tensor Train（TT）/ Matrix Product State（MPS）** 学習用。

TT/MPS基礎の理論と小さいTensorでのPyTorch確認を進めている。NN重みのTT圧縮・学習実験は、ここで完了したとは扱わない。

## 前提として完了した内容

```text
SVD
→ 行列の低rank近似
→ 特異値truncation
→ Fine-tuning

Tucker / HOSVD
→ Tensorのmode
→ modeごとのrank
→ core / factor

HOOI
→ 固定rankでfactorを反復更新
→ sweep
→ convergence
```

特にHOOIで学んだ「初期分解 → 固定rank → 局所的な更新 → sweep → 収束判定」は、TT/MPSからDMRGへ進むときの重要な橋渡しになる。

## 学習NotebookとPyTorch確認ノートの対応

`notebooks/30_tt_mps/00_fundamentals/` では、理論を学習しながら小さいTensorで数値確認している。

`docs/40_TensorTrain_MPS/` では、Notebookの全文やTODO解答を複製せず、**確認した式・shape・代表的な数値結果・結論**をテーマ単位でまとめる。

| 学習Notebook | 主題 | PyTorch確認ノート |
| --- | --- | --- |
| 00〜02 | TT-SVD、TT-rank、rank truncationと誤差 | [[40_TensorTrain_MPS/03_TT_SVDとrank_truncationのPyTorch確認]] |
| 03〜05 | 基底変換、Gauge自由度、左/右QR直交化 | [[40_TensorTrain_MPS/04_基底変換_Gauge自由度_左右直交化のPyTorch確認]] |
| 06 | mixed-canonical form、中心ノルム、中心摂動の等長性 | [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]] |
| 07 | QRによるorthogonality centerの移動 | [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] |
| 08 | SVD center move、Schmidt形 | [[40_TensorTrain_MPS/05_SVD中心移動とSchmidt形のPyTorch確認]] |
| 09 | truncated SVD、単一bond rank truncation | [[40_TensorTrain_MPS/06_単一ボンドSVD打ち切りのPyTorch確認]] |

この対応は「Notebook 1本につきdocs 1本」ではなく、内容が連続するNotebookは1つの検証ノートへまとめる方針とする。

## 現在のPyTorch確認ノート

- [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]]
- [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]]
- [[40_TensorTrain_MPS/03_TT_SVDとrank_truncationのPyTorch確認]]
- [[40_TensorTrain_MPS/04_基底変換_Gauge自由度_左右直交化のPyTorch確認]]
- [[40_TensorTrain_MPS/05_SVD中心移動とSchmidt形のPyTorch確認]]
- [[40_TensorTrain_MPS/06_単一ボンドSVD打ち切りのPyTorch確認]]

## 学習予定

1. Tensorization / reshape
2. TT / MPSのcore shape
3. TT rank / bond dimension
4. TT-SVD
5. reconstructionとrelative error
6. parameter数・圧縮率
7. canonical form / orthogonality center
8. SVD center move / Schmidt form
9. 単一bond truncationと誤差
10. Eckart–Young–Mirskyによる単一bond最適性
11. rank決定とTT rounding
12. NN weightへの対応
13. 必要ならCIFAR-10等でSVD / Tuckerとの比較
14. 局所2-site更新とDMRGへの接続

## 評価軸

SVD / Tuckerで使った評価軸を可能な限り維持する。

```text
weight / tensor relative error
parameters
MACs
圧縮直後accuracy
decomposition time
Fine-tuning後accuracy
```

TT rank / bond dimensionを変えたときも、圧縮率だけでなくtask性能と分けて評価する。

基礎Notebookでは、上記に加えて次を数値確認する。

```text
reconstruction error
orthogonality / Gram error
center norm
Schmidt-state orthogonality
discarded singular-value energy
single-bond truncation error
```

## ノート設計

これまでと同様に、

```text
Notebook
→ 自作して理解
→ shapeと途中式を確認
→ 小さいTensorでsanity check

docs/40_TensorTrain_MPS
→ Notebookで確認した重要結果をテーマ単位で整理

src
→ 学習後に再利用処理だけ共通化
```

とする。

理論ノートでは、reshape / unfolding / SVD / truncate / core / reconstruction / errorを途中式から確認する。

直交中心の移動は [[00_基礎理論/01_数学基礎/02_テンソル代数/41_TT_MPSの直交中心の移動]]、ブロック状態としての意味は [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]]、PyTorchでのshape・QR・収縮の確認は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] を参照する。中心だけの摂動と等長性を検証する場合は [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]] へ進む。既存の学習NotebookのTODOはこの実装ノートでは変更しない。

## 先に読む

- [[00_基礎理論/06_手法間のつながり/14_低ランク学習からテンソルネットワークへの発展]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[06_Tucker基礎実装検証/README]]
- [[README_実装編]]

## 次

TT / MPSでbond dimensionとchain型Tensor表現を理解した後、[[50_DMRG/README]] へ進む。
