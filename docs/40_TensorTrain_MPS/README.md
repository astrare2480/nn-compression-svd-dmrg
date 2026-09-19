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

## 学習予定

1. Tensorization / reshape
2. TT / MPSのcore shape
3. TT rank / bond dimension
4. TT-SVD
5. reconstructionとrelative error
6. parameter数・圧縮率
7. NN weightへの対応
8. 必要ならCIFAR-10等でSVD / Tuckerとの比較
9. 局所2-site更新とDMRGへの接続

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

## ノート設計

これまでと同様に、

```text
Notebook
→ 自作して理解
→ shapeと途中式を確認
→ 小さいTensorでsanity check

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
