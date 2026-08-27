# DMRGへの接続

このディレクトリは、SVD → Tucker / HOSVD / HOOI → TT / MPSで学んだ低rank・Tensor Networkの考え方を、DMRGへ接続するための章。

現時点では**DMRGの独立実装・NN圧縮実験はまだ未着手**。既存の接続ノートは概念・学習ロードマップとして扱う。

## 現在の位置

```text
SVD
完了
↓
Tucker / HOSVD / HOOI
完了
↓
TT / MPS
次の学習フェーズ
↓
DMRG
その後
```

TT / MPSの準備と学習順は [[40_TensorTrain_MPS/README]] を参照。

## 既存ノート

- [[50_DMRG/01_SVDからDMRGへのつながり]]

このノートは、SVD低rank近似からMPS / MPO、bond dimension、DMRG-like sweepへ進む概念的な橋渡し。将来のDMRG実装結果そのものではない。

## DMRGで確認する予定の要素

```text
MPS / TT表現
bond dimension
2-site local tensor
局所最適化
SVDによる再分割・truncate
left-to-right / right-to-left sweep
convergence
```

NNへ適用する段階では、さらに

```text
何をMPS / MPOとして表すか
objectiveを何にするか
局所更新をどう定義するか
gradient-based Fine-tuningとどう比較するか
```

を明示する必要がある。

## これまでとの対応

```text
SVD
→ 1回の低rank近似

HOOI
→ Tucker factorを固定rankで交互更新

TT / MPS
→ chain型Tensorとbond dimension

DMRG
→ local tensorをsweepしながら最適化
```

HOOIで学んだ「初期値 → 局所更新 → sweep → 収束」の考え方はDMRG理解にもつながるが、HOOIとDMRGは同一アルゴリズムではない。

## 実装・評価方針

DMRGを実装するときも、これまでと同様に、

- 数式・shapeを途中式から記録する
- 小さい問題でsanity checkする
- 学習Notebookと再利用srcを分離する
- 圧縮率とtask性能を分ける
- baselineとの比較条件を揃える
- single runの小差を一般化しない

という方針を引き継ぐ。

## 関連

- [[00_基礎理論/06_手法間のつながり/14_低ランク学習からテンソルネットワークへの発展]]
- [[06_Tucker基礎実装検証/README]]
- [[40_TensorTrain_MPS/README]]
- [[README_実装編]]
