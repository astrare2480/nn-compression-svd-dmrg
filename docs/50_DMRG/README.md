# DMRGへの接続

このディレクトリは、SVD → Tucker / HOSVD / HOOI → TT / MPSで学んだ低rank・Tensor Networkの考え方を、DMRGへ接続するための章。

現時点では、横磁場Ising鎖でenvironmentベースの2-site局所更新を左右1往復へ接続し、さらに複数往復、energy安定による停止判定、global residual・energy分散、bond上限比較まで確認済みである。大規模な反復固有値solverとNN圧縮への適用はまだ扱っていない。

## 現在の位置

```text
SVD
完了
↓
Tucker / HOSVD / HOOI
完了
↓
TT / MPS
基礎実装・正準形まで確認
↓
DMRG
複数往復・停止判定・bond上限比較まで確認
```

TT / MPSの準備と学習順は [[08_TT_MPS基礎実装検証/README]] を参照。

## 既存ノート

- [[50_DMRG/01_SVDからDMRGへのつながり]]
- [[50_DMRG/02_中心テンソルの線形写像と2-site局所更新]]
- [[50_DMRG/03_MPOとenvironmentによる1-site局所有効問題]]
- [[50_DMRG/04_environmentを用いた2-site_DMRG_1往復]]
- [[50_DMRG/05_複数往復_停止判定_bond上限比較]]

`01` は、SVD低rank近似からMPS / MPO、bond dimension、DMRG-like sweepへ進む概念的な橋渡しである。

`02` は、固定した環境の中で中心テンソルから全状態への写像が線形になることを出発点に、$x=Pc$、$H_{\mathrm{eff}}=P^{\mathsf T}HP$、2-site局所最適化、SVD再分割、bond dimensionの打ち切りまでを小さい数値例で導出する。

`03` は、横磁場Ising HamiltonianのMPO表現、bra・MPO・ketの左右environment、$LWR$から作る1-site有効Hamiltonian、明示的な $P^{\mathsf T}HP$ との一致、matrix-free作用、固定した中心での1回の局所更新を整理する。

`04` は、隣接2サイトをまとめた $\Theta$、2-site有効Hamiltonian、SVD切り捨てとdiscarded weight、方向別の正準中心移動、左右environmentの逐次更新、2-site DMRGの左右1往復を整理する。

`05` は、左右1往復を複数回反復する制御、energy安定による停止判定、global residualとenergy分散、bond制約内の理想値 $E_\chi^\star$、bond上限2・4・無切り捨ての比較を整理する。

対応する学習用Notebook：

- [[notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb|00_two_site_dmrg_local_update_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb|01_mpo_environment_one_site_local_update_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb|02_environment_two_site_dmrg_sweep_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb|03_dmrg_convergence_and_bond_dimension_exercises.ipynb]]

## DMRGで確認した要素

```text
MPS / TT表現
bond dimension
2-site local tensor
局所最適化
SVDによる再分割・truncate
HamiltonianのMPO表現
left / right environment
1-site effective Hamiltonian
matrix-freeな局所作用
2-site effective Hamiltonian
方向別のSVD分割と正準中心移動
environmentを逐次更新する左右1往復
discarded weightと局所・全体診断の分離
複数往復とenergy安定停止
global residualとenergy分散
bond上限2・4・無切り捨ての比較
```

## 今後確認する要素

```text
複数seed・初期状態での収束確認
反復固有値solver
```

NNへ適用する段階では、さらに

```text
何をMPS / MPOとして表すか
objectiveを何にするか
局所更新をどう定義するか
gradient-based Fine-tuningとどう比較するか
```

を明示する必要がある。

物理DMRGの基礎Notebookは03で一区切りとし、次はFashion-MNIST TT-MLPに対するDMRG-likeな2-core局所更新へ進む。

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
- [[08_TT_MPS基礎実装検証/README]]
- [[README_実装編]]
