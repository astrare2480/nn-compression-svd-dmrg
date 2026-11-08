# DMRGへの接続

このディレクトリは、SVD → Tucker / HOSVD / HOOI → TT / MPSで学んだ低rank・Tensor Networkの考え方を、DMRGへ接続するための章。

現時点では、横磁場Ising鎖でenvironmentベースの2-site局所更新を左右1往復へ接続し、さらに複数往復、energy安定による停止判定、global residual・energy分散、bond上限比較まで確認済みである。加えて、物理DMRGの構造をTT-MLPへ移し、1組のTTLinear coreを結合テンソルとして更新する最小単位から、Fashion-MNIST全データを使う6-core左右sweep、学習率・更新予算の比較、GPU常駐による勾配集計高速化、更新対象ablation、Dense・通常TT・DMRG-like比較、固定rank sweep、adaptive rank、単一bond感度、非一様rank配分、全コア同時GDとのcontrol、データ計量付き切り詰め診断まで確認した。大規模な反復固有値solver、scope別の学習率最適化、固定wall time比較、複数seedでの最終比較はまだ扱っていない。

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
物理系は複数往復・停止判定・bond上限比較まで確認
TT-MLPは固定rank・adaptive rank・非一様配分・全コア同時GD control・計量付き切り詰めまで確認
```

TT / MPSの準備と学習順は [[08_TT_MPS基礎実装検証/README]] を参照。

## 既存ノート

- [[50_DMRG/01_SVDからDMRGへのつながり]]
- [[50_DMRG/02_中心テンソルの線形写像と2-site局所更新]]
- [[50_DMRG/03_MPOとenvironmentによる1-site局所有効問題]]
- [[50_DMRG/04_environmentを用いた2-site_DMRG_1往復]]
- [[50_DMRG/05_複数往復_停止判定_bond上限比較]]
- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
- [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]
- [[50_DMRG/09_全コア同時勾配法と2-core局所スイープの比較設計]]

`01` は、SVD低rank近似からMPS / MPO、bond dimension、DMRG-like sweepへ進む概念的な橋渡しである。

`02` は、固定した環境の中で中心テンソルから全状態への写像が線形になることを出発点に、$x=Pc$、$H_{\mathrm{eff}}=P^{\mathsf T}HP$、2-site局所最適化、SVD再分割、bond dimensionの打ち切りまでを小さい数値例で導出する。

`03` は、横磁場Ising HamiltonianのMPO表現、bra・MPO・ketの左右environment、$LWR$から作る1-site有効Hamiltonian、明示的な $P^{\mathsf T}HP$ との一致、matrix-free作用、固定した中心での1回の局所更新を整理する。

`04` は、隣接2サイトをまとめた $\Theta$、2-site有効Hamiltonian、SVD切り捨てとdiscarded weight、方向別の正準中心移動、左右environmentの逐次更新、2-site DMRGの左右1往復を整理する。

`05` は、左右1往復を複数回反復する制御、energy安定による停止判定、global residualとenergy分散、bond制約内の理想値 $E_\chi^\star$、bond上限2・4・無切り捨ての比較を整理する。

`06` は、TT-matrixの隣接2 coreの結合、更新後rankが増え得る理由、切り捨てなし再分割、rank制約の入れ子、最適化誤差とansatz誤差、NN勾配stepと物理DMRGの違い、$\beta$-smoothnessとSVD打ち切りを合わせた損失低下の十分条件を途中式から整理する。

`07` は、TT-SVD・rank打ち切り・正準化を段階分けした初期化診断、$n_b/N$による全データ平均勾配、同じlossからTT pairと通常parameterの勾配を得るjoint update、候補検証後のatomic commit、方向付きSVD、左右sweepと更新予算、finalとbest、因果解釈の限界を整理する。

`08` は、DataLoaderと前処理済みGPU tensor再利用の違い、batch sizeとの$2\times2$比較、数値的同等性、preloadと定常処理を分けたwall-clock測定、CUDA同期、speed-memory trade-off、ランダムaugmentationやGPU容量による適用限界を整理する。

対応する学習用Notebook：

- [[notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb|00_two_site_dmrg_local_update_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb|01_mpo_environment_one_site_local_update_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb|02_environment_two_site_dmrg_sweep_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb|03_dmrg_convergence_and_bond_dimension_exercises.ipynb]]
- [[notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb|04_tt_mlp_two_core_local_update.ipynb]]

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
TT-matrixの隣接2 core結合
固定minibatchの分類損失による1回の局所勾配更新
切り捨てなし・rank制限SVD再分割
局所更新後のcore列をモデルへ書き戻す確認
全データ平均損失による6-core joint sweep
学習率・往復数・pair内局所step数の固定予算比較
前処理済み学習データのGPU常駐による勾配集計高速化
TT-only・通常parameter-only・共同更新の固定予算比較
```

## 今後確認する要素

```text
複数seed・初期状態での収束確認
反復固有値solver
SVD前inner step数とTT側・通常parameter側の独立学習率の切り分け
計算予算・wall time・peak memoryをそろえた比較
```

NNへ適用する段階では、さらに

```text
何をMPS / MPOとして表すか
objectiveを何にするか
局所更新をどう定義するか
gradient-based Fine-tuningとどう比較するか
```

を明示する必要がある。

物理DMRGの基礎Notebookは03で一区切りとした。Notebook 04でTT-MLPの1 pair・1 stepを確認した後、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]に従ってFashion-MNIST全学習データへ拡張した。Notebook 00〜05では基準実験、初期化診断、学習率・更新予算grid、長時間更新、データ供給高速化、更新対象ablationを行った。Notebook 06〜12ではDense・通常TT・DMRG-like比較、固定rank sweepとPareto、adaptive rank、単一bond感度、非一様rank配分、全コア同時GD control、データ計量付き切り詰めへ進んだ。結果は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]から[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/22_計量を考慮したbond切り詰め診断]]へ分けて整理した。

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
- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]
- [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
- [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]
- [[50_DMRG/09_全コア同時勾配法と2-core局所スイープの比較設計]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/16_Dense_通常TT_DMRG-like比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/17_固定rank_sweepとPareto比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/18_adaptive_bond_rank_DMRG-likeと閾値探索]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/19_単一bond事後切り詰めと分類感度診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/20_非一様bond_rank配分と選択基準比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/21_全コア同時GDとDMRG-like局所スイープ比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/22_計量を考慮したbond切り詰め診断]]
- [[README_実装編]]
