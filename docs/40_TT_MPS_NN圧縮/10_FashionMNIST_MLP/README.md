# Fashion-MNIST MLPのTTLinear圧縮

この章では、Fashion-MNISTの学習済みMLPへTTLinearを適用し、exact置換の等価性確認からrank打ち切り、精度評価、fine-tuningへ進む。

## 現在の到達点

[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]では、`fc1: 784 -> 512`だけを打ち切りなしTTLinearへ置換した。保存済み出力では、対象Linear直後、ReLU直後、最終logitsがfloat64の丸め誤差水準で一致し、predictionも一致した。

一方、exact TT-rankは$(1,16,256,56,1)$、TT重みは470,336要素であり、dense重み401,408要素より大きい。したがって、現在の結果は学習済みモデル上の関数等価性検証であり、圧縮結果ではない。

その後の[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/01_fc1_TT-rank_sweep]]では、`fc1`だけを3-core TTLinearへ置換し、固定tensorizationでのrank sweepを整理した。さらに[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/02_fc1_tensorization比較]]では、08のparameter予算を引き継いだ3種類のtensorization比較を整理した。保存済みvalidation結果では、比較した3つの予算すべてで`reversed_sites`が選択された。ただし、この結果は1つのcheckpointとrank・tensorization選択用validationに限定され、一般的な最適性は示していない。

[10_pretrained_mlp_fc1_tt_finetuning.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/10_pretrained_mlp_fc1_tt_finetuning.ipynb)では、09で選んだrank 16／32／64の3候補を同じdense checkpointから独立に再生成し、通常fine-tuningした。optimizerはTT coresだけでなく、`fc1`のbias、`fc2`、`fc3`を含むmodelの全parameterを更新する。保存済みrank-selection validationでは、aggressiveはaccuracy `0.5116`から`0.8890`へ回復し、balancedは`0.7980`から`0.8850`、conservativeは`0.8872`から`0.8850`へ変化した。test setはまだ使用していない。

設定・保存結果と、fine-tuning後の順位逆転、logits RMSE、過学習傾向、比較の公平性、モデル選択に関する考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/03_fc1_TT-fine-tuning]]に整理した。一般論は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/73_TT圧縮後fine-tuningの評価とモデル選択]]へ分けている。

[11_tt_rank_tradeoff_after_finetuning.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/11_tt_rank_tradeoff_after_finetuning.ipynb)では、09・10の保存結果の接続を検証し、fine-tuning前後のPareto frontierを別々に求めた。beforeではrank 16・32・64がすべてfrontierに残ったが、afterでは最小parameterかつ最高accuracyのrank 16だけが残った。rank 16は探索範囲の下端なので、次の追加候補にはrank 12だけを登録し、このNotebookでは追加学習やtest評価を行っていない。

Notebook 11の保存結果、Pareto変化、追加rank判断、対話で行った考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/04_fine-tuning後のTT-rank_trade-off]]に整理した。一般論は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/74_fine-tuning後のPareto_frontierと追加rank探索]]へ分けている。

[12_epoch0_checkpoint_selection_and_reproducibility.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/12_epoch0_checkpoint_selection_and_reproducibility.ipynb)では、新たな学習やtest評価を行わず、圧縮直後のepoch 0とNotebook 10の保存済みtrained bestを同じvalidation loss基準で比較した。rank 16・32ではtrained best、rank 64ではepoch 0が選ばれた。保存checkpointの再評価値は全候補で許容差内に再現したが、これは保存・復元・再評価の整合性確認であり、独立な再学習の再現ではない。

Notebook 12の保存結果、rank 64でlossとaccuracyが逆方向へ動いた理由、Notebook 10・11との関係、再現確認の範囲は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/05_epoch0を含むcheckpoint再選択と再現確認]]に整理した。一般論は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/75_epoch0を含むcheckpoint選択と保存モデルの再現確認]]へ分けている。

[13_tt_cores_only_ablation.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/13_tt_cores_only_ablation.ipynb)では、rank 16／32／64について、同じepoch 0から全parameter更新と`fc1`のTT coresだけの更新を比較した。rank 16・32ではTT coresだけでもaccuracyが回復したが、指標とrankを通して一貫した勝者はなかった。rank 64では両方式ともepoch 0が選ばれ、TT coresだけを更新した学習途中でもlogits RMSEが大きく増えたため、後段層の更新だけを乖離の原因にはできない。

Notebook 13の保存結果、更新方式ごとの違い、rank 64の学習trajectory、単一試行から言える範囲は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/06_TTコアのみ更新ablation]]に整理した。更新範囲を制限する一般的な意味は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/76_fine-tuning更新範囲のcontrolled_ablation]]へ分けている。

[14_rank12_update_scope_ablation.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/14_rank12_update_scope_ablation.ipynb)では、Notebook 11が追加候補として登録したrank 12について、Notebook 13と同じ全parameter更新／TT coresのみ更新の比較を行った。epoch 0のaccuracyは`0.3950`だったが、全parameter更新で`0.8848`、TT coresのみ更新で`0.8884`まで回復した。後者はdenseと同じrank-validation accuracyで、比較候補中のモデル総parameter数も最小だったため、圧縮を優先する暫定候補とした。ただし、最高accuracy・最短学習時間・test性能を意味しない。

Notebook 14の保存結果とNotebook 11・13を統合した候補整理は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/07_rank12追加と更新範囲比較]]に整理した。rank、モデル総parameter数、学習対象数、学習時間、推論コストの一般的な読み分けは[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/77_TT-rankと学習計算量_推論コストの切り分け]]へ分けている。

[15_integrated_tt_rank_update_scope_tradeoff.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/15_integrated_tt_rank_update_scope_tradeoff.ipynb)では、Notebook 13・14の8 trialを統合し、rank 64の同一checkpointをまとめた7モデル候補について、モデル総parameter数とaccuracy・agreement・logits RMSEのtrade-offを分けて比較した。accuracy Paretoにはrank 12・16・32のTT coresのみ更新が残り、dense accuracyから0.001以内という事後的許容差ではrank 12・TT coresのみ更新が最小候補になった。ただし、最高accuracy、最短学習時間、統計的同等性、test性能を意味しない。rank 12が探索範囲の下端なので、圧縮を優先する場合はさらに低rankを検証する余地が残る。

Notebook 15の統合結果と候補選択の考察は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/08_TT-rankと更新範囲の統合trade-off]]に整理した。許容差、Pareto frontier、knee、学習コストを分ける一般的な方法は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/78_accuracy許容差とParetoによる圧縮候補選択]]へ分けている。

物理DMRGで使った2-site結合・局所更新・SVD再分割・左右sweepをTT-MLPへ移す次段階は、[[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]で理論を、[[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]で1 pair・固定minibatch・1 stepを整理した。Fashion-MNISTの6-core `fc1`を新しいdense初期値からTT-SVDし、各pairで全学習データ平均勾配を蓄積する左右sweepは、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]に従ってNotebook 00へ実装した。5 sweep・50局所更新は全て成功しbest checkpointも再現したが、accuracyは約9.5%でrandom baseline付近だった。実験結果、解釈の限界、次のcontrolled experimentは[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]に整理した。これは既存の学習済みcheckpointに対するfine-tuning系列とは別実験である。

Notebook 01では、denseからrank cap 12のTTへ変換したときの重み・activation・勾配scaleを段階別に測定した。重みnormはTT-SVD・rank制限で`13.0630`から`1.88249`へ下がり、正準化前後の実効重みとlogitsは丸め誤差内で一致した。結果と因果解釈の限界は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]に整理した。

Notebook 02では共同学習率`0.001`・`0.01`・`0.1`、往復数5・20、各pairの局所step数1・4の12条件を同じStage Cから独立に比較した。候補中では学習率`0.1`・800更新が最良で、同じ200更新の配分差は小さかった。完全な結果と考察は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]に整理した。

Notebook 03では共同学習率`0.2`・`0.3`を40往復・各pair 8 step・3,200更新で比較した。`0.3`はvalidation loss `0.342614`、accuracy `0.8710`で候補中最良だったが、終盤の改善は計算量に対して小さくなった。02との間では更新予算も変わるため、学習率だけの効果とは扱わない。結果は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]に整理した。

Notebook 04では、batch size 512・4096とDataLoader・GPU常駐を$2\times2$で比較した。前処理済み全データをGPU上で再利用し、batch size 4096にすると、定常的な10局所更新は現行baseline比で約10.49倍になった。短期数値同等性、測定範囲、memoryとの交換条件は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]に整理した。一般技術は[[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]と[[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]へ分けた。

Notebook 05では、04で採用したGPU常駐・batch size 4,096を共通にし、joint・TT-only・dense-onlyを同じStage C、学習率0.3、3,200 full-batch更新で比較した。jointはvalidation accuracy `0.8716`で両単独更新を上回った。TT-onlyも`0.0946`から`0.6278`へ改善した一方、dense-onlyは途中で`0.6672`へ達した後に大きく振動した。固定予算での性能、到達速度、計算費用、解釈の限界は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]に整理した。

Notebook 06では、Dense MLP、通常Adam TT、DMRG-like TTを共通のデータ分割とearly stopping規則で比較した。rank-selection accuracyは順に`0.8884`、`0.8906`、`0.8968`だった。TTはモデル総parameter数を73.80%削減した一方、現在のdirect contractionでは理論MACsがDenseの約4.316倍、推論latencyも増えた。DMRG-likeは通常TTより高い評価値へ到達したが、学習＋評価時間は約23.5倍だった。結果、計算費用、保存成果物監査、解釈の限界は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/16_Dense_通常TT_DMRG-like比較]]に整理した。

Notebook 07では、rank cap `1, 2, 3, 4, 5, 6, 8, 12`の通常TTとDMRG-likeを比較し、固定rankの性能と費用の基準を作った。DMRG-likeはrank 5・6・12で同rankの通常TTを上回ったが、全rankではなかった。理論MACs対errorの規則は通常TT rank 1、DMRG-like rank 6、DMRG-like rank 12をそれぞれ`aggressive`・`balanced`・`conservative`に選んだが、実測latencyと学習時間では異なる候補が残った。図の読み取り、rank間の非単調性、保存成果物の検証範囲、次のadaptive truncationの比較基準は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/17_固定rank_sweepとPareto比較]]に整理した。

Notebook 08では、rank cap 12のStage Cから、discarded-weight閾値`1e-6`・`1e-4`・`1e-2`でbond rankを選ぶadaptive DMRG-like学習を行った。すべての条件でrank列は`(1,8,12,12,12,12,1)`のままで、最大閾値`1e-2`でもrank 11以下は選ばれなかった。3閾値のbest modelは相互に同一で、07の固定DMRG-like rank 12ともparameter tensorが完全一致した。そのため07のPareto frontierは改善しなかったが、adaptive rank全般を否定する結果ではない。閾値未達率、bond別の違い、図の考察、次のスペクトル診断は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/18_adaptive_bond_rank_DMRG-likeと閾値探索]]に整理した。

Notebook 09では、そのrank 12 best modelのbond 2〜5を一つずつrank 11・10・8へ切り、再学習なしで重み誤差と分類感度を測った。全12候補でlossは増加しaccuracyは低下したが、影響はbondごとに大きく異なった。rank 11ではbond 2の重み相対誤差`13.00%`に対してaccuracyが`0.84`ポイント低下した一方、bond 5は重み誤差`16.85%`でも低下は`0.30`ポイントだった。実画像に対する`fc1`出力、logits、正誤遷移、classification marginまで追跡し、重み再構成誤差だけでは分類性能への影響を順位付けできないことを[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/19_単一bond事後切り詰めと分類感度診断]]に整理した。

Notebook 10では、通常TTとDMRG-likeのrank 12 checkpointについて、bond 2〜5のrankを`8・10・11・12`から選ぶ各256候補、合計512候補を再学習なしで生成した。一様rank 11・10相当の理論MAC予算以下から、実効重み誤差、実入力に対する`fc1`出力誤差、calibration cross-entropy増加の三基準で候補を選んだ。Task基準は主順序の4条件すべてでvalidation loss最小候補と一致し、DMRG-like・r10予算では一様rank 10のaccuracy`0.8236`に対して非一様配分`(11,11,10,8)`で`0.8672`を保った。一方、通常TT・r10では一様rank 10が選ばれ、非一様性自体が改善を保証するわけではなかった。全10図、選択基準、oracle regret、切り詰め順序依存性、08〜10の接続は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/20_非一様bond_rank配分と選択基準比較]]に整理した。

Notebook 11では、同じStage C初期TT-MLP、全学習データ平均勾配、素朴なGD、共通の学習率候補と停止規則を用い、全コア同時更新とDMRG-like局所スイープを比較した。選択後のrank-selection validationでは、全コア同時GDのloss`0.310293`・accuracy`0.8886`に対し、DMRG-likeはloss`0.287969`・accuracy`0.8960`だった。一方、DMRG-likeはbestまでに約4.94倍のfull-dataset gradient evaluationsと約1.69倍の更新処理時間を使い、通常parameterの更新回数も約4.94倍だった。到達性能、同一目標への到達時間、評価費用、GPUメモリ、train-validation gap、loss基準checkpoint選択、因果解釈の限界は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/21_全コア同時GDとDMRG-like局所スイープ比較]]に整理した。

Notebook 12では、Notebook 06のDMRG-like rank 12 best modelについて、bond 2をrank 11へ事後切り詰めし、通常SVD、分離計量SVD、固定rank ALSを比較した。validationのReLU前`fc1`出力相対誤差は順に`12.10%`、`8.52%`、`6.13%`で、同じrank 11でもデータ計量を考慮すると通常SVDの出力二乗誤差を約50%・75%削減できた。ただし、accuracy・loss・test、他bond、2-core更新直後での再現、大域最適性は未確認である。結果と訂正済み考察は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/22_計量を考慮したbond切り詰め診断]]、一般理論は[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/79_データ計量付きTT_bond切り詰めとALS]]に整理した。

一般的なtoy modelでは、第1 Linearの重み誤差をReLUと第2 Linearへ伝播させる合成上界、1 sampleのclassification margin、batch logits errorからsample-wise prediction certificate、minimum marginによるbatch内全sampleの同時保証までを確認している。PyTorch操作は[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]、[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]]に分け、Fashion-MNISTの実測結果とは数えない。

## 読む順序

1. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]
2. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]
3. [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]
4. [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
5. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]
6. [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
7. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]
8. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]
9. [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]
10. [[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]
11. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
12. [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
13. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]
14. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]
15. [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
16. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]
17. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]
18. [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
19. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]
20. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]
21. [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
22. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]
23. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/01_fc1_TT-rank_sweep]]
24. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/02_fc1_tensorization比較]]
25. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/73_TT圧縮後fine-tuningの評価とモデル選択]]
26. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/03_fc1_TT-fine-tuning]]
27. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/74_fine-tuning後のPareto_frontierと追加rank探索]]
28. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/04_fine-tuning後のTT-rank_trade-off]]
29. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/75_epoch0を含むcheckpoint選択と保存モデルの再現確認]]
30. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/05_epoch0を含むcheckpoint再選択と再現確認]]
31. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/76_fine-tuning更新範囲のcontrolled_ablation]]
32. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/06_TTコアのみ更新ablation]]
33. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/77_TT-rankと学習計算量_推論コストの切り分け]]
34. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/07_rank12追加と更新範囲比較]]
35. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/78_accuracy許容差とParetoによる圧縮候補選択]]
36. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/08_TT-rankと更新範囲の統合trade-off]]
37. [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
38. [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
39. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
40. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]
41. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]
42. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]
43. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]
44. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]
45. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]
46. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/16_Dense_通常TT_DMRG-like比較]]
47. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/17_固定rank_sweepとPareto比較]]
48. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/18_adaptive_bond_rank_DMRG-likeと閾値探索]]
49. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/19_単一bond事後切り詰めと分類感度診断]]
50. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/20_非一様bond_rank配分と選択基準比較]]
51. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/21_全コア同時GDとDMRG-like局所スイープ比較]]
52. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/22_計量を考慮したbond切り詰め診断]]
53. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/79_データ計量付きTT_bond切り詰めとALS]]
54. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/80_適応bond_rank選択と非一様rank配分]]
55. [[50_DMRG/09_全コア同時勾配法と2-core局所スイープの比較設計]]
54. [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
55. [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]

通常のTT圧縮系列は [10_fashion_mnist_mlp](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/) に、DMRG-like sweep系列は [10_fashion_mnist_mlp](../../../notebooks/40_dmrg/10_fashion_mnist_mlp/) に置く。

## 未完了の次段階

- SVD前inner step数とTT側・通常parameter側の学習率を別々に切り分け、必要ならscopeごとの学習率や固定wall timeで比較すること
- rank 12未満の少数候補を同じ条件で追加し、圧縮優先時の下限を確認すること
- Notebook 10のTask基準が別のcalibration subset・seedでも同じrank配分を選ぶか確認すること
- 逆順でも全候補を再列挙し、選択rank列とoracleが主順序から変わるか確認すること
- 小さいaccuracy差を主張する場合、選択候補のsample別予測を保存し、対応のある正誤遷移を確認すること
- Weight・Activation・Taskを独立運用した場合の候補選択計算量を分離すること
- 切り詰め後fine-tuningで、選択した非一様rank配分の優位が残るか確認すること
- 単一bondの重み誤差と実入力への作用が逆転した原因を、入力共分散・特異方向・複数seedで切り分けること
- validationで選択したrank・tensorization・fine-tuning条件に対するFashion-MNIST test setでの最終評価
- rank 64について、同じepoch 0から学習率を下げ、fine-tuning不要と更新量過大の仮説を切り分けること
- 更新方式の差を複数seedとsample別の対応のある予測で確認すること
- parameter、MACs、latency、memoryの実測比較
