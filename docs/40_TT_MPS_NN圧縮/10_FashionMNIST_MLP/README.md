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

対応Notebookは [10_fashion_mnist_mlp](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/) に置く。

## 未完了の次段階

- rank 12未満の少数候補を同じ条件で追加し、圧縮優先時の下限を確認すること
- 学習済みFashion-MNIST MLPの打ち切りrankで、ReLUを通るhidden errorを数値確認すること。一般的なtoy例は実装検証ノート15で確認済み
- 学習済みFashion-MNIST MLPの実際の打ち切りrankで、後続Linearを通るlogits errorと合成上界を数値確認すること。random toy modelは実装検証ノート16で確認済み
- 学習済みFashion-MNIST MLPを実際にTT-rank打ち切りしたときのsample別classification margin、minimum margin、batch内prediction agreement。一般的なtoy例は理論ノート65〜67、実装ノート17〜19、toy結果ノート01〜03で確認済み
- validationで選択したrank・tensorization・fine-tuning条件に対するFashion-MNIST test setでの最終評価
- rank 64について、同じepoch 0から学習率を下げ、fine-tuning不要と更新量過大の仮説を切り分けること
- 更新方式の差を複数seedとsample別の対応のある予測で確認すること
- parameter、MACs、latency、memoryの実測比較
