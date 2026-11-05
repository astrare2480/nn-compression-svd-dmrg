# TT/MPSによるニューラルネットワーク圧縮

この章は、TT/MPSの基礎理論と小規模な実装確認を、学習済みニューラルネットワークの圧縮へ接続する教材・検証記録を置く。

配置の境界は次のとおりである。

- 数学的な定義、導出、誤差評価は [[00_基礎理論/README]]
- 小さいTensorによる一般的なPyTorch確認、共通操作、実装contractは [[08_TT_MPS基礎実装検証/README]]
- 学習済みモデルへの置換、データセット上の評価、rank sweep、fine-tuningはこの章
- 再利用可能な実装の設計は [[90_src設計/README]]

## 結果図と保存成果物の記載規則

rank sweepや非一様rank配分の考察ノートでは、図中の各点がどの候補を表すか追跡できるようにする。

- 横軸がrankである図や、各点へrankを直接注記した図は、その図だけで候補を識別できる。
- rank以外を横軸にした散布図で点へrankを表示しない場合は、`results/`に保存したCSVまたはJSONを根拠に、少なくとも図の解釈に必要な候補対応表を図の近くへ載せる。
- 候補数が少ない場合は全候補のrank列、費用、性能を載せる。
- 候補数が多く全件掲載が現実的でない場合は、選択候補、一様対照、oracle、Pareto候補など考察に使う点を表に載せ、全候補を特定できる保存成果物名と対応キーも明記する。
- rank列は図から推測せず、保存済みCSV／JSONの値を転記する。

したがって、図は傾向を読むために使い、表と保存成果物は個々の点を特定するために使う。図だけで候補を識別できない状態のまま、特定の点について考察しない。

## 現在の教材

- [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]]：2層MLPの重み誤差からbatch logits誤差、sample-wise certificate、minimum marginによるbatch内全予測不変の十分条件までの設定・結果・考察を確認する。
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/README]]：小さいConv2dでのTT-matrix表現・direct forward／backward検証から、学習済みCNNの`conv2`に対するrank sweepとcores-only fine-tuningまでを扱う。

toy検証では人工的な重み誤差を使い、第2 Linearまでの恒等式、合成上界、scale sweepを確認した。さらに、1 sampleのmargin、batch Frobenius誤差からsample-wise certificateへのchain、minimum marginを使った1 batch内全sampleの同時保証まで確認した。学習済みFashion-MNIST MLPでは、第1層のTT-rank sweep、tensorization比較、選択した3候補の通常fine-tuning、fine-tuning前後のPareto frontier比較、epoch 0を含むcheckpoint再選択、全parameter更新とTT coresのみ更新のablation、追加候補rank 12の同じablationまで進んでいる。Notebook 11ではafter frontierにrank 16だけが残り、次の追加候補としてrank 12を登録した。Notebook 12ではrank 16・32にtrained best、rank 64にepoch 0が選ばれ、保存checkpointの再評価値も許容差内で再現した。Notebook 13ではrank 16・32はTT coresのみでもaccuracyを回復したが、rank・指標を通した一貫した勝者はなく、rank 64では両方式ともepoch 0が選ばれた。Notebook 14ではrank 12もTT coresのみでaccuracy `0.3950`から`0.8884`へ回復した。Notebook 15では13・14の候補を統合し、accuracy Pareto、dense accuracyから0.001以内の許容差、補助的なkneeを分けて整理した。許容差規則ではrank 12・TT coresのみ更新が圧縮優先の暫定候補になったが、rank 12は探索下端なので、さらに低rankを検証する余地が残る。

Fashion-MNIST CNN系列のNotebook 00・01では、学習済みモデルへ進む前の小さいrandom Convを使い、Conv kernelのTT-matrix表現、exact／rank制限TTからのdense kernel再構成、patchを介したdirect forward、および再構成経路とのcore gradient等価性を確認した。Notebook 02では、そのdirect計算を`TTConv2d`として学習済みCNNの`conv2`へ接続した。CPU・float32、validation 5,000件でexact TT置換後もprediction agreement 1.0、accuracy 0.9194を維持したが、actual rank$(1,24,32,1)$ではparameter数が421,642から423,242へ増えたため、これは圧縮結果ではなくrank制限前の比較基準である。Notebook 03では固定tensorizationでrank sweepし、fine-tuning前にaccuracyをほぼ維持した確認済み境界としてrank 24を得た。Notebook 04ではrank 12・16・20・24のTT coresだけをfine-tuningし、全候補がdense baseline以上の開発用accuracyへ回復した。特にrank 12はaccuracy 0.6910から0.9238へ回復した一方、test性能、全parameter更新との比較、速度・peak memoryの優位性は未評価である。

CNN DMRG系列Notebook 00では、Notebook 04のfine-tuning済みrank 24 TTConvをsourceとして、二つの内部bond rankを`{12,16,20,24}`から選ぶ16候補を追加学習なしで評価した。uniform rank 20相当の予算では、Weight-awareの`(20,20)`に対し、Activation-awareとTask-awareは非一様配分`(16,24)`を選び、validation accuracyを2.26ポイント高く保ちながら対象層理論MACsも約4.36%少なくした。sourceに対してはaccuracyが1.04ポイント低下し、モデル総理論MACsが約32.3%減少した。MLPとのsource学習条件の違い、全16候補表、4図、oracle、順序依存性、適用範囲は[[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/05_事後非一様TTConv_rank配分と選択基準比較]]に整理した。

物理DMRGの2-site更新をTT-MLPへ移す橋渡しとして、1 pair・1 gradient stepの確認後、Fashion-MNISTの6-core `fc1`を全データ勾配で左右sweepする系列を実行した。Notebook 00の5 sweep・50局所更新ではaccuracyがrandom baseline付近だったため、Notebook 01でrank制限時の重み・activation・勾配scaleを診断し、Notebook 02で学習率・往復数・pair内局所step数を比較した。Notebook 03では共同学習率0.2・0.3を3,200更新まで実行し、0.3でvalidation accuracy 0.8710を得た。Notebook 04ではDataLoaderと前処理済みGPU tensor再利用を比較し、現在のbaselineに対する定常的な10局所更新を約10.49倍高速化した。Notebook 05では同じStage Cと3,200更新でjoint・TT-only・dense-onlyを比較し、jointが両単独更新より高いvalidation性能へ早く到達した。設計と結果は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]から[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]へ分け、一般技術は[[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]と[[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]に整理した。

rank 12未満の追加候補、rank 64の学習率診断、複数seedでの更新方式比較、scopeごとの学習率・固定wall time比較、最終候補を固定した後のtest評価、parameter・MACs・latency・memoryの比較は、今後この章へ追加する。

## 関連

- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]
- [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
- [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]
- [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/11_TT初期化スケール診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/12_full-batch更新予算grid]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/13_joint学習率0.2_0.3長時間比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/14_full-batch勾配集計高速化benchmark]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/15_更新対象固定予算ablation]]
