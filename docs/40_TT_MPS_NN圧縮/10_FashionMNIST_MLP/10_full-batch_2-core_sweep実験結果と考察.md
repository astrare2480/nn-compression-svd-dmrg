---
title: Fashion-MNIST TT-MLP full-batch 2-core sweep実験結果と考察
aliases:
  - Fashion-MNIST TT-MLP DMRG-like sweep実験結果
tags:
  - Fashion-MNIST
  - TTLinear
  - DMRG-like
  - two-core sweep
  - full-batch gradient
  - 実験結果
  - 考察
---

# Fashion-MNIST TT-MLP full-batch 2-core sweep実験結果と考察

## このノートの役割

本ノートは、次のNotebook 00の実験設計、保存済み結果、この結果から言えること、次の比較実験を一つにまとめる。

- `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`
- 実装前のcontract：[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
- 局所更新の理論：[[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- 1 pair・1 stepの先行確認：[[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]

Notebook 00では学習からartifact保存、best checkpoint再読込までの出力が保存されており、保存済みerrorはない。ただし、現在のNotebookは11個のcode cellのうち`weighted_batch_loss`を定義する1 cellだけ`execution_count=None`で、成功出力は残っている。Notebook本体の更新時刻はartifactより後なので、現時点のファイルを「全cellが連番のFresh Kernel Run All」とは扱わない。ここで扱う学習結果はNotebookの見た目だけに依存せず、次のartifactをPythonの`pandas`で再集計した値である。完全な再現証跡を固定するときは、もう一度Fresh KernelからRun Allして保存する。

```text
models/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training/
├─ history.csv
├─ local_updates.csv
├─ run_state.json
├─ initial_state.pt
├─ best_state.pt
└─ final_state.pt
```

ただし、Fresh Run Allの成功は「実装した学習経路が最後まで動いた」証拠であり、「分類器が十分に学習した」証拠ではない。この二つを分けて読む。

---

## 1. 実験で確認した問い

Notebook 00の主目的は、次の一連の処理をFashion-MNISTの実データで接続できるか確認することだった。

```text
新規dense MLPを初期化
→ fc1をTT-SVDして6-core TTLinearへ置換
→ 隣接2 coreを結合
→ 全学習データ平均損失の勾配で局所更新
→ directionに応じてSVD再分割
→ atomic validation後に一括commit
→ left-to-right / right-to-left sweep
→ validationによるbest選択とearly stopping
→ checkpoint保存・再読込・値の再現
```

したがって、これは次の実験ではない。

- 学習済みdense checkpointをTT化した後のfine-tuning。
- 物理DMRGの局所有効Hamiltonianを解く実験。
- minibatchごとに即時更新するSGD。
- rank、tensorization、optimizer、学習率の探索。
- test setによる最終性能評価。
- 通常fine-tuningとの速度・memory・accuracy比較。

00で答えようとしたのは、最良の学習法かどうかではなく、設計したfull-batch 2-core sweepの学習経路がcontractどおり成立するかである。

---

## 2. 実際に使った実験設計

### 2.1 モデル、初期化、データ

| 項目 | 実際の設定 |
|---|---|
| モデル | `fc1: 784→512`、ReLU、`fc2: 512→256`、ReLU、`fc3: 256→10` |
| 初期値 | seed 0で新規作成した未学習dense MLP |
| TT化する層 | `fc1` |
| `in_modes` | `(2, 2, 2, 2, 7, 7)`、積は784 |
| `out_modes` | `(4, 2, 2, 2, 4, 4)`、積は512 |
| rank cap | 12 |
| 初期・最終TT-rank | `(1, 8, 12, 12, 12, 12, 1)` |
| dtype / device | `torch.float32` / `cuda:0` |
| 分割 | training 50,000、early-stopping validation 5,000、rank-selection 5,000 |
| batch size | 64 |
| test・rank-selection split | 未使用 |

rank capを12にしても、第1 cutのshape上限は8なので、実rankは先頭だけ8になる。このrank列は、実験中に変化しなかった。

dense `fc1.weight`のFrobenius normは`13.0630`、TT-SVD・canonicalize後に再構成した`fc1.weight`のnormは`1.88246`だった。比は`0.14411`である。したがって、00はdense初期関数をほぼ同じままTT表現へ移した実験ではなく、rank打ち切りにより初期関数が大きく変わった状態から学習を始めている。

### 2.2 各pairで固定した目的関数

training set全体の平均cross-entropyを

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)
$$

とする。minibatchは別々の目的関数ではなく、同じ$L$とその勾配をmemoryに収まる大きさへ分けて計算するために使う。batch $b$のsample数を$n_b$、batch mean lossを$L_b$とすると、

$$
L=\sum_b\frac{n_b}{N}L_b,
\qquad
\nabla L=\sum_b\frac{n_b}{N}\nabla L_b
$$

で蓄積する。最後の小さいbatchも正しく重み付けされ、batch meanを単純平均して二重に割ることはしない。

この方式を最初に選んだ理由は、**全学習データ平均損失という同じ目的関数を固定し、その目的に対してDMRG-likeな左右sweepを繰り返す構造を明確にするため**である。minibatchごとに更新する方式では、更新ごとに見ているbatch lossが変わる。00では、この別方式を混ぜなかった。

これは「full-batchのほうが性能がよい」と実証した結果ではない。固定した一つの目的関数に対する局所更新を、最初の基準実装に選んだという設計上の理由である。

小さい学習率で$T$を$T-\eta\nabla_TL$へ更新すると、滑らかさを仮定した一次近似は

$$
L(T-\eta\nabla_TL)
\approx
L(T)-\eta\|\nabla_TL\|_F^2+O(\eta^2)
$$

となる。ただし、これはSVD打ち切り、他parameterの同時更新、有限学習率まで含めて毎回の損失低下を保証する式ではない。

### 2.3 1 pairの更新

pair $(k,k+1)$ごとに次を行った。

1. 最新の2 coreを結合し、結合テンソル$T$を作る。
2. 他のTT coreを固定し、全training setから$T$の平均勾配を蓄積する。
3. 同じ更新前状態から、`fc1.bias`、`fc2`、`fc3`の勾配も蓄積する。
4. momentumなしの勾配stepを、TT側と通常parameter側へそれぞれ適用する。
5. right sweepでは特異値を右coreへ、left sweepでは左coreへ吸収してSVD再分割する。追加のunit-norm rescalingは行わない。
6. shape、bond、rank cap、dtype、device、有限性、固定diagnostic batchを候補state上で検証する。
7. 検証を通ったTT coreと通常parameterを一括commitする。

設定は次のとおりである。

| 項目 | 値 |
|---|---:|
| TT pair learning rate | $10^{-3}$ |
| 通常parameter learning rate | $10^{-3}$ |
| 1 pairあたりの局所step数 | 1 |
| optimizer相当 | momentumなしの素の勾配step |

pairの途中で通常parameterだけ先に更新すると、TT勾配と通常parameter勾配が別のstateに対する勾配になる。そのため、全勾配を同じ更新前stateから作り、候補全体を検証した後に同時commitする。

固定diagnostic batchのloss低下はcandidateの受理条件ではない。ここで拒否するのはshape不整合、rank違反、非有限値などであり、有限だがlossが増えたcandidateを暗黙にrollbackする設計ではない。

### 2.4 sweepの定義と計算量

6 coreなので、right sweepとleft sweepは次の10 visitで構成される。

```text
right: (0,1) → (1,2) → (2,3) → (3,4) → (4,5)
left : (4,5) → (3,4) → (2,3) → (1,2) → (0,1)
```

折り返しの$(4,5)$は、directionと特異値の吸収先が異なるため、rightの最後とleftの最初で連続して訪れる。1 pairの1 stepごとにtraining set全体を1回走査するので、core数$d$、1 pairの局所step数$L$、sweep数$S$に対する全データ走査回数は

$$
2(d-1)LS
$$

である。今回は$d=6$、$L=1$、$S=5$なので、50回の全データforward・backwardを行った。共有される`fc1.bias`、`fc2`、`fc3`も、各pairと同じく1 sweepあたり10回更新された。ここでいう1 sweepは、通常のminibatch学習の1 epochとは異なる。

### 2.5 validation、停止、保存

- sweep 0を学習前baselineとして評価する。
- 各completed sweep後にtrainとearly-stopping validationを評価する。
- `patience=5`、`min_delta=1e-5`を使う。
- strictなvalidation loss最小値をbest checkpointとして別に保存する。
- 最大5 sweepまで行う。
- partial sweepはcompleted sweepのbest候補に混ぜない。
- `initial_state`、`best_state`、`final_state`を分けて保存する。
- best checkpointを新しいmodel instanceへ読み込み、validation値を再現する。

ここでのearly stoppingは、設定したvalidation規則で改善が止まったことだけを意味する。局所勾配が0になったこと、数学的定常点、物理DMRGのenergy収束を意味しない。

実際の停止理由はearly stoppingではなく`max_sweeps`だった。bestはsweep 5で、再読込後のvalidation loss `2.3038700309753417`、accuracy `0.0952`は保存値と一致した。

---

## 3. 保存済み結果

### 3.1 sweep履歴

| sweep | train loss | train accuracy | validation loss | validation accuracy | 累積成功更新数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.303470 | 0.09358 | 2.303983 | 0.0946 | 0 |
| 1 | 2.303448 | 0.09374 | 2.303961 | 0.0946 | 10 |
| 2 | 2.303425 | 0.09380 | 2.303938 | 0.0948 | 20 |
| 3 | 2.303403 | 0.09400 | 2.303915 | 0.0948 | 30 |
| 4 | 2.303381 | 0.09420 | 2.303893 | 0.0948 | 40 |
| 5 | 2.303358 | 0.09426 | 2.303870 | 0.0952 | 50 |

sweep 0から5までの差は次のとおりである。

- train loss：`-0.00011204`
- validation loss：`-0.00011315`
- train accuracy：`+0.00068`、すなわち`+0.068` percentage point
- validation accuracy：`+0.00060`、すなわち`+0.060` percentage point

### 3.2 局所更新の診断値

- 成功した局所更新：50、失敗：0。
- right：25回、left：25回。
- 各pair index 0〜4：それぞれ10回。
- TT相対更新量：`4.10e-6`〜`2.72e-5`。
- 通常parameter相対更新量：`4.58e-6`前後。
- TT勾配norm：`0.00772`〜`0.05113`。
- 通常parameter勾配norm：`0.04319`〜`0.04320`。
- SVDで捨てた二乗norm：`0`〜`1.85e-9`。
- 固定diagnostic batch loss：最初の成功更新後`2.296782`、最後の成功更新後`2.296674`。
- 最終center：0。
- 最終rank：`(1, 8, 12, 12, 12, 12, 1)`。

相対更新量は

$$
\frac{\|\Delta\theta\|}{\|\theta\|}
$$

であり、$10^{-5}$は1回の更新でparameter normの約`0.001%`だけ動いたことを表す。0ではなく実際に更新されているが、変化量は非常に小さい。

---

## 4. 実験結果からの考察

### 4.1 学習pipelineは成立したが、分類学習はほぼ進んでいない

50回すべての局所更新が成功し、左右sweep、center移動、checkpoint保存、best再読込まで完了した。したがって、**full-batch 2-core sweepの実装経路を実データで最後まで通せた**という00の主目的は達成した。

一方、loss低下は約$1.1\times10^{-4}$、accuracy上昇はtrainで0.068 percentage point、validationで0.060 percentage pointにとどまった。10クラス分類を一様ランダムに答えたときの期待accuracyは10%である。また、一様予測のcross-entropyは

$$
-\log 0.1=\log 10\approx2.302585
$$

である。今回のlossは約2.303、accuracyは約9.5%なので、モデルは各クラスをほとんど区別できていない。lossは数値上低下したが、実用的な分類性能の改善とは読めない。

したがって、結論は次に限定する。

> 今回の初期化、rank、学習率、局所step数、5 sweepという条件では、lossは微減したが、分類性能の明確な改善は確認できなかった。

「DMRG-likeな学習方法はFashion-MNISTで学習できない」と一般化することはできない。

### 4.2 微小な変化の候補は、更新量と初期化の両方にある

TT相対更新量は最大でも約$2.7\times10^{-5}$、通常parameter側は約$4.6\times10^{-6}$だった。したがって、1回ごとのparameter変化が小さく、5 sweepでは大きな関数変化へ至らなかった可能性がある。

同時に、TT-SVD後の`fc1` normは元のdense初期値の約14.4%だった。rank cap 12による初期打ち切りで、最初から信号scaleや表現が大きく変わっていた可能性もある。ただし、1 runだけでは、norm低下が学習停滞の原因だと因果的には確定できない。

「更新回数が少ない」「学習率が小さい」「初期TT近似が厳しすぎる」は、いずれも候補である。00ではこれらを同時に固定したので、どれが支配的だったかは切り分けられない。

### 4.3 lossの低下をTT更新だけの効果とは言えない

各pairで更新したのは結合TT coreだけではない。同じ勾配蓄積中に、`fc1.bias`、`fc2`、`fc3`も更新した。保存されたloss履歴は、それらを同時に更新した合計の結果である。

したがって、今回の微小なloss低下について、次のどれかを判別できない。

- TT core更新で下がった。
- 通常parameter更新で下がった。
- 両方の相互作用で下がった。

TT更新の寄与を分けるには、少なくとも「TTだけ更新」と「通常parameterだけ更新」が必要である。さらに無更新対照を加えた$2\times2$比較にすると、実装・評価だけで生じる差と相互作用も確認しやすい。

### 4.4 full-batch方式の意味と代償

各pairで同じ全training mean lossを目的にしたことで、minibatch抽出の違いを局所更新の解釈へ混ぜずに済んだ。各pairと各sweepが、同じ目的関数に対する更新として比較できる。これは物理DMRGそのものではないが、「同じ全体目的に対して局所blockを順に動かす」というDMRG-likeな構造を明確にする。

代償は計算量である。5 sweep、50局所更新のために、training set全体のforward・backwardを50回行った。それでもvalidation lossの低下は約$1.13\times10^{-4}$だった。このrunでは、使った全データ走査回数に対する目的関数の変化が小さかった。

ただし、wall-clock timeとpeak memoryは保存artifactに記録されていない。したがって、「minibatch方式より遅い」「通常fine-tuningより非効率」と実測比較したわけではない。

### 4.5 rank不変とSVD discarded normの意味

全更新でrankは`(1, 8, 12, 12, 12, 12, 1)`のままで、SVDのdiscarded squared normも最大`1.85e-9`だった。これは、**更新後の結合テンソルを現在のrank capへ再分割する段階では、追加の切り捨てが非常に小さかった**ことを示す。

一方、初期dense `fc1`からrank-12 TTへ移した時点ではweight normが約14.4%まで低下している。したがって、

```text
初期dense → rank-12 TT-SVDで生じた大きな変化
```

と

```text
局所step後のSVD再分割で生じた小さな追加切り捨て
```

を混同してはいけない。後者が小さいことは、rank 12が分類に十分だったことや、初期TT化の情報損失が小さかったことを意味しない。

### 4.6 lossとaccuracyは同じ量ではない

accuracyはargmaxが正解かだけを見る。cross-entropyは、正解classへ割り当てた確率を連続値として見る。そのため、argmaxが変わらない範囲で正解classの確率が少し上がれば、lossだけが下がることはある。

ただし今回は、lossの低下自体が約$10^{-4}$と小さく、accuracyもrandom baseline付近である。この一般的な指標の違いだけを使って、学習が実質的に進んだとは評価しない。

---

## 5. 設計contractと保存artifactの対応

### 5.1 確認できた項目

- $n_b/N$で重み付けした全データmean gradient。
- 固定順序のtraining loader。
- TT pairと通常parameterを同じ更新前stateから更新。
- right / leftで特異値の吸収先を変えるSVD再分割。
- candidateの検証後に一括commit。
- 5 pairずつの左右sweep。
- sweep 0を含むtrain / validation履歴。
- early-stopping rule、strict best、`max_sweeps`停止。
- 50局所更新の成功・方向・pair・相対更新量。
- initial / best / final checkpointの分離。
- bestを新modelへ再読込したvalidation値の一致。
- testとrank-selection splitを走査していないこと。

### 5.2 当初の設計にはあったが、今回のartifactだけでは実測確認できない項目

- sweep別・pair別wall-clock time。
- CUDA同期を含む総学習時間。
- peak memory。
- parameter数と計算量。
- 明示的な累積full-data pass数。
- sweepごとのrank列。
- canonicality errorの数値履歴。
- SVD前後の再構成相対誤差。
- 更新前、gradient step直後、SVD打ち切り後の3種類のdiagnostic loss。
- candidate validation各項目の個別結果。
- 通常fine-tuningやminibatch方式との比較値。

実装testでcontractを検証していても、今回のCSVへ時系列として保存されていない値は、実験結果として後から比較できない。次の比較実験では、少なくともtime、full-data pass数、peak memory、rank、truncation診断を保存対象へ加える。

また、Notebookの教材形式は作成途中で方針を変更し、穴埋めTODOではなく、完成した基準実装を読んで理解する形式にした。これは本Notebookについての明示的な決定であり、TODOがないことを未完成とは扱わない。

---

## 6. 次の比較実験の完全な設計

### 6.1 次に分けるべき問い

00の結果だけでは、学習がほぼ進まなかった理由を切り分けられない。次の実験では、少なくとも次の四つを別の因子として扱う。

1. TT更新と通常parameter更新の寄与。
2. 全体を何sweepするか。
3. 同じpairを何回続けて訪問するか。
4. TT側と通常parameter側の学習率。
5. 同じ結合テンソルを複数回更新してから1回だけSVDするinner optimizationを導入するか。

rank・tensorization・optimizerまで同時に変えると原因を読めなくなるので、最初の比較では固定する。

### 6.2 比較A：更新対象の$2\times2$ ablation

| 条件 | TT pair | `fc1.bias`・`fc2`・`fc3` | 目的 |
|---|---|---|---|
| A0 | 固定 | 固定 | 評価・保存だけで指標が変わらない対照 |
| A1 | 更新 | 固定 | TT局所更新だけの効果 |
| A2 | 固定 | 更新 | 通常parameter更新だけの効果 |
| A3 | 更新 | 更新 | 00と同じ共同更新 |

同じ初期state、同じsplit、同じデータ順、同じsweep数、同じ評価時点を使う。A1、A2、A3の差を見れば主な寄与を切り分けられるが、A3の効果がA1とA2の和になるとは限らない。相互作用があるためである。

### 6.3 比較B：sweep数と同一pairの反復訪問数を別々に変える

00は最大5 sweep、各pair 1 stepだった。次は、次の二軸を別実験で変える。

- sweep数$S$：全pairを何往復するか。
- `local_steps_per_pair` $L$：同じpairで、結合、全データ勾配、SVD再分割、atomic commitまでの一式を何回繰り返すか。

一方を変えるときは他方を固定する。さらに、結果は二つの予算で読む。

1. **同じsweep数での比較**：局所stepを増やしたとき、1 sweepあたりどこまで進むか。
2. **同じfull-data pass数での比較**：計算予算をそろえたとき、sweepを増やすかpair内stepを増やすかのどちらが効率的か。

6 coreでは予算が

$$
N_{\mathrm{full\ pass}}=10LS
$$

なので、必ずこの値を記録する。単に「5 sweep対10 sweep」と比較して、計算量の差を無視しない。

現行APIの`local_steps_per_pair>1`では、各stepの最後にSVD再分割とcommitを行い、次のstepは最新coreから結合テンソルを作り直す。したがって、これは「同じ未分割の$T$を複数gradient step動かし、最後に1回だけSVDする方式」と同じではない。

後者を試す場合は、別因子としてinner step数$K$を導入する。各inner stepで最新の$T$に対して全データ勾配を再計算し、$K$ step後に1回だけSVDする。通常parameterをinner stepごとに更新するのか、pair終了時に1回だけ更新するのかも事前に固定する。この方式と現行の`local_steps_per_pair`を同じ名前で比較しない。

### 6.4 比較C：TT側と通常parameter側の学習率

00では両方を$10^{-3}$にしたが、相対更新量はTT側と通常parameter側で異なった。したがって、二つの学習率を一括で上げ下げせず、

```text
TT learning rateだけを変更
通常parameter learning rateだけを変更
```

を分ける。各条件でgradient norm、parameter norm、相対更新量、非有限発生、validation lossを保存する。学習率を大きくして変化量が増えても、安定性やvalidationが悪化する可能性があるため、「よく動いた」だけで良い条件とはしない。

### 6.5 初期TT化の診断は学習scheduleと分ける

初期TT再構成weight normがdenseの14.4%だったため、学習開始前に次を保存する。

- dense `fc1.weight`とTT再構成weightのFrobenius誤差・相対誤差。
- dense初期modelとTT初期modelのlogits差。
- sweep 0のloss・accuracy。
- cutごとの特異値tailと保持率。

rankや初期化法を変える実験は、sweep数・pair内step数・学習率を切り分けた後の別段階にする。初期scaleを単純に合わせる操作はmodel関数も変えるため、診断なしに混ぜない。

### 6.6 全条件で固定するもの

- 同じ保存済み初期dense state。
- TT-SVD前後の同じseed。
- training / early-stopping / rank-selection split。
- `in_modes`、`out_modes`、rank cap 12。
- batch size 64、`shuffle=False`、`drop_last=False`。
- dtype、device、loss関数。
- right / leftのpair順序とSVD方向規則。
- fixed diagnostic batchのindexと順序。
- candidate validationとatomic commitの規則。
- validation評価時点、strict best、checkpoint形式。
- testとrank-selection splitを未使用にする境界。

### 6.7 必ず保存する値

```text
run id / seed / config
更新対象の条件
sweep数 S / pair内step数 L
full-data pass数
train・validation loss / accuracy
TT・通常parameterのparameter norm / gradient norm / relative update
pairごとのretained rank / discarded norm / truncation relative error
canonicality診断
成功・失敗・rollback数
wall-clock time / peak memory
initial / best / final checkpoint
best reload後の再現値
stop reason / partial sweep位置
```

### 6.8 判断の順序

1. まず単一seedで、実装差と主要因を切り分ける。
2. full-data pass数とwall timeを併記し、改善量だけでなく計算予算も比較する。
3. 有望な少数条件へ絞って複数seedを実行する。
4. validationで設計と候補を固定した後に、最後のtest評価を1回行う。

単一seedの小差、今回のrandom baseline付近のaccuracy、実測していない速度・memoryから一般的な優劣を結論しない。

---

## 7. 00の最終結論

Notebook 00で確認できたのは、全学習データ平均勾配を使うDMRG-like 2-core左右sweepを、6-core TT-MLP、方向付きSVD、atomic commit、validation、checkpoint再現まで接続できたことである。

保存結果では5 sweep・50局所更新のすべてが成功し、lossは一貫して微減した。しかし、accuracyは約9.5%でrandom baseline付近にとどまり、分類学習の明確な改善はなかった。TTと通常parameterを同時更新したため、微小な改善をTT更新単独へ帰属させることもできない。

したがって、00は学習方法の性能結論ではなく、**次のcontrolled experimentを行える基準実装を確立した実験**として位置付ける。次は、更新対象、sweep数、pair内step数、TT側・通常parameter側の学習率を一度に混ぜず、共通初期stateと計算予算の下で切り分ける。

## 関連

- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
- `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

