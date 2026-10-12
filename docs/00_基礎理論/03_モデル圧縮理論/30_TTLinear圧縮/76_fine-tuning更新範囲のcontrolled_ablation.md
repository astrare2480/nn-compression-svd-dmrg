---
title: Fine-tuning更新範囲のcontrolled ablation
tags:
  - model-compression
  - fine-tuning
  - ablation
  - layer-freezing
  - optimization
---

# Fine-tuning更新範囲のcontrolled ablation

## 1. このノートの目的

圧縮した層だけを更新するfine-tuningと、モデル全体を更新するfine-tuningを比較するときの一般的な読み方を整理する。

- 更新対象を減らすこととモデルを小さくすることの違い
- frozen層を通るforward・backward
- 同じ初期値と学習条件を使うcontrolled ablation
- 同じ学習率で比較して言える範囲
- task lossとteacherへの忠実度の違い
- early stopping、Adam、局所最小点の違い
- 単一試行のaccuracy差を評価する方法
- 出力指標から内部特徴を断定しない理由

Fashion-MNIST MLPの実測結果は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/06_TTコアのみ更新ablation]]で扱う。

## 2. 更新対象の2方式

3-core TTLinearのcoresを$G_1,G_2,G_3$、第1層biasを$b_1$、後段dense層を$(W_2,b_2)$、$(W_3,b_3)$とする。

全parameter更新は、

$$
\theta_{\mathrm{all}}
=
\{G_1,G_2,G_3,b_1,W_2,b_2,W_3,b_3\}
$$

を更新する。TT coresのみの更新は、

$$
\theta_{\mathrm{TT}}
=
\{G_1,G_2,G_3\}
$$

だけを更新し、残りを固定する。

ここで「固定」はparameterを削除することではない。`requires_grad=False`により、そのparameter自身の勾配保存とoptimizer更新を止める。予測に必要な層はモデル内に残るため、同じrankなら総parameter数、checkpointの構造、推論時のparameter数は変わらない。

## 3. frozen層にも計算は残る

後段の$W_2,W_3$を固定しても、予測にはforwardが必要である。また、前段のTT coresへlossの勾配を返すには、固定した後段を通した入力勾配が必要である。

$$
\frac{\partial L}{\partial G_k}
=
\frac{\partial L}{\partial z_3}
\frac{\partial z_3}{\partial h_2}
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_1}{\partial G_k}
$$

$W_2,W_3$自身を更新しなくても、これらが定めるJacobianはTT coresへの勾配に現れる。従って、学習可能parameter数を80%減らしても、試行時間が80%短くなるとは限らない。

総学習時間は概ね、

$$
T_{\mathrm{total}}
\approx
T_{\mathrm{epoch}}
\times
N_{\mathrm{executed\ epochs}}
$$

で決まる。更新対象削減による1 epochの短縮があっても、early stoppingまでのepoch数が増えれば試行全体は長くなり得る。

## 4. controlled ablationの条件

更新範囲の影響を分離するには、各方式で次をそろえる。

1. 同じdense checkpointから作った同一の圧縮済みepoch 0
2. 同じデータ分割
3. 同じseedとmini-batch順序
4. 同じoptimizer、学習率、weight decay
5. 同じepoch 0を含むcheckpoint候補
6. 同じearly-stopping規則
7. 同じ評価関数

同じrankの2方式でepoch 0の全state tensorを`torch.equal`で確認すると、結果差を初期値差ではなく更新範囲差へ結びつけやすい。frozen parameterも学習前後で完全一致を検査する。

ただし、比較の問いを区別する。

| 問い | 同じ学習率のablationで答えられるか |
| :-- | :-- |
| 共通の学習条件で更新範囲だけを変えるとどうなるか | 答えられる |
| 各方式に最適なhyperparameterを与えたとき、どちらが最高性能か | 答えられない |

同じ学習率を使うこと自体が不公平なのではない。controlled ablationとして必要な固定である。一方、更新する層と最適化する部分空間が異なるため、各方式に適した学習率が同じとは限らない。

## 5. 更新部分空間と自由度

モデル出力$f$の微小変化を一次近似すると、全parameter更新では、

$$
\Delta f
\approx
J_G\Delta G
+
J_{b_1}\Delta b_1
+
J_{W_2}\Delta W_2
+
J_{b_2}\Delta b_2
+
J_{W_3}\Delta W_3
+
J_{b_3}\Delta b_3
$$

となる。TT coresのみでは、

$$
\Delta f
\approx
J_G\Delta G
$$

に制限される。全parameter更新は圧縮誤差への補正を複数層へ分担できる一方、task lossだけを最適化するとteacherから離れる方向も含めて自由度が大きい。TT coresのみでは、固定した後段が利用できる特徴を前段だけで作る必要がある。

ただし、自由度と収束速度に単調な一般則はない。選択epochが遅いだけでは、同じaccuracyへの到達が遅いとは言えない。選択epochは事前に固定したvalidation lossが最良だった時点であり、accuracyの閾値到達時点とは異なる。

## 6. task lossとdense忠実度は別の目的である

正解labelに対するCross Entropyを最小化するfine-tuningでは、teacherであるdenseモデルのlogitsは目的関数に入らない。

$$
L_{\mathrm{task}}
=
-\frac{1}{N}
\sum_{n=1}^{N}
\log p_{n,y_n}
$$

一方、logits RMSEは、

$$
\operatorname{RMSE}_{\mathrm{logits}}
=
\sqrt{
\frac{1}{NC}
\sum_{n=1}^{N}
\sum_{c=1}^{C}
\left(z^{\mathrm{TT}}_{n,c}-z^{\mathrm{dense}}_{n,c}\right)^2
}
$$

でteacherへの数値的な近さを測る。prediction agreementはargmaxだけを比較し、accuracyは正解labelを比較する。

### 小さい数値例

正解classを2番目とし、dense logitsを$(1,3,0)$、圧縮モデルを$(2,6,0)$とする。両方のargmaxは2番目なのでagreementとaccuracyは保てるが、logits RMSEは0ではない。逆にdenseが誤り、圧縮モデルが別classへ変化して正解した場合、accuracyは上がるがagreementは下がる。

従って、task accuracyの回復はdense logitsの復元を意味しない。dense忠実度も目的にするなら、蒸留lossや複数指標の制約を実験前に設計する必要がある。

## 7. fine-tuning中のRMSE増大を読む

全parameter更新でRMSEが大きく、coresのみで小さい場合、広い更新範囲が追加の乖離へ寄与した可能性はある。しかしcoresのみでも大幅に増えたなら、後段更新だけを原因にはできない。前段の出力が変われば、固定した後段へ入るactivationも変わり、最終logitsも変わる。

raw logits RMSEだけでは、次を区別できない。

- 全classへの共通offset
- logits scaleやmarginの変化
- class間の相対的な形の変化
- argmaxの入替え

agreementも同時に低下していれば、少なくともargmaxの入替えを伴う。一方、内部でどの意味的特徴を失い、何を獲得したかは、accuracy、agreement、RMSEの集計値からは分からない。

内部表現を調べるなら、同じsampleについて層activationを保存し、CKAなどで表現類似度を測る。ただし、表現類似度から特定の意味的特徴を直接命名するには、さらにfeature visualizationやsample分析が必要である。

## 8. Adam、early stopping、局所最小点

Adamはtrain mini-batchのlossから勾配を計算し、parameterを更新する。early stoppingはvalidation lossを観測し、checkpoint選択と停止を行う。この2つは別の役割である。

簡単な関数、

$$
L(x)=x^2,
\qquad
\frac{dL}{dx}=2x
$$

を考える。$x=3$で勾配6を得ても、勾配を計算した時点で最小点$x=0$へ到達したわけではない。同様に、Adamで勾配を使ったことは最小点への到達を保証しない。

early stoppingで分かるのは、今回の更新経路で観測したepoch checkpointの間では、patienceの期間に意味のあるvalidation改善がなかったことである。次は保証しない。

- train lossの勾配が0である。
- validation lossの勾配が0である。
- parameter空間の全方向でlossが増える局所最小点である。
- epoch間の途中に、より良い状態が存在しない。
- 別の学習率・optimizer・seedでも改善しない。

## 9. fine-tuning不要と学習率過大を分ける

epoch 0が良く、最初の1 epochでvalidation lossとRMSEが悪化した場合、少なくとも次の仮説が残る。

1. epoch 0の周辺にvalidation改善余地がほとんどない。
2. 学習率が大きく、良い領域の近傍を細かく調整できなかった。
3. train lossへの適合がvalidationへ一般化せず、過学習した。
4. task lossは下がってもteacher忠実度を保つ目的がない。

切り分ける最初の実験では、同じepoch 0、データ分割、seed、mini-batch順、optimizerから、学習率だけを下げる。例えば元の`1e-3`に対して`3e-4`、`1e-4`を比較する。各試行は前の学習の続きではなく、同じepoch 0からoptimizerも作り直す。

小さい学習率では改善が遅くなる可能性があるため、元と同じpatienceで早く止めた結果を、そのまま「改善しない」と断定しない。まず学習率だけを変え、その後に必要なら学習期間を別要因として検討する。複数seedは学習率効果の切り分け後に行うと、原因を混同しにくい。

## 10. 単一試行のaccuracy差

同じ$N$ sampleに2モデル$A,B$を適用した比較は対応がある。$A$だけ正解した件数を$b$、$B$だけ正解した件数を$c$とすると、accuracy差は、

$$
\operatorname{acc}(B)-\operatorname{acc}(A)
=
\frac{c-b}{N}
$$

である。集計accuracyから分かるのは$c-b$だけであり、$b+c$、すなわち予測がどれだけ入れ替わったかは分からない。

十分な件数があれば、連続性補正付きMcNemar統計量、

$$
\chi^2
=
\frac{\left(|b-c|-1\right)^2}{b+c}
$$

を使える。件数が小さい場合はexact binomial版を用いる。これは固定された同一sample集合上の差を評価する。一方、学習seedを変えたときの変動は別問題なので、重要な手法差は複数seedでも確認する。

test setはhyperparameterや更新方式の選択に繰り返し使わない。validationで条件を固定した後、testを一度だけ最終評価する。

## 11. 実用的な確認手順

1. 同じrankの2方式を完全に同じepoch 0から開始する。
2. 更新するparameter名とoptimizerへ渡したparameterを照合する。
3. frozen parameterの完全一致、TT-rank、core shape、総parameter数を検査する。
4. epoch 0をbest候補に含め、事前に固定したvalidation lossで選ぶ。
5. accuracy、agreement、logits RMSEは役割を分けて記録する。
6. 選択モデルと学習途中のtrajectoryを混同しない。
7. 同じ学習率の結果はcontrolled ablationとして読み、最高性能比較とは分ける。
8. 小さいaccuracy差はsample別の対応と複数seedで確認する。
9. 内部特徴の獲得・喪失は出力集計値だけから断定しない。
10. validationで条件を固定した後にtestを評価する。

## 12. まとめ

- TT coresのみの更新は、圧縮層だけに最適化の自由度を制限するablationである。
- parameterを固定してもモデルから削除されず、forwardと前段への勾配伝播も残る。
- 同じ学習率で条件をそろえる比較は更新範囲の影響を調べられるが、各方式を個別調整した最高性能は比較できない。
- task lossを最小化したaccuracy回復は、dense logitsの復元を意味しない。
- early stoppingは観測trajectory上の選択規則であり、局所最小点への収束証明ではない。
- 単一seedの小さいaccuracy差では明確な優位性を主張せず、対応のある予測比較と複数seedを使う。
- 出力指標の変化から、内部の意味的特徴の獲得・喪失を直接断定しない。
