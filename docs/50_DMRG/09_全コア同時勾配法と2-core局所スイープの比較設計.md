---
title: 全コア同時勾配法と2-core局所スイープの比較設計
aliases:
  - global gradient descent versus DMRG-like local sweep
tags:
  - DMRG-like
  - TTLinear
  - full-batch gradient
  - local sweep
  - controlled comparison
  - computational budget
---

# 全コア同時勾配法と2-core局所スイープの比較設計

## このノートの目的

同じTTモデルと同じ分類目的を使っても、全TTコアを同時更新する方法と、隣接2コアだけを更新してSVD再分割するDMRG-like局所スイープでは、学習経路と計算費用が異なる。

ここでは両者を比較するときに、何をそろえ、何を別指標として報告するかを一般論として整理する。Fashion-MNISTの実測結果は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/21_全コア同時GDとDMRG-like局所スイープ比較]]へ分ける。

---

## 1. 二つの更新方式

TT coresを$G^{(1)},\ldots,G^{(d)}$、通常parameterを$\phi$とする。

### 1.1 全コア同時勾配法

全データ平均loss、

$$
\mathcal L(\theta)
=
\frac1N\sum_{s=1}^N
\ell(f_\theta(x_s),y_s),
\qquad
\theta=(G^{(1)},\ldots,G^{(d)},\phi)
$$

の勾配を全parameterについて求め、一度に更新する。

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta\mathcal L(\theta_t)
$$

1 gradient evaluationで全TTコアと通常parameterが1回ずつ更新される。

### 1.2 DMRG-like 2-core局所スイープ

現在のpair $(k,k+1)$だけを結合し、

$$
T^{(k,k+1)}
=
\operatorname{merge}\left(G^{(k)},G^{(k+1)}\right)
$$

を局所変数として勾配更新する。更新後はSVDで2コアへ戻し、正準中心を移動する。

```text
2-coreを結合
→ 全データ平均勾配
→ 結合テンソルを更新
→ SVD再分割
→ rank制約
→ 正準中心を移動
```

通常parameterを共同更新する設計では、現在のTT pairだけでなく$\phi$も毎局所更新で動く。

---

## 2. 同じモデル容量でも同じ最適化ではない

初期rank列、parameter数、forward関数が同じでも、更新変数の座標系が異なる。

- 全コア同時法：個別coreを同時に更新する。
- 局所法：2-core結合テンソルを直接更新し、SVDで再parameter化する。

TTにはgauge freedomがあるため、個別coreのユークリッド勾配と、結合テンソル上の勾配は同じparameter化ではない。さらに局所法には、

- SVD再分割
- rank制約
- 正準化
- pairの巡回順序

が入る。

したがって、同じ最終accuracyにならなくても実装矛盾ではない。逆に、一方が高accuracyでも、差を「2-core更新だけの因果効果」とは分離できない。

---

## 3. gradient evaluation数とparameter更新回数を分ける

full-dataset gradient evaluationを1回数えても、両方式で更新されるparameter数は異なる。

core数$d$、1 pair当たり局所step数$L$とする。左右1往復のpair visit数は、

$$
2(d-1)
$$

なので、1往復の局所更新数は、

$$
N_{mathrm{local/sweep}}
=
2(d-1)L
$$

である。

左右往復では端のcoreは1方向につき1 pair、内部coreは2 pairに含まれる。1往復当たりの更新回数は、

$$
N_{mathrm{edge}}=2L,
\qquad
N_{mathrm{inner}}=4L
$$

となる。通常parameterを共同更新するなら、

$$
N_{\phi}=2(d-1)L
$$

である。

### 数値例

$d=6$、$L=8$なら1往復は80局所更新である。173往復では、

$$
N_{mathrm{edge}}=173\times16=2,768,
$$

$$
N_{mathrm{inner}}=173\times32=5,536,
$$

$$
N_{\phi}=173\times80=13,840
$$

となる。

同じ時点まで全コア同時法を2,800回実行した場合、全TTコアと通常parameterはすべて2,800回更新される。端coreはほぼ同じ回数でも、内部coreと通常parameterの更新機会は大きく異なる。

---

## 4. 「同じ予算」の候補

比較でそろえられる予算には複数ある。

1. full-dataset gradient evaluation数
2. optimizer stepまたは局所更新数
3. 各parameterの更新回数
4. 学習データを通過したsample数
5. 更新処理時間
6. validation評価を含む時間
7. 固定wall-clock時間

これらは同値ではない。特に局所更新は1回当たりの勾配対象が少ない一方、SVDと正準化を含む。したがって、evaluation数が多い方法が必ず実時間でも遅いとは限らない。

比較目的を先に決める。

- 学習手順全体の比較：各方式を自然な手順で実行する。
- 変数parameter化の因果比較：更新機会や通常parameterの更新頻度を追加controlでそろえる。
- 運用効率の比較：同じ目標性能へのwall timeを測る。

---

## 5. 最終性能と目標性能への到達時間

最終bestだけの比較では、どちらが早く有用な性能へ達したかが消える。共通の目標loss $L_*$を定め、

$$
t_*
=
\min\left\{
t\mid
\mathcal L_{mathrm{val}}(t)\le L_*
\right\}
$$

を比較する。

ただし、評価が80更新ごとなら、真の初回到達時点は区間内にあり、記録点は上界である。

次の時間を分ける。

$$
t_{mathrm{update}}
$$

はforward・backward・parameter更新・SVDなどの更新処理時間、

$$
t_{mathrm{update+eval}}
=
t_{mathrm{update}}+t_{mathrm{validation}}
$$

はvalidation評価を含む時間である。頻繁に評価する方式では、更新処理だけの到達時間が短くても、評価込みでは遅くなることがある。

---

## 6. final、best-loss、最高accuracyを区別する

classification accuracyとcross-entropyは同じモデル選択規則ではない。

- `final`：実行した最後のstate
- `best-loss`：validation lossが最小のcheckpoint
- `max-accuracy`：履歴中でaccuracyが最大の評価点

accuracyはargmaxだけを見るが、cross-entropyは正解classへ割り当てた確率も見る。そのため、loss最小時点とaccuracy最大時点は一致しなくてよい。

考察では、単に`best accuracy`と書かず、

- best-loss checkpointのaccuracy
- 履歴中の最高accuracy
- 別のrank-selection splitでのaccuracy

を区別する。

---

## 7. train–validation gapと正則化説

DMRG-like法はSVDとrank制約を含むため、正則化効果を仮説として考えられる。しかし、validation性能が良かっただけでは正則化を証明しない。

best時点のgapを、

$$
g
=
\mathcal L_{mathrm{val}}
-
\mathcal L_{mathrm{train}}
$$

として比較する。

局所法でtrain lossもvalidation lossも低いが$g$は大きい場合、「gapを縮めたため良かった」という説明とは合わない。一方、gapが大きいだけで過学習と断定することもできない。validation loss自体、更新予算、学習率が異なるためである。

trainとvalidationが同時に悪化した場合も、単純な過学習より最適化の振動・不安定化と整合する。原因を確定するには学習率を変えたcontrolと細かい履歴が必要である。

---

## 8. 選択モデルの費用と探索全体の費用

複数学習率を試して最良条件を選んだ場合、費用を二つに分ける。

1. 選択済み設定を1回学習する費用
2. 設定を選ぶために全候補を実行した探索費用

新しいモデルへ手法を移すと学習率探索が再び必要になる可能性がある。最良1条件の時間だけを、研究全体の計算費用として報告しない。

同様に、推論時のparameter数・MACs・latencyと、学習時のgradient evaluations・時間・peak memoryも別軸である。

---

## 9. どこまで因果的に言えるか

同じ初期state、data split、loss、学習率候補、停止規則をそろえても、DMRG-like法の差には次が同時に含まれる。

- 結合2-coreを直接更新する変数空間
- SVD再分割
- 正準化とgauge移動
- rank制約
- 逐次巡回
- TT coreごとの更新頻度差
- 通常parameterの更新回数差

したがって、実験から直接言えるのは、

> 今回比較した学習手順全体では、どちらがどの性能と費用へ到達したか。

である。個別機構の効果を分けるには、たとえば次の追加controlが必要になる。

- 通常parameter更新頻度をそろえる。
- TT coreごとの更新回数をそろえる。
- SVD再分割だけを行うcontrolを入れる。
- 固定wall timeで比較する。
- 複数seedで同じ判断規則を適用する。

---

## 10. チェックリスト

- 同じ初期checkpointから独立に開始する。
- data splitとsample順を固定する。
- 学習率候補と選択規則を事前に定める。
- gradient evaluation、局所更新、parameter更新回数を分ける。
- 更新処理時間とvalidation評価時間を分ける。
- final、best-loss、max-accuracyを分ける。
- trainとvalidationの履歴を両方保存する。
- 目標性能への初回到達時間を評価間隔付きで報告する。
- 選択モデル1個の費用と探索全体の費用を分ける。
- 高いaccuracyだけから計算効率や正則化効果を主張しない。
- 単一seedの小差を一般化しない。

## 関連

- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[50_DMRG/07_TT-MLPのfull-batch共同更新と更新予算]]
- [[50_DMRG/08_full-batch勾配のGPU常駐高速化とbenchmark設計]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/16_Dense_通常TT_DMRG-like比較]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/21_全コア同時GDとDMRG-like局所スイープ比較]]
