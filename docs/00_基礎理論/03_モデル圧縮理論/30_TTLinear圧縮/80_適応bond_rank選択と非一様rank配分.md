---
title: 適応bond rank選択と非一様rank配分
aliases:
  - adaptive TT bond rank selection
  - nonuniform TT rank allocation
tags:
  - TT
  - MPS
  - TTLinear
  - adaptive rank
  - nonuniform rank
  - discarded weight
  - model selection
---

# 適応bond rank選択と非一様rank配分

## このノートの目的

TTの内部bond rankを決める方法を、次の三段階に分けて整理する。

1. SVD時にdiscarded weightの閾値からrankを選ぶ。
2. 学習済みTTのbondを一つずつ削り、局所感度を測る。
3. 保存量・MAC予算の下で、bondごとに異なるrankを配分する。

ここでは一般技術を扱う。Fashion-MNIST MLPの実測結果は、次へ分ける。

- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/18_adaptive_bond_rank_DMRG-likeと閾値探索]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/19_単一bond事後切り詰めと分類感度診断]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/20_非一様bond_rank配分と選択基準比較]]

---

## 1. rank cap、actual rank、必要rankを区別する

内部bond $k$の上限を$r_k^{\max}$、実際に保持したrankを$r_k$とする。

$$
r_k\le r_k^{\max}
$$

`rank cap = 12`は「12が必要」という意味ではなく、「12より大きくしない」という制約である。SVD後にrank 12が選ばれても、次の三つは区別する。

- 閾値を満たす最小rankが12だった。
- rank 12でも閾値を満たせず、上限に制限された。
- 実装が常に上限rankを残しており、適応選択が働いていない。

したがって、保存すべき値は少なくとも、

- `rank_cap`
- `threshold_required_rank`
- `retained_rank`
- `threshold_met`
- `cap_limited`

である。

---

## 2. discarded weightによる適応rank選択

2-core結合行列の特異値を、

$$
\sigma_1\ge\sigma_2\ge\cdots\ge0
$$

とする。rank $r$で切ったときのrelative discarded weightを、

$$
\delta_r
=
\frac{\sum_{j>r}\sigma_j^2}
{\sum_j\sigma_j^2}
$$

と定義する。閾値$\tau$を使う最も単純な規則は、

$$
r_*
=
\min\left\{
r\le r^{\max}
\mid
\delta_r\le\tau
\right\}
$$

である。集合が空なら$r^{\max}$を残し、`threshold_met=False`と記録する。

### 小さい数値例

特異値の二乗が、

$$
(64,16,4,1)
$$

なら、全エネルギーは85である。rank 2では、

$$
\delta_2=\frac{4+1}{85}\approx0.0588
$$

である。$\tau=0.1$ならrank 2を選べるが、$\tau=0.01$ならrank 3が必要になる。

閾値が小さいほど条件は厳しく、通常は高rankを残す。ただし、実際のrankはbondごとのspectrumに依存するため、同じ閾値でも一様rankになるとは限らない。

---

## 3. 分類lossにはrank削減圧力が自動では入らない

通常の分類学習が最小化するのは、たとえばcross-entropy、

$$
\mathcal L_{mathrm{CE}}
=
-\frac1N\sum_{s=1}^N
\log p_\theta(y_s\mid x_s)
$$

である。この目的には、特異値を小さくする項やrankを減らす項が明示的には含まれない。

したがって、分類lossだけで学習した後にspectrumが自然に急減することはあり得るが、保証されない。適応rankが下がらなかったときに言えるのは、

> 今回の学習経路と閾値規則では、より小さいrankが選ばれなかった。

までである。次の主張は強すぎる。

- 低rankモデルは存在しない。
- 高rankの全方向が分類に必要である。
- adaptive rankという考え方自体が無効である。

rank削減を直接促したい場合は、特異値・核ノルム・rank surrogateへの正則化、または低rankから必要に応じて増やす設計を別に検討する。

---

## 4. 事後単一bond診断

学習済みrank列、

$$
(1,r_1,r_2,\ldots,r_{d-1},1)
$$

から一つのbond $k$だけを$r_k'$へ下げ、他bondを固定する。再学習を行わなければ、現在の解の周囲にある局所的な切り詰め感度を測れる。

この診断では、少なくとも次を分ける。

1. discarded weightまたは重み再構成誤差
2. 実入力に対する対象層出力誤差
3. logits差
4. cross-entropyとaccuracy
5. prediction変更数とclassification margin
6. parameter数と理論MACs

同じrank削減でも、bondごとに削る特異方向が異なる。したがって、rankの数だけから分類影響を予測できない。

また、重み誤差の順位とaccuracy低下の順位が一致する保証もない。重要なのは、削った方向が実入力にどれだけ励起され、後段と判定境界へどう伝わるかである。

---

## 5. 非一様rank配分

複数bondのrank候補集合を、

$$
\mathcal R_k=\{r_{k,1},\ldots,r_{k,m_k}\}
$$

とする。全rank配分は直積、

$$
\mathcal R
=
\mathcal R_1\times\cdots\times\mathcal R_{d-1}
$$

であり、候補数は、

$$
|\mathcal R|
=
\prod_{k=1}^{d-1}m_k
$$

になる。各bondに4候補、対象bondが4本なら$4^4=256$候補である。

予算$B$の下で候補を選ぶ問題は、

$$
\min_{\mathbf r\in\mathcal R}
S(\mathbf r)
\quad
\text{subject to}
\quad
C(\mathbf r)\le B
$$

と書ける。$C$はparameter数または理論MACs、$S$は候補選択用scoreである。

「非一様rankであること」自体が目的ではない。一様rankが同じ予算で最良なら、一様rankを選ぶのが正しい。

---

## 6. 三つの選択基準

### 6.1 Weight-aware

元の実効重み$W_0$と候補$W_{\mathbf r}$の差を使う。

$$
S_{mathrm{weight}}(\mathbf r)
=
\frac{\|W_{\mathbf r}-W_0\|_F}
{\|W_0\|_F}
$$

データ不要で計算できるが、入力分布と後段の感度を見ない。

### 6.2 Activation-aware

calibration入力$x_s$に対する対象層出力を使う。

$$
S_{mathrm{activation}}(\mathbf r)
=
\frac{
\left(
\sum_s
\|z_{\mathbf r}(x_s)-z_0(x_s)\|_2^2
\right)^{1/2}}
{
\left(
\sum_s\|z_0(x_s)\|_2^2
\right)^{1/2}}
$$

実入力への作用を含むが、後段や正解labelは見ない。

### 6.3 Task-aware

calibration上のtask loss増加を使う。

$$
S_{mathrm{task}}(\mathbf r)
=
\mathcal L_{mathrm{cal}}(\mathbf r)
-
\mathcal L_{mathrm{cal}}(\mathrm{source})
$$

最終目的に最も近いが、calibrationへの選択適応が起こる。候補選択と性能報告に同じデータを使わない。

---

## 7. calibration、validation、oracleを分ける

選択規則を比較するときは、

- calibration：Weight以外のscore計算と候補選択
- validation：選択後候補の評価
- test：最終候補を固定した後の一度だけの評価

を分ける。

validation全候補を見て最小loss候補を求める`oracle`は、選択規則のregretを測る診断である。

$$
\operatorname{regret}
=
\mathcal L_{mathrm{val}}(\widehat{\mathbf r})
-
\min_{\mathbf r:C(\mathbf r)\le B}
\mathcal L_{mathrm{val}}(\mathbf r)
$$

oracleを実運用の選択結果として扱ってはならない。validationを使って全候補から選んでいるためである。

---

## 8. 逐次切り詰めは順序に依存する

複数bondを一つずつ切る場合、先に行った切り詰めで後続bondのsupercoreとspectrumが変わる。

$$
\mathcal T_{k_2}\!\left(
\mathcal T_{k_1}(W)
\right)
\ne
\mathcal T_{k_1}\!\left(
\mathcal T_{k_2}(W)
\right)
$$

が一般に起こり得る。ここで$\mathcal T_k$はbond $k$の切り詰め操作である。

したがって、forward順とreverse順で同じrank列を作っても、同じ実効重みになる保証はない。次を分けて保存する。

- rank配分
- 切り詰め順序
- 実効重みまたはcheckpoint hash
- 各段階のspectrumとdiscarded weight
- 最終task指標

一つの順序で最良だったrank配分を、順序に依存しない最適解とは呼ばない。

---

## 9. 単一bond感度から複数bond候補を予測する

全候補評価が高価な場合、単一bond削減のscoreを組み合わせて複数bond候補を近似する。

bond $k$をrank $r_k$へ削った単独scoreを$s_k(r_k)$とする。直接加算は、

$$
\widehat S_{mathrm{sum}}(\mathbf r)
=
\sum_k s_k(r_k)
$$

である。$s_k$がノルム量なら、二乗量を足して最後に平方根を取る、

$$
\widehat S_{mathrm{sq}}(\mathbf r)
=
\sqrt{\sum_k s_k(r_k)^2}
$$

も候補になる。

ただし、複数bondの誤差には交差項と逐次切り詰めの相互作用がある。どちらも厳密な加法則ではない。単一bond診断は、

- 候補順位をどの程度再現したか
- 予算内の最良候補を選べたか
- 選択候補のvalidation性能が一様対照を上回ったか

を別々に検証する。

候補評価数が減っても、wall time短縮を実測するまでは「高速化した」とは言わない。

---

## 10. 結果の読み方

### adaptive rankが下がらなかった場合

- 指定閾値と現在のspectrumの組合せではrank削減が起きなかった。
- rank削減圧力の欠如は原因候補だが、因果は確定しない。
- 事後切り詰め、低rankからのrank増加、正則化は別の設計である。

### 重みscoreとtask scoreが異なる候補を選んだ場合

- scoreが測る段階が異なるため、矛盾ではない。
- Weightは保存忠実度、Activationは入力上の層応答、Taskはlabelを含む目的を測る。

### proxyがoracleと一致した場合

- その候補集合・calibration・予算では成功した。
- 別seed、別source、別予算での一般保証ではない。

### 高rank候補が低rank候補より悪かった場合

表現集合の包含関係は、有限更新・異なる初期値・optimizer経路で得た学習結果の単調性を保証しない。容量と最適化結果を分ける。

---

## 11. チェックリスト

- rank cap、必要rank、保持rankを分けて保存する。
- discarded weightの分母と、二乗量か平方根かを明記する。
- 同じsource checkpointから候補を生成する。
- calibrationとvalidationを分ける。
- 一様対照を必ず残す。
- rank列だけでなく切り詰め順序を保存する。
- Weight・Activation・Task scoreを同じ意味として扱わない。
- oracleは診断に限定する。
- 小さいaccuracy差を単一seedで一般化しない。
- parameter削減、理論MAC削減、latency改善を分ける。
- 事後切り詰めの結果を、fine-tuning後の限界性能とみなさない。

## 関連

- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/72_TT圧縮の重み近似指標と分類性能指標の読み分け]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/78_accuracy許容差とParetoによる圧縮候補選択]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/79_データ計量付きTT_bond切り詰めとALS]]
