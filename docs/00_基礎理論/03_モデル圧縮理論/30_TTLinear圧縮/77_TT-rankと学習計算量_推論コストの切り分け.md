---
title: TT-rankと学習計算量・推論コストの切り分け
tags:
  - model-compression
  - TTLinear
  - fine-tuning
  - computational-cost
  - inference
---

# TT-rankと学習計算量・推論コストの切り分け

## 1. このノートの目的

TTLinearをfine-tuningするとき、次の量は同じではない。

1. モデル総parameter数
2. 学習対象parameter数
3. 1 epoch当たりの学習時間
4. early stoppingまでの総学習時間
5. 推論時のparameter数・演算量・latency

TT-rankを下げる操作と、`requires_grad=False`で更新範囲を狭める操作は、異なる量へ作用する。本ノートでは、この区別を数式と小さい数値例で整理する。

更新範囲のcontrolled ablation、frozen層を通る勾配、task lossとdense忠実度の違いは、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/76_fine-tuning更新範囲のcontrolled_ablation]]を参照する。Fashion-MNISTの実測は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/07_rank12追加と更新範囲比較]]で扱う。

## 2. TT-rankを下げるとモデル構造が小さくなる

3-core TT-matrixのcore shapeを、

$$
G_1 \in \mathbb{R}^{1 \times m_1 \times n_1 \times r},
$$

$$
G_2 \in \mathbb{R}^{r \times m_2 \times n_2 \times r},
$$

$$
G_3 \in \mathbb{R}^{r \times m_3 \times n_3 \times 1}
$$

とする。TT coresのparameter数は、

$$
P_{\mathrm{TT}}(r)
=
m_1n_1r
+
m_2n_2r^2
+
m_3n_3r
$$

である。中央coreに$r^2$の項があるため、rankを下げるとTT層自体のparameter数と収縮計算量が減る。

モデル全体では、TT化していないparameter数を$P_{\mathrm{fixed}}$とすると、

$$
P_{\mathrm{model}}(r)
=
P_{\mathrm{TT}}(r)
+
P_{\mathrm{fixed}}
$$

となる。rankを0へ近づけることはできないが、概念上、TT coresを小さくしても$P_{\mathrm{fixed}}$は残る。この項がモデル全体の圧縮下限を作る。

## 3. 数値例：rank 16から12への変更

Fashion-MNIST MLPの`fc1`で使ったmodeは、

$$
(m_1,m_2,m_3)=(32,8,2),
$$

$$
(n_1,n_2,n_3)=(2,8,49)
$$

である。従って、

$$
P_{\mathrm{TT}}(r)
=
32 \cdot 2 \cdot r
+
8 \cdot 8 \cdot r^2
+
2 \cdot 49 \cdot r
$$

$$
=
64r^2+162r
$$

となる。

rank 12では、

$$
P_{\mathrm{TT}}(12)
=
64 \cdot 12^2
+
162 \cdot 12
=
11{,}160
$$

である。rank 16では、

$$
P_{\mathrm{TT}}(16)
=
64 \cdot 16^2
+
162 \cdot 16
=
18{,}976
$$

となる。TT coresだけを見ると、rank 16から12で、

$$
\frac{18{,}976-11{,}160}{18{,}976}
\approx
0.412
$$

より約41.2%減る。

一方、TT cores以外のparameterが134,410個残るため、モデル全体は153,386から145,570への減少である。

$$
\frac{153{,}386-145{,}570}{153{,}386}
\approx
0.051
$$

モデル全体では約5.1%しか減らない。局所的なTT層の削減率と、ネットワーク全体の削減率を混同してはいけない。

## 4. 更新範囲を狭めてもモデル総parameter数は減らない

同じrankのモデルで、全parameter更新とTT coresのみ更新を比較する。parameter集合を、

$$
\theta
=
\{G,b_1,W_2,b_2,W_3,b_3\}
$$

とする。全parameter更新では全要素をoptimizerへ渡し、TT coresのみ更新では$G$だけを渡す。

しかし、`requires_grad=False`はparameterを削除しない。固定した$b_1,W_2,b_2,W_3,b_3$もforwardに必要で、`state_dict`にも残る。従って、同じrankなら、

$$
P_{\mathrm{model}}^{\mathrm{all}}
=
P_{\mathrm{model}}^{\mathrm{cores}}
$$

である。一方、学習対象parameter数は、

$$
P_{\mathrm{trainable}}^{\mathrm{all}}
>
P_{\mathrm{trainable}}^{\mathrm{cores}}
$$

となる。

更新範囲の変更は学習問題の自由度を変えるが、それだけでは配備モデルを追加圧縮しない。

## 5. 学習対象数と1 epochの時間は比例しない

1 epochの学習時間を概念的に、

$$
T_{\mathrm{epoch}}
=
T_{\mathrm{data}}
+
T_{\mathrm{forward}}
+
T_{\mathrm{backward}}
+
T_{\mathrm{optimizer}}
$$

と分ける。

TT coresのみ更新で削減できるのは、主に固定parameter自身の勾配計算・保存とoptimizer更新である。次は残る。

- 全層のforward
- loss計算
- TT coresへ勾配を返すため、固定後段を通る入力勾配
- data loaderやdevice転送
- TT core収縮とそのbackward

従って、学習対象parameterを90%減らしても、$T_{\mathrm{epoch}}$が90%減るとは限らない。高rankではTT core収縮自体の費用が大きくなり、optimizer更新の削減が全体時間へほとんど現れない場合もある。

## 6. 総学習時間には実行epoch数が効く

early stoppingまでの総学習時間は概ね、

$$
T_{\mathrm{total}}
\approx
T_{\mathrm{epoch}}
\times
N_{\mathrm{executed\ epochs}}
$$

で決まる。

例として、方式Aが4秒/epochで10 epoch、方式Bが3秒/epochで20 epoch実行したとする。

$$
T_A=4 \times 10=40\ \mathrm{s},
$$

$$
T_B=3 \times 20=60\ \mathrm{s}
$$

である。方式Bは1 epoch当たり25%速いが、停止までの総時間は50%長い。

従って、計算効率を述べるときは、少なくとも次を分けて記録する。

- 秒/epoch
- 選択epoch
- 停止epoch
- 停止までの累積時間
- 同じaccuracyまたはloss閾値への到達時間

選択epochはbest checkpointの位置、停止epochはそのbestを採用すると判断するために実際に計算した範囲である。選択モデルがepoch 7でも、patience確認のためepoch 12まで学習したなら、実験時の計算費用は12 epoch分である。

## 7. 選択epochは収束速度そのものではない

early stoppingが選ぶのは、事前に決めたvalidation指標が最良だったepochである。これは次と同じではない。

- train lossの勾配が0になった時点
- 同じaccuracyへ初めて到達した時点
- parameterが動かなくなった時点
- 理論上の局所最小点

更新範囲Aの選択epochが5、Bが15でも、それだけで「Bは3倍収束が遅い」とは言えない。到達したloss水準が異なる場合や、選択指標と比較したいaccuracyが異なる場合がある。

更新対象数と選択epochの因果を調べるなら、複数seed、学習率、lossまたはaccuracyの共通閾値を事前に決めて比較する。

## 8. 推論コストは更新範囲ではなくforward構造で決まる

学習後の推論ではoptimizerを使わず、`requires_grad`も通常のforward演算を変えない。同じrank、同じcore shape、同じ後段構造なら、全parameter更新モデルとTT coresのみ更新モデルが実行する演算の種類とshapeは同じである。

従って、理論上の推論parameter数と演算量は同じである。ただし、parameter値の違い、kernel選択、hardware状態などにより、実測latencyが完全一致するとは限らない。latencyを主張するならwarm-upと反復測定を別に行う。

TT-rankを変える場合はcore shapeが変わるため、推論演算量も変わる。整理すると、

| 操作 | 学習対象数 | モデル総parameter数 | 理論上の推論構造 |
| :-- | :-- | :-- | :-- |
| 同じrankで更新範囲だけを狭める | 減る | 同じ | 同じ |
| TT-rankを下げる | 減る | 減る | 小さくなる |
| 別の層も圧縮する | 設計による | 減り得る | 変わる |

## 9. optimizer stateと総メモリも分ける

Adamは学習対象parameterごとに一次・二次momentを保持するため、学習対象数を減らすとoptimizer stateは小さくなる。一方、総学習メモリには次が含まれる。

- 全parameter本体
- 学習対象parameterの勾配
- optimizer state
- forwardで保存するactivation
- backwardの一時tensor
- frameworkやallocatorの予約領域

従って、学習対象数の削減率から、GPUメモリ全体の削減率を直接計算できない。`torch.cuda.max_memory_allocated()`などで同じ条件を繰り返し測定する必要がある。

## 10. 比較時の実用的な記録項目

rankと更新範囲を比較するときは、次を別列で保存する。

1. requested rankとactual rank
2. core shape
3. モデル総parameter数
4. 学習対象parameter数
5. optimizer stateの対象parameter名
6. 選択epochと停止epoch
7. train部分の累積時間と秒/epoch
8. 評価を含むend-to-end時間
9. peak memory
10. 推論MACsと反復latency
11. accuracy、loss、agreement、logits RMSE

一つの「効率」という語へまとめず、モデルサイズ、学習時資源、停止までの費用、推論時資源を分ける。

## 11. まとめ

- TT-rankを下げるとTT core shapeが変わり、モデル総parameter数と推論構造が小さくなる。
- 同じrankで更新範囲だけを狭めても、固定parameterは残るためモデル総parameter数と推論構造は変わらない。
- 学習対象parameter数の削減率と、1 epochの時間短縮率は一致しない。
- 総学習時間は1 epochの時間だけでなく、early stoppingまでの実行epoch数にも依存する。
- 選択epochだけから収束速度を一般化しない。
- optimizer stateは小さくなり得るが、総メモリ削減率は実測なしに断定しない。
- モデルサイズ、学習計算量、総学習時間、推論コストを別々に記録・評価する。
