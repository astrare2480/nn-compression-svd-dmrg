---
title: Tucker・HOOI圧縮の評価設計
aliases:
  - Tucker評価指標
  - HOSVD HOOI比較設計
  - Tucker rank sweep
tags:
  - Tucker
  - HOOI
  - 実験設計
  - NN圧縮
  - CIFAR10
---

# Tucker・HOOI圧縮の評価設計

## サマリー

Tucker圧縮では、**数学的な近似品質**と**モデル圧縮としての有効性**を分けて評価する。

主な評価軸は、

```text
weight reconstruction
parameters
MACs
validation / test accuracy
fine-tuning recovery
decomposition time
```

である。

HOSVDとHOOIを同rankで比較すると構造は同じなので、parameters / MACsではなく、factor/coreの値、weight error、分解時間、task accuracyの差を見る。

---

## 1. weight relative error

今回の基準指標は

$$
\boxed{
e_W
=
\frac{\lVert W-\hat W\rVert_F}{\lVert W\rVert_F}
}
$$

で統一する。

絶対誤差ではなく相対化することで、weightのスケールが変わっても比較しやすくする。

---

## 2. accuracy drop

$$
\boxed{
\mathrm{accuracy\_drop}
=
\mathrm{accuracy}_{\mathrm{baseline}}
-
\mathrm{accuracy}_{\mathrm{compressed}}
}
$$

- 正：圧縮後accuracyが低い
- 0：同じ
- 負：圧縮後accuracyがbaselineより高い

小さい負値は、single seedや評価揺らぎも考慮し、「精度改善」と強く一般化せず、少なくとも性能を維持したと解釈する。

---

## 3. parameter reduction

$$
\boxed{
\mathrm{parameter\ reduction}
=
1-
\frac{N_{\mathrm{compressed}}}{N_{\mathrm{baseline}}}
}
$$

負なら、圧縮後の方がparameter数が増えたことを表す。

Tucker-2ではrankが大きいと前後1x1 Convの追加コストが効くため、負になる設定が実際に存在する。

---

## 4. MAC reduction

$$
\boxed{
\mathrm{MAC\ reduction}
=
1-
\frac{\mathrm{MACs}_{\mathrm{compressed}}}{\mathrm{MACs}_{\mathrm{baseline}}}
}
$$

Conv2単体とモデル全体の両方を分けて記録する。

```text
conv2_macs_reduction
all_macs_reduction
```

1層だけを大きく圧縮しても、モデル全体削減率は小さくなることがある。

また、

$$
\mathrm{MACs\ reduction}
\not\Rightarrow
\mathrm{latency\ reduction}
$$

である。3層化によるkernel launchや小さい演算の増加、hardware利用効率などがあるため、実測速度は別測定する。

---

## 5. rank sweep

Tucker-2ではrank pair

$$
(R_{\mathrm{out}},R_{\mathrm{in}})
$$

を探索する。

rankごとに最低限、

- model parameters
- target Conv parameters
- parameter reduction
- Conv / model MAC reduction
- validation loss
- validation accuracy
- accuracy drop
- weight relative error

を保存する。

rankの選択はparameter最小だけで決めず、accuracyとのPareto trade-offとして見る。

今回の最終候補は、

```text
aggressive   = (16, 24)
balanced     = (32, 16)
conservative = (32, 24)
```

である。

---

## 6. HOSVD / HOOI比較の公平条件

同じrankで比較する。

| 条件 | HOSVD | HOOI | TensorLy |
| --- | --- | --- | --- |
| 元weight | 同一 | 同一 | 同一 |
| 圧縮mode | 0 / 1 | 0 / 1 | 0 / 1 |
| rank | 同一 | 同一 | 同一 |
| 層構造 | 同一 | 同一 | 同一 |
| parameters / MACs | 同一 | 同一 | 同一 |
| 主比較 | error / time | error / time / history | error / time / history |

TensorLyは、可能な限り同じrank、SVD初期化、最大iteration、停止条件で合わせる。

最終比較の誤差は、各実装が内部報告する値を混ぜず、**返されたfactor/coreからweightを再構成し、同じ自作relative error関数で再計算**する。

---

## 7. decomposition time

GPUでは演算が非同期なので、時間計測の前後で必要に応じて

PyTorchの確認コード：[[06_Tucker基礎実装検証/25_評価設計のPyTorchコード#PyTorch確認-001]]

を入れる。

さらに数十msの単発計測では、

- warm-up
- 初回kernel起動
- backend処理
- Python/API overhead

の影響が相対的に大きい。

したがって、今回の

```text
HOSVD      17.6 ms
self HOOI  41.6 ms
TensorLy   83.8 ms
```

を「自作HOOIはTensorLyより常に2倍速い」と一般化しない。

TensorLyと自作HOOIは同じ6 iterationへ到達しているため、今回の小さいweightではTensorLyの汎用backend/API等の固定overheadが相対的に目立った可能性がある。ただしこれは**今回の計測に対する解釈**であり、大規模Tensorでも同じ速度比になる証拠ではない。

---

## 8. 圧縮直後とfine-tuning後を分ける

分解法そのものを比較する主表は、圧縮直後を使う。

```text
表A: 分解直後
method / rank / weight error / time /
parameters / MACs / val loss / val acc / test acc
```

fine-tuning後は別表にする。

```text
表B: fine-tuning後
initialization / seed / training condition /
post val acc / post test acc / recovery
```

fine-tuning後の精度には、

- learning rate
- optimizer
- epoch
- early stopping
- augmentation
- seed

が入るため、分解品質だけの差ではない。

---

## 9. 同条件fine-tuning比較

HOSVD初期化とHOOI初期化を比較するときは、

- 同じbaseline
- 同じrank
- 同じsplit
- 同じseed
- 同じoptimizer
- 同じlearning rate
- 同じepoch上限
- 同じearly stopping
- 同じshuffle / augmentation乱数系列

を可能な限り揃える。

今回の05ではseed 0のみなので、post test accuracyの

$$
0.7555-0.7508=0.0047
$$

という差は「HOOIが優れている」と統計的に一般化しない。

複数seedを実施していないため、結論は

> HOOI初期化はこのseedでHOSVDと同等以上の最終性能へ到達したが、優位性を保証する証拠は不足している。

とする。

---

## 10. weight errorとtask性能を混同しない

今回の最重要な観察は、

```text
HOOI
→ weight errorはHOSVDより小さい
→ 圧縮直後accuracyはHOSVDより低い
```

だったこと。

さらにfine-tuning後には、元weightからのrelative errorが増えてもaccuracyは大幅に改善した。

したがって、

$$
\boxed{
\text{元weightへの近さ}
\neq
\text{taskにとって最適なweight}
}
$$

を明示的に区別する。

この評価枠組みは次のTT/MPSでも引き継ぐ。

### weight誤差以外の目的を、全要素で区別する

weight relative errorはweightの全要素を同じ重みで比較する。しかし、実データを通した後の振る舞いを評価・学習するときは比較対象が変わる。代表的なfeature-map損失は

$$
\mathcal L_{\mathrm{feature}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\left\|
F_{\mathrm{baseline}}(x_n)
-F_{\mathrm{compressed}}(x_n)
\right\|_F^2
$$

である。1sampleの小さい例として

$$
F_{\mathrm{baseline}}
=
\begin{pmatrix}
1&2\\
0&-1
\end{pmatrix},
\qquad
F_{\mathrm{compressed}}
=
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
$$

なら、差と二乗和は

$$
\begin{aligned}
F_{\mathrm{baseline}}-F_{\mathrm{compressed}}
&=
\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix},\\
\left\|
F_{\mathrm{baseline}}-F_{\mathrm{compressed}}
\right\|_F^2
&=0^2+1^2+(-1)^2+0^2=2.
\end{aligned}
$$

$N=1$ なら $\mathcal L_{\mathrm{feature}}=2$ である。要素平均を使う実装では、さらに要素数4で割って0.5になるため、`sum` か `mean` かも固定する。

最終logitを比較するなら、

$$
\mathcal L_{\mathrm{logit}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\left\|
z_{\mathrm{baseline}}(x_n)
-z_{\mathrm{compressed}}(x_n)
\right\|_2^2.
$$

例えば

$$
z_{\mathrm{baseline}}
=
\begin{pmatrix}
2\\
0
\end{pmatrix},
\qquad
z_{\mathrm{compressed}}
=
\begin{pmatrix}
1.5\\
0.5
\end{pmatrix}
$$

なら、

$$
\left\|
z_{\mathrm{baseline}}-z_{\mathrm{compressed}}
\right\|_2^2
=(2-1.5)^2+(0-0.5)^2
=0.5.
$$

両方とも第1成分が最大なので予測classは同じである。accuracyだけではこの差を0として扱うが、logit損失は出力値の変化を捉える。

温度 $T$ を使うknowledge distillationなら、baselineをteacherとして

$$
\mathcal L_{\mathrm{KD}}
=
T^2
\operatorname{KL}
\left(
\operatorname{softmax}
\left(
\frac{z_{\mathrm{baseline}}}{T}
\right)
\,\middle\|\,
\operatorname{softmax}
\left(
\frac{z_{\mathrm{compressed}}}{T}
\right)
\right)
$$

を使える。正解labelを直接使うtask損失は

$$
\mathcal L_{\mathrm{task}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\operatorname{CE}
\left(
y_n,
f_{\mathrm{compressed}}(x_n)
\right)
$$

である。目的に応じて、

$$
\mathcal L
=
\mathcal L_{\mathrm{task}}
+\lambda_{\mathrm{feature}}\mathcal L_{\mathrm{feature}}
+\lambda_{\mathrm{logit}}\mathcal L_{\mathrm{logit}}
$$

のように組み合わせる。

標準HOOIで上位左特異ベクトルを選べるのは、直交射影に対するweight Frobenius誤差を目的にしているからである。feature-map、logit、KD、Cross Entropyには入力データや後段の非線形処理が入り、同じSVDの閉形式更新へ単純には置き換えられない。HOSVDまたはHOOIでfactor/coreを初期化し、3層Convへ置換した後にbackpropagationでこれらの損失を最適化する、という役割分担になる。

---

## 11. 参照結果

- [[06_Tucker基礎実装検証/02_Tucker2_Convとrank_sweepの確認結果]]
- [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
- [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
