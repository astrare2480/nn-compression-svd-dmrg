---
title: Tucker-2 Convとrank sweepの確認結果
tags: [Tucker, CIFAR10, Conv2d, rank sweep, Fine-tuning]
---

# Tucker-2 Convとrank sweepの確認結果

## 1. 対象とbaseline

対象：`01_tucker2_conv.ipynb`、`02_rank_sweep.ipynb`、`03_finetuning.ipynb`。

$$
W_{conv2}\in\mathbb R^{64\times32\times3\times3}.
$$

baseline：

```text
val acc = 0.7344
test acc = 0.7327
params = 128,842
```

## 2. parameter式を今回の層へ代入

元Conv2 weight数：

$$
64\times32\times3\times3=18432.
$$

Tucker-2：

$$
N=32R_{in}+9R_{out}R_{in}+64R_{out}.
$$

### `(64,32)`

$$
\begin{aligned}
N
&=32\times32+9\times64\times32+64\times64\\
&=1024+18432+4096\\
&=23552.
\end{aligned}
$$

$$
23552-18432=5120
$$

増えるため、full channel rankでTucker形式にしただけでは圧縮ではない。

### balanced `(32,16)`

$$
\begin{aligned}
N
&=32\times16+9\times32\times16+64\times32\\
&=512+4608+2048\\
&=7168.
\end{aligned}
$$

Conv2 weight/MACの同一空間位置での削減率は

$$
1-\frac{7168}{18432}
=0.611111\ldots
\approx61.11\%.
$$

モデル全体parameterは

$$
128842\rightarrow117578
$$

なので、

$$
\begin{aligned}
1-\frac{117578}{128842}
&\approx0.087425\\
&\approx8.74\%.
\end{aligned}
$$

## 3. 現行 `conv_tucker.py` のTucker-2構造

Conv2d weight

```text
(C_out, C_in, kH, kW)
```

のmode 0 / 1を圧縮し、

```text
C_in
→ 1x1 Conv / U_in^T
→ R_in
→ kHxkW Conv / core
→ R_out
→ 1x1 Conv / U_out
→ C_out
```

へ置換する。

対応するshapeは、

```text
U_in  : (C_in,  R_in)
core  : (R_out, R_in, kH, kW)
U_out : (C_out, R_out)
```

である。

現行builderでは、中央core Convだけが元Convの `stride / padding / dilation / padding_mode` を引き継ぎ、前後のprojectionは1x1・stride=1・padding=0・dilation=1。元biasは最後の1x1 Convへ置く。

現在のTucker-2 Convは `groups=1` の通常 `nn.Conv2d` のみを対象とし、grouped convolutionや転置畳み込みを黙って近似しない。

## 4. rank contract

`rank_out` / `rank_in` は独立に検証する。

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

boolは整数として受理せず拒否する。

汎用HOSVD/HOOI側ではmode-n unfolding上の最大rankまで検証するが、Tucker-2 Convではmode 0 / 1のchannel rankとして上記範囲を入口で固定する。

HOOIではさらに、他modeをfactorで射影した後のprojected unfolding上でそのrankが実現可能かを反復前に検証する。

## 5. rank sweep 全20設定

| rank_out | rank_in | params | Conv2 weight | param削減 | val acc | acc drop | Conv2 MAC削減 | 全MAC削減 | weight error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | 133,962 | 23,552 | -3.97% | 0.7350 | -0.0006 | -27.78% | -12.66% | 0.000013 |
| 64 | 24 | 129,098 | 18,688 | -0.20% | 0.7342 | 0.0002 | -1.39% | -0.63% | 0.207197 |
| 64 | 16 | 124,234 | 13,824 | 3.58% | 0.7006 | 0.0338 | 25.00% | 11.39% | 0.361443 |
| 64 | 8 | 119,370 | 8,960 | 7.35% | 0.4398 | 0.2946 | 51.39% | 23.41% | 0.577291 |
| 48 | 32 | 128,330 | 17,920 | 0.40% | 0.7246 | 0.0098 | 2.78% | 1.27% | 0.192191 |
| 48 | 24 | 124,618 | 14,208 | 3.28% | 0.7264 | 0.0080 | 22.92% | 10.44% | 0.267652 |
| 48 | 16 | 120,906 | 10,496 | 6.16% | 0.7000 | 0.0344 | 43.06% | 19.62% | 0.387317 |
| 48 | 8 | 117,194 | 6,784 | 9.04% | 0.4322 | 0.3022 | 63.19% | 28.79% | 0.585418 |
| 32 | 32 | 122,698 | 12,288 | 4.77% | 0.6532 | 0.0812 | 33.33% | 15.19% | 0.344839 |
| 32 | 24 | 120,138 | 9,728 | 6.76% | 0.6532 | 0.0812 | 47.22% | 21.51% | 0.377868 |
| 32 | 16 | 117,578 | 7,168 | 8.74% | 0.6280 | 0.1064 | 61.11% | 27.84% | 0.449042 |
| 32 | 8 | 115,018 | 4,608 | 10.73% | 0.4200 | 0.3144 | 75.00% | 34.17% | 0.609956 |
| 16 | 32 | 117,066 | 6,656 | 9.14% | 0.5062 | 0.2282 | 63.89% | 29.11% | 0.554705 |
| 16 | 24 | 115,658 | 5,248 | 10.23% | 0.5162 | 0.2182 | 71.53% | 32.59% | 0.565492 |
| 16 | 16 | 114,250 | 3,840 | 11.33% | 0.5108 | 0.2236 | 79.17% | 36.07% | 0.590919 |
| 16 | 8 | 112,842 | 2,432 | 12.42% | 0.3816 | 0.3528 | 86.81% | 39.55% | 0.678494 |
| 8 | 32 | 114,250 | 3,840 | 11.33% | 0.3910 | 0.3434 | 79.17% | 36.07% | 0.715727 |
| 8 | 24 | 113,418 | 3,008 | 11.97% | 0.3946 | 0.3398 | 83.68% | 38.12% | 0.719928 |
| 8 | 16 | 112,586 | 2,176 | 12.62% | 0.3978 | 0.3366 | 88.19% | 40.18% | 0.729537 |
| 8 | 8 | 111,754 | 1,344 | 13.26% | 0.3598 | 0.3746 | 92.71% | 42.24% | 0.758660 |

## 6. accuracy dropの具体例

定義：

$$
\mathrm{drop}=\mathrm{baseline\ acc}-\mathrm{compressed\ acc}.
$$

`(64,32)` は

$$
0.7344-0.7350=-0.0006.
$$

validation 5000枚なので、1枚分は

$$
1/5000=0.0002
$$

であり、0.0006は3枚分。これは「本質的に精度が上がった」と断定する大きさではない。

## 7. rank方向は対称ではない

例：

```text
(16,32): val 0.5062
(32,16): val 0.6280
```

同じようにrankの積が小さくても、入力channel側と出力channel側のどちらを絞るかでtask影響が異なる。

## 8. 選択した3候補

```text
aggressive   = (16,24)
balanced     = (32,16)
conservative = (32,24)
```

## 9. HOSVD Tucker-2 fine-tuning

| role | rank | val before | test before | val after | test after | best epoch | params | Conv2 MAC削減 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aggressive | `(16,24)` | 0.5162 | 0.5297 | 0.7462 | 0.7476 | 16 | 115,658 | 71.53% |
| balanced | `(32,16)` | 0.6280 | 0.6378 | 0.7628 | 0.7682 | 30 | 117,578 | 61.11% |
| conservative | `(32,24)` | 0.6532 | 0.6612 | 0.7568 | 0.7571 | 19 | 120,138 | 47.22% |

balancedのtest回復量は、この03 run内では

$$
0.7682-0.6378=0.1304.
$$

baseline testとの差は

$$
0.7682-0.7327=0.0355.
$$

ただし追加学習込みsingle runなので、「Tucker化そのものが3.55 point改善」とは解釈しない。

## 10. MACs contract

Tucker-2の実験表では理論MACsと実測latencyを分けて扱う。

現行 `metrics/macs.py` の圧縮MACs helperでは、生成不可能なrankに対してもっともらしいMACs値を返さないため、rankをbool拒否・正整数・最大rank以下として検証する。また圧縮Convの `out_h / out_w` は正であることを要求する。

SVD向け `compressed_conv2d_macs()` は `groups=1` のみを対象とする。Tucker-2の層MACsも同じく、実際に構築する3層構造の演算量として評価し、parameter削減率とMAC削減率を混同しない。

さらに、MACsが減ってもGPU latencyが必ず短くなるとは限らない。これはSVD編でも確認した通り、kernel launch、メモリアクセス、層分割のoverhead等が影響するためである。

## 11. Autograd / Parameterの扱い

`build_tucker2_conv()` は学習済みConvを新しい3層moduleへ置換する初期化APIなので、元weightをdetachした上で値をcopyする。新しい3層のParameterはleafで、元ConvのParameterと共有しない。

一方 `tucker2_hooi()` は低レベルTensor分解APIなので入力weightをdetachしない。この2つを混同しない。

## 12. 注意

03 balanced `0.7682` と、05のHOSVD同rank `0.7508` は別run。絶対値を横断して同じ実験のように扱わない。

また、現行srcのvalidation強化は入力contractを明示化したもので、上記rank sweep / Fine-tuningの保存済み実験結果を再計算して置き換える変更ではない。
