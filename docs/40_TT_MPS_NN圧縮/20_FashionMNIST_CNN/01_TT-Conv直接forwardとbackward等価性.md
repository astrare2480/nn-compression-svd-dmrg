---
title: TT-Conv直接forwardとbackward等価性の確認
tags:
  - TTConv
  - CNN
  - Conv2d
  - autograd
  - unfold
---

# TT-Conv直接forwardとbackward等価性の確認

## このノートの位置づけ

このノートは、Notebook 01の保存結果と、同じ内容についてChatGPTおよびPerplexityとの対話で確認した事項を統合する。

- [01_ttconv_direct_forward_and_backward_equivalence.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/01_ttconv_direct_forward_and_backward_equivalence.ipynb)

Notebook 00ではTT coresからdense kernelを再構成して`F.conv2d`へ渡した。Notebook 01では、入力をpatch行列へ変換し、denseな重み行列やConv kernelを作らずにTT coresを直接作用させる経路を実装した。

このNotebookの目的は、次の2経路が同じ計算を表すことを、forwardとbackwardの両方で確認することである。

```text
再構成経路: TT cores → dense weight → Conv kernel → Conv出力
direct経路: 入力 → patch行列 → TT coresによる直接計算 → Conv出力
```

学習済みFashion-MNIST CNNへの適用、分類accuracy、速度、peak memoryの優劣は、このNotebookの検証対象ではない。

## 1. 証拠レベル

保存済みNotebookは21セルで、11個のコードセルには実行結果があり、保存されたerror出力はない。TODO 1からTODO 3のshape、forward、gradient比較、および最終確認のassertionは保存出力上で通過している。

ただし、このノート作成時にFresh Restart KernelからRun Allを新たに実行したわけではない。以下は保存済みコード・出力と対話内容に基づく整理である。

## 2. 検証条件

Notebook 00と同じ小さいrandom Convを使った。

| 項目 | 値 |
| :-- | :-- |
| batch size | 2 |
| 入力channel | 4 |
| 出力channel | 8 |
| 入力空間shape | $7\times7$ |
| kernel size | $3\times3$ |
| stride / padding / dilation | $(1,1)$ / $(1,1)$ / $(1,1)$ |
| groups | 1 |
| bias | あり |
| dtype / device | `torch.float64` / CPU |
| TT modes | `OUT_MODES=(2,2,2)`、`IN_MODES=(3,3,4)` |
| rank制限 | `max_rank=2`、actual ranks $(1,2,2,1)$ |

gradient比較では、同じ値から独立に作った2組のTT cores、同じ入力、固定した同じbias、同じtarget、同じMSE lossを使った。各経路のgradientが混ざらないように、別々に`backward()`して比較した。

## 3. 入力patchと出力の並び

`F.unfold`の保存結果は、

$$
(B,C_{\mathrm{in}}K_HK_W,L)=(2,36,49)
$$

であった。ここで、$L=H_{\mathrm{out}}W_{\mathrm{out}}=7\times7=49$である。

TODO 1では、各patchを1行として、

$$
(2,36,49)\rightarrow(2,49,36)\rightarrow(98,36)
$$

へ並べ替えた。要素対応は、

$$
\mathrm{patch\_matrix}[49b+\ell,j]
=
\mathrm{unfolded\_patches}[b,j,\ell]
$$

である。最初の49行が画像0、次の49行が画像1に対応する。

TTによる線形計算後のshapeは$(98,8)$である。これを、

$$
(98,8)\rightarrow(2,7,7,8)\rightarrow(2,8,7,7)
$$

と戻した。並べる側と戻す側が同じ順序規約を使う必要がある。

## 4. forwardの保存結果

| 比較 | 出力RMSE |
| :-- | --: |
| exact TT direct vs 元のdense Conv | $1.235322\times10^{-15}$ |
| rank制限TT direct vs 同じTTの再構成経路 | $7.787415\times10^{-17}$ |
| rank制限TT direct vs 元のdense Conv | $0.4402539$ |

### 4.1 exact TTの結果

exact TT directと元のdense ConvのRMSEは約$1.24\times10^{-15}$であり、今回のfloat64条件では数値誤差水準で一致した。

この比較は、TT縮約だけを単独で調べたものではない。次の一連の実装が、全体として元のConvと整合したことを示す。

- `F.unfold`によるpatch抽出
- patch内の36要素の順序
- batchと49個の出力位置の行順序
- exact TT coresによる直接線形計算
- biasの加算
- $(98,8)$から$(2,8,7,7)$への復元

### 4.2 rank制限TTの2種類の比較

rank制限TT directと、同じrank制限coresからkernelを再構成した経路との差は$7.79\times10^{-17}$だった。従って、同じTT表現を使う限り、direct実装によって新たに生じたforward差は数値誤差水準である。

一方、rank制限TT directと元のdense Convとの差は$0.4402539$だった。この差はdirect計算方式そのものではなく、rank制限されたTT表現と元のdense weightの違いに対応する。

これにより、次の2種類の誤差を分離できた。

1. rank制限による近似誤差
2. direct経路の実装による追加誤差

今回、2は数値誤差水準であり、観測された大きな差は1に由来する。ただし、RMSE $0.4403$の絶対値だけでは、元出力に対する相対的な大きさや分類への影響は判断できない。

## 5. backwardの保存結果

| core | direct gradient norm | 再構成経路gradient norm | 最大絶対差 |
| ---: | --: | --: | --: |
| 0 | 0.117466 | 0.117466 | $9.714451\times10^{-17}$ |
| 1 | 0.118728 | 0.118728 | $4.857226\times10^{-17}$ |
| 2 | 0.220505 | 0.220505 | $6.938894\times10^{-17}$ |

全coreについて、gradientが存在し、shapeが対応するcoreと一致し、値がfiniteであることを確認した。そのうえで、direct経路と再構成経路のgradientが約$10^{-16}$以下の最大絶対差で一致した。

### 5.1 「gradientが存在する」より強い確認

全coreのgradient normが非ゼロであること自体は、通常のsanity checkである。今回のnormは約$0.12$から$0.22$であり、最大絶対差$9.71\times10^{-17}$と同程度に小さいgradientを比較して一致したように見えているわけではない。重要なのは、同じlossに対する2経路のgradient値を直接照合し、数値誤差水準で一致したことである。

より一般には、絶対差に加えて、

$$
\frac{
\lVert\nabla L_{\mathrm{direct}}-\nabla L_{\mathrm{reconstructed}}\rVert_2
}{
\lVert\nabla L_{\mathrm{reconstructed}}\rVert_2
}
$$

のような相対差も確認できる。ただし、分母のgradient normがゼロまたは極端に小さい場合には、相対差は不安定になる。

単純なSGDなら、更新は$G_{\mathrm{new}}=G_{\mathrm{now}}-\eta\nabla_G L$なので、同じ初期coreと学習率では勾配差に$\eta$を掛けた分だけ更新差が生じる。この結果から、今回の1回のbackward時点では、2経路が各coreへ与える更新方向も整合すると読める。momentumやAdamの場合は、optimizerの種類・設定だけでなく内部状態も同じである必要がある。また、optimizer stepを繰り返した複数stepの学習軌跡まで比較したわけではない。

### 5.2 core間のgradient norm差は評価対象ではない

core 2のnormが他より大きいが、random入力・random targetを使った1 batchの値である。これだけから、特定coreが常に重要、学習しやすい、あるいは不安定とは判断しない。

### 5.3 一致にも限界がある

2経路のgradient一致は強い整合性チェックだが、両方が共有するtensorization、cores、入力、targetなどに同じ誤りがあれば、その誤りを独立に排除する証明にはならない。結論は、「設定した条件において、2つのbackward経路が整合した」である。

## 6. Notebook 00から進んだ点

Notebook 00は、TTをweightの表現として使い、計算時にはdense kernelへ戻した。

```text
TT cores → dense kernel → Conv出力 → loss
```

Notebook 01は、TTをweightの保存形式だけでなく、forwardとgradient計算へ直接使った。

```text
入力 → patch行列 → TT coresによる直接計算 → Conv出力 → loss
```

したがって、01で新しく確認した本質は「coreを初めて学習可能にした」ことではない。00でも再構成経路を通してcoreへgradientは届いていた。01の前進は、dense kernelを明示的に再構成しない計算経路でも、出力とcore gradientが参照経路に一致したことである。

## 7. parameter圧縮と実行時性能は別である

rank制限TTのweight部分は52要素で、dense kernelの288要素より少ない。しかしdirect経路では、`F.unfold`によって$(2,36,49)$、すなわち3,528要素のpatch Tensorを作った。

| Tensor | 要素数 | float64での値部分の大きさ |
| :-- | --: | --: |
| 元入力 | 392 | 3,136 bytes |
| dense kernel | 288 | 2,304 bytes |
| rank制限TT cores | 52 | 416 bytes |
| `F.unfold`のpatch Tensor | 3,528 | 28,224 bytes |

patch Tensorは元入力の9倍、省いたdense kernelの12.25倍の要素数である。また、元入力がpatch Tensorへ置き換わって即座に消えるわけではなく、実行中には両方が存在し得る。

この事実から言えるのは、dense kernelを作らないことだけではdirect経路の省メモリ性を主張できない、ということである。一方、この4種類のTensorの要素数だけでdirect経路のpeak memoryが必ず大きいとも断定できない。両経路にはTT縮約の中間Tensor、Convの作業領域、autogradがbackward用に保持するTensorがあるためである。

同様に、dense-freeであることだけから高速化も主張できない。少なくとも次を分けて測定する必要がある。

- 保存parameter数
- forward時間
- forwardとbackwardを含む学習時間
- peak memory
- MACsまたは演算量

測定時は、同じdevice・dtype・batch・thread条件を使い、warm-upを行い、各経路を単独で測る必要がある。今回のNotebookは両経路のTensorを同時に保持する比較用構成なので、Notebook全体の使用メモリをそのまま各方式の値とはみなせない。

## 8. 学習時と固定weightでの推論時

学習時はcoresがoptimizerによって更新される。再構成経路を使うなら、更新後のcoresからdense kernelを再生成する必要がある。

一方、weightを固定した推論時には、dense kernelを一度再構成してcacheし、通常のConvとして繰り返し使う構成も可能である。この場合、「毎回kernelを再構成する経路」と「cache済みkernelを使う経路」と「TT direct経路」は別条件である。

従って、direct経路の実用上の意味は、学習か推論か、weightを何回再利用するかによって変わる。Notebook 01は性能比較ではなく、その前提となる数値的整合性を確認した。

## 9. このNotebookで確認した範囲

- 小さいrandom Conv、`groups=1`、biasあり、float64、CPUで検証した。
- exact TT directは元のdense Convと数値誤差水準で一致した。
- rank制限TT directは、同じcoresの再構成経路と数値誤差水準で一致した。
- rank制限による近似誤差と、direct実装による追加誤差を分離した。
- patchの行順序とConv出力shapeへの復元を含む一連の処理が整合した。
- 直接経路と再構成経路の全core gradientが数値誤差水準で一致した。

学習済みFashion-MNIST CNN、float32またはmixed precision、`groups>1`、別のstride・padding・dilation、分類accuracy、複数optimizer step、速度、peak memoryは未検証である。Convをpatchごとの線形計算として扱う考え方と、現在のコードがそれらの条件でも正しく動くことは分けて扱う。

## 10. 結論

> Notebook 01では、入力を$(98,36)$のpatch行列へ変換し、denseな重み行列やConv kernelを再構成せず、TT coresから直接$(2,8,7,7)$のConv出力を計算した。exact TT directは元のdense ConvとRMSE $1.24\times10^{-15}$、rank制限TT directは同じTTの再構成経路とRMSE $7.79\times10^{-17}$で一致した。さらに、3 coresのgradientも最大絶対差$9.72\times10^{-17}$以下で一致した。これにより、今回の条件ではrank制限による近似誤差とdirect実装の誤差を切り分け、direct forwardとbackwardの数値的整合性を確認できた。ただし、parameter削減は速度・peak memory・分類性能の改善を意味せず、それらは別の実測が必要である。
