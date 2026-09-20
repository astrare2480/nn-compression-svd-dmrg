---
title: Tucker-2 Convの実装契約と注意
tags: [Tucker, Conv2d, PyTorch, API]
---

# Tucker-2 Convの実装契約と注意

## 1. 現行 `conv_tucker.py` のTucker-2構造

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

## 2. rank contract

`rank_out` / `rank_in` は独立に検証する。

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

boolは整数として受理せず拒否する。

汎用HOSVD/HOOI側ではmode-n unfolding上の最大rankまで検証するが、Tucker-2 Convではmode 0 / 1のchannel rankとして上記範囲を入口で固定する。

HOOIではさらに、他modeをfactorで射影した後のprojected unfolding上でそのrankが実現可能かを反復前に検証する。

## 3. MACs contract

Tucker-2の実験表では理論MACsと実測latencyを分けて扱う。

現行 `metrics/macs.py` の圧縮MACs helperでは、生成不可能なrankに対してもっともらしいMACs値を返さないため、rankをbool拒否・正整数・最大rank以下として検証する。また圧縮Convの `out_h / out_w` は正であることを要求する。

SVD向け `compressed_conv2d_macs()` は `groups=1` のみを対象とする。Tucker-2の層MACsも同じく、実際に構築する3層構造の演算量として評価し、parameter削減率とMAC削減率を混同しない。

さらに、MACsが減ってもGPU latencyが必ず短くなるとは限らない。これはSVD編でも確認した通り、kernel launch、メモリアクセス、層分割のoverhead等が影響するためである。

## 4. Autograd / Parameterの扱い

`build_tucker2_conv()` は学習済みConvを新しい3層moduleへ置換する初期化APIなので、元weightをdetachした上で値をcopyする。新しい3層のParameterはleafで、元ConvのParameterと共有しない。

一方 `tucker2_hooi()` は低レベルTensor分解APIなので入力weightをdetachしない。この2つを混同しない。

