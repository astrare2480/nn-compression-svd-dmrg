---
title: Conv2dのTT-matrix表現と直接forward
tags:
  - TTConv
  - CNN
  - Conv2d
  - TT-matrix
  - unfold
  - autograd
---

# Conv2dのTT-matrix表現と直接forward

## 1. このノートの目的

Conv2dのweightを行列として捉え、TT-matrixへ変換し、次の2通りで同じ畳み込みを計算する考え方を整理する。

1. TT coresからdense kernelを再構成して`conv2d`を行う。
2. 入力をpatchへ展開し、TT coresを直接作用させる。

ここでは、CNN、Conv2d、TT-matrix、PyTorch autogradに関する一般的な技術事項を扱う。特定Notebookの実測値と考察は、次へ分ける。

- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/00_TT-Conv表現とdense_kernel再構成]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/01_TT-Conv直接forwardとbackward等価性]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/02_学習済みCNNへのexact_TT-Conv接続]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/03_学習済みCNNのconv2_TT-rank_sweep]]
- [[40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/04_conv2_TT-cores-only_fine-tuning]]

## 2. Conv2dをpatchごとの線形写像として見る

### 2.1 入出力とweightのshape

入力、weight、出力を、

$$
X\in\mathbb{R}^{B\times C_{\mathrm{in}}\times H_{\mathrm{in}}\times W_{\mathrm{in}}},
$$

$$
W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_H\times K_W},
$$

$$
Y\in\mathbb{R}^{B\times C_{\mathrm{out}}\times H_{\mathrm{out}}\times W_{\mathrm{out}}}
$$

とする。`groups=1`の場合、1つの出力位置におけるConv2dは、

$$
Y_{b,o,h,w}
=
b_o
+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
W_{o,c,u,v}
X_{b,c,hs_H-p_H+ud_H,ws_W-p_W+vd_W}
$$

である。$s_H,s_W$はstride、$p_H,p_W$はpadding、$d_H,d_W$はdilationである。入力範囲外はpadding値として扱う。

出力の空間サイズは、

$$
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+2p_H-d_H(K_H-1)-1}{s_H}
\right\rfloor+1,
$$

$$
W_{\mathrm{out}}
=
\left\lfloor
\frac{W_{\mathrm{in}}+2p_W-d_W(K_W-1)-1}{s_W}
\right\rfloor+1
$$

である。`kernel_size=3`は$K_H=K_W=3$を意味する。

### 2.2 patchとは何か

出力位置$(h,w)$を1点計算するときに参照する、全入力channelにわたる$K_H\times K_W$の領域をpatchと呼ぶ。1 patchの要素数は、

$$
N=C_{\mathrm{in}}K_HK_W
$$

である。

ここでchannelは、同じ空間位置に存在する特徴量の軸である。入力channel数が4なら、各出力位置では4枚のfeature mapから局所領域を取り出す。「4層のweightがある」という意味ではない。

## 3. Conv kernelの行列化

Conv weightの出力channel以外の軸をまとめ、

$$
W_{\mathrm{mat}}
\in
\mathbb{R}^{C_{\mathrm{out}}\times(C_{\mathrm{in}}K_HK_W)}
$$

とする。`groups=1`では、各行が1つの出力channelに対応し、各列がpatch内の1要素に対応する。

PyTorchの連続なNCHW weightをそのまま`reshape`する場合、0始まりの添字対応は、

$$
j
=
c(K_HK_W)+uK_W+v
$$

である。$K_H=K_W=3$なら、

$$
j=9c+3u+v
$$

となる。この式をコードへ直接書かなくても、`reshape(C_out, -1)`が同じメモリ順序を使う。

### 数値例

$C_{\mathrm{in}}=4$、$C_{\mathrm{out}}=8$、$K_H=K_W=3$では、

$$
W:(8,4,3,3)
\longrightarrow
W_{\mathrm{mat}}:(8,36)
$$

となる。たとえば$c=2,u=1,v=0$は、

$$
j=9\times2+3\times1+0=21
$$

列目に対応する。

## 4. TT-matrixのmode分解

### 4.1 modeの条件

出力側modeを$(m_1,\ldots,m_d)$、入力側modeを$(n_1,\ldots,n_d)$とし、

$$
\prod_{k=1}^{d}m_k=C_{\mathrm{out}},
\qquad
\prod_{k=1}^{d}n_k=C_{\mathrm{in}}K_HK_W
$$

を満たすように選ぶ。

これはTTの定義そのものというより、現在の行列を要素の欠落・重複なくTT-matrixへ写すために必要なshape条件である。modeの組は一般に一意ではない。

### 4.2 数値例

$8\times36$行列に対し、

$$
(m_1,m_2,m_3)=(2,2,2),
$$

$$
(n_1,n_2,n_3)=(3,3,4)
$$

を選べる。確かに、

$$
2\times2\times2=8,
\qquad
3\times3\times4=36
$$

である。

各siteで出力modeと入力modeを組にすると、TT-SVDへ渡すcombined physical modesは、

$$
(m_1n_1,m_2n_2,m_3n_3)=(6,6,8)
$$

となる。

### 4.3 modeとchannelは違う

`OUT_CHANNELS=8`と`IN_CHANNELS=4`は、元のConv層の入出力構造である。`OUT_MODES=(2,2,2)`と`IN_MODES=(3,3,4)`は、TT表現のために行列の行・列添字を分解する方法である。

今回の`IN_MODES=(3,3,4)`は、3、高さ・3、幅・4、channelを各coreがそのまま担当するという意味ではない。先にまとめた36列の添字を、3個の小さい添字へ表し直している。

### 4.4 「physical index」の意味

スピン鎖のMPSでは、各siteのphysical indexは局所スピン状態などの物理的自由度を表す。一方、NN weightのtensorizationでは、行列の添字を人為的に因数分解したものでもphysical indexと呼ぶ。

従って、NNのTT siteが実在の空間位置やchannel群へ自動的に対応するとは限らない。物理的解釈は、tensorizationの設計として別途与える必要がある。

## 5. TT-matrix coreとrank

TT-matrixの第$k$ coreを、

$$
G^{(k)}
\in
\mathbb{R}^{r_{k-1}\times m_k\times n_k\times r_k}
$$

とする。境界rankは$r_0=r_d=1$である。行添字$(i_1,\ldots,i_d)$と列添字$(j_1,\ldots,j_d)$に対応する行列要素は、

$$
W_{(i_1,\ldots,i_d),(j_1,\ldots,j_d)}
=
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1,i_1,j_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,j_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,j_d,1}
$$

で復元される。

TT parameter数は、

$$
N_{\mathrm{TT}}
=
\sum_{k=1}^{d}r_{k-1}m_kn_kr_k
$$

である。Conv biasを持つ場合は、さらに$C_{\mathrm{out}}$を加える。

### exact TTとrank制限TT

exact TTは、各unfoldingの必要rankを保持し、丸め誤差を除いて元行列を再構成する。exactであることは、parameter削減を意味しない。rankが大きければ、TTのparameter数がdense行列を上回ることもある。

rank制限TTは$r_k$に上限を設け、parameter数を減らす代わりに近似誤差を許容する。

mode分解とrank制限は別の選択である。

- mode分解：行・列添字をどのsiteへ割り当てるか。
- TT-rank：site間でどれだけの結合情報を保持するか。

同じmode分解でも、exact rankなら再構成でき、低rankでは誤差が生じ得る。

## 6. dense kernel再構成経路

TT coresを全て縮約して$W_{\mathrm{mat}}$を復元し、

$$
W_{\mathrm{mat}}
\rightarrow
W_{\mathrm{conv}}
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_H\times K_W}
$$

へreshapeすれば、通常の`F.conv2d`へ渡せる。

```text
TT cores
  ↓ 全coreを縮約
dense weight matrix
  ↓ reshape
dense Conv kernel
  ↓ F.conv2d
Conv出力
```

この経路は、TT表現とConv kernelの対応を検証する参照実装として分かりやすい。一方、forward時にdense kernelを明示的に作るため、TTを直接の計算形式としては使っていない。

## 7. `F.unfold`によるpatch行列

`F.unfold`は、Convのkernel size、stride、padding、dilationに従って各出力位置のpatchを列へ並べる。

$$
X:(B,C_{\mathrm{in}},H_{\mathrm{in}},W_{\mathrm{in}})
\longrightarrow
U:(B,N,L),
$$

$$
N=C_{\mathrm{in}}K_HK_W,
\qquad
L=H_{\mathrm{out}}W_{\mathrm{out}}
$$

である。

このpatch Tensorは元画像そのものではなく、元画像からConvが1出力位置の計算に使う領域を列挙した、中間入力である。TT-matrix forwardへ渡すには、各patchを1行にする。

```python
import torch
import torch.nn.functional as F

# 例: B=2, Cin=4, H=W=7, kernel=3, padding=1
x = torch.randn(2, 4, 7, 7, dtype=torch.float64)

# shape: (B, Cin * Kh * Kw, Hout * Wout) = (2, 36, 49)
unfolded = F.unfold(x, kernel_size=3, padding=1, stride=1)

# patchごとに1行へ並べる。shapeは(2 * 49, 36) = (98, 36)
patch_matrix = unfolded.permute(0, 2, 1).reshape(-1, unfolded.shape[1])

assert unfolded.shape == (2, 36, 49)
assert patch_matrix.shape == (98, 36)
```

この行順序では、最初の$L$行がbatch 0、次の$L$行がbatch 1である。

## 8. TT coresによる直接forward

patch行列$P\in\mathbb{R}^{(BL)\times N}$に対し、TT-matrixが表す線形写像を直接作用させる。

$$
Z=PW_{\mathrm{mat}}^{\mathsf{T}}+b,
\qquad
Z\in\mathbb{R}^{(BL)\times C_{\mathrm{out}}}
$$

ただし実装では$W_{\mathrm{mat}}$を構成せず、$P$を入力modeへreshapeし、各TT coreと順次縮約する。

出力をConv形式へ戻すには、

$$
(BL,C_{\mathrm{out}})
\rightarrow
(B,H_{\mathrm{out}},W_{\mathrm{out}},C_{\mathrm{out}})
\rightarrow
(B,C_{\mathrm{out}},H_{\mathrm{out}},W_{\mathrm{out}})
$$

とする。

```text
入力画像
  ↓ F.unfold
patch行列
  ↓ TT-matrix direct forward
patchごとの出力
  ↓ reshape + permute
Conv出力
```

「direct」は、denseな$W_{\mathrm{mat}}$やConv kernelを明示的に再構成しないという意味である。入力patchは`F.unfold`で明示的に作っている。

既存APIを使う最小の接続部分は、次の形になる。

```python
from nn_compression.compression import tt_linear_forward

# dense weightを作らず、各patchへTT-matrixを直接作用させる
patch_output = tt_linear_forward(
    patch_matrix,
    cores,
    out_modes,
    in_modes,
    bias=bias,
)

# (B * Hout * Wout, Cout)からConv2dの出力配置へ戻す
direct_output = patch_output.reshape(
    batch_size,
    out_height,
    out_width,
    out_channels,
).permute(0, 3, 1, 2).contiguous()
```

`tt_linear_forward`は、入力をinput modesへreshapeした後、各coreと`torch.tensordot`で順に縮約する。内部で`tt_matrix_to_dense`を呼ばないため、denseな$C_{\mathrm{out}}\times(C_{\mathrm{in}}K_HK_W)$ weightを構築しない。一方、patch行列と縮約途中のTensorは存在する。

## 9. forwardで何を比較するか

少なくとも次の2比較を分ける。

### 9.1 exact TT direct vs dense Conv

exact TTを使い、元のdense Convと比較する。この差は、主にpatch順序、TT縮約、bias、出力復元を含むdirect実装全体の整合性を調べる。

### 9.2 rank制限TT direct vs 同じTTの再構成経路

同じrank制限coresを使い、direct経路とdense kernel再構成経路を比較する。この比較では、両方に同じrank近似が含まれるため、rank制限による元weightとの差を相殺し、計算経路の差を調べられる。

rank制限TT directと元のdense Convだけを比較すると、rank近似誤差とdirect実装誤差が混ざる。

## 10. PyTorchでTT coresを学習可能にする

### 10.1 `nn.Parameter`と`nn.ParameterList`

`nn.Parameter`は、`nn.Module`へ属性として登録されたときに学習対象として扱われるTensorである。複数のcoresは`nn.ParameterList`に入れると、順序を保ちながらModuleのparameterとして登録できる。

```python
import torch.nn as nn

# 分解元のTensorから独立した学習用coresを作る
learnable_cores = nn.ParameterList([
    nn.Parameter(core.detach().clone())
    for core in truncated_cores
])
```

`detach()`は元のTT-SVD計算グラフから切り離し、`clone()`はstorageを共有しないコピーを作る。`nn.Parameter(...)`は各coreをparameterとして扱い、`nn.ParameterList(...)`はそれらをModuleへ登録できる入れ物にする。

型注釈だけでは変換されない。たとえば、

```python
learnable_cores: nn.ParameterList = truncated_cores
```

は型の想定を示すだけで、通常のlistを`ParameterList`へ変換しない。

### 10.2 `list(cores)`とautograd

`list(learnable_cores)`はPythonのcontainerを変えるだけで、中の`Parameter`を`detach`しない。したがって、そのTensorを演算へ使えばautograd graphは維持される。

### 10.3 登録とgradient計算は別

`ParameterList`に入れただけでgradientが生じるわけではない。coreを使った計算がscalar lossまでつながり、`loss.backward()`が呼ばれる必要がある。

## 11. `nn.Module`とfunctional APIの使い分け

`nn.Conv2d`は、自身に登録されたweightとbiasを使う層である。通常のモデル定義やoptimizerへのparameter登録に適している。

`torch.nn.functional.conv2d`は、引数として渡した入力、weight、biasを使って計算する関数である。TT coresからその場で再構成したkernelを使いたい場合や、2種類のweightを同じConv条件で比較したい場合に扱いやすい。

どちらを使っても、入力からlossまでautograd graphがつながっていればgradientは計算される。functional APIを使うこと自体がgradientを無効にするわけではない。

## 12. backward等価性の検証

direct経路と再構成経路のgradientを公平に比較するには、次をそろえる。

- 初期値が同じだが、互いに独立した2組のcores
- 同じ入力
- 同じbias
- 同じtarget
- 同じscalar loss
- 同じdtypeとdevice

この検証の`gradient_target`は、2経路の出力に対して同じMSE lossを作るための固定Tensorである。分類labelそのものではなく、backward結果を比較するための共通の参照値である。

出力要素数を$M$、targetを$T$とすれば、MSE lossは、

$$
L(Y,T)=\frac{1}{M}\sum_{q=1}^{M}(Y_q-T_q)^2
$$

である。scalar lossにすることで、各core要素に対する$\partial L/\partial G^{(k)}$を`backward()`で計算できる。このlossは分類性能を測るためではなく、2つの計算経路の微分を同条件で比較するために使う。

各経路で別々にforwardと`backward()`を行い、各coreについて次を確認する。

1. `.grad is not None`
2. gradient shapeがcore shapeと同じ
3. 全値がfinite
4. gradient norm
5. 2経路間の最大絶対差または相対差

全coreのgradientが存在することは計算グラフがつながっているというsanity checkである。gradientの正しさをより強く確認するには、参照経路との値の一致を調べる。

同じforward関数と同じscalar lossが数値的に一致し、各parameterのgradientも一致すれば、現在のparameter点における1 step分の更新方向は整合する。単純なSGDでは、

$$
G_{\mathrm{new}}=G_{\mathrm{now}}-\eta\nabla_G L
$$

なので、同じ初期値と学習率なら、更新量の差は勾配差に$\eta$を掛けた大きさになる。momentumやAdamでは、optimizerの種類と設定に加えて内部状態もそろえる必要がある。ただし、丸め誤差の蓄積や複数stepのtrajectoryまで同じとは、1点でのgradient比較だけでは言えない。

絶対差だけでは、gradient自体が非常に小さい場合の方向の一致を評価しにくい。必要に応じて、

$$
\frac{
\lVert\nabla L_{\mathrm{direct}}-\nabla L_{\mathrm{reconstructed}}\rVert_2
}{
\lVert\nabla L_{\mathrm{reconstructed}}\rVert_2
}
$$

も確認する。分母がゼロまたは極端に小さい場合は、相対差を単独で使わない。

## 13. 保存parameter数と実行時性能を分ける

TT rankを制限すれば、weightを保持するparameter数を減らせる。しかし、次は別の指標である。

- 保存parameter数とmodel file size
- forward/backward中のpeak memory
- forward latency
- training step time
- MACsまたはFLOPs

direct経路はdense kernelを作らないが、`F.unfold`のpatch TensorやTT縮約の中間Tensorを作る。そのため、dense-freeという実装上の性質だけから高速化や省メモリ化は導けない。

### patch展開の数値例

$B=2$、$C_{\mathrm{in}}=4$、入力$7\times7$、kernel$3\times3$、padding 1なら、

$$
\text{input elements}=2\times4\times7\times7=392,
$$

$$
\text{patch elements}=2\times36\times49=3528
$$

であり、patchは入力の9倍の要素数になる。dense kernelが$8\times4\times3\times3=288$要素なら、patchはkernelの12.25倍である。

これはpatchだけの比較であり、経路全体のpeak memoryではない。Conv実装のworkspace、TT縮約の一時Tensor、backward保存Tensor、Tensorの寿命を含めて実測する必要がある。

## 14. 性能測定の設計

direct経路と再構成経路を比較する場合は、次をそろえる。

- device、dtype、batch size、入力shape
- stride、padding、dilation、groups、bias
- 同じcoresと出力精度条件
- CPU thread数またはGPU環境
- warm-up回数と測定反復数

forwardだけでなく学習用途なら、zero-grad、forward、loss、backwardを含むstep時間を測る。GPU時間は非同期なので、適切な同期またはCUDA eventを使う。peak memoryは各経路を単独で動かし、測定前にpeak統計をresetする。

## 15. 学習と推論で異なる比較

### 学習時

coresが更新されるため、再構成経路は原則として各forwardで現在のcoresからdense kernelを作る。direct経路は現在のcoresをそのまま縮約へ使う。forwardとbackwardの両方を含めて比較する必要がある。

### 固定weightでの推論時

coresを更新しないなら、dense kernelを一度だけ再構成してcacheし、通常の最適化されたConvを繰り返し使える。したがって、次は異なる運用条件である。

- 毎回dense kernelを再構成する。
- 事前に再構成したdense kernelをcacheする。
- TT coresから毎回direct計算する。

モデルを一度学習して長く推論へ使う場合と、頻繁に再学習する場合では、再構成コストの重要度が異なる。

## 16. 適用範囲と次の検証

小さいrandom Conv、`groups=1`、float64での等価性確認は、direct TT-Conv実装の基礎検証になる。ただし、学習済みCNNへ適用する際は少なくとも次を再確認する。

- 対象Conv層の実際のweight、input shape、stride、padding、dilation
- 実運用dtypeでのforward誤差とgradient誤差
- `groups`を含む実装対応範囲
- rank制限後のモデル全体のlogits、prediction、accuracy、loss
- training stabilityと複数step後の挙動
- parameter数、latency、peak memory、演算量

学習済みweightでもConvをpatchごとの線形写像として扱う数学は同じである。しかし、数学的に適用できることと、現在のコードが対象条件で正しく、実用上有利に動くことは別々に検証する。

## 17. `TTConv2d`を`nn.Module`として構成する

小さいTensorで検証したdirect forwardをモデルへ組み込むには、TT cores、bias、Conv設定を保持する`nn.Module`へまとめる。

### 17.1 parameter登録

coresは`nn.ParameterList`、biasは`nn.Parameter`または登録済みの`None`として保持する。

```python
from torch import nn

# 分解時のTensorと履歴・storageを共有しないparameterを作る
self.cores = nn.ParameterList([
    nn.Parameter(core.detach().clone())
    for core in cores
])

if bias is None:
    # biasなしもModuleのparameter contractへ明示的に登録する
    self.register_parameter("bias", None)
else:
    # 元Convのbiasとは独立した学習可能parameterを作る
    self.bias = nn.Parameter(bias.detach().clone())
```

direct TT-Convとして使うなら、denseな`weight` parameterは登録しない。parameter登録はoptimizerから見える状態を作るが、学習が成功することやgradient値が正しいことを単独では保証しない。

### 17.2 Conv設定の正規化

PyTorchのConv設定は、`3`のようなintと`(3, 3)`のようなtupleの両方で指定される。内部では2要素tupleへ正規化すると、出力shape計算と`F.unfold`へ同じ値を渡しやすい。

```python
def as_pair(
    value: int | tuple[int, int],
    *,
    name: str,
    minimum: int,
) -> tuple[int, int]:
    """Convのintまたは2要素tuple設定を正規化する。"""
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an int or a tuple of two ints.")

    if not all(
        isinstance(v, int) and not isinstance(v, bool)
        for v in value
    ):
        raise TypeError(f"{name} must contain integers.")

    if any(v < minimum for v in value):
        raise ValueError(f"Each value in {name} must be >= {minimum}.")

    return value
```

`kernel_size`、`stride`、`dilation`は1以上、`padding`は0以上とする。Pythonでは`bool`が`int`のsubclassなので、`True`や`False`を整数設定として誤って受理しないように除外している。

### 17.3 初期化時に検証するcontract

少なくとも次を検証する。

- `in_channels`、`out_channels`、`groups`は正の整数である。
- 実装範囲外の`groups`や`padding_mode`を明示的に拒否する。
- coresとmodesは空でなく、個数が一致する。
- 各coreは実数浮動小数点の4階Tensorである。
- core shapeは$(r_{k-1},m_k,n_k,r_k)$である。
- 隣接bondと境界rankが整合する。
- $\prod_k m_k=C_{\mathrm{out}}$である。
- $\prod_k n_k=C_{\mathrm{in}}K_HK_W$である。
- 全coresとbiasのdtype・deviceが一致する。
- biasがある場合はshape$(C_{\mathrm{out}},)$である。

未対応条件を暗黙に処理するより、`NotImplementedError`で境界を明示した方が、一般的な`Conv2d`互換と誤解されにくい。

### 17.4 `forward`の責務

`forward`は次を行う。

1. 入力が$(B,C_{\mathrm{in}},H,W)$の4階Tensorで、channel、dtype、deviceがcoresと整合することを確認する。
2. Convの式から$H_{\mathrm{out}}$と$W_{\mathrm{out}}$を計算する。
3. `F.unfold`でpatch Tensorを作る。
4. 各patchを1行へ並べる。
5. `tt_linear_forward`でTT-matrixを直接作用させる。
6. $(BH_{\mathrm{out}}W_{\mathrm{out}},C_{\mathrm{out}})$をConv出力shapeへ戻す。

出力サイズ計算で使う`//`は整数の切り下げ除算である。たとえば`5 // 2 + 1`は$2+1=3$となる。これはConv出力サイズ式のfloorに対応する。

## 18. 学習済みモデルの1層だけを安全に置換する

### 18.1 checkpointから元モデルを復元する

モデル構造を作り、学習済み`state_dict`を読み込んだものを比較基準とする。TT分解は学習履歴を逆伝播する操作ではないため、対象weightを`detach()`してから行列化できる。

```python
# 学習済みweightの値だけを使い、Conv行列へ並べ直す
weight_matrix = dense_model.conv2.weight.detach().reshape(
    out_channels,
    in_channels * kernel_height * kernel_width,
)
```

### 18.2 元モデルと比較モデルを分ける

```python
import copy

# 元モデルを変更しない独立した比較モデルを作る
tt_model = copy.deepcopy(dense_model)

# 比較モデルの対象層だけをTT層へ置換する
tt_model.conv2 = tt_conv2
```

`tt_model = dense_model`はコピーではなく同じオブジェクトへの別名である。この状態で`conv2`を交換すると、比較基準の元モデルも変更される。

置換前に元モデルの`state_dict()`をcloneし、置換後に次を確認する。

- 元モデル自体のstateが変化していない。
- 比較モデルの対象層以外のstateが元モデルと一致する。
- 元モデルと比較モデルが別オブジェクトである。
- 元モデルの対象層は元の型のままである。
- 比較モデルの対象層だけがTT層になっている。

`state_dict`にはparameterだけでなくpersistent bufferも含まれる。対象外stateを確認することで、出力差の原因を層置換へ限定しやすくなる。

### 18.3 `eval()`と`inference_mode()`の役割

`model.eval()`は、DropoutやBatchNormなどを評価時の動作へ切り替える。これはgradient記録を無効にしない。

`torch.inference_mode()`は評価中のautograd記録を無効にするが、Moduleを評価モードへ切り替えない。そのため、推論比較では両方を使う。

```python
# DropoutやBatchNormを評価時の動作へ切り替える
dense_model.eval()
tt_model.eval()

# 評価ループだけでautograd記録を無効にする
with torch.inference_mode():
    dense_logits = dense_model(images)
    tt_logits = tt_model(images)
```

学習可能なcoresやbiasの作成は、評価用の`inference_mode()`へ入る前に行う。

## 19. validation全体で等価性を評価する

### 19.1 層単体とモデル全体を分ける

層単体の出力を比べる場合は、元モデルで対象層の直前まで計算した同一feature mapを、dense層とTT層の両方へ入力する。これにより、前段の差を混ぜずに対象層だけを比較できる。

モデル全体では、同じ画像を2モデルへ入力し、最終logits、予測class、正解ラベルに対するaccuracyを比較する。

### 19.2 RMSEは全要素から集計する

batch $b$の二乗誤差合計を$S_b$、比較した要素数を$N_b$とすると、validation全体のRMSEは、

$$
\mathrm{RMSE}
=
\sqrt{
\frac{\sum_b S_b}{\sum_b N_b}
}
$$

である。

$$
\frac{1}{N_{\mathrm{batch}}}
\sum_b\mathrm{RMSE}_b
$$

のようにbatchごとのRMSEを単純平均すると、最後のbatchの大きさが異なる場合に各要素を同じ重みで扱えない。

### 19.3 指標が答える問い

| 指標 | 確認する内容 |
| :-- | :-- |
| 層出力RMSE | 対象層単体の出力がどの程度近いか |
| logits RMSE | 全sample・全classの最終数値出力がどの程度近いか |
| logits最大絶対差 | 一部の要素に大きな差が隠れていないか |
| prediction agreement | 2モデルが同じclassを選んだsampleの割合 |
| accuracy | 各モデルの予測が正解labelと一致した割合 |

agreementが1.0なら、共通の正解ラベルに対する2モデルのaccuracyも一致する。逆にaccuracyが同じでも、sampleごとに別の誤予測をしている可能性があるため、agreementは保証されない。

予測不一致が発生した場合は、そのsampleについて次を記録すると原因を調べやすい。

- validation内のsample index
- 正解label
- 両モデルの予測class
- denseモデルの上位2class間margin
- sample内の最大logits差

微小なlogits差でも上位2classがほぼ同点ならargmaxが変わり得るため、不一致を直ちに実装ミスと決めない。

## 20. exact TT置換をrank制限前の対照にする

最初からrank制限TTだけを評価すると、出力差が、

- TT層の実装・モデル接続の誤り
- rankを削ったことによるweight近似

のどちらから生じたかを切り分けにくい。そこで、まずexact TT層を学習済みモデルへ接続し、対象外stateを維持した条件で推論の数値的一致を確認する。

exact TTは、元行列を表現するために必要なrankを保持する。従って、parameter削減を保証しない。

### 学習済みConvの数値例

$64\times32\times3\times3$のConv weightを、out modes$(4,4,4)$、in modes$(6,6,8)$でexact TT化し、actual rankが$(1,24,32,1)$だった場合、

$$
P_{\mathrm{dense}}=64\times32\times3\times3=18{,}432,
$$

$$
P_{\mathrm{TT}}
=1\times4\times6\times24
+24\times4\times6\times32
+32\times4\times8\times1
=20{,}032
$$

となり、TTの方が1,600 parameters多い。

これはTT化の失敗ではなく、このweight、mode分解、exact rankでは圧縮にならないという結果である。元weightが低いTT-rank構造を持つ場合は、exactのままparameter数が少なくなる可能性もある。

exact置換で推論を維持できたという対照を作った後で、同じ接続と評価処理を使ってrank制限を導入すれば、parameter削減と出力・予測・accuracyの変化を比較しやすくなる。ただし、exact rankで動いたことは、任意のrankや設定での正しさを自動的に保証しない。

## 21. TT-Convのrank sweepを設計する

### 21.1 requested rankとactual ranksを分ける

複数の内部bondを持つTTで、全rankへ同じ上限$r$を指定しても、actual rankがすべて$r$になるとは限らない。各unfoldingの最大rank、mode積、元weightのrankによって飽和するためである。

3-core TT-matrixのactual ranksが$(1,r_1,r_2,1)$なら、parameter数は、

$$
P_{\mathrm{TT}}
=m_1n_1r_1+r_1m_2n_2r_2+r_2m_3n_3
$$

である。たとえばout modes$(4,4,4)$、in modes$(6,6,8)$なら、

$$
P_{\mathrm{TT}}=24r_1+24r_1r_2+32r_2
$$

となる。共通上限28に対してactual ranksが$(1,24,28,1)$なら、

$$
P_{\mathrm{TT}}
=24\times24+24\times24\times28+32\times28
=17{,}600
$$

である。requested rankだけでなく、actual ranksと実parameter数を保存する。

### 21.2 同じ元weightから各候補を作る

rank sweepでは、すべての候補を同じdense weightから独立に分解する。

```text
元のdense weight → rank 4のTT-SVD
元のdense weight → rank 8のTT-SVD
元のdense weight → rank 12のTT-SVD
```

rank 4の近似weightをrank 8へ再分解すると、元weightに対するrank 8近似ではなくなる。exact coresを単純に切り取る操作も、元weightへTT-SVDを適用して上限rankで打ち切る操作と同じとは限らない。

### 21.3 共通上限sweepの探索範囲

内部rankが2つあるとき、共通上限$r$だけを変えるsweepは、任意の$(r_1,r_2)$を調べる探索ではない。$(8,16)$や$(12,20)$のような非対称rankは直接比較していない。従って、得られた候補は「今回の探索系列内での候補」と表現する。

## 22. rank制限の影響とdirect実装誤差を分ける

元dense weightを$W$、rank制限coresが表す近似weightを$\widehat W$とする。次の3経路を区別する。

| 出力 | 使用するweight | 計算経路 |
| :-- | :-- | :-- |
| 元dense Conv | $W$ | 通常のConv |
| TT direct | TT cores | coresを直接縮約 |
| 再構成Conv | $\widehat W$ | coresからkernelを再構成してConv |

圧縮による変化を測る比較は、

$$
\mathrm{Conv}(X,W)
\quad\text{vs}\quad
\mathrm{TTConv}(X,\mathrm{cores})
$$

である。一方、direct実装の経路確認は、

$$
\mathrm{TTConv}(X,\mathrm{cores})
\quad\text{vs}\quad
\mathrm{Conv}(X,\widehat W)
$$

である。後者が一致すれば、元weightから離れていても、同じ近似weightをdirect経路が意図どおり計算できている。後者が一致しない場合は、rank truncationとは別に縮約、reshape、軸順序、出力shapeの復元を疑う。

## 23. weight誤差からaccuracyまでを比例関係として扱わない

rank制限後の診断は、次の異なる段階を測る。

```text
weightの差
  ↓ 実入力へ作用
対象層の出力差
  ↓ 後段ネットワーク
logitsの差
  ↓ argmax
予測classの差
  ↓ 正解labelとの比較
accuracyの差
```

この並びは評価段階の順序であり、各指標間の単純な比例関係や因果量を保証しない。

相対Frobenius誤差はweight全体の数値差を測るが、誤差方向が実入力でどれだけ使われるか、分類にどれだけ重要かは含まない。層出力RMSEは実入力を通した差、logits RMSEは後段まで通した全class scoreの差である。Tensorのshapeとscaleが異なるため、RMSEの比だけから後段が誤差を何倍へ増幅・減衰したかを断定しない。

agreementはdenseモデルとの予測一致率、accuracyは正解labelとの一致率である。denseモデルも誤分類するため、denseとの予測不一致には、

- denseの正答を壊す
- denseの誤答を直す
- 両方誤答だが、誤答classを変える

がある。accuracy差は最初の2種類の差し引きであり、3種類目はagreementだけを下げる。

たとえば5,000件中49件で予測が変わり、正解数がdenseより6件多くても、「6件だけ変わってすべて改善した」とは言えない。49件の変化全体について、改善と悪化を差し引いた純増が6件である。

## 24. 層単体とモデル全体の圧縮率

モデル全体のparameter数を$P$、対象層の元parameter数を$P_L$、圧縮後を$\widehat P_L$とすると、全体の削減率は、

$$
\frac{P_L-\widehat P_L}{P}
=
\frac{P_L}{P}
\left(1-\frac{\widehat P_L}{P_L}\right)
$$

である。

モデル全体421,642 parametersのうち対象weightが18,432で、それを4,128へ減らす場合、

$$
\frac{18{,}432}{421{,}642}
\left(1-\frac{4{,}128}{18{,}432}\right)
\approx3.39\%
$$

となる。対象weight自体は4.47倍に圧縮されても、モデル全体の削減率は約3.39%である。層単体の圧縮率だけでモデル全体の効果を表さない。

## 25. Pareto frontierとknee

### 25.1 Pareto frontier

parameter数とvalidation errorをともに小さくしたいとする。候補Aのparameter数とerrorが候補B以下で、少なくとも一方が厳密に小さければ、BはAに支配される。どの候補にも支配されない点の集合がPareto frontierである。

対象層以外のparameter数が全候補で固定なら、core parameter数とモデル全体parameter数は定数差なので、どちらを横軸にしても支配関係は同じになる。複数層を変える比較には、この性質をそのまま使えない。

TT core parameter数が元dense weight以上なら、保存parameter数の意味では対象層を圧縮していない。exact controlは実装確認には必要だが、圧縮候補のfrontierから分離する。

### 25.2 kneeは採用条件ではない

Pareto点を各軸で0から1へ正規化し、両端を結ぶ直線からの距離が最大の内部点をkneeとする方法がある。このkneeは幾何学的規則による代表点であり、accuracy低下の許容幅を含まない。

従って、`balanced`という名前だけからaccuracyが十分維持されたとは判断しない。kneeと、「denseから0.1 percentage points以内」などのtoleranceを満たす最小候補は別のrankになり得る。

frontierが1点または2点、正規化幅が0、全点がほぼ直線上、最大距離が同率、kneeの左右に候補がない場合は、無理に3候補を作らない。

圧縮直後のkneeも最終rankではない。fine-tuning後の回復量によって候補の順位は変わり得るため、学習後に再評価する。

## 26. validation error、loss、logits RMSEを分ける

$$
\mathrm{validation\ error}
=1-\mathrm{validation\ accuracy}
$$

と定義した値は誤分類率であり、cross-entropy lossではない。

cross-entropy lossは、logits$z_{i,c}$と正解class$y_i$から、

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{\exp(z_{i,y_i})}{\sum_c\exp(z_{i,c})}
$$

を計算する。正解classの順位だけでなくscoreの余裕にも反応する。

logits RMSEはdense logits$L$と近似モデルのlogits$\widehat L$の差、

$$
\sqrt{
\frac{1}{BC}
\sum_{i=1}^{B}
\sum_{c=1}^{C}
(\widehat L_{i,c}-L_{i,c})^2
}
$$

を測り、正解labelを参照しない。結果表に誤分類率とlogits RMSEしかない場合、validation lossの回復を確認したとは書かない。

## 27. validationによる候補選択と独立評価

gradient更新へ使わなくても、validation結果を見てrank、tensorization、checkpointを選べば、そのdataはモデル選択に使われている。

```text
validation結果を見る
    ↓
rank・tensorization・checkpointを選ぶ
    ↓
同じvalidationの値を最終性能として扱う
```

この値は候補選択用の観測値としては有効だが、未知dataへの独立評価ではない。基本的には、validationで候補を固定した後、未使用test setでdenseモデルと比較する。test結果を見て候補を選び直すと、そのtest setも選択用dataになる。

## 28. TT-SVD初期値とtask向けfine-tuning

TT-SVDによるrank制限は、元weight $W$と再構成weight $\widehat W$のFrobenius誤差を小さくする近似を作る。一方、分類のfine-tuningでは、正解labelに対するcross-entropyなどのtask lossを小さくする。

```text
TT-SVD：元weightを数値的によく近似する初期値
fine-tuning：正解labelに対してtask lossを下げるparameterへ更新
```

従って、圧縮直後のaccuracyが低いことだけから、そのrankの表現能力が不足しているとは断定できない。同じrankとparameter数のままfine-tuningして回復できれば、圧縮直後の値は「その構成で到達可能な性能の上限」ではなく、「TT-SVDで得た特定の初期値の性能」だったと分かる。

ただし、回復したことはrank制約が一切影響しないことを意味しない。表現能力、初期値、optimizer、学習率、更新範囲、停止条件を一度の結果だけで分離しない。

最終的にfine-tuningする構成を選ぶ場合、圧縮直後のaccuracyだけで小さいrankを除外すると、学習後には有力になる候補を見落とす可能性がある。rank候補は必要に応じてfine-tuning後の同一条件でも比較する。

## 29. cores-only fine-tuningの意味

TT層のcoresだけを更新し、biasと他層を固定すると、他層の適応による補償を除いた状態で「TT表現内部の調整だけでどこまで回復するか」を調べられる。

PyTorchでは、optimizerへ渡すparameterを限定するだけでなく、対象外parameterの`requires_grad`、学習後の値の完全一致、全coresが実際に変化したことを確認する。

cores-onlyで目標accuracyへ回復した場合は、次のように読む。

- その条件と目標に対しては、他parameterを変更せずに回復できた。
- 全parameter更新が不要または不利だと証明したわけではない。
- 全parameter更新ならさらに改善するか、別rankやtest性能で有利かは別の比較である。

逆に回復しなかった場合も、そのrankのあらゆるTT-Convで回復不能とは言えない。tensorization、初期化、学習率、optimizer、更新範囲などの影響が残る。

## 30. task lossとdense fidelityを同じ目的とみなさない

分類fine-tuningで通常最小化するcross-entropyは、正解classを$y$、モデルの予測確率を$p(y\mid x)$として、

$$
\mathcal L_{\mathrm{CE}}
=-log p(y\mid x)
$$

である。元denseモデルのlogitsや予測classへ近づける項を目的関数へ入れていなければ、学習中にdenseから離れることは許される。

そのため、次は同時に起こり得る。

- 正解labelに対するaccuracyやcross-entropyは改善する。
- denseとのprediction agreementは低下する。
- denseとのlogits RMSEは増加する。

agreementはargmax classだけを比較する。logits RMSEは全classのscore差を測るため、agreementが改善してもRMSEが悪化し得る。accuracyはさらに別の、正解labelとの一致である。

denseの出力再現を学習目的に含めたい場合は、task lossとは別にlogitsや中間特徴のdistillation lossなどを設計する必要がある。ただし、その導入は分類accuracyだけを目的としたfine-tuningとは別実験である。

## 31. epoch 0を含むcheckpoint選択とearly stopping

fine-tuningが不要な候補を識別するには、学習前のepoch 0を最初のbest checkpointとして扱う。学習後のmonitor lossがepoch 0を改善しなければ、未更新モデルを選択できる。

monitor lossを選択基準とする場合、rank-validation accuracyはcheckpoint選択へ使わず、epoch 0と選択済みcheckpointの診断に限定すると役割を分けやすい。

`min_delta=\delta`なら、単にlossがわずかに小さくなっただけではbest更新とみなさず、所定量以上の改善を要求する。`patience=p`は、改善条件を満たさないepochが$p$回連続するまで待つ。従って、停止直前に一時停滞があると、patienceの違いで選択checkpointが変わり得る。

early stoppingの結果を読むときは、少なくとも次を保存する。

- epochごとのtrain lossとmonitor loss
- best更新の有無と未改善回数
- 選択epochと実行epoch
- epoch 0を候補へ含めたか
- `min_delta`と`patience`

選択epochは最良checkpointのepoch、実行epochは停止判定まで実際に学習したepochであり、同じではない。

## 32. fine-tuning後の候補選択と学習時間

fine-tuning後は候補のaccuracy順位が変わり得るため、圧縮直後に作ったPareto frontierやkneeを最終選択へそのまま流用しない。parameter数と学習後accuracyで再び支配関係を確認する。

ただし、目的へ学習時間を追加すると支配関係が変わる場合がある。学習コストには少なくとも次の異なる値がある。

- 1 epoch当たりの時間
- 選択checkpointへ到達するまでの時間
- patience待機を含むtrial全体の累積時間
- monitor評価やcheckpoint保存を含むwall-clock時間

小さいrankは1 epoch当たりの計算を減らしても、収束までに多くのepochを要すれば累積時間は短くならない。異なる停止時点と異なる到達accuracyのtrialを比べた時間は、「同じaccuracyへ到達する効率」の比較ではない。また学習時間から推論latencyを推定しない。

## 33. fine-tuning用data、monitor、rank-validation、testの役割

fine-tuning実験では、次の役割を分けて記録する。

| data | 主な役割 |
| :-- | :-- |
| 更新用data | gradient更新 |
| monitor data | early stoppingとcheckpoint選択 |
| rank-validation | rankや構成の比較 |
| test | 候補固定後の最終評価 |

分割が別でも、そのdataが研究の前段で使われていれば完全に独立とは限らない。たとえばmonitor dataが元denseモデルの学習に含まれていた場合、fine-tuning更新用dataからは分離されていても、dense checkpointにとって未知ではない。rank-validationを以前のrank選択に使っていれば、今回のepoch選択に未使用でも、研究全体を通じて未使用ではない。

単一seedかつ数件程度のaccuracy差から、未知dataでの順位や統計的優位性を結論しない。開発用dataで候補を固定した後、未使用test setで比較する。さらに順位の再現性を論じるなら、複数seedやsplitも必要になる。

## 34. 要点

1. `groups=1`のConv2dは、各patchに対する$C_{\mathrm{out}}\times(C_{\mathrm{in}}K_HK_W)$行列の線形写像として表せる。
2. TT-matrixのmodeは行・列添字の因数分解であり、channelや空間軸との物理的対応が自動的に決まるわけではない。
3. exact TTとrank制限TT、mode分解とrank制限、TT表現とparameter圧縮は区別する。
4. dense kernel再構成経路は参照実装として有用である。
5. direct経路は`F.unfold`したpatchへTT coresを直接作用させ、dense weightを明示的に作らない。
6. rank制限による誤差とdirect実装による誤差は、同じcoresの再構成経路を使って分離する。
7. backwardはgradientの存在だけでなく、同条件の参照経路との値の一致を調べる。
8. parameter削減、実行時間、peak memory、MACsは別々に測る。
9. 学習時と、dense kernelをcacheできる固定weight推論時では、公平な比較条件が異なる。
10. `TTConv2d`ではcoresとbiasをparameterとして登録し、対応範囲とshape contractを初期化時に検証する。
11. 学習済みモデルを比較するときは元モデルを残し、対象層以外のparameter・bufferが一致することを確認する。
12. `eval()`と`inference_mode()`は役割が異なり、推論比較では両方を使用する。
13. validation全体のRMSEは二乗誤差合計と全要素数から計算し、batch RMSEを単純平均しない。
14. exact TT置換は圧縮を保証せず、rank制限による近似を評価する前の対照として使える。
15. rank sweepではrequested upper bound、actual ranks、実parameter数を分けて保存する。
16. 各rank候補は同じ元dense weightから独立に分解する。
17. 共通rank上限sweepは内部rankペア全体の探索ではない。
18. rank制限による元モデルとの差と、同じ近似weightを使うdirect・再構成経路の差を分ける。
19. weight誤差、層出力RMSE、logits RMSE、agreement、accuracyは異なる段階を測り、単純な比例関係を仮定しない。
20. 対象層の圧縮率とモデル全体の圧縮率を分ける。
21. Pareto kneeは幾何学的候補であり、accuracy維持や最終採用を保証しない。
22. `validation_error = 1 - accuracy`、cross-entropy loss、logits RMSEを区別する。
23. validationで候補を選んだ後は、候補を固定して未使用test setで評価する。
24. TT-SVD初期値の性能と、同じrankでfine-tuning後に到達できる性能を区別する。
25. cores-onlyで回復できたことは、全parameter更新が不要または不利である証明ではない。
26. task lossがdense再現を含まなければ、accuracy改善とagreement・logits RMSEの悪化は両立する。
27. epoch 0をcheckpoint候補に含め、`min_delta`、`patience`、選択epoch、実行epochを記録する。
28. fine-tuning後はparameter数とaccuracyのPareto関係を再評価し、学習時間を別の目的として扱う。
29. 更新用data、monitor、rank-validation、testの役割と、過去の使用履歴を区別する。
