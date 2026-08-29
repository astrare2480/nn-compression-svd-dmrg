---
title: compression設計
---

# compression設計

## 1. 役割

`nn_compression.compression` は、学習済みNN weightを低rank表現へ変換する処理を担当する。

```text
SVD
├─ matrix-level SVD
├─ Linear factorization
├─ Conv2d factorization
└─ model / named layer replacement

Tucker
├─ HOSVD
├─ HOOI
├─ Tucker parameter metrics
└─ Tucker-2 Conv
```

実験評価そのものは `metrics`、mode-n演算は `tensor` へ分離する。

---

# 2. SVD core

## `truncated_svd`

```python
truncated_svd(matrix, rank)
```

2次元行列

$$
W\in\mathbb{R}^{m\times n}
$$

に対し、

$$
W\approx U_r\Sigma_rV_r^{\mathsf T}
$$

を返す。

PyTorchの返り値に合わせて、

```text
U_r  : (m, r)
S_r  : (r,)
Vh_r : (r, n)
```

とする。

rankは、

$$
1\le r\le \min(m,n)
$$

を要求し、範囲外を黙ってclampしない。

### scalar contract

rankでは、

- Python `bool` を拒否
- Tensor scalarを拒否
- integer protocolに従うscalarを受理
- NumPy integer scalarは受理

とする。

---

## retained energy

```python
retained_energy_from_matrix(matrix, rank)
retained_energy(layer, rank)
```

$$
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_i\sigma_i^2}
$$

を返す。

ゼロ行列では総energyが0なので、現在は `1.0` を返す。

これはweight Frobenius normの保持率であり、task accuracyを保証する指標ではない。

`retained_energy(layer, rank)` は `layer.weight.detach()` を2次元化して評価するだけで、学習graphへ参加しない。

---

# 3. Linear SVD

## `rebuild_linear_from_svd`

SVD成分から元と同じshapeの1つのLinearを構築する。

```text
用途
→ reconstruction error確認
→ parameter削減はしない
```

$$
W_r=U_r\Sigma_rV_r^{\mathsf T}
$$

を新しい `nn.Linear` のweightへコピーする。

---

## `factorize_linear_layer`

```text
Linear(D_in → D_out)
```

を、

```text
Linear(D_in → r, bias=False)
→ Linear(r → D_out, bias=original)
```

へ変換する。

重み配置：

```text
first.weight  = Vh_r
second.weight = U_r @ diag(S_r)
```

biasは最終出力側だけへ置く。

### Module contract

- input layerを変更しない
- 新しい `nn.Sequential` を返す
- device維持
- dtype維持
- weight `requires_grad` 維持
- bias `requires_grad` 維持
- weight copyは `torch.no_grad()` 内で行う

分解時は学習済み `layer.weight.detach()` を用いる。ここは「新しいModuleの初期値を作る」境界であり、元Parameterまで逆伝播する用途ではない。

---

## named Linear

`factorize_named_linear()` はmodelを `deepcopy()` し、copy側だけを置換する。

```text
baseline Parameter
≠
compressed Parameter
```

を保証する。

nested pathも `get_submodule / set_submodule(strict=True)` 系のutilityで扱う。

---

# 4. Conv2d SVD

## unfolding

```python
conv2d_weight_matrix(weight)
```

$$
(C_{out},C_{in},K_h,K_w)
\rightarrow
(C_{out},C_{in}K_hK_w)
$$

へreshapeする。

この行列にSVDを適用する。

---

## `factorize_conv2d_layer`

元Conv：

```text
Conv(C_in → C_out, kH×kW)
```

を、

```text
Conv(C_in → r, kH×kW)
→ Conv(r → C_out, 1×1)
```

へ置換する。

### semantic mapping

1層目：

- 元 `kernel_size`
- 元 `stride`
- 元 `padding`
- 元 `dilation`
- 元 `padding_mode`
- biasなし

2層目：

- 1x1
- stride 1
- padding 0
- biasは元Convと同じ有無

### 対応範囲

現在は `groups=1` の通常 `nn.Conv2d` のみ正式対応する。

grouped convolutionを数学的に別扱いせずflattenすると意味が変わるため、明示的にrejectする。

---

# 5. model-level SVD

## `factorize_named_layers`

```python
factorize_named_layers(
    model,
    *,
    conv_ranks={...},
    linear_ranks={...},
)
```

複数層を一度に置換する。

重要なのは、各層の分解元を**圧縮途中のmodelではなく元baseline modelから取る**こと。

```text
baseline
├─ conv1 → factorize
├─ conv2 → factorize
└─ fc1   → factorize
```

とし、前の置換結果を次のSVD入力に使わない。

---

## MLP convenience API

`make_one_layer_svd_model` / `make_two_layer_svd_model` は現行MLPの `fc1 / fc2 / fc3` 構造に依存するexperiment-support API。

generic compression APIではないため、TT/MPSから新しいモデル構造へ流用する前提にはしない。

`r1 / r2` はhistorical Notebook互換のみ。

---

# 6. HOSVD

## `hosvd`

```python
hosvd(X, ranks)
```

`ranks` は、

```python
{mode: rank}
```

のMapping。

全modeを指定すればtruncated HOSVD、一部だけならpartial HOSVDになる。

### factor計算順序

各factorは必ず**元のX**から求める。

```text
X → unfold(mode 0) → SVD → U0
X → unfold(mode 1) → SVD → U1
...
```

factor計算中に、先にprojectしたcoreを次modeのSVD入力へ使わない。

これは標準HOSVDと別アルゴリズムになるため、Core API v1では固定する。

### core

factorを求めた後、

$$
G
=
X
\times_{n_1}U_{n_1}^{\mathsf T}
\times_{n_2}U_{n_2}^{\mathsf T}
\cdots
$$

としてcoreを得る。

---

## Tucker rank上限

mode $n$ のunfoldingは、

$$
I_n
\times
\prod_{m\neq n}I_m
$$

なので最大rankは、

$$
\boxed{
\min\left(I_n,\prod_{m\neq n}I_m\right)
}
$$

とする。

単に `rank <= shape[mode]` だけではない。

HOSVD・parameter count・ratio・compression factorでこの同じ数学的上限を使う。

---

## empty ranks

HOSVD / parameter count系では、

```python
ranks = {}
```

をidentity指定として許す。

```text
core = X
factors = {}
```

に相当する。

HOOIでは反復更新するmodeが無いため空ranksを許さない。

---

# 7. Tucker parameter metrics

## `tucker_parameter_count`

保存要素数：

$$
N
=
|G|
+
\sum_{n\in\text{compressed modes}}I_nR_n
$$

partial Tuckerでは未圧縮modeをcoreに元dimensionのまま残す。

## `parameter_ratio`

$$
\frac{N_{Tucker}}{N_{original}}
$$

小さいほど圧縮。

## `compression_factor`

$$
\frac{N_{original}}{N_{Tucker}}
$$

大きいほど圧縮。

この3つはweight表現の要素数であり、モデル全体parametersやMACsとは別指標。

---

# 8. HOOI

## 初期化

`hooi()` はHOSVDから開始する。

```text
HOSVD
→ initial reconstruction error
→ history[0]
→ sweep
→ core更新
→ reconstruction
→ error
→ convergence
```

`history[0]` の意味を変えない。

---

## 1 sweep

`hooi_sweep()` では各target modeについて、target以外のfactorでXをprojectする。

```text
target = mode 0
→ mode 1,2,...をproject
→ unfold(mode 0)
→ SVD
→ U0更新
```

sweep内では既に更新済みfactorを後続modeで使う。

更新順は `ranks` Mappingの挿入順。

Tucker-2では通常、

```python
{0: rank_out, 1: rank_in}
```

なので mode 0 → mode 1。

---

## factor validation

HOOIでは、

```text
factor keys == rank keys
U_n.shape == (X.shape[n], rank[n])
U_n.dtype == X.dtype
U_n.device == X.device
```

を要求する。

また、各mode更新前にprojected unfoldingで可能な最大rankを計算し、指定rankが実現可能か確認する。

この確認は `max_iter` に依存せず、`max_iter=0` でも行う。

---

## convergence

```python
abs(error - prev_error)
<= abs_tol + rel_tol * abs(prev_error)
```

を使う。

`abs_tol / rel_tol` は、

```text
有限値
>= 0
bool拒否
```

`max_iter` は、

```text
bool以外の整数
>= 0
```

とする。

`max_iter=0` ではHOSVD初期値だけを返し、historyは初期誤差1要素になる。

---

# 9. 実数dtype contract

現行HOSVD/HOOIのfactor射影は

```python
U.T
```

を使う。

実数では正しいが、複素数なら共役転置 `U.mH` が必要になる。

`torch.linalg.svd` 自体がcomplexを処理できるためsilent failureを防ぐ目的で、**HOSVD/HOOI分解入口では複素TensorをTypeErrorで拒否する**。

対象：

- `hosvd`
- `hooi`
- `hooi_sweep`
- `core_from_factors`
- HOSVD/HOOIを内部利用するTucker-2 decomposition APIs

### 注意

`reconstruct_tucker()` はfactorを掛け戻すutilityであり、独自のreal-dtype guardは持たない。

また、`build_tucker2_conv_from_components()` は既に与えられたcomponentsからModuleを作るため、分解入口と同じcomplex guardを直接持っていない。

したがってCore API v1での表現は、

> **Tucker/HOOIの分解アルゴリズムは実数Tensor限定**

とし、「`tucker2_*` という名前の全関数が一律complex拒否」とは定義しない。

---

# 10. Tucker-2 Conv

Conv2d weight：

$$
W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

mode 0 / 1だけを圧縮する。

$$
W
\approx
G\times_0U_{out}\times_1U_{in}
$$

shape：

```text
U_out : (C_out, R_out)
U_in  : (C_in,  R_in)
core  : (R_out, R_in, kH, kW)
```

---

## 3層構造

```text
input projection
Conv2d(C_in -> R_in, 1x1, bias=False)
        ↓
core convolution
Conv2d(R_in -> R_out, original k/stride/padding/dilation, bias=False)
        ↓
output projection
Conv2d(R_out -> C_out, 1x1, bias=original)
```

重み：

```text
input.weight  = U_in.T[:, :, None, None]
core.weight   = core
output.weight = U_out[:, :, None, None]
```

---

## rank contract

Tucker-2 channel rankは、

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

とする。

Conv weightのmode 0 / 1では空間modeの積が存在するため、通常これが各channel modeのunfolding上限と整合する。

boolは拒否する。

---

## Conv semantic contract

- `groups=1`
- transposed convolution未対応
- core layerが元 `kernel_size`
- core layerが元 `stride`
- core layerが元 `padding`
- core layerが元 `dilation`
- core layerが元 `padding_mode`
- input/output projectionは1x1、stride1、padding0、dilation1
- biasはoutput projectionのみ

---

# 11. autograd境界

## Tensor-level decomposition

```python
tucker2_hooi(weight, ...)
```

では入力 `weight` をdetachしない。

これは低レベルTensor APIとして、分解計算自体のgraphを利用側が必要なら保持できるようにするため。

## Module build

```python
build_tucker2_conv(conv, ...)
```

では、

```python
conv.weight.detach()
```

を分解へ渡す。

学習済みParameterから新しい3層Parameterの**初期値を作る処理**であり、元Convへ逆伝播する必要はない。

componentsのcopyは `torch.no_grad()` 内で行う。

新しく構築されたParameterはleafであり、その後のFine-tuningで通常どおりgradientを持つ。

元weightとstorageを共有しない。

---

# 12. `tucker2_effective_weight`

Fine-tuning後の3層からeffectiveな4階weightを再構成する評価utility。

```text
input projection
core layer
output projection
→ U_in / core / U_outを取得
→ reconstruct_tucker
→ effective weight
```

ここでは各layer weightを `detach()` してから再構成する。

用途は評価であり、effective weightを通したgradient計算を目的にしない。

Sequential構造は厳密に、

```text
3層
全てConv2d
1x1 → core → 1x1
channel接続整合
projectionのstride/padding/dilation/groups/bias contract
```

を検証する。

---

# 13. rank sweep

`rank_sweep.py` はアルゴリズム本体より実験支援に近い。

```text
rank候補
→ baselineからcandidate deepcopy
→ 1層だけ置換
→ collect_compression_metrics
→ retained energy
→ numeric record
```

全candidate modelを保持しない。

Fine-tuningする候補はrankから再構築する。

この設計により、

- GPU/CPU memory保持量
- pandas object column
- CSV非serializable

を避ける。

---

# 14. compatibility names

以下はhistorical Notebook互換のため残す。

```text
SVD
RebuildSVD
factorize_Conv2d_layer
sweep_conv_svd_ranks
```

新しい実装ではprimary APIへ寄せる。

---

# 15. TT/MPSへの拡張時に守ること

TT/MPSを実装するとき、SVD/Tucker APIの意味をTensor Network向けに変更しない。

再利用候補：

```text
truncated_svd
relative_frobenius_error
count_parameters
benchmark_inference
training loops
seed/path utilities
```

一方、TT rank / bond dimensionのcontractはTucker rankと意味が異なるので、新しいvalidation/APIとして追加する。

例：

```text
compression/tt.py
or
compression/tensor_train.py
```

に新規処理を追加し、既存 `hosvd / hooi` のsignatureを広げてTTまで処理させない。
