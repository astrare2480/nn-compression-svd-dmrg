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

2次元行列 $W\in\mathbb{R}^{m\times n}$ に対し、

$$
W\approx U_r\Sigma_rV_r^{\mathsf T}
$$

を返す。

```text
U_r  : (m, r)
S_r  : (r,)
Vh_r : (r, n)
```

rankは、

$$
1\le r\le \min(m,n)
$$

を要求し、範囲外を黙ってclampしない。

rank scalarはPython `bool` とTensor scalarを拒否し、`operator.index()` に従う整数scalarを受理する。NumPy integer scalarは受理する。

---

## retained energy

```python
retained_energy_from_matrix(matrix, rank)
retained_energy(layer, rank)
```

$$
E(r)=\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_i\sigma_i^2}
$$

を返す。

ゼロ行列では総energyが0なので、現在は `1.0` を返す。

これはweight Frobenius normの保持率であり、task accuracyを保証する指標ではない。

`retained_energy(layer, rank)` は `layer.weight.detach()` を評価用に用いる。

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

```text
first.weight  = Vh_r
second.weight = U_r @ diag(S_r)
```

biasは最終出力側だけへ置く。

### Module contract

- input layerを変更しない
- 新しい `nn.Sequential` を返す
- device / dtype維持
- weight / biasの `requires_grad` 維持
- weight copyは `torch.no_grad()` 内

分解時は学習済み `layer.weight.detach()` を用いる。新しいModuleの初期値を作る境界であり、元Parameterまで逆伝播する用途ではない。

---

## named Linear

`factorize_named_linear()` はmodelを `deepcopy()` し、copy側だけを置換する。

```text
baseline Parameter
≠
compressed Parameter
```

nested pathもnamed module utilityで扱う。

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

## `factorize_conv2d_layer`

```text
Conv(C_in → C_out, kH×kW)
```

を、

```text
Conv(C_in → r, kH×kW)
→ Conv(r → C_out, 1×1)
```

へ置換する。

1層目は元 `kernel_size / stride / padding / dilation / padding_mode` を継承し、biasなし。2層目は1x1、stride 1、padding 0で元biasを保持する。

現在は `groups=1` の通常 `nn.Conv2d` のみ正式対応する。

rank上限は、flatten後の行列shapeから

$$
r\le\min(C_{out},C_{in}K_hK_w)
$$

とする。

---

# 5. model-level SVD

`factorize_named_layers()` は複数named layerを一度に置換する。

重要なのは、各層の分解元を**圧縮途中のmodelではなく元baseline modelから取る**こと。

```text
baseline
├─ conv1 → factorize
├─ conv2 → factorize
└─ fc1   → factorize
```

前の置換結果を次のSVD入力に使わない。

`make_one_layer_svd_model` / `make_two_layer_svd_model` は現行MLPの `fc1 / fc2 / fc3` 構造向けexperiment-support APIで、`r1 / r2` はhistorical Notebook互換。

---

# 6. HOSVD

## `hosvd`

```python
hosvd(X, ranks)
```

`X` は2階以上の実数Tensor、`ranks` は `{mode: rank}` のMapping。

全modeを指定すればtruncated HOSVD、一部だけならpartial HOSVDになる。

### factor計算順序

各factorは必ず**元のX**から求める。

```text
X → unfold(mode 0) → SVD → U0
X → unfold(mode 1) → SVD → U1
...
```

factor計算中に、先にprojectしたcoreを次modeのSVD入力へ使わない。

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
I_n\times\prod_{m\neq n}I_m
$$

なので最大rankは、

$$
\boxed{\min\left(I_n,\prod_{m\neq n}I_m\right)}
$$

とする。

単に `rank <= shape[mode]` だけではない。

HOSVD・parameter count・ratio・compression factorでこの同じ数学的上限を使う。

---

## empty ranks / reconstruction

HOSVD / parameter count系では `ranks={}` をidentity指定として許す。

```text
core = X
factors = {}
```

`reconstruct_tucker(core,{})` もidentityとしてcoreを返すが、Tucker系のshape contractに合わせて `core.ndim >= 2` を要求する。

HOOIでは反復更新modeが無いため空ranksを許さない。

---

# 7. Tucker parameter metrics

` tucker_parameter_count` は、

$$
N=|G|+\sum_{n\in\text{compressed modes}}I_nR_n
$$

を返す。

`parameter_ratio`：

$$
\frac{N_{Tucker}}{N_{original}}
$$

`compression_factor`：

$$
\frac{N_{original}}{N_{Tucker}}
$$

これらはweight表現の要素数であり、モデル全体parametersやMACsとは別指標。

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

`hooi_sweep()` では各target modeについて、target以外の最新factorでXをprojectする。

sweep内では更新済みfactorを後続modeで使う。更新順は `ranks` Mappingの挿入順。

Tucker-2では通常 `{0: rank_out, 1: rank_in}` なので mode 0 → mode 1。

---

## validation

HOOIでは、

```text
X.ndim >= 2
Xは実数dtype
factor keys == rank keys
U_n.shape == (X.shape[n], rank[n])
U_n.dtype == X.dtype
U_n.device == X.device
```

を要求する。

各mode更新前にprojected unfoldingで可能な最大rankを計算し、指定rankが実現可能か反復前に確認する。この確認は `max_iter=0` でも行う。

`core_from_factors()` も単独Public APIとして `X.ndim >= 2` と実数dtypeを要求する。ただし、factor key集合とranksの一致を確認するAPIではなく、渡された各factorを順に `mode_dot` するutilityである。

---

## convergence

```python
abs(error - prev_error)
<= abs_tol + rel_tol * abs(prev_error)
```

`abs_tol / rel_tol` は有限・0以上・bool拒否。

`max_iter` はbool以外の整数・0以上。

`max_iter=0` ではHOSVD初期値だけを返し、historyは初期誤差1要素になる。

---

# 9. 実数dtype contract

現行HOSVD/HOOIのfactor射影は `U.T` を使う。

複素Tensorでは `U.mH` が必要になるため、`torch.linalg.svd` がcomplexを受理しても、HOSVD/HOOI分解入口では複素TensorをTypeErrorで拒否する。

対象：

- `hosvd`
- `hooi`
- `hooi_sweep`
- `core_from_factors`
- HOSVD/HOOIを内部利用するTucker-2 decomposition APIs

`reconstruct_tucker()` はfactorを掛け戻すutilityであり、独自のreal-dtype guardは持たない。

`build_tucker2_conv_from_components()` も既に与えられたcomponentsからModuleを作るため、分解入口と同じcomplex guardを直接持たない。

したがって、Core API v1では

> **Tucker/HOOIの分解アルゴリズムは実数Tensor限定**

と定義し、「`tucker2_*` 全関数が一律complex拒否」とは定義しない。

---

# 10. Tucker-2 Conv

Conv2d weight：

$$
W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

mode 0 / 1だけを圧縮する。

$$
W\approx G\times_0U_{out}\times_1U_{in}
$$

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

```text
input.weight  = U_in.T[:, :, None, None]
core.weight   = core
output.weight = U_out[:, :, None, None]
```

---

## rank contract

component shapeとしては、

```text
1 <= rank_out <= C_out
1 <= rank_in  <= C_in
```

を要求する。

ただし**分解API**ではgeneric HOSVD/HOOIのmode-unfolding上限も適用する。

4階weight `(C_out,C_in,kH,kW)` なら、

$$
R_{out}\le\min(C_{out}, C_{in}K_hK_w)
$$

$$
R_{in}\le\min(C_{in}, C_{out}K_hK_w)
$$

である。

対象：

- `tucker2_decompose_conv_weight`
- `tucker2_hooi`
- `tucker2_hooi_sweep`
- 内部で分解を行う `build_tucker2_conv`

一方 `build_tucker2_conv_from_components` は既に計算されたcomponentsからModuleを構築するAPIなので、channel shape整合を確認し、分解時のunfolding feasibilityを再計算しない。

bool rankは拒否する。

---

## Conv semantic contract

- `groups=1`
- transposed convolution未対応
- core layerが元 `kernel_size / stride / padding / dilation / padding_mode`
- input/output projectionは1x1、stride1、padding0、dilation1
- biasはoutput projectionのみ

---

# 11. autograd境界

## Tensor-level decomposition

`tucker2_hooi(weight, ...)` は入力 `weight` をdetachしない。

低レベルTensor APIとして、分解計算自体のgraphを利用側が必要なら保持できるようにするため。

## Module build

`build_tucker2_conv(conv, ...)` は `conv.weight.detach()` を分解へ渡す。

学習済みParameterから新しい3層Parameterの**初期値を作る処理**であり、元Convへ逆伝播する必要はない。

componentsのcopyは `torch.no_grad()` 内で行う。

新しく構築されたParameterはleafで、その後のFine-tuningで通常どおりgradientを持つ。元weightとstorageを共有しない。

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

各layer weightを `detach()` してから再構成する。用途は評価であり、effective weightを通したgradient計算を目的にしない。

Sequential構造は、3層・全てConv2d・1x1→core→1x1・channel接続・projectionのstride/padding/dilation/groups/biasを検証する。

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

全candidate modelを保持しない。Fine-tuningする候補はrankから再構築する。

---

# 14. compatibility names

historical Notebook互換：

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

TT rank / bond dimensionのcontractはTucker rankと意味が異なるので、新しいvalidation/APIとして追加する。

既存 `hosvd / hooi` のsignatureを広げてTTまで処理させない。
