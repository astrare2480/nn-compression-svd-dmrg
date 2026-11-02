---
title: TTLinear 2-core局所更新のPyTorch確認
aliases:
  - TT-matrix結合テンソルの局所勾配更新
tags:
  - PyTorch
  - TTLinear
  - DMRG-like
  - two-site update
  - autograd
---

# TTLinear 2-core局所更新のPyTorch確認

## このノートの位置づけ

`notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb` で行った、TT-MLP第1層の隣接2 coreに対する1回の局所勾配更新を整理する。

対象Notebookが確認するのは、次の最小単位である。

```text
保存済みTT-MLPを読み込む
→ 第1層をmixed-canonical formにする
→ 1組の隣接coreを結合する
→ 結合テンソルを独立leafにする
→ 固定minibatchの分類損失で1回だけSGD更新する
→ SVDで再分割する
→ rank制限の影響を測る
→ core列をモデルへ書き戻す
```

このNotebookは、full sweep、複数sweep、epoch学習、early stopping、Fashion-MNIST上の性能比較を行わない。理論は[[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]、全データsweepの設計は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]に分ける。

本ノートの数値はNotebookに保存された出力の記録である。このノートを作成した時点で新しい環境からFresh Run Allを行った証明ではない。

---

## 1. 物理MPS用の2-site関数をそのまま使わない

物理MPS coreの典型shapeは

$$
(r_{k-1},d_k,r_k)
$$

であり、2-site結合は4階テンソルになる。

TT-matrix coreのshapeは

$$
(r_{k-1},m_k,n_k,r_k)
$$

なので、2-core結合は6階テンソルになる。

$$
(r_{k-1},m_k,n_k,m_{k+1},n_{k+1},r_{k+1})
$$

さらに、物理状態では状態ノルムを1へ正規化する処理を入れる場合があるが、NN重みのcoreを再正規化するとLinear層の重みscaleを変えてしまう。このため、TT-matrix専用のAPIを使う。

一方、正準化そのものは

$$
q_k=m_kn_k
$$

として各coreを

$$
(r_{k-1},q_k,r_k)
$$

へ一時的に見れば、通常のTT/MPS正準化へ帰着できる。

---

## 2. 対象モデルとcore shape

読み込んだcheckpointは

```text
models/30_tt_mps/10_fashion_mnist_mlp/
14_rank12_update_scope_ablation/tt_cores_only_selected.pt
```

の `state_dict` である。モデルは

```text
784 -> 512 -> 256 -> 10
```

のMLPで、Linear層の間にReLUを持つ。`fc1` をTTLinearへ差し替えてから `state_dict` を読み込む。

checkpointには `out_modes` と `in_modes` の設定値が独立には保存されていないため、

```text
fc1.cores.k.shape[1] -> out mode m_k
fc1.cores.k.shape[2] -> in mode n_k
```

としてcore shapeから読み、Notebook 14の記録と積512・784を照合した。

Fashion-MNISTの60,000件をseed 0で50,000 / 5,000 / 5,000へ分け、学習subsetのindex 0から63までを固定batchとして使った。shuffle loaderの `next(iter(loader))` は使っていない。

`model.eval()` はdropoutやbatch normalizationの挙動を評価用へ切り替えるが、autogradを無効にはしない。記録用forwardだけを `torch.no_grad()` で囲み、局所学習用forwardはgraphを有効にする。一方、モデル内の全parameterは `requires_grad_(False)` とし、モデル外のleaf $T$ だけを更新対象にした。

保存済みNotebook出力では、第1TTLinear層のmodeとrankは次だった。

```text
out_modes = (32, 8, 2)
in_modes  = (2, 8, 49)
ranks     = (1, 12, 12, 1)
```

積は

$$
32\cdot8\cdot2=512,
$$

$$
2\cdot8\cdot49=784
$$

なので、dense weightのshapeは

$$
512\times784
$$

である。3 coreのshapeは

$$
G^{(1)}:(1,32,2,12),
$$

$$
G^{(2)}:(12,8,8,12),
$$

$$
G^{(3)}:(12,2,49,1)
$$

だった。

Notebook 04はpair index 0、すなわち $G^{(1)}$ と $G^{(2)}$ を更新対象にした。このとき結合テンソルは

$$
T:(1,32,2,8,8,12)
$$

である。

行列化後のshapeは

$$
(1\cdot32\cdot2)
\times
(8\cdot8\cdot12)
=
64\times768
$$

なので、還元SVDが返す形状上の最大成分数は

$$
\min(64,768)=64
$$

である。これは数値rankが64であるという意味ではない。

---

## 3. 2 coreの結合

実装上の要素対応は

$$
\begin{aligned}
&T[
\alpha,
i_k,
j_k,
i_{k+1},
j_{k+1},
\gamma
]
\\
&\qquad=
\sum_{\beta=1}^{r_k}
G_k[
\alpha,i_k,j_k,\beta
]
G_{k+1}[
\beta,i_{k+1},j_{k+1},\gamma
]
\end{aligned}
$$

である。PyTorchでは概念的に

```python
# G_left:  (r_left, m_left, n_left, r_mid)
# G_right: (r_mid, m_right, n_right, r_right)
T = torch.einsum("aijb,bklc->aijklc", G_left, G_right)
```

と縮約する。

ここで文字対応は次である。

```text
a = alpha
i = i_k
j = j_k
b = beta
k = i_{k+1}
l = j_{k+1}
c = gamma
```

結合直後に、通常のcore列を通るforwardと、pairだけを $T$ へ置き換えたforwardを比較する。結合は縮約順序の変更だけなので、同じdtypeで同じ演算順に近ければ、差は浮動小数点丸め誤差scaleになる。

---

## 4. 結合テンソルを独立leafにする

`T` を元coreからそのまま作ると、`T` は元coreへつながる非leaf tensorである。Notebook 04では、現在の値を保ったまま元coreから計算graphを切り、局所変数として独立させる。

```python
# 現在の結合テンソルの値だけを引き継ぎ、独立した局所変数にする。
T_local = T.detach().clone().requires_grad_(True)
```

各操作の役割は次である。

- `detach()`：元coreへ続くgraphを切る。
- `clone()`：元coreとstorageを共有しない独立tensorを作る。
- `requires_grad_(True)`：`T_local.grad` を受け取れるleafにする。

この構成では、forwardとbackwardを行っても元モデルのTT core parameterは更新されない。局所更新後にSVDで分割し、明示的に書き戻すまでモデル本体は不変である。

---

## 5. $T$ を通るforward

局所更新に必要なのは、dense weightを再構成して `F.linear` へ渡す経路ではなく、pairだけを $T$ に置き換えたTTLinear direct forwardである。

$$
x
\longrightarrow
\text{TT-matrix縮約}
\longrightarrow
h_1
\longrightarrow
\operatorname{ReLU}
\longrightarrow
h_2
\longrightarrow
\operatorname{ReLU}
\longrightarrow
\text{logits}
\longrightarrow
L
$$

この経路で、

$$
\frac{\partial L}{\partial T}
$$

をautogradに計算させる。

Notebook 04の局所確認では、TT core以外のparameterも含めて元モデルを更新しない。したがって、更新対象は外部leafの $T$ だけである。

---

## 6. 1回の手動SGD更新

固定minibatchの損失を $L(T)$ とし、

```python
loss.backward()
g = T_local.grad
```

により勾配を得る。

手動SGDは、更新式をgraphの外で計算する。

```python
# 更新式自体を次の高階微分graphへ含めない。
with torch.no_grad():
    T_step = T_local - learning_rate * g

# SVD前の更新済み局所テンソルとして固定する。
T_new = T_step.detach()
```

数学的には

$$
T_{\mathrm{new}}
=
T-eta g
$$

である。丸め誤差を無視すれば、

$$
\lVert T_{\mathrm{new}}-T\rVert_F
=
\eta\lVert g\rVert_F
$$

となる。

保存済み出力では

$$
\lVert g\rVert_F
\approx
0.3077406,
$$

$$
\eta=0.001,
$$

$$
\eta\lVert g\rVert_F
\approx
3.077406\times10^{-4},
$$

$$
\lVert T_{\mathrm{new}}-T\rVert_F
\approx
3.07767\times10^{-4}
$$

だった。差はfloat32の丸めの範囲である。

---

## 7. 勾配更新前後の損失

保存済み出力の固定batch sizeは64だった。局所更新の損失は

$$
L_{\mathrm{before}}
\approx
0.1406570226,
$$

$$
L_{\mathrm{after\ gradient}}
\approx
0.1405630112
$$

であり、1回のSGD step後に

$$
L_{\mathrm{after\ gradient}}
-
L_{\mathrm{before}}
\approx
-9.40114\times10^{-5}
$$

だけ低下した。

1次のTaylor項が予測する変化は

$$
-\eta\lVert g\rVert_F^2
\approx
-0.001\cdot(0.3077406)^2
\approx
-9.47\times10^{-5}
$$

であり、観測値 $-9.40\times10^{-5}$ と近い。これはこの小さいstepで高次項が小さかったことを示す局所的な観測であり、別のstepでも同じ近似精度になる保証ではない。

これは、この固定batch、この初期状態、この学習率、このpairで観測された結果である。任意のbatch、pair、学習率での単調減少を意味しない。

また、checkpointを通常forwardした最初の表示は

$$
L_{\mathrm{checkpoint}}
\approx
0.1406568587
$$

だった。局所forward側の更新前損失との差は、演算経路とfloat32の丸めによる微小差である。

---

## 8. 更新後テンソルのSVD再分割

$T_{\mathrm{new}}$ を

$$
M_{\mathrm{new}}
\in
\mathbb{R}^{64\times768}
$$

へreshapeし、

$$
M_{\mathrm{new}}
=
U\Sigma V^{\mathsf T}
$$

とする。

### 8.1 切り捨てなし再分割

右へ進む分割は

$$
G_k'=\operatorname{reshape}(U),
$$

$$
G_{k+1}'
=
\operatorname{reshape}(\Sigma V^{\mathsf T})
$$

である。還元SVDの全64成分を保持したため、保存済み表示の `retained_rank=64` は数値rank判定ではなく、**shapeが許す全成分を保持した**という意味である。

保存済み再構成残差は、

$$
\lVert
M_{\mathrm{new}}-U\Sigma V^{\mathsf T}
\rVert_F
\approx
2.72\times10^{-12}
$$

をfloat64診断で確認した。一方、同じ確認をfloat32で行った残差表示は約

$$
1.22\times10^{-3}
$$

だった。

ここでfloat64診断は、**同じfloat32入力値をfloat64へcastしてSVDを再計算したときの数値安定性**を確認するものであり、元のfloat32分割の誤差を後からfloat64へcastした値ではない。この2つを同じ意味で比較しない。

float32分割後の左coreについて、保存済みの左直交性誤差は

$$
\lVert U^{\mathsf T}U-I\rVert_F
\approx
7.49\times10^{-5}
$$

だった。切り捨てなし再分割後のlogitsの最大絶対差は約 $4.08\times10^{-4}$、損失差は約 $2.38\times10^{-7}$ だった。logitsは絶対誤差だけでなく、保存済み検証で指定した相対許容差も含めて合格している。

### 8.2 rank 12への再分割

同じ $T_{\mathrm{new}}$ から独立にrank 12分割を作った。捨てた特異値のtailは

$$
\delta_{12}^2
=
\sum_{j>12}\sigma_j^2
\approx
7.910722\times10^{-8},
$$

$$
\delta_{12}
\approx
2.8132\times10^{-4}
$$

だった。相対残差は

$$
\frac{\delta_{12}}{\lVert T_{\mathrm{new}}\rVert_F}
\approx
2.3498\times10^{-6}
$$

だった。

勾配step scale

$$
s=\eta\lVert g\rVert_F
\approx
3.0774\times10^{-4}
$$

との比は、

$$
\frac{\delta_{12}}{s}
\approx
0.914
$$

である。理論ノートの十分条件 $\delta_r\le s/8$ は満たさない。それでも実測では打ち切り後損失が更新前より低かった。このことは、十分条件が必要条件ではないことを示す。

更新前の行列 $M$ 自身はrank 12以下なので、rank 12行列集合までの最短距離には

$$
\delta_{12}
\le
\lVert M_{\mathrm{new}}-M\rVert_F
=s
$$

が必ず成り立つ。比が約0.914ということは、このstepでrank 12集合から離れた距離がstep normの約91%に達し、同じrankへ戻すSVDが更新の大部分を失わせたことを示す。ただし、rank制約集合は線形部分空間ではないので、「stepの91%が厳密な直交成分だった」とまでは言えない。

$\delta_{12}>0$ なので、更新後行列には元bond 12ではexactに表せない成分が実際に生じている。切り捨てなしの `retained_rank=64` だけでは数値rank増加の証拠にならないが、非零のtailは元rankを超えたことの数値的証拠になる。

ただし、最終的には元bondと同じrank 12へ戻したため、モデルが高いrankを保持した実験ではない。2-core更新によるrank拡張を実際にモデルへ残すには、`R_KEEP` またはrank capを12より大きくし、parameter数・損失・validation指標を改めて評価する必要がある。

正準化されたpair外environmentを含めた重み全体の差は、保存済み出力で

$$
\lVert W(T_{\mathrm{new}})-W(T_r)\rVert_F
\approx
2.819219\times10^{-4}
$$

だった。局所tail $\delta_r\approx2.813202\times10^{-4}$ と近い。float32の正準化誤差と縮約誤差があるため保存値は完全には一致しない。これは $T\to W$ の等長性の数値確認であり、logits誤差との一致ではない。

この照合にNotebookが使う許容差は `rtol=1e-3, atol=1e-4` であり、比較値約 $2.8\times10^{-4}$ に対して絶対許容差が大きい。実際の2値の差は約

$$
6.0\times10^{-7}
$$

と小さいが、検証contractとしてはfloat64で比較するか、scaleに合わせて許容差を絞る方が強い。保存済みの「検証成功」を、厳密な等長性検証が完了したという意味に広げない。

---

## 9. 打ち切り前後の損失を分けて記録する

保存済み出力では、rank 12打ち切り後の損失は

$$
L_{\mathrm{after\ trunc}}
\approx
0.1406413317
$$

だった。

打ち切りが増やした分は

$$
\begin{aligned}
&L_{\mathrm{after\ trunc}}
-
L_{\mathrm{after\ gradient}}
\\
&\qquad\approx
0.1406413317
-
0.1405630112
\\
&\qquad\approx
7.83205\times10^{-5}
\end{aligned}
$$

である。一方、更新前との比較は

$$
\begin{aligned}
&L_{\mathrm{after\ trunc}}
-
L_{\mathrm{before}}
\\
&\qquad\approx
0.1406413317
-
0.1406570226
\\
&\qquad\approx
-1.56909\times10^{-5}
\end{aligned}
$$

なので、打ち切りが勾配更新の利得の多くを失わせたが、この例では更新前よりは低い損失が残った。

この3点を1つの「更新後損失」にまとめると、勾配stepとrank制限の効果を分離できない。

---

## 10. 左右方向のSVD contract

共通APIは、右方向と左方向の両方を扱う。

### 10.1 `direction="right"`

$$
G_k'=\operatorname{reshape}(U_r),
$$

$$
G_{k+1}'
=
\operatorname{reshape}(\Sigma_rV_r^{\mathsf T})
$$

とし、左coreを左直交にして中心を右へ運ぶ。

### 10.2 `direction="left"`

$$
G_k'
=
\operatorname{reshape}(U_r\Sigma_r),
$$

$$
G_{k+1}'
=
\operatorname{reshape}(V_r^{\mathsf T})
$$

とし、右coreを右直交にして中心を左へ運ぶ。

既定値を右方向のまま維持すれば、既存呼び出しとの後方互換性を保てる。NN重みでは分割後にcore normを1へ正規化しない。

---

## 11. coreの書き戻し

rankを変えない場合、既存parameterへ同じshapeの値をcopyできる。

```python
with torch.no_grad():
    module.cores[k].copy_(new_left)
    module.cores[k + 1].copy_(new_right)
```

rankが変わる場合、parameterのshapeも変わるので、既存parameterへの `copy_` はできない。新しい `nn.Parameter` へ置き換える必要がある。

```python
# shapeが変わる場合の概念例。
module.cores[k] = torch.nn.Parameter(new_left)
module.cores[k + 1] = torch.nn.Parameter(new_right)
```

この置換は、既に作成済みoptimizerのparameter参照とstateを自動では移行しない。Adamのmomentum、weight decay、step countなどを使う学習へ拡張するときは、optimizer再構築またはstate移行を別途設計する必要がある。

Notebook 04は外部leafの手動SGD更新であり、optimizer state移行を扱わない。

---

## 12. 外部core列の確認とモデルへの書き戻し

Notebook 04では、rank 12分割結果をまずモデル外の `cores_truncated` として作り、そのcore列を使う一時forwardを確認した。その後に `assign_tt_linear_cores` で実モデルへ書き戻し、通常forwardとの一致を確認した。書き戻し前の `state_dict` も保存しているが、このNotebookは独立candidate modelへ全parameterを適用してからatomic commitする構成ではない。

保存済み出力では、書き戻し後モデルの通常forwardと、rank 12分割core列を直接与えたforwardの差は

$$
\lVert
\operatorname{logits}_{\mathrm{candidate}}
-
\operatorname{logits}_{\mathrm{cores}}
\rVert_F
=0
$$

であり、損失差も0だった。

今後のsweepでは、TT coreだけでなく、同じlocal blockで更新するbiasや後段dense layerも含めて独立candidate stateを作り、全検証後にまとめてcommitする。これはNotebook 04の実施済み事項ではなく、Fashion-MNIST Notebook 00の追加contractである。

---

## 13. 保存済み数値結果の要約

| 項目 | 保存済み値 |
| --- | ---: |
| device / dtype | CUDA / float32 |
| physical shape | out `(32, 8, 2)`, in `(2, 8, 49)` |
| TT ranks | `(1, 12, 12, 1)` |
| 固定batch size | 64 |
| checkpoint通常forward損失 | `0.1406568587` |
| 局所forward更新前損失 | `0.1406570226` |
| 勾配Frobenius norm | `0.3077406` |
| 学習率 | `0.001` |
| 更新step norm | `0.000307767` |
| 勾配更新後損失 | `0.1405630112` |
| 切り捨てなし保持成分数 | 64 |
| float64 SVD診断残差 | `2.72e-12` |
| rank 12の $\delta_r^2$ | `7.910722e-08` |
| rank 12の $\delta_r$ | `2.8132e-04` |
| rank 12相対残差 | `2.3498e-06` |
| rank 12打ち切り後損失 | `0.1406413317` |
| candidate書き戻し後logits差 | `0` |
| 保存済み検証表示 | 22 success、failureなし、skipなし |

ここで「22 success」は保存済みセル出力の集計であり、現環境でのFresh Run All結果とは区別する。

また、`rank制限の記録` のように診断量を正常に計算・表示できた項目も「検証成功」に含まれる。$\delta_r\le s/8$ の理論十分条件を満たした、または分類損失の改善が保証された、という22件ではない。

---

## 14. このNotebookで確認していないこと

Notebook 04が確認したのは局所更新の最小単位である。次は未確認である。

- pair 0以外の全pair。
- pair 0の左側は空なので、非自明な左environmentを持つ内側pairの正準性。
- 左方向のfull sweep。
- 1往復後のvalidation指標。
- 複数sweepとearly stopping。
- full-dataset gradient accumulation。
- rankを増加させる更新。
- 学習率やrank capの比較。
- 通常のTT-MLP fine-tuningとの公平な速度・精度比較。
- CPUとCUDAをまたぐ再現性。
- 複数seed。
- optimizer stateを伴うparameter shape変更。
- test set評価。

保存済み数値は「局所経路が通った」という証拠であり、DMRG-like学習が通常fine-tuningより優れる、またはFashion-MNISTで収束するという証拠ではない。

---

## 15. 共通APIへ求める境界

局所2-core処理をNotebookの一時コードだけに置かず、再利用可能な関数へ分けるときは、少なくとも次の境界を持たせる。

```text
TT-matrix 2-core merge
merged tensor <-> matrix reshape
direction付きSVD split
任意pairをmerged tensorで置換したTTLinear forward
core列を直接与えるmodule forward
rank変更を含むcore pair置換
mixed-canonical化と直交性誤差
```

各関数は次を検証する。

- coreが4階である。
- 隣接bond dimensionが一致する。
- mode shapeがmodule contractと一致する。
- dtypeとdeviceが一致する。
- `rank_cap>=1` である。
- directionが明示された集合の要素である。
- 出力coreを再結合したshapeが元のmerged shapeと一致する。
- non-finiteを黙って通さない。

現行実装でこの境界を担当する公開APIは次である。

```text
tt_matrix_cores_to_tt_cores
canonicalize_tt_matrix_cores
tt_matrix_canonicality_errors
merge_tt_matrix_cores
merged_tt_matrix_to_matrix
tt_linear_forward_with_merged_pair
module_forward_with_merged_tt_linear_pair
module_forward_with_tt_linear_cores
split_merged_tt_matrix
replace_tt_matrix_core_pair
assign_tt_linear_cores
```

`split_merged_tt_matrix` は `direction="right"` と `direction="left"` を扱い、既定の右方向を維持する。Notebook固有のseed、固定batch、plot、表示、history、停止判定は共通APIへ押し込まず、実験側の責務に残す。

## 関連

- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]]
- [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/09_full-batch_2-core_sweep学習設計]]
