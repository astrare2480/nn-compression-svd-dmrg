# TT/MPS の SVD 中心移動：実装と shape の補助導出

> 対象：二サイトをまとめた中心テンソルに対する **exact SVD center move**。ここでは切り捨てを行わず、再構成誤差が丸め誤差だけになる場合を扱う。

## 1. 記号と全体像

左から右へ並ぶ MPS/TT を考える。隣接する二つの site テンソルを

\[
A^{[1]} \in \mathbb{R}^{r_0\times n_1\times r_1},\qquad
A^{[2]} \in \mathbb{R}^{r_1\times n_2\times r_2}
\]

と書く。開放端なら通常 \(r_0=1\) だが、以下では shape を明示するため残しておく。

左側ブロックの等長基底を

\[
L \in \mathbb{R}^{d_L\times r_0},
\]

右側ブロックの等長基底を

\[
R \in \mathbb{R}^{r_2\times d_R}
\]

とし、\(L^\mathsf{T}L=I_{r_0}\)、\(RR^\mathsf{T}=I_{r_2}\) を仮定する。このとき全状態は

\[
\Psi_{\alpha,i_1,i_2,\beta}
=
\sum_{a=1}^{r_0}\sum_{b=1}^{r_1}\sum_{c=1}^{r_2}
L_{\alpha a}
A^{[1]}_{a i_1 b}
A^{[2]}_{b i_2 c}
R_{c\beta}
\]

である。ここで \(\alpha\) は左環境、\(\beta\) は右環境、\(i_1,i_2\) は物理添字である。

SVD 中心移動では、二サイトの局所部分を一つの行列にして

\[
M=U\Sigma V^\mathsf{H}
\]

と分解する。exact move では特異値を捨てない。よって実効 rank を

\[
\rho=\operatorname{rank}(M)\le \min(r_0n_1,n_2r_2)
\]

とすると、SVD の shape は

\[
M\in\mathbb{R}^{(r_0n_1)\times(n_2r_2)},\quad
U\in\mathbb{R}^{(r_0n_1)\times\rho},\quad
\Sigma\in\mathbb{R}^{\rho\times\rho},\quad
V^\mathsf{H}\in\mathbb{R}^{\rho\times(n_2r_2)}.
\]

複素数の場合は \(\mathsf{T}\) を共役転置 \(\mathsf{H}\) に置き換える。

## 2. left unfolding と right unfolding

### 2.1 二サイト中心テンソル

二つの site を縮約して

\[
\Theta_{a,i_1,i_2,c}
=
\sum_{b=1}^{r_1}A^{[1]}_{a i_1 b}A^{[2]}_{b i_2 c}
\]

を作ると、

\[
\Theta\in\mathbb{R}^{r_0\times n_1\times n_2\times r_2}
\]

である。添字の順序を \((a,i_1,i_2,c)\) と固定する。

### 2.2 left unfolding

ボンドを \(i_1\) と \(i_2\) の間に置くなら、左側に \((a,i_1)\)、右側に \((i_2,c)\) を置く。

\[
M_{(a,i_1),(i_2,c)}=\Theta_{a,i_1,i_2,c}.
\]

これが left unfolding（左側の複合添字を行、右側の複合添字を列にした展開）であり、shape は

\[
M\in\mathbb{R}^{(r_0n_1)\times(n_2r_2)}.
\]

NumPy の C-order を前提に

```python
M = Theta.reshape(r0 * n1, n2 * r2)
```

と書けばよい。行 index は \((a,i_1)\)、列 index は \((i_2,c)\) である。

### 2.3 right unfolding

逆に \((a,i_1)\) を列側、\((i_2,c)\) を行側に置けば

\[
M_{\rm right,(i_2,c),(a,i_1)}=\Theta_{a,i_1,i_2,c}
\]

となり、shape は

\[
M_{\rm right}\in\mathbb{R}^{(n_2r_2)\times(r_0n_1)}.
\]

これは left unfolding の転置（複素数なら随伴）に対応する。どちらが「正しい」かではなく、**どちら側を次の左正規テンソル／右正規テンソルにしたいか**で選ぶ。

## 3. `reshape(r1 * n2, r2)` が左展開になる理由

左から右へ一サイトずつ左正規化している途中を考える。現在の site テンソルを

\[
C\in\mathbb{R}^{r_1\times n_2\times r_2}
\]

とする。ここで \(r_1\) は左 bond、\(n_2\) は site 2 の物理次元、\(r_2\) は右 bond である。

この site を左正規化するには、左に残す添字を \((r_1,n_2)\)、右へ送る添字を \(r_2\) にする。したがって

\[
C_{a,i_2,c}\longmapsto
C_{(a,i_2),c}
\in\mathbb{R}^{(r_1n_2)\times r_2}
\]

と行列化する。

```python
C_left = C.reshape(r1 * n2, r2)
Q, R = np.linalg.qr(C_left, mode="reduced")
A_left = Q.reshape(r1, n2, rho)
```

このとき \(Q\in\mathbb{R}^{(r_1n_2)\times\rho}\) なので

\[
\sum_{a,i_2}
(A_{\rm left})_{a i_2 \mu}^*
(A_{\rm left})_{a i_2 \nu}
=\delta_{\mu\nu}.
\]

すなわち新しい右 bond \(\mu\) に関して列が直交し、`A_left` は左正規化される。

重要なのは、`reshape(r1 * n2, r2)` は「左から右へ見る」ための展開である点である。行側に \((r_1,n_2)\) をまとめるので、Q の列空間が新しい右 bond 基底になる。これに対して

```python
C_right = C.reshape(r1, n2 * r2)
```

は左 bond を行、\((n_2,r_2)\) を列に置く right-oriented な展開であり、右正規化で用いる向きである。

## 4. exact SVD center move の全 shape

二サイト中心テンソルを直接 SVD して、中心を左 site から右 site へ移す場合を順に書く。

### 4.1 縮約

\[
A^{[1]}\in\mathbb{R}^{r_0\times n_1\times r_1},\qquad
A^{[2]}\in\mathbb{R}^{r_1\times n_2\times r_2}.
\]

共有する \(r_1\) を縮約する。

```python
Theta = np.tensordot(A1, A2, axes=(2, 0))
```

shape は

\[
(r_0,n_1,r_1)\;\times\;(r_1,n_2,r_2)
\quad\longrightarrow\quad
(r_0,n_1,n_2,r_2).
\]

### 4.2 なぜ `tensordot` の中間出力は4階か

縮約されるのは内部 bond 添字 \(b\) 一つだけである。

\[
\Theta_{a,i_1,i_2,c}
=\sum_b A^{[1]}_{a i_1 b}A^{[2]}_{b i_2 c}.
\]

元の添字は \((a,i_1,b)\) と \((b,i_2,c)\) の計6本で、そのうち \(b\) の二本が縮約で消える。残るのは

\[
(a,i_1,i_2,c)
\]

の4本である。従って中間テンソルは必ず4階になる。これは「二つの三階テンソルを一つの bond で縮約すると、\(3+3-2=4\) 階」という添字数の数え上げそのものである。

### 4.3 行列化と SVD

```python
M = Theta.reshape(r0 * n1, n2 * r2)
U, s, Vh = np.linalg.svd(M, full_matrices=False)
rho = s.size
```

exact SVD では数値的に保持する特異値本数を \(\rho\) として、

\[
\begin{aligned}
M &: (r_0n_1,\;n_2r_2),\\
U &: (r_0n_1,\;\rho),\\
s &: (\rho,),\\
V^\mathsf{H} &: (\rho,\;n_2r_2).
\end{aligned}
\]

再構成は

```python
M_reconstructed = (U * s[None, :]) @ Vh
```

であり、\(U\operatorname{diag}(s)V^\mathsf{H}\) と同じである。

### 4.4 中心を右へ移す実装

左 site を左正規テンソルにし、特異値を右 site 側に吸収する。

```python
A1_left = U.reshape(r0, n1, rho)
A2_center = (s[:, None] * Vh).reshape(rho, n2, r2)
```

shape は

\[
A^{[1]}_{\rm left}\in\mathbb{R}^{r_0\times n_1\times\rho},\qquad
A^{[2]}_{\rm center}\in\mathbb{R}^{\rho\times n_2\times r_2}.
\]

添字で書けば

\[
\Theta_{a,i_1,i_2,c}
=
\sum_{\mu=1}^{\rho}
(A^{[1]}_{\rm left})_{a i_1 \mu}
(A^{[2]}_{\rm center})_{\mu i_2 c}.
\]

ここで \(A^{[1]}_{\rm left}\) は左正規化される。

\[
\sum_{a,i_1}
(A^{[1]}_{\rm left})_{a i_1\mu}^*
(A^{[1]}_{\rm left})_{a i_1\nu}
=\delta_{\mu\nu}.
\]

### 4.5 中心を左へ移す実装

逆に特異値を左 site 側に吸収する。

```python
A1_center = (U * s[None, :]).reshape(r0, n1, rho)
A2_right = Vh.reshape(rho, n2, r2)
```

\[
A^{[1]}_{\rm center}\in\mathbb{R}^{r_0\times n_1\times\rho},\qquad
A^{[2]}_{\rm right}\in\mathbb{R}^{\rho\times n_2\times r_2}.
\]

\(A^{[2]}_{\rm right}\) は右正規化される。

\[
\sum_{i_2,c}
(A^{[2]}_{\rm right})_{\mu i_2c}
(A^{[2]}_{\rm right})_{\nu i_2c}^*
=\delta_{\mu\nu}.
\]

## 5. `Vh @ R_old` による右 bond 基底回転

右にすでに環境ブロックがある場合を考える。旧右ブロックを

\[
R_{\rm old}\in\mathbb{R}^{r_2\times d_R}
\]

とする。二サイト中心の SVD で得た

\[
V^\mathsf{H}\in\mathbb{R}^{\rho\times(n_2r_2)}
\]

をまず三階テンソルに戻す。

\[
V^\mathsf{H}_{\mu,(i_2,c)}
\longrightarrow
V^\mathsf{H}_{\mu,i_2,c}
\in\mathbb{R}^{\rho\times n_2\times r_2}.
\]

この右因子と旧右ブロックを \(c\) で縮約すると

\[
\widetilde R_{\mu,i_2,\beta}
=
\sum_{c=1}^{r_2}V^\mathsf{H}_{\mu,i_2,c}(R_{\rm old})_{c\beta}.
\]

つまり `Vh` が、旧 right-bond 基底 \(c\) を新しい Schmidt/bond 基底 \(\mu\) に回転させる。

`Vh @ R_old` という表現は、物理添字を含めた右側を行列化した見方である。実装では次のように書ける。

```python
Vh_tensor = Vh.reshape(rho, n2, r2)
R_new_tensor = np.tensordot(Vh_tensor, R_old, axes=(2, 0))
# shape: (rho, n2, dR)
R_new = R_new_tensor.reshape(rho, n2 * dR)
```

あるいは、\(V^\mathsf{H}\) をそのまま \((\rho,n_2r_2)\) と見て、site 2 と旧右環境を含む行列へ作用させる見方もできる。要点は、**右側の表現空間に対する基底変換が \(V^\mathsf{H}\) であり、その新しい bond 添字が \(\mu\) になる**ことである。

## 6. `L_tensor`、`L_block`、Schmidt 左状態

### 6.1 左ブロックのテンソル表示

左環境の基底を

\[
L_{\rm old}\in\mathbb{R}^{d_L\times r_0}
\]

とし、左正規テンソル

\[
A^{[1]}_{\rm left}\in\mathbb{R}^{r_0\times n_1\times\rho}
\]

を付け加えると、拡張された左ブロックは

\[
(L_{\rm tensor})_{\alpha,i_1,\mu}
=
\sum_{a=1}^{r_0}(L_{\rm old})_{\alpha a}
(A^{[1]}_{\rm left})_{a i_1\mu}.
\]

この `L_tensor` の shape は

\[
L_{\rm tensor}\in\mathbb{R}^{d_L\times n_1\times\rho}
\]

である。

```python
L_tensor = np.tensordot(L_old, A1_left, axes=(1, 0))
# shape: (dL, n1, rho)
```

### 6.2 `L_block` は物理・環境添字を一つに畳んだ行列

Schmidt 分解では、左側全体 \((\alpha,i_1)\) を一つの複合添字として扱う。したがって

```python
L_block = L_tensor.reshape(dL * n1, rho)
```

とし、

\[
(L_{\rm block})_{(\alpha,i_1),\mu}
=(L_{\rm tensor})_{\alpha,i_1,\mu}
\]

と定める。shape は

\[
L_{\rm block}\in\mathbb{R}^{(d_Ln_1)\times\rho}.
\]

この行列の第 \(\mu\) 列が、cut の左側 Hilbert 空間における Schmidt 左状態 \(|\mu_L\rangle\) の係数である。

\[
|\mu_L\rangle
=
\sum_{\alpha,i_1}
(L_{\rm block})_{(\alpha,i_1),\mu}
|\alpha\rangle\otimes|i_1\rangle.
\]

したがって、`L_tensor` は添字構造を保った表現、`L_block` は線形代数上の行列表現、Schmidt 左状態はその各列を ket と見たものである。

## 7. `reshape(n1 * n2, rho)` が必要な理由

左端から二つの物理 site だけを取り出す簡単な場合を考える。\(d_L=1\) なら

\[
L_{\rm tensor}\in\mathbb{R}^{n_1\times n_2\times\rho}
\]

であり、左側の物理 Hilbert 空間は

\[
\mathcal H_{1}\otimes\mathcal H_{2}
\]

である。Schmidt 左状態を列ベクトルとして比較・直交性検査するには、二つの物理添字 \((i_1,i_2)\) を一つの行 index に畳む必要がある。

```python
Lblock = L_tensor.reshape(n1 * n2, rho)
```

すなわち

\[
L_{\rm block}\in\mathbb{R}^{(n_1n_2)\times\rho}.
\]

この変形はデータを変更しない。\((i_1,i_2)\) というテンソル添字を、Schmidt 左状態の成分ラベル \(I=(i_1,i_2)\) に付け替えているだけである。

一般には左環境次元も含めて

\[
L_{\rm block}=\operatorname{reshape}(L_{\rm tensor},(d_Ln_1,\rho))
\]

となる。`reshape(n1 * n2, rho)` は、左側がちょうど二物理 site であるケースの特殊形である。

## 8. \(L_{\rm block}^\mathsf{T}L_{\rm block}=I_\rho\) の途中式

複素数なら \(\mathsf{T}\) を \(\mathsf{H}\) に読み替える。左環境基底が直交し

\[
L_{\rm old}^\mathsf{T}L_{\rm old}=I_{r_0}
\]

かつ新しい左 site が左正規化されている、すなわち

\[
\sum_{a,i_1}
(A^{[1]}_{\rm left})_{a i_1\mu}
(A^{[1]}_{\rm left})_{a i_1\nu}
=\delta_{\mu\nu}
\]

とする。

定義より

\[
(L_{\rm block})_{(\alpha,i_1),\mu}
=
\sum_a(L_{\rm old})_{\alpha a}
(A^{[1]}_{\rm left})_{a i_1\mu}.
\]

よって

\[
\begin{aligned}
(L_{\rm block}^\mathsf{T}L_{\rm block})_{\mu\nu}
&=
\sum_{\alpha,i_1}
(L_{\rm block})_{(\alpha,i_1),\mu}
(L_{\rm block})_{(\alpha,i_1),\nu}\\
&=
\sum_{\alpha,i_1}
\left(\sum_a(L_{\rm old})_{\alpha a}
(A^{[1]}_{\rm left})_{a i_1\mu}\right)
\left(\sum_{a'}(L_{\rm old})_{\alpha a'}
(A^{[1]}_{\rm left})_{a' i_1\nu}\right)\\
&=
\sum_{a,a',i_1}
(A^{[1]}_{\rm left})_{a i_1\mu}
\left(\sum_\alpha(L_{\rm old})_{\alpha a}(L_{\rm old})_{\alpha a'}\right)
(A^{[1]}_{\rm left})_{a' i_1\nu}\\
&=
\sum_{a,a',i_1}
(A^{[1]}_{\rm left})_{a i_1\mu}
\delta_{aa'}
(A^{[1]}_{\rm left})_{a' i_1\nu}\\
&=
\sum_{a,i_1}
(A^{[1]}_{\rm left})_{a i_1\mu}
(A^{[1]}_{\rm left})_{a i_1\nu}\\
&=\delta_{\mu\nu}.
\end{aligned}
\]

従って

\[
L_{\rm block}^\mathsf{T}L_{\rm block}=I_\rho.
\]

これは SVD の \(U^\mathsf{T}U=I\) と、左環境ブロックの等長性をテンソル縮約で合成した結果である。

実装上の検査は例えば次のように書ける。

```python
orth_err_L = np.linalg.norm(Lblock.conj().T @ Lblock - np.eye(rho))
```

## 9. exact move と \(\Sigma\) を bond 上に残す表現

### 9.1 特異値を右テンソルへ吸収する表現

中心を右に置く標準的な実装では

\[
A^{[1]}_{\rm left}=U,\qquad
A^{[2]}_{\rm center}=\Sigma V^\mathsf{H}
\]

とする。コードでは

```python
A1_left = U.reshape(r0, n1, rho)
A2_center = (s[:, None] * Vh).reshape(rho, n2, r2)
```

である。この表現では、左 site は左正規だが右 site は一般に右正規ではない。中心は右 site にある。

### 9.2 \(\Sigma\) を独立した bond 行列として残す表現

Schmidt 形を明示したいなら

\[
\Theta=U\Sigma V^\mathsf{H}
\]

をそのまま

\[
A^{[1]}_{\rm left}=U,\qquad
\Lambda=\Sigma,\qquad
A^{[2]}_{\rm right}=V^\mathsf{H}
\]

として保つ。

```python
A1_left = U.reshape(r0, n1, rho)
Lambda = np.diag(s)                    # (rho, rho)
A2_right = Vh.reshape(rho, n2, r2)
```

このとき左右の site はそれぞれ左正規・右正規であり、\(\Lambda\) が Schmidt 係数を表す独立の bond オブジェクトになる。

### 9.3 実装上の差

| 表現 | 保持するもの | 利点 | 注意点 |
|---|---|---|---|
| `U` と `ΣVh` | 中心テンソルを右 site に吸収 | 二つの三階テンソルだけで演算できる。局所最適化の中心を site に置きやすい | 右 site 単独は右正規でない。Schmidt 係数を読むには再分解または別管理が必要 |
| `U`, `Σ`, `Vh` | 明示的な Schmidt bond | 正準形とエンタングルメントスペクトルが直接見える。左右の等長性を分離して検査できる | 二サイト演算・縮約時に `Λ` をどちら側へ掛けるかを常に管理する必要がある |

両者は同じ状態を表す。違いは数値表現と、正規化・観測量・更新式をどこに置くかである。

## 10. Notebook 08 の数値検証を記録する欄

このファイル作成時点では、Notebook 08 の実行出力そのもの（具体的な浮動小数点値）はこの会話に提示されていない。そのため、数値を推測して埋めない。Notebook 08 で得た値を、下の欄にそのまま転記する。

```text
reconstruction_error = <Notebook 08 の実測値>
left_orthogonality_error = <Notebook 08 の実測値>
right_orthogonality_error = <Notebook 08 の実測値>
norm_error = <Notebook 08 の実測値>
```

exact SVD center move で確認する代表的な量は以下である。

```python
# 元の二サイト中心テンソル
Theta0 = np.tensordot(A1, A2, axes=(2, 0))

# SVD 後（Σを右へ吸収）
Theta1 = np.tensordot(A1_left, A2_center, axes=(2, 0))

reconstruction_error = np.linalg.norm(Theta0 - Theta1)
left_orthogonality_error = np.linalg.norm(
    U.conj().T @ U - np.eye(rho)
)
right_orthogonality_error = np.linalg.norm(
    Vh @ Vh.conj().T - np.eye(rho)
)
norm_error = abs(np.linalg.norm(Theta0) - np.linalg.norm(Theta1))
```

理想的な exact move では、これらはすべて浮動小数点丸め誤差の範囲にある。特に、特異値を切り捨てていない限り、再構成誤差が大きくなる理由はない。

## 11. 最小実装例

```python
import numpy as np

# A1: (r0, n1, r1), A2: (r1, n2, r2)
r0, n1, r1 = A1.shape
r1_check, n2, r2 = A2.shape
assert r1_check == r1

# 1) 二サイトを縮約： (r0, n1, n2, r2)
Theta = np.tensordot(A1, A2, axes=(2, 0))

# 2) left unfolding： rows=(r0,n1), cols=(n2,r2)
M = Theta.reshape(r0 * n1, n2 * r2)

# 3) exact SVD
U, s, Vh = np.linalg.svd(M, full_matrices=False)
rho = s.size

# 4a) 中心を右へ移す：左 site は左正規
A1_left = U.reshape(r0, n1, rho)
A2_center = (s[:, None] * Vh).reshape(rho, n2, r2)

# 4b) あるいは Schmidt 係数を bond 上に独立保持
Lambda = np.diag(s)
A2_right = Vh.reshape(rho, n2, r2)

# 5) 検証
Theta_reconstructed = np.tensordot(A1_left, A2_center, axes=(2, 0))
reconstruction_error = np.linalg.norm(Theta - Theta_reconstructed)
left_orthogonality_error = np.linalg.norm(
    U.conj().T @ U - np.eye(rho)
)
right_orthogonality_error = np.linalg.norm(
    Vh @ Vh.conj().T - np.eye(rho)
)
norm_error = abs(np.linalg.norm(Theta) - np.linalg.norm(Theta_reconstructed))

print({
    "Theta_shape": Theta.shape,
    "M_shape": M.shape,
    "U_shape": U.shape,
    "s_shape": s.shape,
    "Vh_shape": Vh.shape,
    "A1_left_shape": A1_left.shape,
    "A2_center_shape": A2_center.shape,
    "reconstruction_error": reconstruction_error,
    "left_orthogonality_error": left_orthogonality_error,
    "right_orthogonality_error": right_orthogonality_error,
    "norm_error": norm_error,
})
```

## 12. shape チェックリスト

- 元の二 site：`A1.shape == (r0, n1, r1)`、`A2.shape == (r1, n2, r2)`
- 縮約後：`Theta.shape == (r0, n1, n2, r2)`
- left unfolding：`M.shape == (r0 * n1, n2 * r2)`
- exact SVD：`U.shape == (r0 * n1, rho)`、`s.shape == (rho,)`、`Vh.shape == (rho, n2 * r2)`
- 右へ中心移動：`A1_left.shape == (r0, n1, rho)`、`A2_center.shape == (rho, n2, r2)`
- 明示的 Schmidt 形：`Lambda.shape == (rho, rho)`、`A2_right.shape == (rho, n2, r2)`
- 左ブロック：`L_tensor.shape == (dL, n1, rho)`、`L_block.shape == (dL * n1, rho)`

この対応を保てば、`reshape` は単なる配列操作ではなく、どの添字群を Schmidt cut の左・右 Hilbert 空間として選んだかを明示する操作になる。