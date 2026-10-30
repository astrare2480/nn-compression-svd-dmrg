---
title: environmentを用いた2-site DMRG 1往復
aliases:
  - 2-site DMRG sweep
  - environmentの逐次更新
  - 2-site局所更新の左右pass
tags:
  - DMRG
  - MPS
  - MPO
  - environment
  - 2-site
  - sweep
  - SVD
---

# environmentを用いた2-site DMRG 1往復

## このノートの位置づけ

DMRG学習Notebookの役割は、次のように進んでいる。

```text
Notebook 00
明示的なPとfull Hamiltonianを使った、1回の2-site局所更新
    ↓
Notebook 01
MPO・左右environmentを使った、固定位置での1-site局所更新
    ↓
Notebook 02
environmentベースの2-site局所更新を左右へ接続した1往復
```

対応するNotebookは次である。

- [[notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb|02_environment_two_site_dmrg_sweep_exercises.ipynb]]

本ノートでは、Notebook 02で追加された次の内容を整理する。

- 隣接2サイトをまとめた中心Tensor $\Theta$。
- 左右environmentと2個のMPO coreから作る2-site有効Hamiltonian。
- mixed-canonical formと通常の局所固有値問題。
- 最小固有ベクトルが局所エネルギー最小値を与える理由。
- SVDによる再分割、bond dimensionの制限、discarded weight。
- 右向き・左向きで特異値を置く側を変える理由。
- environmentを逐次更新して行う左右1往復。
- 局所診断と全体診断の区別。

確認した範囲は**右向き1 passと左向き1 passからなる1往復**である。複数往復による収束判定や、matrix-freeな2-site反復固有値solverはまだ扱わない。

---

## 1. 数値例と実装条件

Notebook 01と同じ、サイト数 $L=6$ の開境界横磁場Ising鎖を用いる。

$$
H
=
-J\sum_{i=1}^{L-1}Z_iZ_{i+1}
-g\sum_{i=1}^{L}X_i,
\qquad
J=1,
\qquad
g=1.2.
$$

主な固定条件は次である。

- 実数・`torch.float64`。
- 初期MPSの内部bond dimensionは2。
- SVD後に保持する上限も `MAX_BOND = 2`。
- dense Hamiltonianは小さい系での参照計算だけに使う。
- 実際の局所有効問題はMPOとenvironmentから構成する。
- 局所固有値問題は `torch.linalg.eigh` で厳密に解く。

---

## 2. なぜ隣接2サイトをまとめるのか

左coreを

$$
A^{[n]}_{\alpha s\beta}
$$

右coreを

$$
A^{[n+1]}_{\beta t\gamma}
$$

とする。共有bond $\beta$ を縮約して、

$$
\boxed{
\Theta_{\alpha st\gamma}
=
\sum_\beta
A^{[n]}_{\alpha s\beta}
A^{[n+1]}_{\beta t\gamma}
}
$$

を作る。

### 2.1 まとめるだけでは状態は変わらない

現在の2 coreを縮約して $\Theta$ と書き直しただけなら、物理状態も自由度も変わらない。これは、

```text
A[n] ─ A[n+1]
       ↓ 共有bondを縮約
Theta
```

という同一状態の再表現である。

### 2.2 2-site更新では$\Theta$全体を新しい変数にする

2-site局所更新では、作った $\Theta$ を現在の積 $A^{[n]}A^{[n+1]}$ に固定せず、4階Tensor全体を局所最適化変数として選び直す。

最適化後の $\Theta_{\mathrm{opt}}$ をSVDで再分割するため、

- 2サイト間のbond basisを選び直せる。
- 保持rankを選び直せる。
- bond dimensionを上限まで増減できる。

という点が1-site更新と異なる。

ただし、sweepに2-site更新が必須という意味ではない。1-site sweepも存在する。Notebook 02で2-site更新を使うのは、中心bondの更新とSVD切り捨てを同時に学ぶためである。

### 2.3 PyTorchでの結合

添字順を $(\alpha,s,t,\gamma)$ に保つ実装は、

```python
# left_core:  (alpha, s, beta)
# right_core: (beta, t, gamma)
two_site_tensor = torch.einsum(
    "abc,cde->abde",
    left_core,
    right_core,
)
```

である。縮約するのは、左coreの右bondと右coreの左bondだけである。

---

## 3. 2-site有効Hamiltonian

中心ペアより左を $L$、右を $R$ に縮約する。添字を

$$
L_{a\mu b},
\qquad
(W_n)_{\mu su\nu},
\qquad
(W_{n+1})_{\nu tv\rho},
\qquad
R_{c\rho d}
$$

とすると、2-site有効Hamiltonianは

$$
\boxed{
(H_{\mathrm{eff}})_{astc,buvd}
=
\sum_{\mu,\nu,\rho}
L_{a\mu b}
(W_n)_{\mu su\nu}
(W_{n+1})_{\nu tv\rho}
R_{c\rho d}
}
$$

である。

残る添字は、

- $(a,s,t,c)$：中心 $\Theta$ のbra側で、行になる。
- $(b,u,v,d)$：中心 $\Theta$ のket側で、列になる。

縮約される $\mu,\nu,\rho$ は、MPOを左environmentから右environmentまでつなぐvirtual bondである。

実装では、

```python
effective_tensor = torch.einsum(
    "apb,psuq,qtvr,crd->astcbuvd",
    left_environment,
    left_mpo_core,
    right_mpo_core,
    right_environment,
)
```

とし、先頭4軸を行、後半4軸を列へまとめる。

```python
row_dimension = math.prod(effective_tensor.shape[:4])
column_dimension = math.prod(effective_tensor.shape[4:])
effective_hamiltonian = effective_tensor.reshape(
    row_dimension,
    column_dimension,
)
```

$\Theta$ 自体は $H_{\mathrm{eff}}$ の構築に入れない。$H_{\mathrm{eff}}$ は、固定した周囲とHamiltonianが中心候補へどう作用するかを表す行列であり、$\Theta$ は後から入力する局所変数である。

---

## 4. mixed-canonical formと中心ノルム

中心ペア以外を固定すると、全状態は

$$
|\psi\rangle
=
\sum_{\alpha,s,t,\gamma}
\Theta_{\alpha st\gamma}
|\ell_\alpha\rangle
|s\rangle
|t\rangle
|r_\gamma\rangle
$$

と書ける。

中心より左が左正準、右が右正準なら、

$$
\langle\ell_{\alpha'}|\ell_\alpha\rangle
=
\delta_{\alpha'\alpha},
\qquad
\langle r_{\gamma'}|r_\gamma\rangle
=
\delta_{\gamma'\gamma}.
$$

物理基底も正規直交なので、

$$
\begin{aligned}
\langle\psi|\psi\rangle
&=
\sum_{\substack{\alpha',s',t',\gamma'\\
                  \alpha,s,t,\gamma}}
\Theta^*_{\alpha's't'\gamma'}
\Theta_{\alpha st\gamma}
\delta_{\alpha'\alpha}
\delta_{s's}
\delta_{t't}
\delta_{\gamma'\gamma}
\\
&=
\sum_{\alpha,s,t,\gamma}
|\Theta_{\alpha st\gamma}|^2.
\end{aligned}
$$

$I=(\alpha,s,t,\gamma)$ として $\Theta$ をベクトル $\theta$ へ並べれば、

$$
\boxed{
\langle\psi|\psi\rangle
=
\theta^\dagger\theta
}
$$

となる。

ここで単位行列として働くのは、**MPOを含まない状態内積の左右Gram行列**である。Hamiltonian MPOを含むHamiltonian environment $L,R$ が単位行列になるわけではない。

同じ局所基底で、

$$
(H_{\mathrm{eff}})_{IJ}
=
\langle B_I|H|B_J\rangle
$$

と定義すれば、局所Rayleigh商は

$$
E(\theta)
=
\frac{\theta^\dagger H_{\mathrm{eff}}\theta}
{\theta^\dagger\theta}
$$

となる。

---

## 5. なぜ最小固有ベクトルが局所最適解になるのか

### 5.1 未定乗数法は停留条件を与える

正規化条件 $\theta^\dagger\theta=1$ を課し、

$$
\mathcal F
=
\theta^\dagger H_{\mathrm{eff}}\theta
-
\lambda(\theta^\dagger\theta-1)
$$

を考える。$\theta_K^*$ で微分すると、

$$
\sum_J
(H_{\mathrm{eff}})_{KJ}\theta_J
-
\lambda\theta_K
=
0
$$

である。したがって、停留する局所状態は

$$
H_{\mathrm{eff}}\theta
=
\lambda\theta
$$

を満たす。

ただし、この導出だけでは最小・最大・その他の停留点を区別していない。正規化されたすべての固有ベクトルが候補になる。

### 5.2 固有値の重み付き平均が最小性を示す

$H_{\mathrm{eff}}$ の正規直交固有ベクトル $v_k$ と固有値 $\varepsilon_k$ を使い、

$$
\theta
=
\sum_k c_kv_k,
\qquad
\sum_k|c_k|^2=1
$$

と展開する。このとき、

$$
E(\theta)
=
\sum_k|c_k|^2\varepsilon_k
$$

である。最小固有値を $\varepsilon_1$ とすれば、

$$
E(\theta)-\varepsilon_1
=
\sum_k|c_k|^2
(\varepsilon_k-\varepsilon_1)
\geq0.
$$

また、$\theta=v_1$ を選べば $E(\theta)=\varepsilon_1$ となる。したがって、

$$
\boxed{
\min_{\theta^\dagger\theta=1}
\theta^\dagger H_{\mathrm{eff}}\theta
=
\varepsilon_1
}
$$

であり、対応する最小固有ベクトルが局所最適な中心状態である。

### 5.3 Lanczos法の役割

Lanczos法は、最小固有ベクトルが局所最適になる理由ではない。すでに定義された大きなHermitian局所固有値問題について、密行列を保持せず最小固有対を数値的に求める手段である。

Notebook 02は小さい問題なので `torch.linalg.eigh` を使う。matrix-freeな2-site作用とLanczos法は次の段階に残す。

---

## 6. SVDによる再分割とbond dimension制限

最適化した $\Theta_{\alpha st\gamma}$ を、

$$
M_{(\alpha,s),(t,\gamma)}
$$

という行列へreshapeし、

$$
M
=
U\Sigma V^\dagger
$$

と分解する。

### 6.1 保持rank

利用可能な特異値数と、指定した上限 `max_bond` から、

利用可能な特異値数を $N_\sigma$ とすると、

$$
r
=
\min(N_\sigma,\texttt{max\_bond})
$$

を選ぶ。`max_bond=None` ならすべて保持する。

PyTorchでは、`singular_values.numel()` は特異値Tensorに含まれる要素数、すなわち利用可能な特異値の本数を返す。

### 6.2 discarded weight

捨てた特異値の二乗和を、

$$
\delta^2
=
\sum_{\beta>r}\sigma_\beta^2
$$

とする。全二乗ノルムに対する割合は、

$$
\boxed{
w_{\mathrm{discard}}
=
\frac{\sum_{\beta>r}\sigma_\beta^2}
{\sum_\beta\sigma_\beta^2}
}
$$

である。

これは、SVDで捨てた**状態成分の割合**である。エネルギーの相対誤差ではない。

### 6.3 切り捨てと再正規化

切り捨て前の $\Theta$ が正規化されていれば、

$$
\sum_\beta\sigma_\beta^2=1.
$$

rank $r$ で切ると、保持部分のノルムは

$$
\|\Sigma_r\|_F
=
\sqrt{\sum_{\beta\le r}\sigma_\beta^2}
=
\sqrt{1-w_{\mathrm{discard}}}
$$

となる。保持特異値を

$$
\widehat\sigma_\beta
=
\frac{\sigma_\beta}
{\sqrt{\sum_{\gamma\le r}\sigma_\gamma^2}}
$$

で再正規化する。

再正規化は、捨てた情報を回復する処理ではない。切り捨て後の近似状態をノルム1へ戻すだけである。そのため、切り捨て前の局所最小固有値と、切り捨て・再正規化後に保存されたMPSのエネルギーは区別する。

### 6.4 関数の再正規化契約

Notebookの `split_two_site_tensor` は、切り捨ての有無にかかわらず保持成分をノルム1へ戻す。

- 正規化済み入力・切り捨てなし：再結合すると元の $\Theta$ に戻る。
- 非正規化入力・切り捨てなし：再結合すると $\Theta/\|\Theta\|_F$ になる。
- 再正規化前の分解：切り捨てなしなら元の入力をそのまま再構成する。
- 切り捨てあり：再正規化前の二乗誤差が捨てた特異値の二乗和と一致する。

sweep本体で入力する局所固有ベクトルは `torch.linalg.eigh` により正規化されている。

---

## 7. 移動方向と特異値の配置

保持部分を

$$
M_r
=
U_r\widehat\Sigma_rV_r^\dagger
$$

とする。

### 7.1 右へ進む場合

$$
A_n=U_r,
\qquad
A_{n+1}=\widehat\Sigma_rV_r^\dagger.
$$

$U_r^\dagger U_r=I$ なので、中心から外れる左core $A_n$ は左正準になる。特異値を右coreへ持たせることで、中心を右へ移す。

### 7.2 左へ進む場合

$$
A_n=U_r\widehat\Sigma_r,
\qquad
A_{n+1}=V_r^\dagger.
$$

$V_r^\dagger$ の保持行は正規直交するので、中心から外れる右core $A_{n+1}$ は右正準になる。特異値を左coreへ持たせることで、中心を左へ移す。

### 7.3 PyTorchのbroadcast

保持rankを $r$、行列の左右次元を $m,n$ とすると、

```text
U[:, :r]  : (m, r)
sigma     : (r,)
Vh[:r, :] : (r, n)
```

である。

右向きでは、

```python
left_factor = left_singular_vectors[:, :retained_rank]
right_factor = (
    normalized_singular_values.unsqueeze(1)
    * right_singular_vectors_h[:retained_rank, :]
)
```

とする。`unsqueeze(1)` により特異値を $(r,1)$ にして、$V^\dagger$ の各行へ1個ずつ掛ける。

左向きでは、

```python
left_factor = (
    left_singular_vectors[:, :retained_rank]
    * normalized_singular_values.unsqueeze(0)
)
right_factor = right_singular_vectors_h[:retained_rank, :]
```

とする。`unsqueeze(0)` により特異値を $(1,r)$ にして、$U$ の各列へ1個ずつ掛ける。

最後に、

```python
left_core = left_factor.reshape(
    left_bond,
    left_physical,
    retained_rank,
)
right_core = right_factor.reshape(
    retained_rank,
    right_physical,
    right_bond,
)
```

として3階MPS coreへ戻す。

---

## 8. 1回の2-site局所更新

1個のペアに対する更新順は次である。

```text
現在の2 coreをTheta_beforeへ結合
                ↓
現在のL・W[n]・W[n+1]・RからH_effを構成
                ↓
更新前の局所Rayleigh商を記録
                ↓
H_effの最小固有対を計算
                ↓
最小固有ベクトルをTheta_optへreshape
                ↓
SVD分割・rank制限・再正規化
                ↓
更新した2 coreだけをMPSへ戻す
```

`update_two_site_pair` が変更するのはMPSの中心2 coreである。Hamiltonianを表す `mpo_cores` は固定なので、cloneして更新する必要はない。

局所最小固有値は、切り捨て前の $\Theta_{\mathrm{opt}}$ のエネルギーである。切り捨て後のMPSで再計算した `energy_after` とは別の値として記録する。

---

## 9. environmentを逐次更新する右向きpass

右向きpassの開始時には、現在のMPSから全right environmentを作る。

ペア $(n,n+1)$ に必要なのは、

$$
R^{[n+2]},
$$

すなわちサイト $n+2$ 以降だけを含む

```python
right_environment = right_environments[pair_start + 2]
```

である。

局所更新後、次のペアは $(n+1,n+2)$ なので、更新済みの左core $A^{[n]}$ は中心から外れる。これをleft environmentへ取り込む。

```python
left_environment = update_left_environment(
    left_environment,
    cores[pair_start],
    mpo_cores[pair_start],
)
```

したがって、各ステップは

```text
現在のleft environmentを用意
        ↓
現在のペアより右のright environmentを選ぶ
        ↓
中心ペアをdirection="right"で更新
        ↓
更新済み左coreをleft environmentへ取り込む
        ↓
次のペアへ
```

となる。

pass開始時に作ったright environmentを再利用できるのは、右向きに進む間、現在の中心より右側のcoreがまだ更新されていないからである。

---

## 10. 折り返しと左向きpass

右向きpassが終わったら、**右向きpass後のMPS**から全left environmentを作り直す。更新前の古いMPSに対応するenvironmentは再利用しない。

左向きpassでは、ペアの左端indexを $n$ とすると、

```python
left_environment = left_environments[pair_start]
```

を選び、

```python
update_result = update_two_site_pair(
    cores,
    mpo_cores,
    pair_start,
    left_environment,
    right_environment,
    direction="left",
    max_bond=max_bond,
)
```

で更新する。

更新後、次のペアは $(n-1,n)$ なので、中心から外れる右core $A^{[n+1]}$ をright environmentへ取り込む。

```python
right_environment = update_right_environment(
    right_environment,
    cores[pair_start + 1],
    mpo_cores[pair_start + 1],
)
```

左右の対応は次のとおりである。

| 処理 | 左から右 | 右から左 |
|---|---|---|
| ペア順序 | $(0,1)\to(1,2)\to\cdots$ | $\cdots\to(1,2)\to(0,1)$ |
| 最初にまとめて作る環境 | right environments | left environments |
| 選ぶ環境 | `right_environments[n + 2]` | `left_environments[n]` |
| SVD方向 | `"right"` | `"left"` |
| 逐次更新する環境 | left environment | right environment |
| 取り込む更新済みcore | 左core `cores[n]` | 右core `cores[n + 1]` |

進行方向が変わっても、`pair_start` は常に「ペアの左サイトindex」を意味する。

---

## 11. なぜsweepし、なぜ何往復もするのか

### 11.1 1か所だけの局所更新では全体を最適化できない

1回の局所更新で変わるのは中心ペアだけである。他の場所のMPS coreは固定されているため、全状態に残る改善可能性を取り除けない。

中心を動かして各ペアを順番に最適化することで、更新結果を次のenvironmentと有効Hamiltonianへ反映できる。

### 11.2 折り返す理由

右向きにあるペアを最適化した後、さらに右側を更新すると、先ほどのペアから見た右environmentが変わる。右から左へ戻ることで、更新済みの右側を反映した局所問題として同じ領域を再び最適化できる。

### 11.3 1往復でも同時最適にはならない

元のHamiltonian $H$ は変わらないが、周囲のMPSが更新されるため、同じ位置でも後から作る有効Hamiltonianは一般に

$$
H_{\mathrm{eff}}^{(\mathrm{previous})}
\neq
H_{\mathrm{eff}}^{(\mathrm{next})}
$$

となる。前回の最小固有ベクトルは、新しい周囲に対して再び最適とは限らない。

そのため通常は、例えば

$$
|E_{k+1}-E_k|<\varepsilon_E
$$

や、状態・局所量・残差の変化が十分小さくなるまで往復を繰り返す。

Notebook 02で確認するのは、左右1往復の接続が正しいことまでであり、収束を確認したわけではない。

---

## 12. 検証設計

### 12.1 2-site有効Hamiltonian

- 強制対称化せず、$H_{\mathrm{eff}}=H_{\mathrm{eff}}^\dagger$ を確認する。
- 任意の $\Theta$ について、全状態側と局所空間側の分子を比較する。
- mixed-canonical formで、全状態ノルムと $\theta^\dagger\theta$ を比較する。
- 両者から作ったRayleigh商を比較する。

任意の $\Theta$ を使うのは、現在の最適化済み中心だけで偶然一致する実装を排除するためである。

### 12.2 局所固有値solver

- $H_{\mathrm{eff}}\theta=E\theta$ の局所残差を確認する。
- 切り捨て前の局所最小値が更新前の局所エネルギー以下であることを確認する。
- 切り捨てなしなら、保存後の全体エネルギーと局所最小固有値が一致することを確認する。

### 12.3 SVD

- 切り捨てなしの再構成を確認する。
- 捨てた特異値の二乗和と、再正規化前の二乗再構成誤差を比較する。
- 再正規化後のノルムを確認する。
- 右向きでは左core、左向きでは右coreの正準条件を確認する。

### 12.4 environment更新

各逐次更新直後に、現在のMPSからenvironmentを最初から再構築し、逐次更新した値と比較する。これは効率化のためのincremental updateが、古いMPSのenvironmentを誤って使っていないことを検証する参照計算である。

### 12.5 局所診断と全体診断

各ステップでは、

- `local_minimum_energy`：切り捨て前の局所最小固有値。
- `discarded_weight`：SVDで捨てた状態成分の割合。
- `energy_after`：切り捨て・再正規化後に保存されたMPSの全体エネルギー。
- `local_residual`：その時点の局所固有値問題を解いた精度。

を分けて記録する。

1往復後には、

$$
E
=
\frac{\psi^\dagger H\psi}{\psi^\dagger\psi},
$$

$$
r_{\mathrm{global}}
=
\frac{\|H\psi-E\psi\|}{\|\psi\|}
$$

と、厳密基底エネルギーとの差を確認する。

局所残差が小さいことは、その時点の $H_{\mathrm{eff}}$ を正しく解いたことを示す。全体残差が小さいことや、全状態が十分収束したことを直接は示さない。

---

## 13. 実装上の補足

### 13.1 `mps_cores`と`mpo_cores`

- `mps_cores`：最適化対象である状態 $|\psi\rangle$ の各サイトcore。
- `mpo_cores`：固定されたHamiltonian $H$ の各サイトcore。

sweep関数ではMPSをcloneして更新するが、MPOは読み取り専用なのでcloneしない。

### 13.2 keyword-only引数

関数定義の `*` より後ろにある

```python
*,
max_bond: int | None,
atol: float,
rtol: float,
```

は、呼び出し時に名前を明示するkeyword-only引数である。bond上限と数値検証の許容誤差を、位置だけで取り違えないための設計である。

### 13.3 `atol`と`rtol`

- `atol`：絶対誤差。局所残差、正準性誤差、ゼロ付近の比較に効く。
- `rtol`：相対誤差。スケールを持つTensor同士の `assert_close` に効く。

これらは最適化の精度を変更するhyperparameterではなく、検証を合格とみなす数値許容値である。

---

## 14. Notebook 02の保存済み結果

以下は、正式位置へ置き換えたNotebookに保存されている出力である。このノート作成時にFresh Restart KernelからRun Allした結果ではない。

### 14.1 段階別確認

保存済み出力では、次がすべて `OK` になっている。

- dense Hamiltonian・MPO・初期MPSの参照設定。
- 2 coreから $\Theta$ への結合。
- 2-site有効Hamiltonian。
- 局所固有値solver。
- SVD分割・切り捨て・再正規化。
- 1回の2-site局所更新。
- left-to-right pass。
- right-to-left pass。
- 左右1往復後の診断。

### 14.2 1回の内部ペア更新

内部ペアの診断では、

$$
E_{\mathrm{local,before}}
=
1.1577915655716953,
$$

$$
E_{\mathrm{local,min}}
=
-3.93360382392339,
$$

$$
E_{\mathrm{after\ truncation}}
=
-3.9336030983772803
$$

であった。discarded weightは、

$$
w_{\mathrm{discard}}
=
1.3121093472265412\times10^{-7}
$$

である。

局所最小値と切り捨て後エネルギーの差は小さいが、同一ではない。これは保存済み結果でも、切り捨て前の局所最適化と切り捨て後の状態を区別すべきことを示している。

### 14.3 1往復後

初期TT-ranksは、

```text
(1, 2, 2, 2, 2, 2, 1)
```

であり、1往復後も同じrankである。これは `MAX_BOND = 2` を用いたためである。

保存値は、

$$
\|\psi_{\mathrm{final}}\|^2
=
1.0000000000000004,
$$

$$
E_{\mathrm{final}}
=
-8.268340204869851,
$$

$$
E_0
=
-8.269344348226955,
$$

$$
E_{\mathrm{final}}-E_0
=
0.0010041433571039704,
$$

$$
r_{\mathrm{global}}
=
0.07518843974191038
$$

である。

全局所更新のlocal residualは約 $10^{-15}$ であり、各局所有効問題は高精度に解かれている。一方、global residualは $0.0752$ 程度である。したがって、局所solverの精度と全状態の収束は別であることが、保存値にも現れている。

### 14.4 各更新のログ

| 方向 | ペア | 保持rank | 局所最小値 | discarded weight | 切り捨て後エネルギー | 局所残差 |
|---|---:|---:|---:|---:|---:|---:|
| right | (0, 1) | 2 | -4.1207489987 | 0 | -4.1207489987 | $2.465\times10^{-15}$ |
| right | (1, 2) | 2 | -5.1535976542 | $7.334\times10^{-6}$ | -5.1535572381 | $1.990\times10^{-15}$ |
| right | (2, 3) | 2 | -6.3190988704 | $1.535\times10^{-5}$ | -6.3190010768 | $2.736\times10^{-15}$ |
| right | (3, 4) | 2 | -8.2620533516 | $4.150\times10^{-5}$ | -8.2617951722 | $4.514\times10^{-15}$ |
| right | (4, 5) | 2 | -8.2618012227 | 0 | -8.2618012227 | $5.909\times10^{-15}$ |
| left | (4, 5) | 2 | -8.2618012227 | 0 | -8.2618012227 | $5.909\times10^{-15}$ |
| left | (3, 4) | 2 | -8.2620533516 | $4.150\times10^{-5}$ | -8.2617951722 | $9.415\times10^{-15}$ |
| left | (2, 3) | 2 | -8.2655406740 | $6.174\times10^{-5}$ | -8.2651719692 | $5.306\times10^{-15}$ |
| left | (1, 2) | 2 | -8.2686088538 | $4.461\times10^{-5}$ | -8.2683334342 | $4.966\times10^{-15}$ |
| left | (0, 1) | 2 | -8.2683402049 | 0 | -8.2683402049 | $3.297\times10^{-15}$ |

切り捨てがある行では、局所最小値と切り捨て後エネルギーが異なる。さらに、rightの `(4, 5)` からleftの `(3, 4)` では、切り捨て後エネルギーがわずかに上がっている。Notebookは、切り捨てありの各更新に単調減少を要求していないため、これは検証条件と整合する。

---

## 15. この段階で確認したこと・していないこと

### 確認したこと

- 2サイトを自由な局所変数 $\Theta$ として最適化できる。
- environmentから2-site $H_{\mathrm{eff}}$ を正しく構成できる。
- 正準形により通常のHermitian固有値問題として解ける。
- SVDでbond basisと保持rankを選び直せる。
- 進行方向に合わせて正準中心を移動できる。
- environmentを逐次更新し、左右1往復へ接続できる。
- 局所最小値、discarded weight、切り捨て後エネルギー、局所残差を分離して記録できる。

### まだ確認していないこと

- 複数往復でのエネルギー・状態・残差の収束。
- 収束閾値と停止条件。
- `MAX_BOND`を変えたbond dimension依存性。
- matrix-freeな2-site $H_{\mathrm{eff}}$ 作用。
- Lanczos法などの反復固有値solver。
- 大規模系での時間・memory効率。
- NN目的関数に置き換えたDMRG-like学習。

1往復を正しく実行できたことと、基底状態へ十分収束したことは同じではない。Notebook 02の達成点は、局所更新をenvironmentの逐次更新とともに左右へ接続できたことである。
