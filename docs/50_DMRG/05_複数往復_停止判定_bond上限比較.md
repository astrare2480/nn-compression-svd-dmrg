---
title: DMRGの複数往復・停止判定・bond上限比較
aliases:
  - DMRG convergence
  - energy安定停止
  - bond dimension比較
tags:
  - DMRG
  - MPS
  - convergence
  - residual
  - energy variance
  - bond dimension
---

# DMRGの複数往復・停止判定・bond上限比較

## このノートの位置づけ

物理DMRGの学習Notebookは、次の順序で進んだ。

```text
Notebook 00
明示的なPを使った1回の2-site局所更新
    ↓
Notebook 01
MPO・environmentを使った固定位置での1-site局所更新
    ↓
Notebook 02
2-site局所更新を左右へ接続した1往復
    ↓
Notebook 03
複数往復・停止判定・全系診断・bond上限比較
```

対応するNotebookは次である。

- [[notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb|03_dmrg_convergence_and_bond_dimension_exercises.ipynb]]

Notebook 03では、新しいTensor Network縮約を追加するのではなく、Notebook 02で完成した左右passを反復する外側の仕組みを学ぶ。

主題は次の4点である。

1. 局所問題を正確に解くことと、反復全体が安定することは別である。
2. energy変化は停止基準に使えるが、解の正しさの証明ではない。
3. global residualの二乗はenergy分散に等しい。
4. 反復不足とbond上限による表現制約は、概念的には異なる誤差である。

---

## 1. 1-siteでもsweepはできる

Notebook 01が固定位置で1回の1-site更新だけを扱ったのは、1-site法がsweepできないからではない。Notebookの学習範囲を、MPO・environment・1-site有効Hamiltonianの構築へ限定したためである。

1-site DMRGでも、中心coreを移動しながら局所最適化を繰り返せる。ただし、通常の1-site更新ではbond dimensionをその場で増やしにくい。

Notebook 02・03が2-site更新を使う理由は、

```text
隣接2 coreをThetaへ結合
        ↓
Theta全体を局所最適化
        ↓
SVDで再分割
        ↓
bond basisと保持rankを更新
```

という処理を通じて、中心bondの表現も更新できるからである。

---

## 2. 1往復を反復単位として捉える

bond上限を $\chi$ とし、左右1往復する処理を $\mathcal F_\chi$ と書く。往復 $k$ 後の状態を $|\psi^{(k)}\rangle$ とすれば、

$$
|\psi^{(k+1)}\rangle
=
\mathcal F_\chi
\left(|\psi^{(k)}\rangle\right)
$$

と表せる。

$\mathcal F_\chi$ は「bond上限 $\chi$ のもとでleft-to-right passとright-to-left passを実行する処理」を記号化したものであり、線形写像という意味ではない。

### なぜ複数往復が必要か

ある中心ペアの局所有効Hamiltonianを解いた後、隣のペアを更新すると、その中心から見たenvironmentが変わる。したがって、同じペアを次の往復で訪れたとき、一般には前回と異なる局所問題を解く。

```text
ペアAを最適化
    ↓
別のペアを更新
    ↓
ペアAから見たenvironmentが変化
    ↓
ペアAは新しい周囲に対して再び最適とは限らない
```

全ペアを1回ずつ更新したことと、さらに往復しても状態や診断値がほぼ変わらないことは別である。

---

## 3. energy変化による停止判定

完了した往復番号を $k$、right-to-left pass終了後の全energyを $E_k$ とする。往復間のenergy変化を

$$
\boxed{
\delta E_k
=
|E_k-E_{k-1}|
}
$$

と定義する。

同じ方向のpassまで完了し、正準中心が同じ左端へ戻った状態同士を比較することが重要である。

### 3.1 安定条件

Notebook 03では、

$$
\delta E_k
\leq
\varepsilon_E
$$

をenergy安定条件とする。$\varepsilon_E$ はenergy変化に対して利用者が設定する許容値である。

使用した設定は、

```text
energy tolerance       : 1.0e-10
minimum round trips    : 2
required consecutive   : 2
maximum round trips    : 12
```

である。

### 3.2 連続成立回数

単発の小さい変化で停止しないよう、安定条件が連続して成立した回数 $c_k$ を記録する。

$$
c_k
=
\begin{cases}
c_{k-1}+1,
& \delta E_k\leq\varepsilon_E,\\
0,
& \delta E_k>\varepsilon_E.
\end{cases}
$$

初期状態 $k=0$ には直前の比較対象がないため、$\delta E_0$ は0ではなく未定義 `None` とする。

### 3.3 終了理由

`converged`とする条件は、

$$
k\geq k_{\min}
\quad\text{かつ}\quad
c_k\geq q
$$

である。ここで $k_{\min}$ は最小往復数、$q$ は必要な連続成立回数である。

条件を満たさないまま $k_{\max}$ に到達した場合は、`max_round_trips_reached` とする。最終往復で両条件が同時に成立した場合は、energy安定を表す `converged` を優先する。

非有限値や検証失敗は、正常な終了理由へ変換せず例外として扱う。

---

## 4. `converged`が意味すること・意味しないこと

Notebook 03の `converged` は、

> 指定した初期状態、bond上限、更新法、energy許容値のもとで、往復境界のenergy変化が指定回数連続して閾値以下になった。

という意味に限定する。

これだけでは、次を保証しない。

- MPSの各要素が変化しなくなった。
- bond制約内の大域的な最小energyへ到達した。
- 全Hamiltonianの固有状態へ到達した。
- 厳密基底状態へ到達した。
- 別の初期状態からも同じ解へ到達する。

したがって、「energy安定条件で停止した」と「正しい基底状態へ収束した」は分けて記述する。

### 4.1 energy変化が小さくても状態変化が小さいとは限らない

正規化状態 $|\psi\rangle$ に微小変化 $|\delta\psi\rangle$ を加えたとき、energyの一次変化は概念的に

$$
\delta E
=
2\operatorname{Re}
\langle\delta\psi|r\rangle
+
O(\|\delta\psi\|^2)
$$

と書ける。ここで

$$
|r\rangle
=
(H-E)|\psi\rangle
$$

は全残差である。

$|\delta\psi\rangle$ と $|r\rangle$ の内積が小さければ、状態が変化していてもenergy変化は小さくなり得る。また、energyは停留点付近で状態誤差に対して二次的に変化することがある。このため、$\delta E_k$だけで状態差や全残差を評価できない。

---

## 5. global residualとenergy分散

### 5.1 正規化状態

$|\psi\rangle$ を正規化状態、

$$
E
=
\langle\psi|H|\psi\rangle
$$

とする。全Hamiltonianに対する残差ベクトルを

$$
|r\rangle
=
(H-EI)|\psi\rangle
$$

と定義し、そのノルムを

$$
\rho
=
\|r\|
$$

とする。

Hermitian性を用いると、

$$
\begin{aligned}
\rho^2
&=
\langle\psi|(H-E)^2|\psi\rangle\\
&=
\langle H^2\rangle_\psi
-2E\langle H\rangle_\psi
+E^2\\
&=
\langle H^2\rangle_\psi
-\langle H\rangle_\psi^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\rho^2
=
\operatorname{Var}_\psi(H)
}
$$

である。

### 5.2 非正規化入力

実装では非正規化状態にも対応する。$s=\langle\psi|\psi\rangle$ とすると、

$$
E
=
\frac{\langle\psi|H|\psi\rangle}{s},
$$

$$
\rho
=
\frac{\|(H-E)|\psi\rangle\|}{\sqrt{s}},
$$

$$
\langle H^2\rangle
=
\frac{\langle H\psi|H\psi\rangle}{s}.
$$

ここから、

$$
\operatorname{Var}(H)
=
\langle H^2\rangle-E^2
$$

を計算する。

理論上非負の分散が浮動小数点の丸め誤差だけで僅かに負になった場合は0へ補正する。数値誤差の許容範囲を超えて負なら、実装または入力の異常として例外にする。

### 5.3 何を診断できるか

global residualまたはenergy分散が小さいことは、全Hamiltonianの固有状態に近いことを診断する。しかし、励起固有状態でも残差と分散は0になり得るため、それだけでは基底状態だと断定できない。

見る量の役割は異なる。

| 指標 | 確認すること |
|---|---|
| $\delta E_k$ | 往復境界でenergyが安定したか |
| global residual $\rho_k$ | 全系の固有方程式をどの程度満たすか |
| energy variance | global residualと同じ固有状態診断を二乗量で表す |
| $E_k-E_0$ | toy問題の厳密基底energyとの差 |

Notebook 03ではenergy変化を停止条件に使い、global residualと分散は診断値として記録する。厳密基底energyも停止判定には使わない。

---

## 6. bond制約と反復不足を区別する

bond上限を $\chi$ とし、その上限で表現できる正規化MPSの集合を $\mathcal M_\chi$ とする。その集合内の理想的な最小energyを

$$
E_\chi^\star
=
\min_{|\psi\rangle\in\mathcal M_\chi}
\langle\psi|H|\psi\rangle
$$

と定義する。

厳密基底energyを $E_0$、現在の反復結果を $E_k$ とすれば、変分原理から

$$
\boxed{
E_0
\leq
E_\chi^\star
\leq
E_k
}
$$

である。

$E_\chi^\star$ は、現在の計算結果ではなく、bond上限 $\chi$ の状態集合内で完全に最適化できたと仮定した理想値である。現在の状態 $|\psi^{(k)}\rangle$ も $\mathcal M_\chi$ の要素なので、そのenergy $E_k$ は集合内最小値より小さくなれない。

energy誤差は形式的に、

$$
\boxed{
E_k-E_0
=
\underbrace{E_k-E_\chi^\star}_{\text{制約内の最適値までの隔たり}}
+
\underbrace{E_\chi^\star-E_0}_{\text{表現制約による隔たり}}
}
$$

と分けられる。

- 往復を増やすことは、主として第1項を減らす試みである。
- bond上限を増やすことは、状態集合を広げ、第2項を減らせる可能性を作る。

ただし通常は $E_\chi^\star$ を知らないため、実測した $E_k$ と $E_0$ だけから2項を厳密に分離できない。また、非凸な交互最適化なので、往復を増やせば第1項が必ず0になるとも限らない。

### 6.1 理想値の単調性と実測値の違い

$\chi_1\leq\chi_2$ なら、

$$
\mathcal M_{\chi_1}
\subseteq
\mathcal M_{\chi_2}
$$

なので、理想的な最小値には

$$
E_{\chi_2}^\star
\leq
E_{\chi_1}^\star
$$

が成り立つ。

しかしこれは理想値についての保証であり、有限往復・特定初期値から得た実測energyがbond上限に対して必ず単調になる、という意味ではない。そのためNotebookは、実測結果の単調性をassertしない。

---

## 7. Notebook 03の停止・診断設計

### 7.1 初期状態をround-trip 0として記録する

初期状態を履歴の $k=0$ とする。

- `energy_change=None`：直前状態がなく、差を測定していない。
- `max_discarded_weight=None`：局所SVDをまだ行っていない。
- 0を入れないのは、「測定値が0」と「未定義」を区別するためである。

### 7.2 往復単位で記録する値

各往復終了時に次を記録する。

- `energy`：dense Hamiltonianによる全energy。
- `energy_change`：直前の往復境界からの絶対変化。
- `global_residual`：全Hamiltonianに対する残差ノルム。
- `energy_variance`：全残差の二乗と対応する分散。
- `exact_energy_gap`：toy問題の厳密基底energyとの差。
- `state_norm_squared`：$\langle\psi|\psi\rangle$。
- `actual_bond_dimensions`：実際の内部bond dimension。
- `max_discarded_weight`：その往復に含まれる全局所SVDのdiscarded weight最大値。
- `stability_count`：energy安定条件の連続成立回数。

`max_discarded_weight`は、1往復中の局所切り捨てを要約した値である。最終状態の全誤差や基底energy誤差ではない。

### 7.3 1往復のwrapper

Notebook 02のpassを、

```python
right_pass_result = sweep_left_to_right(...)
left_pass_result = sweep_right_to_left(
    right_pass_result.cores,
    ...,
)
```

と接続する。右向きpassの更新済みMPSを左向きpassへ渡し、両passの局所ログを実行順に連結する。

次の往復には、前の往復終了時のMPSを渡す。毎回共通初期状態へ戻してはいけない。

### 7.4 複数往復loop

反復処理は次の順序である。

```text
初期状態をk=0として診断・記録
        ↓
現在のMPSから左右1往復
        ↓
更新後MPSを次の現在状態にする
        ↓
全energy・残差・分散・bondを診断
        ↓
delta Eと連続成立回数を更新
        ↓
今回の最大discarded weightを記録
        ↓
終了理由を判定
        ↓
継続なら次の往復へ
```

exact ground energyはgapの診断と変分下限のsanity checkに使うが、停止条件には使わない。

---

## 8. 停止判定の単体検証

停止判定は、DMRGの実データだけでなく人工的な入力でも確認する。

### 8.1 連続成立回数

- $\delta E=\varepsilon_E/2$：回数を1増やす。
- 2回連続で成立：0から1、1から2へ増える。
- $\delta E=2\varepsilon_E$：0へリセットする。
- $\delta E=\varepsilon_E$：境界を条件成立に含める。
- `energy_change=None`：0へ戻す。
- 非有限値：例外にする。

### 8.2 終了理由

- 最小往復数の直前：連続回数が足りても継続。
- 最小往復数到達時：連続回数が不足なら継続。
- 最大往復数より前：両安定条件を満たせば `converged`。
- 最大往復数で安定条件も成立：`converged`を優先。
- 最大往復数で安定条件が不成立：`max_round_trips_reached`。

連続回数の処理と終了理由の処理を別々にready判定するため、片方だけを穴埋めした途中状態でも、未実装側のassertを誤って実行しない。

---

## 9. bond上限比較の設計

比較条件は、

```text
max_bond = 2
max_bond = 4
max_bond = None
```

である。

- `2`：Notebook 02と同じ強いbond制約。
- `4`：強い制約と無切り捨てcontrolの中間。
- `None`：各局所SVDで利用可能な特異値を意図的に捨てないcontrol。

6サイト二準位系では、中央cutの最大Schmidt rankは8である。`None`はrankを8へ固定する指定ではなく、各SVDで上限による切り捨てを行わないという意味である。

### 公平性

条件間では、

- Hamiltonian。
- dtype・device。
- 乱数seed。
- 初期MPS。
- 最小・最大往復数。
- energy tolerance。
- 必要な連続成立回数。

を揃え、bond上限だけを変える。各条件の可変MPS coreはcloneし、Tensorを共有しない。

共通初期MPSが最小の有限bond上限2でも表現できることを先にassertする。また有限上限の条件では、全履歴について実際の内部bond dimensionが上限以下であることを確認する。

比較セルが既存の `run_dmrg_until_energy_stable` を呼ぶだけなのは、新しいアルゴリズムを実装する問題ではなく、共通条件を保った実験設計を確認する統合問題だからである。

---

## 10. 保存済み結果

以下はNotebookに保存された出力であり、このノート作成時にFresh Restart KernelからRun Allした結果ではない。

### 10.1 実行段階

保存済み出力では次がすべて `OK` になっている。

- Notebook 02依存関係。
- global residualとenergy分散。
- 左右1往復wrapper。
- energy安定の連続回数。
- 終了理由の境界条件。
- 複数往復と停止。
- bond上限比較。
- 履歴表と3種類の可視化。
- 最終契約検証。

主条件 `max_bond=2` は、

```text
stop reason          : converged
completed round trips: 5
```

で終了した。ここでの `converged` はenergy安定条件を満たしたという意味である。

### 10.2 bond上限別の最終結果

| 条件 | 完了往復数 | 終了理由 | 最終energy | 厳密energy差 | global residual | energy variance | 実際の内部bond | 最終往復の最大discarded weight |
|---|---:|---|---:|---:|---:|---:|---|---:|
| `max_bond_2` | 5 | converged | -8.268341 | $1.002944\times10^{-3}$ | $7.562557\times10^{-2}$ | $5.719227\times10^{-3}$ | (2, 2, 2, 2, 2) | $6.542568\times10^{-5}$ |
| `max_bond_4` | 3 | converged | -8.269344 | $4.541877\times10^{-8}$ | $5.734174\times10^{-4}$ | $3.288075\times10^{-7}$ | (2, 4, 4, 4, 2) | $6.314047\times10^{-9}$ |
| `no_truncation` | 3 | converged | -8.269344 | $-8.881784\times10^{-15}$ | $8.232886\times10^{-15}$ | 0 | (2, 4, 8, 4, 2) | 0 |

厳密基底energyは、

$$
E_0
=
-8.269344348226955
$$

である。

`no_truncation`のenergy gapが僅かに負なのは、変分原理の破れを示す大きさではなく、浮動小数点丸め誤差の範囲である。Notebookも `ATOL` の範囲で許容している。

`no_truncation`のenergy varianceが0と表示される一方でresidualが約 $8.2\times10^{-15}$ なのも、$\langle H^2\rangle-E^2$ という近い数同士の差が機械精度で0になったためである。理論上の関係と矛盾する有意な差ではない。

---

## 11. 実験結果から読めること

### 11.1 `max_bond=2`

5往復後にenergy安定条件を満たしたが、

$$
E-E_0
\approx
1.0\times10^{-3},
\qquad
\rho
\approx
7.6\times10^{-2}
$$

が残っている。

したがって、energy変化による停止と、全系固有状態・厳密基底状態への到達は別であることが実測値に現れている。

この停滞はbond上限2の表現制約と整合するが、この1条件だけから $E_k-E_\chi^\star$ と $E_\chi^\star-E_0$ を厳密に分離することはできない。

### 11.2 `max_bond=4`

bond上限4では、3往復でenergy gapが約 $4.5\times10^{-8}$ まで減り、global residualも約 $5.7\times10^{-4}$ まで下がった。

energyが非常に厳密値へ近くても、global residualは同じ桁で0になるとは限らない。energyと固有方程式残差が異なる診断量であることを示している。

### 11.3 無切り捨てcontrol

無切り捨てでは中央bondが8まで成長し、3往復でtoy問題の厳密基底状態と数値誤差内で一致した。

これは今回の小さいHamiltonian・初期状態・更新法で得た結果であり、一般の問題で3往復あれば十分という主張ではない。

### 11.4 bond上限比較からの限定的な推論

今回の固定条件では、bond上限を2から4、無切り捨てへ緩めるほど、最終energy、global residual、discarded weightが改善した。

この結果は、bond表現能力が今回の `max_bond=2` の停滞に関与しているという解釈を支持する。ただし、有限反復結果だけから誤差分解の2項を数値的に確定したり、すべての初期値で同じ順位になると一般化したりはできない。

---

## 12. Notebook 03で確認したこと・していないこと

### 確認したこと

- Notebook 02の左右passを、状態を引き継ぐ複数往復へ接続した。
- energy変化、最小往復数、連続成立回数、最大往復数を使う停止処理を実装した。
- global residualとenergy分散を全状態から計算した。
- 停止条件と解の精度診断を分離した。
- 共通初期状態からbond上限2・4・無切り捨てを比較した。
- 実際の内部bond dimensionが指定上限を守ることを確認した。

### 確認していないこと

- 複数seed・複数初期MPSでの再現性。
- より大きい系での収束・計算量・memory。
- matrix-freeな2-site有効Hamiltonian作用。
- Lanczos法などの反復固有値solver。
- 1-site DMRG sweepとの実測比較。
- NN目的関数を使うDMRG-like最適化。

---

## 13. 次の学習段階

物理DMRGの基礎Notebookは03で一区切りとし、新しい物理DMRG Notebook 04は作らない方針である。

次はFashion-MNISTのTT-MLPへ移り、00〜03で学んだ

```text
隣接2 coreを結合
    ↓
局所変数として更新
    ↓
SVDで再分割
    ↓
bondを移動しながら反復
```

という構造を、教師あり学習へどう接続するかを整理する。

ただし、物理DMRGのHamiltonian最小固有値問題をそのまま移植するわけではない。NN側では、

- 目的関数を教師あり損失へ変更する。
- 固定した他coreとデータから局所問題を定義する。
- 交差エントロピーでは一般に最小固有値問題にならない。
- 局所勾配法・最小二乗法・Gauss-Newton型近似など、損失に応じた更新法を選ぶ。
- DMRG-like更新と通常のgradient-based Fine-tuningを比較する。

必要がある。
