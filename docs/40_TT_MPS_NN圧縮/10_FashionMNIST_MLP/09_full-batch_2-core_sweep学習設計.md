---
title: Fashion-MNIST TT-MLP full-batch 2-core sweep学習設計
aliases:
  - Fashion-MNIST TT-MLP DMRG-like学習設計
tags:
  - Fashion-MNIST
  - TTLinear
  - DMRG-like
  - two-core sweep
  - full-batch gradient
  - early stopping
---

# Fashion-MNIST TT-MLP full-batch 2-core sweep学習設計

## このノートの目的

`notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb` で行う実験について、実装前に固定したcontractを残す。

このNotebookは、`notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb` で確認した

```text
2 coreを結合
→ 結合テンソルを分類損失の勾配で1回更新
→ SVDで再分割
→ modelへ書き戻す
```

という最小単位を、Fashion-MNISTの全学習データ、6 core、左右sweep、複数sweep、validation early stoppingへ拡張する設計である。

本ノートは実装前のアルゴリズムと実験contractとして残す。2026年11月1日にNotebook 00で学習・保存・best再読込まで実行し、保存済み結果と考察は[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]へ分けた。現在保存されているNotebookは1 code cellだけ実行番号がないため、全cell連番のFresh Run All証跡とは区別する。局所更新の数式は[[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]、04の保存済み実装確認は[[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]を参照する。

---

## 1. アルゴリズムの正確な呼び方

本設計は、次のアルゴリズムである。

> 全学習データの平均勾配を使う、重なり合う局所blockのDMRG-like 2-core gradient sweep

次のいずれとも同一ではない。

- 物理DMRGの局所有効Hamiltonian固有値問題。
- 各minibatch直後にparameterを更新する確率的SGD。
- disjoint blockだけを1回ずつ更新する単純なblock coordinate descent。
- dense weight全体を更新してからTTへ射影する方法。
- TT多様体上のRiemannian gradient法。

隣り合うpairは1 coreを共有する。例えばpair $(0,1)$ と $(1,2)$ はcore 1を共有する。そのため、各局所blockは互いに重なる。

また、本設計では局所pairだけでなく、第1層biasと後段dense layerも同時に更新する。したがって、厳密には「coreだけの局所block最適化」ではない。

---

## 2. 2つの独立な設計軸

会話中に使われたA/Bという呼び方は、異なる2軸に現れた。実装名では曖昧なA/Bを避ける。

### 2.1 何を更新変数にするか

本設計は、結合テンソルを直接更新する。

$$
T^+
=
T-eta_T\nabla_TL
$$

dense weight全体を更新した後にTTへ射影する

$$
W^+
=
W-eta_W\nabla_WL,
\qquad
\min_{\widetilde W\in\mathcal{M}_{\mathrm{TT}}}
\lVert W^+-\widetilde W\rVert_F^2
$$

という方式は使わない。

### 2.2 どのデータで1回の勾配を作るか

本設計は、各pairで全学習データを走査し、全データ平均勾配を作ってから1回だけ更新する。

minibatchごとにpairを更新して次のbatchへ進む方式は、別Notebookへ延期する。

したがって本設計は、

```text
結合テンソルの局所更新
+
各pairで全学習データ平均勾配
```

の組合せである。

---

## 3. モデルの初期化

### 3.1 pretrained checkpointから始めない

この実験では、新しいdense MLPを通常のPyTorch初期化で作る。既存の学習済みTT-MLP checkpointを初期状態にしない。

手順は次である。

1. seedを固定する。
2. dense MLPを生成する。
3. dense第1層 `fc1.weight` をTT-SVDで6-core TT-matrixへ変換する。
4. `fc1` をTTLinearへ置換する。
5. 直交中心をsite 0へ移す。
6. この状態をsweep 0の初期状態として保存・評価する。

rank上限付きTT-SVDなので、初期TTLinearが元dense `fc1` をexactに再現するとは限らない。初期化時に少なくとも次を記録する。

```text
dense fc1 weight norm
TT再構成weight norm
weight absolute / relative error
初期actual ranks
初期train / validation loss
初期train / validation accuracy
```

### 3.2 6-core mode分解

入力次元784は

$$
784
=
2\cdot2\cdot2\cdot2\cdot7\cdot7
$$

と分ける。

$$
\mathrm{IN\_MODES}
=(2,2,2,2,7,7)
$$

出力次元512は

$$
512
=
4\cdot2\cdot2\cdot2\cdot4\cdot4
$$

と分ける。

$$
\mathrm{OUT\_MODES}
=(4,2,2,2,4,4)
$$

である。

以前の候補

$$
(4,2,2,2,2,4)
$$

の積は

$$
4\cdot2\cdot2\cdot2\cdot2\cdot4
=256
$$

であり、512にならない。この誤った候補を実装へ残さない。

各siteのTT-matrix physical sizeは

$$
q_k=m_kn_k
$$

なので、

$$
(q_1,q_2,q_3,q_4,q_5,q_6)
=(8,4,4,4,28,28)
$$

である。

### 3.3 rank capとactual rankを区別する

$$
\mathrm{RANK\_CAP}=12
$$

とする。各cutの形状上限とcapから期待される最大rank列は

$$
(1,8,12,12,12,12,1)
$$

である。

第1cutでは左次元が8しかないので、

$$
r_1\le8
$$

である。第2cut以降は形状上限が12より大きいので、capが上限になる。

ただし、actual rankは入力行列の特異値とrank判定方針で決まる。`rank_cap=12` と表示されたことだけを根拠に、actual rankが必ず12だと書かない。

---

## 4. 全学習データ平均損失と勾配蓄積

学習データ総数を $N$、第 $b$ minibatchのsample数を $n_b$、そのbatchのmean lossを $L_b$ とする。全データ平均損失は

$$
L
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell_i
$$

である。各batchがmean reductionなら、

$$
L
=
\sum_b
\frac{n_b}{N}
L_b
$$

となる。

微分の線形性により、任意の更新対象 $	heta$ について

$$
\nabla_\theta L
=
\sum_b
\frac{n_b}{N}
\nabla_\theta L_b
$$

である。

したがって、各batchで

```python
# batch lossはmean reduction。
scaled_loss = batch_loss * (batch_size / total_num_samples)
scaled_loss.backward()
```

とすれば、全batch終了時の `.grad` は全データmean lossの勾配になる。

### 4.1 最終batchが小さい場合

`drop_last=False` では、最後のbatchだけsample数が小さい場合がある。全てのbatch lossを単純に

$$
\frac{1}{B}
\sum_{b=1}^{B}L_b
$$

と平均すると、sampleごとの平均とは一致しない。

正しい重みは

$$
\frac{n_b}{N}
$$

である。

### 4.2 二重除算をしない

各batchで既に $n_b/N$ を掛けてbackwardした後に、蓄積勾配をもう一度batch数や $N$ で割ってはいけない。そうすると勾配scaleが過小になる。

### 4.3 graphをbatch間で保持しない

各batchのforward graphは、そのbatchのbackward後に解放する。`retain_graph=True` で全データ分のgraphを保持しない。

$$
g
=
\sum_b g_b
$$

に必要なのは勾配値の加算であり、全batchのgraphを同時に保持することではない。

### 4.4 gradient loader

最初の基準実験では、勾配蓄積用loaderを次のように固定する。

```text
shuffle=False
drop_last=False
固定seed
```

全データを1回ずつ使う限り、数学的な全データmean gradientはデータ順序に依存しない。しかし、浮動小数点加算順序と再現性を安定させるため、基準Notebookでは順序を固定する。

---

## 5. 1 pairで同時更新するparameter

pair $k,k+1$ では、結合テンソル

$$
T_k
$$

に加えて、次を更新する。

```text
fc1.bias
fc2.weight
fc2.bias
fc3.weight
fc3.bias
```

pair外の `fc1` TT coreは固定する。

局所テンソルとdense parameterの学習率は、最初の値が同じでも設定名を分ける。

$$
T_k^+
=
T_k
-
\eta_{\mathrm{TT}}
\nabla_{T_k}L,
$$

$$
\theta_{\mathrm{dense}}^+
=
\theta_{\mathrm{dense}}
-
\eta_{\mathrm{dense}}
\nabla_{\theta_{\mathrm{dense}}}L
$$

とし、

```text
TT_PAIR_LEARNING_RATE
DENSE_PARAMETER_LEARNING_RATE
```

を別に持つ。

### 5.1 同じ更新前状態から勾配を計算する

1 pairの全データ走査中に、dense parameterを途中更新しない。$T_k$ とdense parameterの勾配は、同じ更新前モデルに対する全データmean lossから計算する。

全batchの蓄積後に初めて、

```text
T candidate
dense parameter candidates
SVD分割core candidates
```

を作る。

$d=6$、$L=1$ では、共有dense parameterは1 sweepに10回更新される。したがって、`DENSE_PARAMETER_LEARNING_RATE` を通常fine-tuningの「1 minibatchにつき1回」の学習率と数値だけで比較してはいけない。また、性能変化をTT pair更新だけの効果として解釈できない。

### 5.2 `LOCAL_STEPS_PER_PAIR`

$$
\mathrm{LOCAL\_STEPS\_PER\_PAIR}=L
$$

を設定可能にする。最初の基準は

$$
L=1
$$

である。

$L>1$ の場合、2回目以降は必ず

1. 直前にcommitした最新coreから $T_k$ を再結合する。
2. 最新のdense parameterを使う。
3. 全学習データを再走査する。
4. 新しい勾配を計算する。

古い $T$ や古い勾配を繰り返し使わない。

---

## 6. 方向付きSVDと中心移動

局所更新後の結合テンソルを行列化して

$$
M_k^+
=
U\Sigma V^{\mathsf T}
$$

とする。

### 6.1 右向き更新

rank $r$ へ制限し、

$$
G_k^+
=
\operatorname{reshape}(U_r),
$$

$$
G_{k+1}^+
=
\operatorname{reshape}(\Sigma_rV_r^{\mathsf T})
$$

とする。左coreは左直交になり、直交中心は $k+1$ へ移る。

### 6.2 左向き更新

$$
G_k^+
=
\operatorname{reshape}(U_r\Sigma_r),
$$

$$
G_{k+1}^+
=
\operatorname{reshape}(V_r^{\mathsf T})
$$

とする。右coreは右直交になり、直交中心は $k$ へ移る。

TTLinear重みのscaleを変えないため、SVD後にcoreを単位ノルムへ正規化しない。

---

## 7. 6-coreの1往復sweep

core数を $d=6$ とする。

右向きでは

$$
(0,1)
\to
(1,2)
\to
(2,3)
\to
(3,4)
\to
(4,5)
$$

の順に更新する。

左向きでは

$$
(4,5)
\to
(3,4)
\to
(2,3)
\to
(1,2)
\to
(0,1)
$$

の順に更新する。

端で折り返すpair $(4,5)$ は、右向きの最後と左向きの最初の両方で更新する。この定義を採ると、1往復のpair visit数は

$$
2(d-1)=10
$$

である。

各pairで $L$ local stepを行うなら、1 sweepの全データforward/backward回数は

$$
2(d-1)L
=
10L
$$

である。$L=1$ でも、1 sweepは全学習データ10 passに相当する。

### 7.1 `MAX_SWEEPS` はepoch数ではない

通常SGDの1 epochは全学習データを1回通る。一方、本設計の1 sweepは、各pairごとに全学習データを通る。

したがって、

$$
1\ \mathrm{sweep}
\neq
1\ \mathrm{epoch}
$$

である。計算量を比較するときはsweep数だけでなく、全データforward/backward pass数を併記する。

---

## 8. candidate更新をatomicにする

1 pairの更新では、

```text
更新済みT
SVD分割後の2 core
同時更新するdense parameter
```

を全てcandidateとして作る。元モデルへ順番に直接書き込まない。

### 8.1 candidate modelの独立性

candidate modelは元モデルとstorageを共有しない。

```python
# 概念例。実装ではmodel構築方法とdevice移動も明示する。
candidate = copy.deepcopy(model)
```

単なるPython list copyやtensor viewでは不十分である。

### 8.2 commit前に確認する項目

少なくとも次を確認する。

1. 全candidate tensorがfinite。
2. coreが4階で、隣接bondが一致。
3. mode shapeが設定と一致。
4. rankがcap以下。
5. dtypeとdeviceがモデルと一致。
6. 再結合したpairのshapeが元のmerged shapeと一致。
7. 方向に応じた左または右直交性誤差が有限。
8. 固定diagnostic batchのlogitsとlossがfinite。
9. candidate stateが元モデルのstorageを共有しない。

対象pair以外の不変性は、実験開始時ではなく**その局所更新の直前**の `state_dict` と比較する。前のpairで更新済みのcoreまで初期状態と一致させる必要はない。parameterだけでなくbufferも確認対象に含める。

全項目が通ったときだけ、2 coreとdense parameterをまとめて元モデルへ反映する。

### 8.3 loss低下をcommit条件にしない

本設計は各pairの全データ勾配を使うが、rank打ち切り後の損失低下は無条件には保証されない。また、診断batchは全データ目的と一致しない。

したがって、

```text
candidate loss < current loss
```

を一般的なaccept/reject条件にしない。loss変化は記録するが、candidate validationは主にshape・finite・contract・再構成整合性へ使う。

### 8.4 例外を全て `non_finite` にしない

停止理由は区別する。

```text
non_finite_loss
non_finite_gradient
non_finite_svd
shape_error
rank_error
device_or_dtype_error
explicit_numerical_failure
```

`except Exception: stop_reason = "non_finite"` のように、実装バグまで数値異常へまとめない。

---

## 9. 固定diagnostic batch

candidate確認に使うbatchは、学習gradient loaderから `next(iter(loader))` で毎回取らない。

shuffle loaderに対する `next(iter(loader))` は、random samplerの状態を進め、比較するsample自体も変え得る。

基準実験では、固定index集合を最初に決める。

```text
DIAGNOSTIC_INDICES = (...)
```

同じsample、同じ順序、同じ前処理で、全candidateを比較する。

このbatchはcommit前の有限値・forward contract診断用であり、学習データ全体の目的関数やvalidation setの代わりではない。

---

## 10. validation early stopping

### 10.1 sweep 0をbaselineにする

局所更新を1回も行っていない初期TTモデルをsweep 0として評価し、historyへ保存する。

### 10.2 `MIN_DELTA` とbest保存を分ける

validation lossを $V_s$、これまでのearly-stopping基準値を $V_{\mathrm{ref}}$ とする。

有意な改善は

$$
V_s
<
V_{\mathrm{ref}}-\mathrm{MIN\_DELTA}
$$

で判定し、成立したらpatience counterを0へ戻し、基準値を更新する。成立しなければcounterを1増やす。

一方、best checkpointは、

$$
V_s<V_{\mathrm{best}}
$$

なら保存する。改善が `MIN_DELTA` より小さくても、実際に最小のvalidation lossならbest stateとして残す。

`best_state` は実モデルのparameterへの参照ではなく、tensorをcloneした独立snapshotとして保持する。参照のまま持つと、後続sweepの更新でbest候補まで書き換わる。

つまり、

- `MIN_DELTA`：early stoppingのpatience判定。
- `best`：観測した最小validation lossの保存。

であり、役割を分ける。

### 10.3 停止理由

正常停止を少なくとも次に分ける。

```text
early_stopping
max_sweeps
```

異常停止は前節の数値・shape・device等の理由を記録する。

`early_stopping` はvalidation指標が設定した規則で改善しなくなったことを意味するだけであり、局所勾配が0、物理DMRGのエネルギー収束、数学的定常点への収束を意味しない。

往復途中で異常停止したpartial sweepは、completed sweepのvalidation候補やbest更新へ混ぜない。最後に成功したpairまでを `final_state` として保存し、completed sweep数とpartial successful update数を別々に記録する。

---

## 11. データ分割の役割

最初のNotebookでは、データを次の2役へ限定する。

### 11.1 training set

- 全データmean gradientの計算。
- sweep終了時のtrain loss・accuracy評価。

### 11.2 validation set

- sweep単位のearly stopping。
- best checkpointの選択。

次は扱わない。

- validationによるpairごとのrank選択。
- test setによる途中選択。
- test accuracyの最終報告。

test setはアルゴリズム設計と候補選択が固定された後の別段階に残す。

---

## 12. 保存する状態

Notebookは、少なくとも次を保存する。

### 12.1 checkpoint

```text
initial_state
best_state
final_state
```

異常停止やearly stoppingが起きても、最後に成功した更新までを含む `final_state` を先に保存する。その後、評価用モデルへ `best_state` をreloadする。

### 12.2 history

sweep 0を含め、各sweepについて次を保存する。

```text
sweep index
train loss / accuracy
validation loss / accuracy
actual TT ranks
TT parameter count
wall time
full-data forward/backward pass count
peak memory
patience counter
best sweep index
stop reason if any
```

pairごとの診断には次を保存する。

```text
sweep index
direction
pair index
local step index
gradient norm of T
gradient norms of dense parameters
eta * ||g_T||
retained rank
discarded tail squared
discarded tail norm
relative truncation error
canonicality error
candidate finite checks
diagnostic batch loss before / after gradient / after truncation
```

### 12.3 再開・監査用メタデータ

```text
configuration
seed
dataset split identifiers
mode decomposition
rank cap
completed sweeps
number of successful pair updates in partial sweep
last successful direction
last successful pair
current orthogonality center
final ranks
stop reason
```

このNotebookでは完全なmid-pair resumeまでは実装しなくてよい。しかし、異常停止した位置を後から判断できる記録は残す。

### 12.4 best reloadの再現確認

保存したbest checkpointを新しいmodel instanceへ読み込み、validation loss・accuracyが保存値と許容差内で一致することを確認する。

---

## 13. 公平な比較のための測定

通常のTT-MLP fine-tuningと比較するときは、更新回数だけをそろえない。本設計の1 sweepは全データを10回以上通るからである。

最低限、次をそろえるか併記する。

- 同じ初期model state。
- 同じtrain / validation split。
- 同じrank capとmode分解。
- 同じデータ順序。
- 同じseed集合。
- warmup後のwall time。
- CUDA使用時の計測前後の同期。
- full-data forward/backward pass数。
- peak device memory。
- 最終TT parameter数。
- train / validation loss・accuracy。

主指標はwall timeとし、forward/backward pass数、sweep数、peak memoryを補助指標にする。1回の短い実行だけで速度優位を結論づけず、複数seedまたは複数反復の中央値を使う。

---

## 14. 必須テスト

実装後は、少なくとも次を確認する。

1. mode積が入力784、出力512に一致する。
2. rank cap 12で第1bondが形状上8以下になる。
3. 2-core mergeのshapeと要素が明示式に一致する。
4. full-rank split後の再結合が入力merged tensorを再現する。
5. `direction="right"` で左coreが左直交になる。
6. `direction="left"` で右coreが右直交になる。
7. 既定directionが既存の右方向動作と後方互換。
8. minibatch蓄積勾配が、全sampleを1 batchにしたmean lossの勾配と一致する。
9. 上の一致を $T$ と全dense parameterの両方で確認する。
10. 最終batchが小さい場合も $n_b/N$ 重みで一致する。
11. 全batch終了前にparameterが更新されない。
12. `LOCAL_STEPS_PER_PAIR=1` で1 pairにつき全データpassが1回。
13. `LOCAL_STEPS_PER_PAIR=2` で最新coreを再結合し、勾配を再計算する。
14. 6 core、1往復、$L=1$ の成功更新数が10で、正常終了時の中心がsite 0へ戻る。
15. 右向き・左向きとも直前更新済みcoreを使い、現在pair以外のTT coreはその局所更新直前から変化しない。
16. candidate failure時に元modelのcoreとdense parameterが全て不変。
17. early stoppingのpatience、`MIN_DELTA`、strict best保存が仕様どおり。
18. best checkpointを新modelへreloadしてvalidation値を再現する。

テストでは、損失低下やaccuracy向上を全random inputに対する必須条件にしない。

---

## 15. Notebookの教材形式

当初は学習用TODOを残す予定だったが、作成途中の合意により、このNotebookは穴埋め式ではなく、完成した基準実装を読んで理解する形式へ変更した。説明セルでは次を示す。

- このセルで確認する数学的対象。
- 入出力shape。
- 実装する式。
- 期待する検証結果。
- 失敗時に見る診断値。

特に次の核を、数式・shape・診断値と対応させて読む。

```text
1: n_b / Nで重み付けした全データ勾配蓄積
2: directionに応じた特異値の吸収先
3: candidate stateのatomic validation
4: sweep 0を含むearly stopping更新
5: best reloadの再現確認
```

完成コードを使う場合も、shape、勾配一致、再構成、rollback、保存再現のassertまたはtestを通し、実装の誤りを成功扱いにしない。

---

## 16. 最初のNotebookに入れないもの

次は意図的に別段階へ送る。

- minibatchごとにsweepする方式。
- adaptive rank cap。
- 別のmode順序・tensorizationとの比較。
- 特異値tailに基づくpairごとの可変rank選択。
- Adam、momentum、weight decay。
- optimizer stateを伴うrank変更。
- pair内の局所収束までの反復。
- dense weight全体を更新してTTへ再射影する方式。
- Riemannian TT最適化。
- 通常fine-tuningとの本比較。
- 複数seedの統計比較。
- test set評価。
- 異常停止位置からの完全resume。

最初のNotebookの目的は、全データ勾配による2-core左右sweepのcontractを正しく通すことであり、全方式の性能比較を1冊で終えることではない。

---

## 17. 1 sweepの疑似コード

```python
# 構造を示す疑似コード。実Notebookでは検証済みAPIを呼び出す。
for direction in ("right", "left"):
    pair_indices = right_pairs if direction == "right" else left_pairs

    for pair_index in pair_indices:
        for local_step in range(local_steps_per_pair):
            # 1. 最新coreから結合テンソルを作る。
            T = build_latest_merged_leaf(model, pair_index)

            # 2. Tと共有dense parameterの勾配を一度だけ初期化する。
            zero_local_gradients(T, dense_parameters)

            # 3. 全学習データのmean gradientを蓄積する。
            for x_batch, y_batch in gradient_loader:
                batch_loss = forward_loss_with_merged_pair(
                    model,
                    x_batch,
                    y_batch,
                    pair_index,
                    T,
                )
                weight = x_batch.shape[0] / total_num_samples
                (weight * batch_loss).backward()

            # 4. 同じ更新前状態からcandidateを作る。
            candidate_T = sgd_step_T(T)
            candidate_dense = sgd_step_dense(dense_parameters)
            candidate_left, candidate_right = split_by_direction(
                candidate_T,
                direction=direction,
                rank_cap=rank_cap,
            )

            # 5. 独立candidate modelで全項目を検証する。
            candidate_model = build_independent_candidate(
                model,
                pair_index,
                candidate_left,
                candidate_right,
                candidate_dense,
            )
            validate_candidate(candidate_model, fixed_diagnostic_batch)

            # 6. 全検証後に一括commitする。
            commit_candidate(model, candidate_model)
```

この疑似コードは、1 pairの処理順とatomic commitの境界を示す。

---

## 18. 文献との対応範囲

隣接MPS tensorを結合し、学習データから局所勾配を作り、更新後にSVDで再分割してsweepする一般構造は、次の一次資料で確認できる。

- [Stoudenmire and Schwab, Supervised Learning with Quantum-Inspired Tensor Networks](https://arxiv.org/abs/1605.05775)
- [Jahromi and Orús, Variational tensor neural networks for deep learning](https://www.nature.com/articles/s41598-024-69366-8)

本設計の6-core mode、ReLU MLP、dense parameter同時更新、全データgradient accumulation、atomic candidate、early stoppingはプロジェクト固有の設計である。論文の実験結果を本Notebookの予想結果として流用しない。

---

## 19. 現行の再利用APIとの対応

共通処理は `src/nn_compression/training/tt_linear_full_batch_sweep.py` に分ける。現行の公開境界は次である。

```text
NonFiniteLocalUpdate
TTLinearLocalUpdate
TTLinearFullBatchSweepResult
accumulate_tt_linear_pair_full_batch_gradients
update_tt_linear_two_core_pair
sweep_tt_linear_pairs_left_to_right
sweep_tt_linear_pairs_right_to_left
train_tt_linear_full_batch_sweeps
save_tt_linear_full_batch_sweep_run
load_tt_linear_sweep_checkpoint
```

役割対応は次である。

- `accumulate_tt_linear_pair_full_batch_gradients`：$n_b/N$ で重み付けした全データ勾配を作る。
- `update_tt_linear_two_core_pair`：候補作成、方向付きSVD、候補検証、1 pairのcommitを担当する。
- 左右sweep関数：最新coreと直交中心を次のpairへ引き継ぐ。
- `train_tt_linear_full_batch_sweeps`：sweep 0、複数sweep、validation、early stopping、best・finalの管理を担当する。
- save / load関数：設定・履歴・checkpointの永続化と再読み込みを担当する。
- `NonFiniteLocalUpdate`：数値の非有限を、shape errorや実装例外と区別する。

Notebook側には、mode・rank・データ・seed・学習率の設定、読みどころ、結果表、plot、保存先、結果の考察を残す。共通APIが存在することと、Fashion-MNIST NotebookがFresh Run Allを通って結果を保存したことは別の完了条件である。

## 関連

- [[50_DMRG/06_TT-MLPのDMRG-like_2-core局所更新]]
- [[08_TT_MPS基礎実装検証/46_TTLinear_2-core局所更新のPyTorch確認]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/01_fc1_TT-rank_sweep]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/06_TTコアのみ更新ablation]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/10_full-batch_2-core_sweep実験結果と考察]]
- [[90_src設計/05_Core_API_v1/training/update_tt_linear_two_core_pair]]
- [[90_src設計/05_Core_API_v1/training/train_tt_linear_full_batch_sweeps]]
- [[90_src設計/05_Core_API_v1/training/save_tt_linear_full_batch_sweep_run]]
