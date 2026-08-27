---
title: Tucker実験で得た設計原則と考察
tags:
  - Tucker
  - HOOI
  - HOSVD
  - 実験設計
  - NN圧縮
---

# Tucker実験で得た設計原則と考察

## 1. Tucker化は自動的に圧縮ではない

Conv Tucker-2は元の1層を3層へ分ける。

$$
N_{\mathrm{Tucker2}}
=
C_{in}R_{in}
+
R_{out}R_{in}K_hK_w
+
C_{out}R_{out}
$$

なので、rankが大きいと

$$
N_{\mathrm{Tucker2}}
>
N_{\mathrm{original}}
$$

になり得る。

実際 `(64,32)` ではparameterとMACsが増加した。

---

## 2. rank_outとrank_inは独立な設計変数

Tucker-2では

$$
(R_{out},R_{in})
$$

を別々に選べる。

両者を同じ割合で減らす必要はない。

rank sweepでは `rank_in=8` の劣化が大きく、入力channel側の強い圧縮がtaskへ大きく影響する設定が観察された。

したがって、multilinear rankは1つのスカラーrankとして扱わない。

---

## 3. weight errorとaccuracyは別の目的

HOOIの直接目的は

$$
\min
\left\|
W-\hat W
\right\|_F^2
$$

である。

CNNで欲しいのは、最終的には

$$
\min\mathcal L_{task}
$$

である。

今回、HOOIは

```text
weight error: 0.449042 → 0.442504
```

と改善したが、圧縮直後validationは

```text
0.6280 → 0.6202
```

と低下した。

さらにfine-tuningではweight errorが増えながらaccuracyが改善した。

この2つの実験から、

$$
\boxed{
\text{weight approximation quality}
\neq
\text{task performance}
}
$$

を明確に確認できた。

---

## 4. HOSVDは弱い方法ではない

今回の設定では、HOSVDは

- HOOIより高速
- rank sweepへ使いやすい
- 圧縮直後accuracyはHOOIより高かった
- fine-tuning後も十分回復した

という結果だった。

したがって、

```text
rank探索       → HOSVD
初期モデル作成 → HOSVD
最終候補精密化 → 必要ならHOOI
```

という使い分けが合理的。

HOOIが弱いのではなく、**数学的に改善する対象がtask accuracyと一致しない場合、追加コストの費用対効果が小さくなる**。

---

## 5. HOOIの価値

HOOIは同rankでfactorを協調させ、HOSVDよりweight再構成を精密化する。

今回、

- HOSVDより低い最終error
- 6 sweepで収束
- TensorLyと同一の外部再計算error
- TensorLyと同じvalidation/test accuracy

を得た。

したがって、学習目的としては

```text
初期分解
vs
反復最適化
```

を実装レベルで理解できたことが大きい。

これはTT-SVDからDMRGへ進むときの

```text
初期分解
→ 固定rankで局所的に更新
→ sweep
→ 収束判定
```

という考え方への橋渡しになる。

---

## 6. fine-tuningは別の最適化

分解は元weightを近似する初期化。

fine-tuningは、低rank構造を保ったままtask lossへ再最適化する。

```text
HOSVD / HOOI
→ weight-space initialization

Fine-tuning
→ task-space optimization
```

そのためfine-tuning後に元weightとの差が増えること自体は矛盾ではない。

---

## 7. 分解法比較とfine-tuning比較を混ぜない

主比較を分ける。

```text
分解法の比較
→ 圧縮直後
→ weight error / decomposition time / accuracy

実用モデルの比較
→ fine-tuning後
→ val/test accuracy / recovery / learning curve
```

fine-tuning後の数値だけでHOOI/HOSVDの分解品質を評価しない。

---

## 8. single seedの扱い

05のHOSVD/HOOI fine-tuning比較はseed 0のみ。

HOOI test 0.7555、HOSVD 0.7508の差は

$$
0.0047
$$

である。

これは今回のrunの観測値として記録するが、統計的な優位性を主張しない。

複数seedは「HOOIが最終accuracyで優れる」と強く主張する場合に必要で、Tucker/HOOIの学習を終えるための必須条件ではない。

---

## 9. TensorLy timingの扱い

今回TensorLyは自作HOOIより遅かったが、

- 問題Tensorが小さい
- 同じ6 iteration
- 数十ms規模のGPU計測
- 汎用libraryの固定overhead

という条件がある。

速度比をライブラリ一般の結論にしない。

性能比較の主目的は、自作HOOIの**数値結果がTensorLyと整合したこと**である。

---

## 10. src化で得た設計原則

最終的に責務を次のように分けた。

```text
operations.py
→ Tensorの基本演算

tucker.py
→ HOSVD / Tucker再構成

hooi.py
→ layerに依存しない汎用HOOI

conv_tucker.py
→ Conv2d固有のTucker-2
```

重要なのは、

- `ranks` のkeyを更新modeとしてpartial HOOIを表現
- Notebookの自作実装は学習履歴として残す
- 実験Notebookはsrc化後に重複処理を共通化
- public helperとprivate validationを分ける
- pandasを汎用compression moduleへ持ち込まない
- 入力factorを破壊しない

という点。

---

## 11. Tucker/HOOI編の到達点

今回までで、

```text
Tensor mode演算
→ Tucker / HOSVD
→ Conv Tucker-2
→ rank sweep
→ fine-tuning
→ HOOI
→ TensorLy照合
→ HOSVD/HOOI同条件fine-tuning
→ src共通化
→ 回帰テスト
```

まで完了した。

次のTT/MPSでも、同じ評価軸

```text
weight relative error
parameters
MACs
圧縮直後accuracy
decomposition time
必要ならfine-tuning後accuracy
```

を維持する。

特に今回得た

> Frobenius weight errorを下げる分解法が、必ずtask accuracyを改善するわけではない。

という観察を、TT/MPS・DMRGでも意識する。
