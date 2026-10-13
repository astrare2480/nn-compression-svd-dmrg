---
title: accuracy許容差とParetoによる圧縮候補選択
tags:
  - model-compression
  - model-selection
  - Pareto-frontier
  - knee-point
  - validation
---

# accuracy許容差とParetoによる圧縮候補選択

## 1. 何を決めるための方法か

圧縮モデルの候補選択では、通常、単一の指標だけを最大化しない。少なくとも次の目的が競合する。

- モデル総parameter数を小さくする。
- validation accuracyを高くする。
- 学習時間やmemoryを小さくする。
- 元モデルのpredictionやlogitsを保つ。

これらを最初から1つの合成scoreへ押し込むと、重み付けに結論が依存する。まず目的を分けて候補を整理し、その後に運用上の優先順位を適用する方が解釈しやすい。

このノートでは、次の3つを区別する。

1. Pareto frontier：明らかに劣る候補を除く。
2. accuracy tolerance：必要なaccuracyを満たす中で最小モデルを選ぶ。
3. knee：frontier上の折衷点を補助的に探す。

## 2. Pareto frontier

### 2.1 支配の定義

候補$i$のモデルサイズを$P_i$、validation errorを$E_i = 1 - A_i$とする。どちらも小さい方がよいとする。

候補$j$が候補$i$を支配するとは、

$$
P_j \leq P_i,
\qquad
E_j \leq E_i
$$

が成り立ち、少なくとも一方が狭義不等号であることをいう。どの候補にも支配されない点の集合がPareto frontierである。

### 2.2 同じモデルサイズの候補

同じ$P$でaccuracyが異なる2候補があるなら、accuracyが低い方は支配される。

一方、$P$もaccuracyも完全に同じなら、accuracy Paretoだけでは一方を選べない。更新方式や名称を使って暗黙に片方へ絞らず、同率候補として残す。別指標で選ぶなら、その規則を明示する。

### 2.3 小さい数値例

次の4候補を考える。

| 候補 | parameter数 | accuracy | error |
| :-- | ---: | ---: | ---: |
| A | 100 | 0.880 | 0.120 |
| B | 100 | 0.890 | 0.110 |
| C | 140 | 0.895 | 0.105 |
| D | 180 | 0.892 | 0.108 |

Aは同じ100 parameterでaccuracyが高いBに支配される。Dは、より小さくaccuracyも高いCに支配される。Pareto frontierにはBとCが残る。

Pareto frontierは「最良の1点」を自動で決めない。Bは小ささ、Cはaccuracyで優位であり、どちらを選ぶかには追加の目的が必要である。

## 3. accuracy tolerance

### 3.1 規則

dense baselineのaccuracyを$A_{\mathrm{dense}}$、許容低下を$\delta$とする。cutoffは、

$$
A_{\mathrm{cutoff}}
=
A_{\mathrm{dense}} - \delta
$$

である。候補集合を、

$$
\mathcal{C}_{\delta}
=
\left\{
i
\mid
A_i \geq A_{\mathrm{dense}} - \delta
\right\}
$$

とし、その中でモデルサイズが最小の候補を選ぶ。

$$
P_{\min}
=
\min_{i \in \mathcal{C}_{\delta}} P_i
$$

同じ最小サイズの候補が複数あれば、事前に定めたtie-breakがない限り、すべて残す。

### 3.2 数値例

dense accuracyを`0.900`、許容低下を`0.005`とすると、cutoffは`0.895`である。

| 候補 | parameter数 | accuracy | cutoffを満たすか |
| :-- | ---: | ---: | :-- |
| A | 100 | 0.893 | いいえ |
| B | 120 | 0.895 | はい |
| C | 120 | 0.897 | はい |
| D | 160 | 0.901 | はい |

最小サイズは120なので、tie規則がなければBとCを両方残す。Dはaccuracyが最も高いが、「cutoffを満たした後はサイズを優先する」という目的では選ばれない。

### 3.3 許容差は統計的同等性ではない

$\delta$は運用上許容する性能低下であり、次のものではない。

- 信頼区間
- 標準誤差
- 同等性検定のmarginと検定結果
- 理論的な近似誤差上界

結果を見てから$\delta$を決めると、候補に都合のよい境界になりやすい。探索段階の整理に事後的許容差を使うことはできるが、「事後的な運用規則」と明記する。確認実験では、値とtie規則を実験前に固定する。

### 3.4 pandasによる最小実装例

次の例は、モデルサイズを最小化し、accuracyを最大化するPareto抽出と、許容差選択を別々に実行する。同じサイズ・同じaccuracyの完全同率候補は両方残る。

```python
import pandas as pd


def extract_pareto_frontier(
    df: pd.DataFrame,
    size_col: str,
    score_col: str,
) -> pd.DataFrame:
    """サイズ最小化・score最大化のPareto frontierを返す。"""
    keep_indices: list[int] = []

    for index, row in df.iterrows():
        # 現在の候補以上に小さく、かつscoreが高い候補を探す。
        weakly_better = (
            (df[size_col] <= row[size_col])
            & (df[score_col] >= row[score_col])
        )
        strictly_better = (
            (df[size_col] < row[size_col])
            | (df[score_col] > row[score_col])
        )
        dominated = (weakly_better & strictly_better).any()

        if not dominated:
            keep_indices.append(index)

    return (
        df.loc[keep_indices]
        .sort_values([size_col, score_col], ascending=[True, False])
        .reset_index(drop=True)
    )


# 小さい例をDataFrameとして用意する。
candidates = pd.DataFrame(
    {
        "candidate": ["A", "B", "C", "D"],
        "model_parameters": [100, 100, 140, 180],
        "accuracy": [0.880, 0.890, 0.895, 0.892],
    }
)

# Pareto抽出ではBとCが残る。
pareto = extract_pareto_frontier(
    candidates,
    size_col="model_parameters",
    score_col="accuracy",
)
assert pareto["candidate"].tolist() == ["B", "C"]

# 許容差選択は、cutoffを満たす中で最小サイズを求める。
dense_accuracy = 0.895
accuracy_tolerance = 0.005
cutoff = dense_accuracy - accuracy_tolerance

eligible = candidates.loc[candidates["accuracy"] >= cutoff].copy()
minimum_size = eligible["model_parameters"].min()
selected = eligible.loc[
    eligible["model_parameters"] == minimum_size
].reset_index(drop=True)

assert selected["candidate"].tolist() == ["B"]
```

Pareto抽出前に同じサイズ内で1行へ集約する方法もあるが、完全同率候補まで消さないよう注意する。安全側では上のように全行へ支配判定を行い、候補数が多い場合だけ、同じサイズ・同じscoreの同率情報を保持した集約へ置き換える。

## 4. Paretoとtoleranceは別の問いに答える

Pareto frontierは、候補集合の中で他候補に明確に劣る点を除く。baselineや許容差を必要としない。

accuracy toleranceは、「ここまでaccuracyを保てば十分」という外部要件を置き、その条件を満たす中で最小モデルを選ぶ。frontier上の最高accuracy点を選ぶ規則ではない。

したがって、Pareto表とtolerance候補表の行数が異なっても矛盾ではない。ある候補がPareto frontierには残らなくてもcutoffを満たすことはあり、逆もあり得る。

## 5. knee point

### 5.1 端点直線からの距離

モデルサイズとerrorのスケールが異なるため、まずfrontier上の値を0から1へ正規化する。

$$
\widetilde{P}_i
=
\frac{P_i - P_{\min}}{P_{\max} - P_{\min}},
\qquad
\widetilde{E}_i
=
\frac{E_i - E_{\min}}{E_{\max} - E_{\min}}
$$

frontierの両端を結ぶ直線からの垂直距離が最大の点を、kneeの候補とする方法がある。2次元点を$\mathbf{x}_i$、端点を$\mathbf{a}, \mathbf{b}$とすれば、距離は、

$$
d_i
=
\frac{
\left|
(\mathbf{b}-\mathbf{a})
\times
(\mathbf{x}_i-\mathbf{a})
\right|
}{
\lVert \mathbf{b}-\mathbf{a} \rVert_2
}
$$

で計算できる。2次元の$\times$は外積の大きさに相当する。

### 5.2 kneeを主規則にしない場合

kneeには次の依存性がある。

- 正規化方法
- 距離や曲率の定義
- 候補点の密度
- 探索したrank範囲
- 外れ値や端点

frontierが3点だけなら、中間点が選ばれやすく、安定した曲率推定とは言いにくい。kneeは「折衷点の参考」であり、精度下限のような明示要件を置けるなら、tolerance規則の方が採用理由を説明しやすい。

## 6. 指標を混ぜない

分類モデルでは、accuracy、prediction agreement、logits RMSEは別の対象を測る。

- accuracy：正解labelに対して正しいか。
- agreement：基準モデルと同じargmaxを出すか。
- logits RMSE：基準モデルのlogits全体に数値的に近いか。

基準モデル自体が誤るため、高agreementや低RMSEが高accuracyを保証しない。反対に、圧縮モデルが基準モデルと異なるlogitsを出しても、labelに対するaccuracyが高い場合がある。

そのため、次のように目的ごとのfrontierを別に作る。

1. モデルサイズ対accuracy
2. モデルサイズ対agreement
3. モデルサイズ対logits RMSE

どのfrontierを採用に使うかは用途で決める。通常の分類精度と圧縮率が主目的ならaccuracyを主にし、agreementとRMSEはdense忠実度の診断として残す。蒸留や回帰互換性が目的なら、dense忠実度側の重みが大きくなる。

## 7. モデルサイズと学習コストを同じ軸とみなさない

TT-rankはTT core shapeを変えるため、モデル総parameter数と推論構造へ影響する。一方、更新範囲を`all_parameters`から`tt_cores_only`へ狭めても、固定parameterはモデルに残る。

従って、次は別の量である。

- モデル総parameter数
- 学習対象parameter数
- 1 epoch当たりの時間
- early stoppingまでの累積時間
- optimizer stateのmemory
- 推論MACs・latency

学習対象parameter数を大きく減らしても、forward計算は残る。前段parameterへ勾配を返すには、後段が固定されていても入力勾配を計算する必要がある。そのため、trainable parameter削減率と秒/epoch短縮率は一致しない。

また、1 epochが速くても、early stoppingまでのepoch数が増えれば総学習時間は長くなる。逆に、モデルを一度学習して長期間推論するなら、学習時間差よりモデルサイズや推論コストが重要になる。頻繁に再学習するなら、累積学習時間を独立した目的としてPareto解析へ追加する。

## 8. validation上の小差を読むときの注意

validation sample数を$N$とすると、accuracyの最小刻みは、

$$
\frac{1}{N}
$$

である。$N=5000$なら1件は`0.0002`、差`0.0006`は3件分である。

ただし、同じsampleを評価した2モデルの差は対応のあるデータである。各accuracyの二項標準誤差だけを比較するより、sample別の正誤表を作り、McNemar検定やpaired bootstrapを使う方が問いに合う。さらに、学習seedを変えたばらつきと、固定checkpointを同じvalidation setで比較したばらつきも分ける。

小差を採用根拠にする場合は、少なくとも次を記録する。

1. validation sample数と正解件数差
2. sample別の正誤対応
3. 複数seedの平均とばらつき
4. 許容差を決めた時点
5. test setを候補選択に使っていないこと

## 9. 再現可能な候補選択の手順

1. 比較するcheckpoint、data split、seed、tensorization、optimizer、loss、early stopping規則を揃える。
2. trial表と、重複checkpointを統合したモデル候補表を分ける。
3. 同じモデルサイズ内で指標が劣る候補を正しく支配判定する。
4. accuracy、agreement、RMSEのPareto frontierを別々に保存する。
5. 主目的とaccuracy許容差を明示し、同率候補を保持する。
6. kneeは補助結果として保存し、候補点数と正規化依存性を書く。
7. 学習コストはモデルサイズとは別表にする。
8. 暫定候補をvalidationで固定した後、未使用test setを一度だけ評価する。

## 10. まとめ

- Pareto frontierは明らかに劣る候補を除くが、最良の1点を自動で決めない。
- accuracy toleranceは、明示したaccuracy下限を満たす中で最小モデルを選ぶ運用規則である。
- 事後的な許容差は統計的同等性や理論保証ではない。
- 同じモデルサイズ・同じ指標値の候補は、根拠なく一方へ絞らない。
- kneeは正規化、定義、候補点数に依存する補助情報である。
- accuracy、agreement、logits RMSEは目的が異なるため、frontierを分ける。
- モデル総parameter数、trainable parameter数、秒/epoch、累積時間、推論コストを別々に評価する。
- 小さいaccuracy差は件数へ戻し、対応のある比較と複数seedで不確実性を確認する。
