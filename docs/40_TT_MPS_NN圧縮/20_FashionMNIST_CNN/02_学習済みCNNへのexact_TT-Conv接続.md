---
title: 学習済みCNNへのexact TT-Conv接続
tags:
  - TTConv
  - CNN
  - Fashion-MNIST
  - exact-TT
  - model-replacement
---

# 学習済みCNNへのexact TT-Conv接続

## このノートの位置づけ

このノートは、Notebook 02の保存結果と、同じ内容についてChatGPTおよびPerplexityとの対話で確認した事項を統合する。

- [02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb)

Notebook 01では、小さいrandom Convを対象に、TT coresを使うdirect forwardとdense kernel再構成経路の出力・gradientを比較した。Notebook 02では、そのdirect計算を`TTConv2d`という`nn.Module`へまとめ、学習済み`FashionMNISTCNN`の`conv2`だけを置換した。

```text
Notebook 01
小さいConv層単体でdirect forwardとbackwardを検証

Notebook 02
学習済みCNNのconv2へexact TTConv2dを接続
    ↓
conv2出力、モデル全体のlogits、予測class、accuracyを検証
```

モデル全体をTT化したわけではなく、置換対象は`conv2`だけである。また、rank制限による圧縮ではなく、情報を打ち切らないexact TTによる接続確認である。

## 1. 証拠レベル

保存済みNotebookは20セルで、11個のコードセルにはすべて実行番号があり、保存されたerror出力はない。次の完了表示も保存されている。

```text
TODO 1 class definition: OK
TODO 2 structural checks: OK
validation-wide acceptance criteria: OK
```

ただし、このノート作成時にFresh Restart KernelからRun Allを新たに実行したわけではない。以下は保存済みコード・出力と対話内容に基づく整理である。

Notebook末尾の確認項目は表示上`[ ]`のままだが、各条件に対応するassertionと保存結果は通過している。

## 2. 実験条件

| 項目 | 設定 |
| :-- | :-- |
| モデル | 学習済み`FashionMNISTCNN` |
| checkpoint | `fashion_mnist_cnn_baseline.pt` |
| 置換対象 | `conv2`のみ |
| device / dtype | CPU / `torch.float32` |
| seed | 0 |
| validation | Fashion-MNIST train 60,000件から固定分割した5,000件 |
| validation batch size | 32 |
| shuffle | `False` |
| 評価モード | `eval()`と`torch.inference_mode()` |
| test set | loaderを作らず、評価していない |

元モデルの総parameter数は421,642である。validation accuracyなどを事前の固定値として使うのではなく、同じ実行で得たdense baselineをTT置換モデルの比較基準にした。

## 3. conv2のexact TT-matrix化

対象weightのshapeは、

$$
(C_{\mathrm{out}},C_{\mathrm{in}},K_H,K_W)
=(64,32,3,3)
$$

である。1出力位置の入力要素をまとめ、

$$
W_{\mathrm{conv2}}
\rightarrow
W_{\mathrm{mat}}
\in
\mathbb{R}^{64\times288}
$$

とした。

採用したtensorizationは、

$$
\mathrm{OUT\_MODES}=(4,4,4),
\qquad
\mathrm{IN\_MODES}=(6,6,8)
$$

である。

$$
4\times4\times4=64,
\qquad
6\times6\times8=288
$$

を満たす。`max_rank=None`で打ち切りを行わず、保存結果では次のrankとcore shapeになった。

| 項目 | 値 |
| :-- | :-- |
| actual TT-rank | $(1,24,32,1)$ |
| core 0 | $(1,4,6,24)$ |
| core 1 | $(24,4,6,32)$ |
| core 2 | $(32,4,8,1)$ |

このtensorizationでexact置換ができることは確認したが、mode構成がparameter効率のよい選択であるとは評価していない。

## 4. `TTConv2d`としての接続

`TTConv2d`は、TT coresを`nn.ParameterList`、biasを`nn.Parameter`として登録し、denseな`weight` parameterを保持しない。forwardは、

```text
入力feature map
  ↓ F.unfold
各patchを1行にした行列
  ↓ tt_linear_forward
patchごとの出力
  ↓ reshape・permute
Conv形式のfeature map
```

という経路である。通常forwardではdense kernelを再構成しない。

現在の実装範囲は`groups=1`と`padding_mode="zeros"`である。`kernel_size`、`stride`、`padding`、`dilation`はintまたは2要素tupleとして正規化・検証される。

## 5. モデル置換の構造確認

元の`dense_model`を変更せず、独立した`tt_model`を作り、`conv2`だけを`TTConv2d`へ置換した。保存済みの構造チェックでは次を確認している。

- `dense_model`と`tt_model`は別オブジェクトである。
- `dense_model.conv2`は元の`nn.Conv2d`のままである。
- `tt_model.conv2`は`TTConv2d`である。
- `tt_model.conv2`はdenseな`weight`を持たない。
- coresとbiasは学習可能parameterとして登録されている。
- 元のdenseモデルのparameter・buffer値は置換前後で変わっていない。
- `tt_model`の`conv2`以外のstateはdenseモデルと一致する。
- actual TT-rankは$(1,24,32,1)$である。

この確認により、両モデルの差を`conv2`の表現・計算経路の違いへ限定できる。

## 6. parameter数

| 対象 | dense | exact TT |
| :-- | --: | --: |
| conv2 weightまたはcores | 18,432 | 20,032 |
| モデル全体 | 421,642 | 423,242 |

TT coresの要素数は、

$$
\begin{aligned}
P_{\mathrm{TT}}
&=1\times4\times6\times24 \\
&\quad+24\times4\times6\times32 \\
&\quad+32\times4\times8\times1 \\
&=576+18{,}432+1{,}024 \\
&=20{,}032
\end{aligned}
$$

である。dense weightの、

$$
P_{\mathrm{dense}}
=64\times32\times3\times3
=18{,}432
$$

より1,600多い。bias数は変わらないため、モデル全体でも1,600増えた。

rankを打ち切らない変換は圧縮を目的としていない。ただし、exact TTなら常にdenseより大きくなるわけではない。元weightの構造、mode分解、必要なexact rankによってparameter数は変わる。今回はactual rank$(1,24,32,1)$を保持した結果、denseより大きくなった。

## 7. validation全体の保存結果

層単体の比較には、元CNNで`conv2`直前まで処理して得た同一feature mapを、dense `conv2`とexact `TTConv2d`へ入力した。モデル全体の比較では、同じ画像をdenseモデルとTTモデルへ入力した。

| 指標 | 実測値 | 事前の受け入れ基準 | 判定 |
| :-- | --: | --: | :-- |
| conv2出力RMSE | $5.581125\times10^{-7}$ | $10^{-5}$以下 | 合格 |
| モデル全体のlogits RMSE | $3.266623\times10^{-6}$ | $10^{-5}$以下 | 合格 |
| logits差の最大絶対値 | $1.239777\times10^{-5}$ | $10^{-4}$以下 | 合格 |
| prediction agreement | 1.0 | 1.0 | 合格 |

| モデル | validation loss | validation accuracy |
| :-- | --: | --: |
| dense baseline | 0.223251 | 0.9194 |
| exact TT-Conv | 0.223251 | 0.9194 |

RMSEはbatchごとの値を単純平均せず、validation全体の二乗誤差合計を全要素数で割ってから平方根を取っている。最大logits差も全batchを通した最大値である。

## 8. 結果からの考察

### 8.1 01の単層検証を学習済みモデルへ接続した

02は、01で確認した処理を単に再実行したNotebookではない。01のdirect TT forwardを`nn.Module`として学習済みCNNの1層へ組み込み、後段のReLU、pool、Linearを通した最終logitsと予測まで確認した。

exact TTにより数学的には同じconv2 weightを表しているが、dense ConvとTT縮約ではparameter表現と計算経路が異なる。validation 5,000件では微小な数値差に収まり、全sampleで同じ予測classを返した。

ここから速度や演算量の優劣は言えない。02で確認したのは計算経路が違っても推論結果を維持できたことであり、latencyやMACsは測っていない。

### 8.2 conv2の差は後段へ伝わったが予測は変わらなかった

conv2出力RMSEは$5.58\times10^{-7}$、logits RMSEは$3.27\times10^{-6}$、最大logits差は$1.24\times10^{-5}$だった。後段を通したlogitsにも差は残ったが、事前の許容範囲内で、argmaxは5,000件すべて一致した。

conv2出力とlogitsではshapeと値のscaleが異なるため、RMSEの比から「後段が誤差を約6倍へ増幅した」とは結論しない。言えるのは、最終logitsでも許容範囲に収まり、今回のvalidationでは予測classへの影響が観測されなかったことである。

prediction agreementが1.0なのは、denseモデルとTTモデルが同じclassを選んだという意味であり、全sampleが正解という意味ではない。正解ラベルに対するaccuracyは両モデルとも0.9194である。また、accuracyが同じだけではsampleごとの予測一致を保証しないため、agreementとaccuracyを分けて確認した。

### 8.3 exact TTモデルは圧縮結果ではなく比較基準である

今回のexact TTモデルはparameter数が増えたため、圧縮モデルではない。役割は、

> 今回の条件では、rankを打ち切らずにconv2をTT層へ置換しても、元モデルの推論を維持できた。

という検証済みの基準を作ることである。

最初からrank制限して差が出た場合、TT層への接続ミスとrank近似の影響を区別しにくい。exact置換が成功したことで、次のrank制限実験では「rankを削らない段階ですでに壊れていた」という原因を先に除ける。

ただし、02の成功が任意の低rank実装の正しさを自動的に保証するわけではない。各rankでも構造と評価条件を確認する必要がある。

### 8.4 conv2以外のstateを固定した意味

比較対象は、

```text
denseモデル: 同じ前段 → dense conv2 → 同じ後段
TTモデル:    同じ前段 → TTConv2d    → 同じ後段
```

である。conv2以外のparameterやbufferも変わっていた場合、logits差の原因をTT置換へ限定できない。さらに、`tt_model = dense_model`のように同じオブジェクトを共有すると、比較基準のdenseモデル自体を変更してしまう。

別オブジェクトであること、元モデルが未変更であること、置換対象外のstateが一致することを確認したため、今回の差をconv2の表現・計算経路の違いとして解釈できる。これは次のrank制限実験でも維持すべき比較条件である。

## 9. 02から次へ引き継げること

- 学習済みCNNへ接続済みの`TTConv2d`と置換手順
- dense kernelを保持・再構成しないforward経路
- conv2以外のstateを維持する比較方法
- CPU・float32・固定validation 5,000件でのdense基準
- conv2出力RMSE、logits RMSE、最大logits差、agreement、accuracyの集計方法
- exact TTのactual rank$(1,24,32,1)$とparameter数423,242
- exact置換でaccuracy 0.9194、prediction agreement 1.0だったこと

これらは、rank制限前のbaselineである。rank制限モデルにexact TTと同じ微小誤差やprediction agreement 1.0を要求するという意味ではなく、parameter削減との交換関係を測るための比較基盤として使う。

## 10. 02からはまだ言えないこと

- rankを削ってもaccuracyを維持できること
- どのrankが適切で、どれだけparameterを削減できるか
- 学習済みCNN全体でbackwardが正しいこと
- fine-tuningで安定して学習でき、性能が回復すること
- direct TT-Convの方が高速または省メモリであること
- test setでも同じ結果になること
- 別のdevice、dtype、層、tensorization、`groups`でも同様に動くこと

coresとbiasが学習可能parameterとして登録されていることは、学習が成功することの検証ではない。02の評価は`inference_mode()`によるvalidation推論だけである。

## 11. 結論

> Notebook 02では、学習済みFashionMNIST CNNの`conv2`を、dense weightを保持しないexact `TTConv2d`へ置換した。conv2以外のparameter・bufferを維持した条件で、validation 5,000件におけるconv2出力RMSEは$5.58\times10^{-7}$、logits RMSEは$3.27\times10^{-6}$、最大logits差は$1.24\times10^{-5}$となり、prediction agreementは1.0、accuracyは両モデルとも0.9194だった。今回はactual rank$(1,24,32,1)$を保持したためモデルparameter数は421,642から423,242へ増えており、圧縮ではない。02の成果は、rank制限前のTT層接続と推論維持を確認し、次のrank制限実験で近似の影響を切り分ける基準を確立したことである。
