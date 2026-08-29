# `factorize_linear_layer`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

1つの`nn.Linear(in → out)`を、SVDを使って`in → rank → out`の2層Linearへ置換できる形に分解する。

## Signature

```python
factorize_linear_layer(
    layer: nn.Linear,
    rank: int,
) -> nn.Sequential
```

## 引数

- `layer`: 分解対象`nn.Linear`。
- `rank`: 2層間の中間dimension。

## 戻り値

```text
Linear(in_features → rank, bias=False)
→ Linear(rank → out_features, bias=original)
```

の`nn.Sequential`。

## 使用場面

Linearのparameter数・理論MACsを実際に削減するSVD圧縮、圧縮後Fine-tuningの初期化。

## 処理概要

1. **rankが元Linear weightで実現可能か検証する。**  
   最大rankは`min(in_features, out_features)`。範囲外をclipせずエラーにする。
2. **元layerのdevice/dtypeを記録する。**  
   新しい2層を元と同じdevice・dtypeで生成するため。
3. **学習済みweightをdetachしてtruncated SVDする。**  
   Module再構築の初期値を作る処理なので、元Parameterへのautograd graphはここで切る。
4. **1層目`Linear(in → rank)`を作る。**  
   biasは持たせず、weightへ`Vh_r`を配置する。入力特徴をrank次元へ射影する役割。
5. **2層目`Linear(rank → out)`を作る。**  
   weightへ`U_r @ diag(S_r)`を配置する。元layerにbiasがあれば、この出力側の層だけへbiasを持たせる。
6. **weightとbiasを`no_grad`でcopyする。**  
   初期値転写を学習graphへ入れず、新しいParameterをleafとして保つ。
7. **各Parameterの`requires_grad`を引き継ぐ。**  
   元weight/biasがfrozenなら新しい対応Parameterもfrozenにする。
8. **2層`nn.Sequential`を返す。**  
   入力`layer`自体は変更しない。

## 主なcontract / 注意事項

- 元layerは非破壊。
- device/dtype/requires_gradを維持する。
- biasは最終出力側だけへ置く。
- 元weightとのautograd graphはModule初期化時に切る。

## 関連API

`truncated_svd`, `factorize_named_linear`, `factorize_named_layers`, `compressed_linear_macs`
