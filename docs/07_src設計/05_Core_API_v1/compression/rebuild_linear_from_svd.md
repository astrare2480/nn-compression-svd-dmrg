# `rebuild_linear_from_svd`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

truncated SVD成分から、元layerと同じ入出力shapeを持つ単一`nn.Linear`を再構築する。層数・Parameter shapeは圧縮前と同じなので、parameter削減ではなく低rank近似誤差だけを評価するためのAPI。

## Signature

```python
rebuild_linear_from_svd(
    U_r,
    S_r,
    Vh_r,
    layer: nn.Linear,
) -> nn.Linear
```

## 引数

- `U_r`, `S_r`, `Vh_r`: truncated SVD成分。
- `layer`: shape・bias・device/dtype・trainabilityの基準となる元Linear。

## 戻り値

元と同じ `in_features/out_features` の新しい`nn.Linear`。

## 使用場面

「低rank近似したweightに置き換えた場合のaccuracy低下」を、2層化によるparameter削減とは切り分けて確認するとき。

## 処理概要

1. **SVD成分から近似weightを再構成する。**  
   `W_r = U_r @ diag(S_r) @ Vh_r`を計算し、元Linearと同shapeの低rank近似行列を作る。
2. **元layerのdeviceとdtypeを取得する。**  
   新しいModuleをCPU/float32へ勝手に戻さず、元layerと同じ配置・精度で作るため。
3. **元と同じ入出力dimensionのLinearを新規作成する。**  
   元layerにbiasがある場合だけ、新layerにもbiasを持たせる。
4. **近似weightを新layerへcopyする。**  
   `torch.no_grad()`内でParameterへ値をコピーし、copy操作自体をautograd graphへ入れない。
5. **biasを必要に応じてコピーする。**  
   元biasが存在する場合だけ、その値を新layerへ引き継ぐ。
6. **`requires_grad`を元layerから引き継ぐ。**  
   frozen layerを意図せずtrainableにしない。
7. **新しい独立layerを返す。**  
   元layer自体は変更しない。

## 主なcontract / 注意事項

- 新しいlayerを返し、入力layerは変更しない。
- 元biasがある場合のみbiasを持つ。
- device/dtype/requires_gradを維持する。

## 関連API

`truncated_svd`, `factorize_linear_layer`, `make_one_layer_svd_model`
