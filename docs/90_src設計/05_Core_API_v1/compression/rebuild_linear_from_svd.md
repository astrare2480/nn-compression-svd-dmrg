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

- `U_r`: truncated SVDの左特異ベクトル。元weightの出力dimensionとrankを結ぶ成分で、shapeは通常`(out_features, rank)`。
- `S_r`: 残した特異値。各rank成分の強さを持つ1次元Tensorで、shapeは`(rank,)`。
- `Vh_r`: truncated SVDの右特異ベクトル転置。rankと元weightの入力dimensionを結ぶ成分で、shapeは通常`(rank, in_features)`。
- `layer`: 再構築先の外形・属性の基準となる元Linear。`in_features/out_features`、bias、device/dtype、`requires_grad`を参照する。

## 戻り値

`U_r @ diag(S_r) @ Vh_r`で再構成した低rank近似weightを持つ、新しい単一`nn.Linear`。入出力dimensionは元`layer`と同じで、元layer自体やParameter storageは変更・共有しない。

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
