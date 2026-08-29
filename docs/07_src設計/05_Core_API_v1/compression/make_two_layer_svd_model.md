# `make_two_layer_svd_model`

**Stability:** B  
**定義:** `src/nn_compression/compression/mlp_svd.py`

## 責務

現行MLPの`fc1`・`fc2`を、それぞれrank次元の2層factorized Linearへ置換した独立model copyを作る。

## Signature

```python
make_two_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
)
```

## 引数

- `model`: 現行MLP。
- `fc1_rank`, `fc2_rank`: primaryな圧縮rank。
- `r1`, `r2`: historical互換名。

## 戻り値

`fc1`・`fc2`が2層Linearへ置換されたmodel copy。

## 使用場面

MNIST/Fashion-MNIST MLPで実際にparameter/MACsを減らし、Fine-tuningするSVD実験。

## 処理の流れ（日本語）

1. **rank引数をprimary名へ解決する。**  
   primary/legacyの二重指定を拒否し、`fc1`・`fc2`両方のrankを確定する。
2. **model全体を`deepcopy`する。**  
   圧縮modelのFine-tuningでbaselineが変化しないよう独立copyを作る。
3. **元modelの`fc1`を`factorize_linear_layer()`で2層化する。**  
   SVD成分の配置、bias、device/dtype、requires_grad処理は層単位APIへ委譲する。
4. **copy側`fc1`をfactorized Moduleへ差し替える。**
5. **元modelの`fc2`も同じように2層化してcopyへ差し替える。**
6. **`fc3`など圧縮対象外の層はdeepcopyされたまま保持する。**
7. **圧縮model copyを返す。**

### 処理フロー（短縮版）

```text
rank alias解決
→ model deepcopy
→ fc1をfactorize_linear_layer
→ fc2をfactorize_linear_layer
→ fc3等はcopyのまま
→ compressed MLP
```

## 主なcontract / 注意事項

- `fc3`を含めbaselineとのParameter共有なし。
- generic MLP APIではなく既存実験構造向け。

## 関連API

`factorize_linear_layer`, `make_one_layer_svd_model`, `estimate_mlp_macs`
