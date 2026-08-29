# `make_torch_generator`

**Stability:** B  
**定義:** `src/nn_compression/utils/seed.py`

## 責務

DataLoader shuffleや`random_split`などへ個別に渡す、seed固定の`torch.Generator`を新規作成する。

## Signature

```python
make_torch_generator(
    seed: int = 0,
) -> torch.Generator
```

## 引数

`seed`: 新Generatorへ設定するseed。

## 戻り値

`manual_seed(seed)`済みの新しい`torch.Generator`。

## 使用場面

split乱数とtraining shuffle乱数を分離したいとき、candidate間でmini-batch順を再現したいとき。

## 処理の流れ（日本語）

1. **新しい`torch.Generator()`を生成する。**  
   global PyTorch RNGとは別の独立したstateを持つ。
2. **`generator.manual_seed(seed)`を呼ぶ。**
3. **seed設定済みGeneratorを返す。**
4. **以後、そのGeneratorを使う処理が乱数を生成するたびに内部stateが進む。**  
   関数が毎回自動resetするわけではない。

### 処理フロー（短縮版）

```text
new torch.Generator
→ manual_seed(seed)
→ independent RNG generator
```

## 主なcontract / 注意事項

`next(iter(loader))`等でshuffle付きloaderを先読みすると、そのloaderに渡したGenerator stateが消費される。benchmark用batch取得ではこの点を意識する。

## 関連API

`set_seed`, `take_inference_batch`, `split_fashion_mnist_dataset`
