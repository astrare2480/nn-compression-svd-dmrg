# `set_seed`

**Stability:** B  
**定義:** `src/nn_compression/utils/seed.py`

## 責務

Python/NumPy/PyTorch/CUDAの主要乱数源とdeterministic設定をまとめて初期化する。

## Signature

```python
set_seed(seed: int = 0) -> None
```

## 引数

`seed`: 共通seed。

## 戻り値

なし。

## 使用場面

実験開始時に候補比較・学習の再現性を高めるとき。

## ざっくりした処理

```text
CUBLAS_WORKSPACE_CONFIG設定
→ random.seed
→ np.random.seed
→ torch.manual_seed / cuda.manual_seed_all
→ deterministic algorithms
→ cudnn benchmark=False / deterministic=True
```

## 主なcontract / 注意事項

既に作成済みDataLoader専用Generatorのstateはresetしない。

## 関連API

`make_torch_generator`
