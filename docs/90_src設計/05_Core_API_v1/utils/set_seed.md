# `set_seed`

**Stability:** B  
**定義:** `src/nn_compression/utils/seed.py`

## 責務

Python / NumPy / PyTorch / CUDAの主要乱数源と、PyTorchのdeterministic設定をまとめて初期化する。

## Signature

```python
set_seed(seed: int = 0) -> None
```

## 引数

`seed`: 共通seed。

## 戻り値

なし。process-globalな乱数・backend設定を変更する。

## 使用場面

実験開始時に学習・candidate比較の再現性を高めたいとき。

## 処理概要

1. **`CUBLAS_WORKSPACE_CONFIG`を未設定時だけ設定する。**  
   CUDAの決定的演算を使うため、`setdefault(":4096:8")`で既存ユーザー設定は上書きしない。
2. **Python標準`random`のseedを設定する。**
3. **NumPyのglobal RNG seedを設定する。**
4. **PyTorch CPU RNGを`torch.manual_seed()`で設定する。**
5. **利用可能なCUDA device群のseedを`torch.cuda.manual_seed_all()`で設定する。**
6. **PyTorchへdeterministic algorithmを優先するよう指定する。**  
   `warn_only=True`なので、完全対応していない演算では警告しつつ実行を継続できる。
7. **cuDNN benchmarkを無効化する。**  
   入力ごとの高速algorithm探索による非決定性を避ける。
8. **cuDNN deterministicを有効化する。**

## 主なcontract / 注意事項

- 既に作成済みの`torch.Generator`の内部stateはresetしない。
- DataLoader shuffle順を候補間で厳密に管理する場合は`make_torch_generator()`を別途使う。

## 関連API

`make_torch_generator`
