# `count_parameters`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

modelが保持するすべての`torch.nn.Parameter`の総要素数を数える。

## Signature

```python
count_parameters(model) -> int
```

## 引数

`model`: `nn.Module`。

## 戻り値

`model.parameters()`に含まれる各Parameterの`numel()`合計。

## 使用場面

baseline/compressed modelの保存Parameter量を比較するとき、圧縮率recordを作るとき。

## 処理の流れ（日本語）

1. **`model.parameters()`でParameterを順に取得する。**
2. **各Parameterの`numel()`を求める。**  
   shapeに関係なく、そのParameterが保持するscalar数へ変換する。
3. **全Parameterの要素数を合計する。**
4. **整数として返す。**

### 処理フロー（短縮版）

```text
model.parameters()
→ 各parameter.numel()
→ sum
→ total parameters
```

## 主なcontract / 注意事項

- `requires_grad=False`のParameterも数える。これは「学習可能Parameter数」ではなく「modelが保持するParameter総数」の指標。
- bufferは含めない。

## 関連API

`parameters_reduction`, `collect_compression_metrics`
