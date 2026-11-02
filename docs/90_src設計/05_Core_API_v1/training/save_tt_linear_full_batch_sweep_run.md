# `save_tt_linear_full_batch_sweep_run`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

初期モデル、最終モデル、bestモデル、往復履歴、局所更新ログ、設定と停止状態をディレクトリへ書く。

## Signature

```python
save_tt_linear_full_batch_sweep_run(
    directory: Path | str,
    model: nn.Module,
    result: TTLinearFullBatchSweepResult,
    config: Mapping,
) -> None
```

## 引数

- `directory`: 出力先。無ければ作る。
- `model`: いまの実モデル。最後に成功した局所更新までを最終モデルとして保存する。
- `result`: 学習結果。初期stateとbest stateを別ファイルにする。
- `config`: JSONにできる設定。`layer_name`、`out_modes`、`in_modes`を含む。

## 戻り値

戻り値はない。`initial_model.pt`、`final_model.pt`、`best_model.pt`、`history.csv`、`local_updates.csv`、`run_state.json`を書く。

## 使用場面

- スイープ学習のあと、再読み込みと比較するために成果物を残すとき。

## 処理概要

1. 初期・最終・bestのstate_dictを、modeと層名つきで別ファイルに保存する。
2. 往復履歴と局所ログをCSVにする。局所ログには TT と通常parameterの norm、勾配 norm、相対更新量を含める。これらは診断値であり、空欄は失敗行である。
3. 停止理由、完了往復、成功回数、中心、最終rankをJSONにする。

## 主なcontract / 注意事項

- 最終モデルをbestの読み込みで上書きしない。bestは別ファイルである。
- 学習の再開に必要なoptimizer stateは保存しない。
- `config`にTensorやdeviceオブジェクトを入れると`TypeError`になる。

## 関連API

[[90_src設計/05_Core_API_v1/training/load_tt_linear_sweep_checkpoint]]、[[90_src設計/05_Core_API_v1/training/TTLinearFullBatchSweepResult]]、[[90_src設計/05_Core_API_v1/Internal_API]]
