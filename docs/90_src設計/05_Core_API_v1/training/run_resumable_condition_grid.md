# `run_resumable_condition_grid`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/02_full_batch_sweep_update_budget_grid.ipynb`

## 責務

条件グリッドを、集計CSVへ記録済みの条件は検証してスキップし、未実行条件は終了ごとに集計へ追記しながら実行する。

## Signature

```python
run_resumable_condition_grid(
    planned: Sequence[Mapping],
    *,
    summary_path: Path | str,
    models_dir: Path | str,
    fieldnames: Sequence[str],
    setting_fields: Sequence[str],
    run_one: Callable,
    artifacts: Sequence[str] = (
        "initial_model.pt",
        "final_model.pt",
        "best_model.pt",
        "history.csv",
        "local_updates.csv",
        "run_state.json",
    ),
) -> list[dict[str, str]]
```

## 引数

- `planned`: 実行予定。各要素は`condition_id`と`setting_fields`の値を持つ。
- `summary_path`: 集計CSV。同じディレクトリの一時ファイルへ書いてから置き換える。
- `models_dir`: 条件ディレクトリ`models_dir / condition_id`の親。
- `fieldnames`: 集計CSVの列順。
- `setting_fields`: 記録と今回の予定で一致を確認する設定名。
- `run_one`: 未実行条件を1つ実行し、集計行を返す関数。
- `artifacts`: スキップ前に存在を確認するファイル名。

## 戻り値

更新後の集計CSVを読み直した行のリスト。

## 使用場面

- 12条件の本実験を、途中から失わずに再開するとき。

## 処理概要

1. 既存の集計CSVを読む。`condition_id`が重複していれば止まる。
2. 記録済みの条件は、成果物、集計行の設定、`run_state.json`の設定が今回の予定と一致するときだけスキップする。
3. 条件ディレクトリがあるのに集計行がない場合は、削除・上書き・スキップをせず`RuntimeError`で止まる。
4. 未実行条件は`run_one`へ渡す。返った行の設定と成果物を確認してから、集計CSVを原子的に更新する。
5. `run_one`の例外は捕捉しない。それ以前に終了した条件の集計は残る。

## 主なcontract / 注意事項

- 集計の置き換えは、同じディレクトリの一時ファイルを`os.replace`で行う。
- 想定外の例外を数値異常やスキップへ読み替えない。
- smoke runの一括保存はこの関数を使わない。

## 関連API

[[90_src設計/05_Core_API_v1/training/load_condition_summary]]、[[90_src設計/05_Core_API_v1/training/save_tt_linear_full_batch_sweep_run]]、[[90_src設計/05_Core_API_v1/Internal_API]]
