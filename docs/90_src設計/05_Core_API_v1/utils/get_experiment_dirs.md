# `get_experiment_dirs`

**Stability:** B  
**定義:** `src/nn_compression/utils/paths.py`

## 責務

実験成果物のdata / models / results directoryを、`method → case → experiment`という共通分類規則で生成・取得する。

## Signature

```python
get_experiment_dirs(
    project_root: Path | str,
    method_name: str,
    case_name: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> tuple[Path, Path, Path]
```

## 引数

- `project_root`: repository/projectのroot path。`data / models / results`各directoryを組み立てる基準位置。
- `method_name`: 圧縮手法を表す階層名。例: `svd`, `tucker`, `tt_mps`。
- `case_name`: 手法内の対象model/Dataset/ケースを識別する階層名。
- `experiment_name`: 同じ手法・ケース内の個別実験を識別する最下位directory名。
- `create`: `True`なら返却前に必要directoryを実際に作成し、`False`ならpath計算だけを行う。

## 戻り値

`(data_dir, models_dir, results_dir)`の3つの`Path`。

- `data_dir`: `project_root/data`。共通Dataset・入力dataの基準directory。
- `models_dir`: `project_root/models/method_name/case_name/experiment_name`。checkpoint等のmodel成果物保存先。
- `results_dir`: `project_root/results/method_name/case_name/experiment_name`。表・JSON・plot等の実験結果保存先。

## 使用場面

SVD / Tucker / TT-MPS / DMRG Notebookで、保存先を同じ階層規則へ揃えるとき。

## 処理概要

1. **`project_root`を`Path`へ変換する。**
2. **data directoryを`root/data`として決める。**
3. **model保存先を組み立てる。**  
   `root/models/method_name/case_name/experiment_name`。
4. **results保存先を同じ分類軸で組み立てる。**  
   `root/results/method_name/case_name/experiment_name`。
5. **`create=True`なら3directoryを作成する。**  
   `parents=True, exist_ok=True`で途中directoryも含めて安全に作る。
6. **3つの`Path`をtupleで返す。**

## 主なcontract / 注意事項

Notebook階層と成果物階層で`method → case → experiment`の分類軸を揃えることが目的。実験固有ファイル名まではこの関数で決めない。

## 関連API

`find_project_root`
