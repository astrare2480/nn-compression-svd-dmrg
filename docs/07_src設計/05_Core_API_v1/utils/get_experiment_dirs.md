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

project root、手法名、ケース名、実験名、directoryを実際に作成するか。

## 戻り値

`(data_dir, models_dir, results_dir)`。

## 使用場面

SVD / Tucker / TT-MPS / DMRG Notebookで、保存先を同じ階層規則へ揃えるとき。

## 処理の流れ（日本語）

1. **`project_root`を`Path`へ変換する。**
2. **data directoryを`root/data`として決める。**
3. **model保存先を組み立てる。**  
   `root/models/method_name/case_name/experiment_name`。
4. **results保存先を同じ分類軸で組み立てる。**  
   `root/results/method_name/case_name/experiment_name`。
5. **`create=True`なら3directoryを作成する。**  
   `parents=True, exist_ok=True`で途中directoryも含めて安全に作る。
6. **3つの`Path`をtupleで返す。**

### 処理フロー（短縮版）

```text
project_root
→ data = root/data
→ models = root/models/method/case/experiment
→ results = root/results/method/case/experiment
→ create=Trueならmkdir
→ 3 Pathを返す
```

## 主なcontract / 注意事項

Notebook階層と成果物階層で`method → case → experiment`の分類軸を揃えることが目的。実験固有ファイル名まではこの関数で決めない。

## 関連API

`find_project_root`
