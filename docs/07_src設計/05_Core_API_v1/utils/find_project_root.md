# `find_project_root`

**Stability:** B  
**定義:** `src/nn_compression/utils/paths.py`

## 責務

開始pathから上位directoryを順に辿り、`.git`が存在する最初のdirectoryをproject rootとして返す。

## Signature

```python
find_project_root(
    start_path: Path | str,
) -> Path
```

## 引数

`start_path`: 探索開始path。

## 戻り値

project rootの絶対`Path`。

## 使用場面

Notebookのcwdが深い階層でも、data/models/results等をrepository root基準で解決したいとき。

## 処理の流れ（日本語）

1. **`start_path`を`Path`へ変換し、`resolve()`する。**  
   相対pathや`..`を解消して探索基準を明確にする。
2. **開始directory自身を最初の候補にする。**
3. **続けて全parent directoryを近い順に候補へ並べる。**
4. **各候補で`.git`の存在を確認する。**
5. **最初に`.git`が見つかったdirectoryをproject rootとして返す。**
6. **最後まで見つからなければ`FileNotFoundError`を送出する。**

### 処理フロー（短縮版）

```text
start_path
→ resolve
→ self → parent → parent ...
→ .git存在確認
→ 最初の一致をreturn
→ 無ければFileNotFoundError
```

## 主なcontract / 注意事項

`.git`を持たない単純なzip展開物等ではproject rootを特定できない。

## 関連API

`get_experiment_dirs`
