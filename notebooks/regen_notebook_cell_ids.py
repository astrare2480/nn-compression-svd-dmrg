"""
ノートブックのセル ID を振り直す。

ノートをコピーするとセル ID もコピーされ、Cursor / Jupyter で
「実行できない」状態になることがある。そのときの修復用。

使い方:
  python regen_notebook_cell_ids.py 06_mnist_rank_accuracy_tradeoff.ipynb
  python regen_notebook_cell_ids.py *.ipynb
  python regen_notebook_cell_ids.py 20_fashion_mnist/*.ipynb
  python regen_notebook_cell_ids.py --check          # 重複チェックのみ
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path


def regen(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    for cell in cells:
        cell["id"] = uuid.uuid4().hex[:8]
    path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(cells)


def collect_ids(paths: list[Path]) -> dict[str, list[str]]:
    """cell_id -> [notebook, ...]"""
    owner: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            cid = cell.get("id")
            if cid:
                owner[cid].append(path.name)
    return owner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jupyter ノートのセル ID を振り直す / 重複チェックする"
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="対象の .ipynb（省略時はカレントの *.ipynb）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="振り直さず、ノート間のセル ID 重複だけ報告する",
    )
    args = parser.parse_args()

    # PowerShell だと *.ipynb が展開されないので、自分で glob する
    # 注意: Path.name だけだと親ディレクトリが落ちるので、str(item) 全体を使う
    raw = args.notebooks or [Path("*.ipynb")]
    paths: list[Path] = []
    for item in raw:
        pattern = str(item).replace("\\", "/")
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(item)
    paths = [p.resolve() for p in paths if p.suffix == ".ipynb" and p.is_file()]

    if not paths:
        raise SystemExit("対象の .ipynb がありません")

    if args.check:
        owner = collect_ids(paths)
        dups = {cid: names for cid, names in owner.items() if len(names) > 1}
        if not dups:
            print(f"OK: {len(paths)} ノート間でセル ID の重複なし")
            return
        print(f"重複あり: {len(dups)} 個の ID")
        for cid, names in sorted(dups.items(), key=lambda x: (-len(x[1]), x[0])):
            print(f"  {cid}: {', '.join(names)}")
        raise SystemExit(1)

    for path in paths:
        n = regen(path)
        print(f"rewrote {n:3d} cell ids  <- {path.name}")


if __name__ == "__main__":
    main()
