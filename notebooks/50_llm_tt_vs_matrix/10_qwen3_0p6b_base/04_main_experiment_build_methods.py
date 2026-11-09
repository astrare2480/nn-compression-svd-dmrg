"""42 セルの M1〜M4r を作る。

本番:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_build_methods.py

1 セル:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_build_methods.py --only L14_k_25
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

from nn_compression.llm.qwen3_main_run import build_requested_methods
from nn_compression.utils.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    rows = build_requested_methods(find_project_root(Path(__file__)), only=args.only)
    print(f"cells={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
