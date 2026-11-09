"""較正・検証の Gram と float64 の重みを書く。

本番:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_prepare_data.py

数分の確認:
  python notebooks/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/04_main_experiment_prepare_data.py --limit 2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

from nn_compression.llm.qwen3_main_run import prepare_experiment_data
from nn_compression.utils.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = prepare_experiment_data(find_project_root(Path(__file__)), limit=args.limit, device=args.device)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
