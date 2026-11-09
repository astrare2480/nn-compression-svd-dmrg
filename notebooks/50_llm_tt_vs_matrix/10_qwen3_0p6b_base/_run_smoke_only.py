"""検算のやり直しを省き、保存済みの重みと Gram から動作確認だけ実行する。"""

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import nbformat

NOTEBOOK = Path(__file__).with_name("01_single_layer_pilot.ipynb")


def cell_source(notebook, index):
    return notebook.cells[index].source


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__"}
    print("cell 2", flush=True)
    exec(compile(cell_source(notebook, 2), "cell_2", "exec"), namespace)
    print("load saved weights and gram", flush=True)
    import torch

    weights = torch.load(namespace["EXPERIMENT_DATA"] / "weights_float64.pt", weights_only=False)
    namespace["W"] = weights["W"]
    namespace["W_GATE"] = weights["W_gate"]
    grams = torch.load(namespace["EXPERIMENT_DATA"] / "calibration_grams.pt", weights_only=False)
    namespace["G"] = grams["total"]
    namespace["N_CALIBRATION_TOKENS"] = int(grams["n_tokens"])
    namespace["G_LAMBDA"] = namespace["regularize_gram"](namespace["G"], namespace["LAMBDA"])

    class _Layer:
        in_features = 1024
        out_features = 3072

    namespace["UP"] = _Layer()
    print("perplexity ids", flush=True)
    from nn_compression.llm.hf_loading import load_tokenizer, load_wikitext

    tokenizer = load_tokenizer(namespace["RECORD"])
    validation = load_wikitext(namespace["RECORD"], "wikitext-2-raw-v1", "validation")
    text = "\n\n".join(validation["text"])
    namespace["PERPLEXITY_IDS"] = torch.tensor(
        tokenizer.encode(text, add_special_tokens=False, truncation=False),
        dtype=torch.long,
    )
    print("perplexity tokens", int(namespace["PERPLEXITY_IDS"].numel()), flush=True)
    print("cell 8", flush=True)
    exec(compile(cell_source(notebook, 8), "cell_8", "exec"), namespace)
    print("cell 10", flush=True)
    exec(compile(cell_source(notebook, 10), "cell_10", "exec"), namespace)
    print("cell 16", flush=True)
    exec(compile(cell_source(notebook, 16), "cell_16", "exec"), namespace)
    print("cell 24", flush=True)
    exec(compile(cell_source(notebook, 24), "cell_24", "exec"), namespace)
    print("SMOKE_ONLY_FINISHED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
