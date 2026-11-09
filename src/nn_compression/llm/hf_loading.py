"""``download_record.json`` に記録した revision で、モデルと WikiText を読む。

読み込みはオフライン（ローカルのスナップショットとキャッシュ）だけを使う。
WikiText の ``test`` split は、ここで拒否する。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def load_download_record(path: Path | str) -> dict:
    """ダウンロード記録の JSON を辞書で返す。"""
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise TypeError("download_record.json の最上位は object である必要があります。")
    return record


def _prepare_offline_env(record: dict) -> None:
    hf_home = record.get("HF_HOME")
    if isinstance(hf_home, str) and hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def load_tokenizer(record: dict):
    """記録されたスナップショットから tokenizer だけを読む。

    モデル本体は読まない。``local_files_only=True``。
    """
    _prepare_offline_env(record)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(record["model"]["path"], local_files_only=True)


def load_causal_lm_and_tokenizer(record: dict, *, dtype: torch.dtype, device: torch.device | str):
    """記録されたスナップショットから因果言語モデルと tokenizer を読む。

    transformers 5 の ``from_pretrained(..., dtype=...)`` を使う。
    ``torch_dtype`` は使わない。戻り値は ``(model, tokenizer)``。
    ``device`` へ移してから返す。
    """
    _prepare_offline_env(record)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = record["model"]["path"]
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model.to(device)
    return model, tokenizer


def load_wikitext(record: dict, config: str, split: str):
    """記録された revision の WikiText を 1 split だけ読む。

    ``split="test"`` は ``ValueError``。使う config は
    ``wikitext-103-raw-v1`` と ``wikitext-2-raw-v1`` だけ。
    """
    if split == "test":
        raise ValueError("WikiText の test split は読み込まない。")
    if config not in {"wikitext-103-raw-v1", "wikitext-2-raw-v1"}:
        raise ValueError(f"未対応の WikiText config です: {config}")
    if split not in {"train", "validation"}:
        raise ValueError(f"split は train または validation です: {split}")
    _prepare_offline_env(record)
    from datasets import load_dataset

    return load_dataset(
        "Salesforce/wikitext",
        config,
        split=split,
        revision=record["wikitext"]["revision"],
    )
