"""LLM の読み込み、文書、活性化、perplexity。

モデルの重みを圧縮する計算はここには置かない。
"""

from .activation_capture import accumulate_input_grams, capture_forward_pre_inputs, collect_sequence_inputs
from .hf_loading import load_causal_lm_and_tokenizer, load_download_record, load_wikitext
from .perplexity import sliding_window_perplexity, temporary_linear_weight
from .text_documents import (
    document_sha256,
    hash_overlaps,
    is_wikitext_level1_heading,
    normalize_document_text,
    read_c4_texts,
    restore_wikitext_articles,
    select_token_prefix_sequences,
)

__all__ = [
    "accumulate_input_grams",
    "capture_forward_pre_inputs",
    "collect_sequence_inputs",
    "load_causal_lm_and_tokenizer",
    "load_download_record",
    "load_wikitext",
    "sliding_window_perplexity",
    "temporary_linear_weight",
    "document_sha256",
    "hash_overlaps",
    "is_wikitext_level1_heading",
    "normalize_document_text",
    "read_c4_texts",
    "restore_wikitext_articles",
    "select_token_prefix_sequences",
]
