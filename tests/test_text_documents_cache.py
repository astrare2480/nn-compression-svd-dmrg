"""文書をまとめてトークン化して覚える encode。"""

from __future__ import annotations

import pytest

from nn_compression.llm.text_documents import cached_token_encoder


class BatchTokenizer:
    def __init__(self):
        self.batch_calls = 0

    def __call__(self, batch, **kwargs):
        assert kwargs["add_special_tokens"] is False and kwargs["padding"] is False
        self.batch_calls += 1
        return {"input_ids": [[len(word) for word in text.split()] for text in batch]}


class EncodeOnlyTokenizer:
    def encode(self, text, add_special_tokens, truncation):
        return [ord(ch) for ch in text]


def test_batches_are_cached_and_progress_is_reported():
    tokenizer = BatchTokenizer()
    seen = []
    encode = cached_token_encoder(tokenizer, ["a bb", "ccc", "d", "ee ff g"], batch_size=3, progress=lambda d, t: seen.append((d, t)))
    assert tokenizer.batch_calls == 2
    assert encode("a bb") == [1, 2]
    assert encode("ee ff g") == [2, 2, 1]
    assert seen == [(3, 4), (4, 4)]
    with pytest.raises(KeyError):
        encode("unknown")
    with pytest.raises(ValueError):
        cached_token_encoder(tokenizer, [], batch_size=0)


def test_falls_back_to_encode():
    encode = cached_token_encoder(EncodeOnlyTokenizer(), ["ab"], batch_size=4)
    assert encode("ab") == [97, 98]


class FailingBatchTokenizer:
    def __call__(self, batch, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def encode(self, text, add_special_tokens, truncation):
        raise AssertionError("1 文書ずつの経路へ逃げてはいけない")


class ShortBatchTokenizer:
    def __call__(self, batch, **kwargs):
        return {"input_ids": [[1]]}


def test_batch_failures_are_reported_not_hidden():
    with pytest.raises(RuntimeError, match="out of memory"):
        cached_token_encoder(FailingBatchTokenizer(), ["a", "b"])
    with pytest.raises(KeyError):
        cached_token_encoder(lambda batch, **kwargs: {"ids": []}, ["a"])
    with pytest.raises(ValueError):
        cached_token_encoder(ShortBatchTokenizer(), ["a", "b"])
