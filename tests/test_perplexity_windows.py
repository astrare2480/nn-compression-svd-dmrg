"""窓を自分で回す部品が ``sliding_window_perplexity`` と同じ NLL になること。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nn_compression.llm.perplexity import (
    shifted_nll_sum,
    sliding_window_perplexity,
    sliding_windows,
    window_target_labels,
)


class TinyCausalLM(nn.Module):
    """Hugging Face の因果 LM と同じ呼び方（labels で 1 トークンずらした平均 loss）をする小さいモデル。"""

    def __init__(self, vocab: int = 11) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, input_ids, labels=None, use_cache=False):
        logits = self.head(torch.tanh(self.embed(input_ids)).cumsum(dim=1))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_windows_cover_every_token_once():
    windows = sliding_windows(23, max_length=8, stride=5)
    assert windows[0] == (0, 8, 8)
    assert windows[-1][1] == 23
    assert sum(target for _b, _e, target in windows) == 23
    assert sliding_windows(4, max_length=8, stride=5) == [(0, 4, 4)]
    with pytest.raises(ValueError):
        sliding_windows(10, max_length=1, stride=1)


@pytest.mark.parametrize("n_tokens", [2, 9, 20, 41])
@pytest.mark.parametrize("stride", [1, 3, 4])
def test_every_token_is_scored_exactly_once(n_tokens, stride):
    covered = []
    for begin, end, target in sliding_windows(n_tokens, max_length=4, stride=stride):
        assert 1 <= target <= end - begin
        covered.extend(range(end - target, end))
    assert covered == list(range(n_tokens))


def test_inputs_that_would_skip_tokens_are_rejected():
    # stride > max_length だと窓の間に評価しないトークンが残る。
    with pytest.raises(ValueError):
        sliding_windows(20, max_length=4, stride=8)
    with pytest.raises(ValueError):
        sliding_windows(1, max_length=4, stride=2)
    chunk = torch.arange(6).reshape(1, 6)
    for target in (0, -1, 7):
        with pytest.raises(ValueError):
            window_target_labels(chunk, target)
    with pytest.raises(ValueError):
        window_target_labels(torch.arange(6), 2)


@pytest.mark.parametrize("n_tokens", [7, 23, 40])
def test_manual_windows_match_sliding_window_perplexity(n_tokens):
    model = TinyCausalLM().eval()
    ids = torch.randint(0, 11, (1, n_tokens), generator=torch.Generator().manual_seed(n_tokens))
    total = 0.0
    counted = 0
    with torch.no_grad():
        for begin, end, target in sliding_windows(n_tokens, max_length=8, stride=5):
            chunk = ids[:, begin:end]
            value, scored = shifted_nll_sum(model(chunk).logits, window_target_labels(chunk, target))
            total += value
            counted += scored
    reference = sliding_window_perplexity(model, ids, max_length=8, stride=5)
    assert counted == reference["n_tokens"]
    assert total / counted == pytest.approx(reference["mean_nll"], rel=1e-6)


def test_labels_and_empty_window():
    chunk = torch.arange(6).reshape(1, 6)
    assert window_target_labels(chunk, 2).tolist() == [[-100, -100, -100, -100, 4, 5]]
    assert window_target_labels(chunk, 6).tolist() == chunk.tolist()
    with pytest.raises(RuntimeError):
        shifted_nll_sum(torch.zeros(1, 3, 4), torch.full((1, 3), -100))
