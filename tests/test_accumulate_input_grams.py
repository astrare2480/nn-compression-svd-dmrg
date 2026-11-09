"""順伝播中に入力 Gram を足す関数が、保存した活性化の gram_matrix と一致すること。"""

from __future__ import annotations

import torch
from torch import nn

from nn_compression.compression.metric_low_rank import gram_matrix
from nn_compression.llm.activation_capture import accumulate_input_grams, collect_sequence_inputs


class _TwoLinears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.up = nn.Linear(4, 5, bias=False)
        self.down = nn.Linear(5, 3, bias=False)

    def forward(self, token_ids: torch.Tensor, **_unused) -> torch.Tensor:
        hidden = self.up(self.embed(token_ids))
        return self.down(hidden)


def test_accumulated_grams_match_gram_matrix_and_do_not_keep_activations():
    generator = torch.Generator().manual_seed(0)
    model = _TwoLinears()
    with torch.no_grad():
        model.embed.weight.copy_(torch.randn(model.embed.weight.shape, generator=generator, dtype=torch.float32))
        model.up.weight.copy_(torch.randn(model.up.weight.shape, generator=generator, dtype=torch.float32))
        model.down.weight.copy_(torch.randn(model.down.weight.shape, generator=generator, dtype=torch.float32))
    sequences = [
        torch.randint(0, 11, (5,), generator=generator),
        torch.randint(0, 11, (3,), generator=generator),
    ]
    document_ids = ["a", "b"]
    accumulated = accumulate_input_grams(
        model, {"up": model.up, "down": model.down}, sequences, device="cpu"
    )
    for name, module in (("up", model.up), ("down", model.down)):
        captured = collect_sequence_inputs(model, module, sequences, document_ids, device="cpu")
        stacked = torch.cat([record["activation"] for record in captured], dim=0)
        assert torch.equal(accumulated[name]["total"], gram_matrix(stacked))
        assert accumulated[name]["n_tokens"] == stacked.shape[0]
    assert model.training is True
