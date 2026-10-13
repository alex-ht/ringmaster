"""Collator-side sharding must produce exactly what the forward pre-hook produces."""

from types import SimpleNamespace

import pytest
import torch

from ringmaster.shard import (
    CP_SHARD_KEY,
    prepare_global_batch,
    shard_batch,
    shard_collated_batch,
    varlen_from_shard_info,
    varlen_meta,
)
from ringmaster.sp_context import ContextParallelContextManager


def _packed_batch(seed=0, rows=2, length=11):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(5, 50, (rows, length), generator=gen)
    labels = input_ids.clone()
    labels[:, :2] = -100
    # two documents per row, different split points
    position_ids = torch.stack(
        [
            torch.cat([torch.arange(4), torch.arange(length - 4)]),
            torch.cat([torch.arange(7), torch.arange(length - 7)]),
        ]
    )
    return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}


def _hook_path(batch, cp_rank, cp_size, load_balance):
    batch = prepare_global_batch(dict(batch))
    global_pos = batch.get("position_ids")
    sharded, info = shard_batch(
        batch, cp_rank=cp_rank, cp_size=cp_size, load_balance=load_balance
    )
    return sharded, info, varlen_meta(global_pos, info.original_seq_len + info.pad_len)


@pytest.mark.parametrize("load_balance", ["contiguous", "head_tail"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_collator_shard_matches_pre_hook_shard(load_balance, cp_size):
    for cp_rank in range(cp_size):
        want, info, want_varlen = _hook_path(
            _packed_batch(), cp_rank, cp_size, load_balance
        )
        got = shard_collated_batch(
            _packed_batch(), cp_rank=cp_rank, cp_size=cp_size, load_balance=load_balance
        )
        shard = got.pop(CP_SHARD_KEY)
        assert set(got) == set(want)
        for key in want:
            assert torch.equal(got[key], want[key]), key
        assert (shard["original_seq_len"], shard["pad_len"]) == (
            info.original_seq_len,
            info.pad_len,
        )
        assert shard["load_balance"] == load_balance
        cu, maximum = varlen_from_shard_info(shard, torch.device("cpu"))
        assert torch.equal(cu, want_varlen[0]) and maximum == want_varlen[1]


def test_collator_shard_derives_positions_from_segment_mask():
    batch = _packed_batch()
    mask = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], [1] * 7 + [2] * 4])
    batch.pop("position_ids")
    batch["attention_mask"] = mask
    batch["cu_seq_lens_q"] = torch.tensor([0, 4, 11, 18, 22], dtype=torch.int32)
    batch["max_length_q"] = 7

    got = shard_collated_batch(batch, cp_rank=0, cp_size=2, load_balance="contiguous")

    assert "attention_mask" not in got and "cu_seq_lens_q" not in got
    assert "max_length_q" not in got
    want, _, _ = _hook_path(_packed_batch(), 0, 2, "contiguous")
    assert torch.equal(got["position_ids"], want["position_ids"])


def test_collator_shard_single_rank_is_identity():
    batch = _packed_batch()
    assert shard_collated_batch(dict(batch), cp_rank=0, cp_size=1) is not None
    assert CP_SHARD_KEY not in shard_collated_batch(dict(batch), cp_rank=0, cp_size=1)


def test_unsupported_layout_is_rejected():
    with pytest.raises(ValueError, match="unsupported shard layout"):
        shard_collated_batch(_packed_batch(), cp_rank=0, cp_size=2, load_balance="ptrr")


class _Model(torch.nn.Module):
    def forward(self, input_ids=None, labels=None, position_ids=None, **kwargs):
        return SimpleNamespace(
            seen=dict(kwargs, input_ids=input_ids, position_ids=position_ids)
        )


def _context(cp_rank, cp_size, load_balance):
    ctx = ContextParallelContextManager([_Model()], None, load_balance=load_balance)
    ctx.cp_size, ctx.cp_rank = cp_size, cp_rank  # no process group in a CPU test
    return ctx


def test_pre_hook_accepts_pre_sharded_batch_without_resharding(monkeypatch):
    ctx = _context(1, 2, "contiguous")
    sharded = shard_collated_batch(_packed_batch(), cp_rank=1, cp_size=2)
    import ringmaster.sp_context as sp

    monkeypatch.setattr(
        sp, "broadcast_batch", lambda *a, **k: pytest.fail("broadcast ran")
    )
    monkeypatch.setattr(sp, "shard_batch", lambda *a, **k: pytest.fail("re-sharded"))
    runtime = SimpleNamespace(varlen="stale")
    monkeypatch.setattr(sp, "maybe_runtime", lambda: runtime)
    hook = ctx._make_pre_hook(["input_ids", "labels", "position_ids"])
    model = ctx.models[0]
    model.eval()

    _, kwargs = hook(model, (), dict(sharded))

    assert CP_SHARD_KEY not in kwargs
    assert torch.equal(kwargs["input_ids"], sharded["input_ids"])
    assert ctx._orig_seq_len == 11 and ctx._pad_len == 1
    want = varlen_from_shard_info(sharded[CP_SHARD_KEY], torch.device("cpu"))
    assert torch.equal(runtime.varlen[0], want[0]) and runtime.varlen[1] == want[1]


def test_pre_hook_rejects_mismatched_shard():
    ctx = _context(0, 2, "contiguous")
    sharded = shard_collated_batch(_packed_batch(), cp_rank=1, cp_size=2)
    hook = ctx._make_pre_hook(["input_ids", "labels", "position_ids"])
    ctx.models[0].eval()
    with pytest.raises(ValueError, match="does not match this context"):
        hook(ctx.models[0], (), dict(sharded))
