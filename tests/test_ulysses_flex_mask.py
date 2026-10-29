"""Unit tests for the Ulysses flex BlockMask builder and its per-step cache (CPU)."""

from types import SimpleNamespace

import torch

from ringmaster.strategies import ulysses as uly


def _dense(block_mask, batch, seq):
    idx = torch.arange(seq)
    b = torch.arange(batch).view(batch, 1, 1)
    return block_mask.mask_mod(
        b, torch.zeros_like(b), idx.view(1, seq, 1), idx.view(1, 1, seq)
    ).expand(batch, seq, seq)


def _module(layer_type=None, chunk=None):
    cfg = SimpleNamespace(
        layer_types=[layer_type] if layer_type else None,
        attention_chunk_size=chunk,
    )
    return SimpleNamespace(config=cfg, layer_idx=0)


def test_causal_mask_without_segments():
    uly._clear_flex_mask_cache()
    bm = uly.ulysses_flex_block_mask(_module(), None, 2, 8, "cpu")
    dense = _dense(bm, 1, 8)[0]
    assert torch.equal(dense, torch.ones(8, 8, dtype=torch.bool).tril())


def test_sliding_window_mask():
    uly._clear_flex_mask_cache()
    bm = uly.ulysses_flex_block_mask(_module(), None, 1, 8, "cpu", sliding_window=3)
    dense = _dense(bm, 1, 8)[0]
    pos = torch.arange(8)
    expected = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < 3)
    assert torch.equal(dense, expected)


def test_packed_mask_is_causal_and_same_document():
    uly._clear_flex_mask_cache()
    packed = (torch.tensor([0, 3, 8], dtype=torch.int32), 5)
    bm = uly.ulysses_flex_block_mask(_module(), packed, 1, 8, "cpu")
    dense = _dense(bm, 1, 8)[0]
    doc = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    pos = torch.arange(8)
    expected = (pos[:, None] >= pos[None, :]) & (doc[:, None] == doc[None, :])
    assert torch.equal(dense, expected)


def test_non_causal_mask():
    uly._clear_flex_mask_cache()
    bm = uly.ulysses_flex_block_mask(_module(), None, 1, 4, "cpu", causal=False)
    assert bool(_dense(bm, 1, 4).all())


def test_cache_shared_within_step_and_reset_on_new_step():
    uly._clear_flex_mask_cache()
    packed = (torch.tensor([0, 4, 8], dtype=torch.int32), 4)
    a = uly.ulysses_flex_block_mask(_module(), packed, 1, 8, "cpu")
    b = uly.ulysses_flex_block_mask(_module(), packed, 1, 8, "cpu")
    w = uly.ulysses_flex_block_mask(_module(), packed, 1, 8, "cpu", sliding_window=2)
    assert a is b  # layers of one step share the mask
    assert w is not a  # sliding vs global layers get their own
    assert len(uly._FLEX_MASK_CACHE["masks"]) == 2
    new_step = (torch.tensor([0, 2, 8], dtype=torch.int32), 6)
    c = uly.ulysses_flex_block_mask(_module(), new_step, 1, 8, "cpu")
    assert c is not a and len(uly._FLEX_MASK_CACHE["masks"]) == 1


def test_local_heads_slices_sinks():
    sinks = torch.arange(8.0)
    assert torch.equal(uly._local_heads(sinks, 2, 1, 8), torch.arange(4.0, 8.0))
    assert uly._local_heads(None, 2, 0, 8) is None
