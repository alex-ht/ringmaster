"""Sequence sharding across the CP group (dense / v1).

Each rank gets a contiguous (or zigzag) slice with GLOBAL position_ids (RoPE is
applied per-shard before attention). Packing/varlen is v2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShardInfo:
    original_seq_len: int
    pad_len: int


# Key under which a batch sharded ahead of the forward (in the collator) carries its
# shard description; the CP pre-hook pops it and skips re-sharding.
CP_SHARD_KEY = "_cp_shard"

_GLOBAL_METADATA_KEYS = (
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
    "cu_seqlens",
    "max_seqlen",
)


def _pad_multiple(seq_len: int, cp_size: int) -> int:
    rem = seq_len % cp_size
    return 0 if rem == 0 else cp_size - rem


def cu_seqlens_from_position_ids(position_ids):
    """Global flash ``cu_seqlens`` + max_seqlen from packed ``position_ids`` (which reset
    to 0 at each document start). Returns None when not packed (<= one segment per row,
    i.e. plain dense/batched attention). Mirrors transformers'
    ``prepare_fa_kwargs_from_position_ids``."""
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=False).view(-1)
    if starts.numel() <= position_ids.shape[0]:
        return None
    cu = torch.cat(
        [
            starts.to(torch.int32),
            torch.tensor([pos.numel()], dtype=torch.int32, device=pos.device),
        ]
    )
    return cu, int(cu.diff().max().item())


def varlen_meta(global_position_ids, total_padded_len):
    """Global document boundaries, including per-row CP padding segments."""
    if global_position_ids is None:
        return None
    rows, length = global_position_ids.shape
    boundaries = []
    packed = False
    for row in range(rows):
        starts = (global_position_ids[row] == 0).nonzero().flatten()
        starts = torch.unique(torch.cat((starts.new_zeros(1), starts)))
        packed |= starts.numel() > 1
        boundaries.append(starts + row * total_padded_len)
        if total_padded_len > length:
            boundaries.append(starts.new_tensor([row * total_padded_len + length]))
    if not packed:
        return None
    cu = torch.cat(
        (*boundaries, boundaries[0].new_tensor([rows * total_padded_len]))
    ).to(torch.int32)
    return cu, int(cu.diff().max().item())


def attention_chunk_size(module):
    config = getattr(module, "config", None)
    layer_types = getattr(config, "layer_types", None)
    layer_idx = getattr(module, "layer_idx", None)
    if layer_types and layer_idx is None and "chunked_attention" in layer_types:
        raise ValueError("Chunked attention requires a layer index")
    if (
        layer_types is None
        or layer_idx is None
        or layer_types[layer_idx] != "chunked_attention"
    ):
        return None
    chunk_size = getattr(config, "attention_chunk_size", None)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Chunked attention requires a positive attention_chunk_size")
    return chunk_size


def attention_segments(
    module, packed, batch_size, seq_len, device, *, sliding_window=None
):
    """Global document/chunk boundaries for an attention layer."""
    chunk_size = attention_chunk_size(module)
    if chunk_size is None and packed is None and not sliding_window:
        return None
    if packed is None:
        boundaries = list(range(0, (batch_size + 1) * seq_len, seq_len))
    else:
        boundaries = packed[0].tolist()
    if chunk_size is not None:
        boundaries = sorted(
            set(boundaries).union(
                position
                for row in range(batch_size)
                for position in range(row * seq_len, (row + 1) * seq_len, chunk_size)
            )
        )
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    return cu, int(cu.diff().max().item())


def _ensure_global_shift_labels(batch):
    """Shift once on the full sequence before sharding: per-shard the boundary target
    (first token of the next rank's shard) is unreachable and would train vs -100."""
    if "shift_labels" not in batch and batch.get("labels") is not None:
        labels = batch["labels"]
        shift = torch.full_like(labels, -100)
        shift[:, :-1] = labels[:, 1:]
        batch["shift_labels"] = shift


def _pad_to(batch, seq_len, pad_len):
    if not pad_len:
        return seq_len
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == seq_len:
            pad_value = -100 if key in ("labels", "shift_labels") else 0
            pad = torch.full(
                (val.size(0), pad_len, *val.shape[2:]),
                pad_value,
                dtype=val.dtype,
                device=val.device,
            )
            if key == "position_ids" and val.ndim == 2:
                pad = val[:, -1:] + torch.arange(
                    1, pad_len + 1, device=val.device, dtype=val.dtype
                )
            batch[key] = torch.cat([val, pad], dim=1)
    return seq_len + pad_len


def _zigzag_shard(batch, cp_rank, cp_size):
    """Zigzag (head_tail) shard: rank holds chunks [r, 2W-1-r] (balances the causal ring)."""
    bsz, seq_len = batch["input_ids"].shape[:2]
    device = batch["input_ids"].device
    pad_len = _pad_multiple(seq_len, 2 * cp_size)

    _ensure_global_shift_labels(batch)
    if batch.get("position_ids") is None:
        batch["position_ids"] = (
            torch.arange(0, seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    total = _pad_to(batch, seq_len, pad_len)
    half = total // (2 * cp_size)
    lo, hi = cp_rank, 2 * cp_size - 1 - cp_rank
    idx = torch.cat(
        [
            torch.arange(lo * half, (lo + 1) * half, device=device),
            torch.arange(hi * half, (hi + 1) * half, device=device),
        ]
    )
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == total:
            batch[key] = val.index_select(1, idx).contiguous()
    return batch, ShardInfo(original_seq_len=seq_len, pad_len=pad_len)


def shard_batch(
    batch: dict[str, torch.Tensor],
    *,
    cp_rank: int,
    cp_size: int,
    load_balance: str | None = None,
) -> tuple[dict[str, torch.Tensor], ShardInfo]:
    """Shard sequence-dim tensors in ``batch`` for this CP rank.

    ``load_balance=None`` derives the layout from the active runtime (the single
    source of truth shared with the attention, so they can't drift); falls back to
    contiguous when no runtime is active.
    """
    if cp_size == 1:
        return batch, ShardInfo(original_seq_len=batch["input_ids"].size(1), pad_len=0)
    if load_balance is None:
        from ringmaster.runtime import maybe_runtime

        rt = maybe_runtime()
        load_balance = rt.shard_load_balance if rt is not None else "contiguous"
    if load_balance == "head_tail":
        return _zigzag_shard(batch, cp_rank, cp_size)

    bsz, seq_len = batch["input_ids"].shape[:2]
    device = batch["input_ids"].device
    pad_len = _pad_multiple(seq_len, cp_size)

    _ensure_global_shift_labels(batch)
    if batch.get("position_ids") is None:
        batch["position_ids"] = (
            torch.arange(0, seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

    total = _pad_to(batch, seq_len, pad_len)
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() > 1 and val.size(1) == total:
            batch[key] = val.chunk(cp_size, dim=1)[cp_rank].contiguous()

    return batch, ShardInfo(original_seq_len=seq_len, pad_len=pad_len)


def prepare_global_batch(batch: dict) -> dict:
    """Normalize a full (unsharded) batch before sharding.

    Validates a 2D right-padded causal ``attention_mask`` and derives packed
    ``position_ids`` from a segment-numbered mask or from global ``cu_seqlens`` /
    ``cu_seq_lens_q`` when they are missing, then drops the mask and the global
    collator metadata, which cannot describe a local shard. Device-agnostic, so it
    runs in a dataloader worker as well as in the forward pre-hook.
    """
    ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    if mask is not None:
        if (
            mask.ndim != 2
            or torch.any(mask < 0)
            or (
                mask.is_floating_point()
                and torch.any(~torch.isfinite(mask) | (mask != mask.round()))
            )
            or torch.any((mask[:, 1:] > 0) & (mask[:, :-1] == 0))
        ):
            raise ValueError(
                "Context parallelism requires a right-padded causal 2D mask"
            )
        if torch.any(mask > 1) and batch.get("position_ids") is None:
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            starts = torch.ones_like(mask, dtype=torch.bool)
            starts[:, 1:] = mask[:, 1:] != mask[:, :-1]
            offsets = torch.where(starts, positions, 0).cummax(dim=1).values
            batch["position_ids"] = positions - offsets
        batch.pop("attention_mask")
    if batch.get("position_ids") is None:
        cu = batch.get("cu_seqlens")
        if cu is None:
            cu = batch.get("cu_seq_lens_q")
        if cu is not None:
            cu = cu.to(device=ids.device, dtype=torch.long)
            if (
                cu.ndim != 1
                or cu.numel() < 2
                or int(cu[0]) != 0
                or int(cu[-1]) != ids.numel()
                or torch.any(cu[1:] <= cu[:-1])
            ):
                raise ValueError("Invalid global packed document boundaries")
            positions = torch.arange(ids.numel(), device=ids.device)
            documents = torch.bucketize(positions, cu[1:], right=True)
            batch["position_ids"] = (positions - cu[documents]).reshape_as(ids)
    for key in _GLOBAL_METADATA_KEYS:
        batch.pop(key, None)
    return batch


def shard_collated_batch(
    batch: dict,
    *,
    cp_rank: int,
    cp_size: int,
    load_balance: str = "contiguous",
) -> dict:
    """Shard a collated batch for ``cp_rank`` ahead of the forward.

    Meant for the collator: the CP ranks of a data-parallel group receive the same
    global batch, so each can cut its own shard in the dataloader worker instead of
    the forward pre-hook doing it on device every step. The result carries a
    :data:`CP_SHARD_KEY` entry (plain Python values only, so it survives pickling and
    device transfer) with the shard geometry and the global packed-document
    boundaries the Ulysses varlen path needs; the pre-hook consumes it and skips
    broadcasting and re-sharding. ``load_balance`` must be the runtime's
    ``shard_load_balance`` (``"contiguous"`` or ``"head_tail"``).
    """
    if cp_size == 1:
        return batch
    if load_balance not in ("contiguous", "none", "head_tail"):
        raise ValueError(
            f"unsupported shard layout for collator-side sharding: {load_balance}"
        )
    batch = prepare_global_batch(batch)
    global_pos = batch.get("position_ids")
    layout = "head_tail" if load_balance == "head_tail" else "contiguous"
    batch, info = shard_batch(
        batch, cp_rank=cp_rank, cp_size=cp_size, load_balance=layout
    )
    total = info.original_seq_len + info.pad_len
    varlen = varlen_meta(global_pos, total)
    batch[CP_SHARD_KEY] = {
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "load_balance": layout,
        "original_seq_len": info.original_seq_len,
        "pad_len": info.pad_len,
        "varlen": None if varlen is None else (varlen[0].tolist(), varlen[1]),
    }
    return batch


def varlen_from_shard_info(shard: dict, device) -> tuple | None:
    """Rebuild the runtime ``varlen`` tuple from a :data:`CP_SHARD_KEY` entry."""
    varlen = shard.get("varlen")
    if varlen is None:
        return None
    cu, maximum = varlen
    return torch.tensor(cu, dtype=torch.int32, device=device), int(maximum)
