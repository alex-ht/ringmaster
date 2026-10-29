"""CPU (gloo, 2 ranks) parity: Ulysses + flex_attention == single-process eager.

After the Ulysses all-to-all every rank holds the full sequence for a head subset,
and the flex inner runs with a BlockMask built over the global positions (causal,
same-document for packed batches, per-layer sliding window). These tests check
forward logits and parameter gradients against an eager single-process reference
for tiny Llama (MHA, unpacked and packed), Mistral (GQA, sliding window) and Gemma-4
(hybrid sliding/global layers with different head dims, KV sharing, GQA).

Runs eagerly on CPU; set TORCHDYNAMO_DISABLE=1 if no C++ compiler is available.
PyTorch's FlexAttention has no CPU backward, so each case runs twice:

* real ``flex_attention`` forward (no grad) -> checks the BlockMask's block sparsity
  and the routing through the registered flex kernel;
* forward + backward with a dense reference kernel registered as
  ``flex_attention`` that evaluates the SAME BlockMask's ``mask_mod`` over all
  (q, kv) pairs -> checks the mask semantics and the gradients end to end.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch

SEQ = 32
DOC_LENS = (10, 14, 8)  # packed case: three documents in one row


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build(kind: str):
    torch.manual_seed(0)
    common = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "attention_dropout": 0.0,
    }
    if kind.startswith("llama"):
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=8,
            **common,
        )
        return LlamaForCausalLM(cfg), 8
    if kind.startswith("mistral"):
        from transformers import MistralConfig, MistralForCausalLM

        cfg = MistralConfig(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            sliding_window=8,
            **common,
        )
        return MistralForCausalLM(cfg), 4
    if kind.startswith("gemma4"):
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        cfg = Gemma4TextConfig(
            num_hidden_layers=6,
            layer_types=["sliding_attention", "full_attention"] * 3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            sliding_window=8,
            num_kv_shared_layers=2,
            vocab_size_per_layer_input=128,
            hidden_size_per_layer_input=8,
            final_logit_softcapping=30.0,
            **common,
        )
        return Gemma4ForCausalLM(cfg), 2
    raise ValueError(kind)


def _inputs(packed: bool):
    torch.manual_seed(123)
    ids = torch.randint(0, 128, (1, SEQ))
    if packed:
        pos = torch.cat([torch.arange(n) for n in DOC_LENS]).unsqueeze(0)
    else:
        pos = torch.arange(SEQ).unsqueeze(0)
    weight = torch.randn(1, SEQ, 128)  # fixed cotangent: loss = sum(logits * weight)
    return ids, pos, weight


def _reference(model, ids, pos, weight, packed):
    """Eager single process; packed documents are run independently (exact)."""
    model.set_attn_implementation("eager")
    model.zero_grad(set_to_none=True)
    if packed:
        outs, start = [], 0
        for n in DOC_LENS:
            outs.append(
                model(
                    input_ids=ids[:, start : start + n],
                    position_ids=pos[:, start : start + n],
                ).logits
            )
            start += n
        logits = torch.cat(outs, dim=1)
    else:
        logits = model(input_ids=ids, position_ids=pos).logits
    (logits * weight).sum().backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return logits.detach(), grads


def _dense_flex_reference(
    module, query, key, value, attention_mask, scaling=None, softcap=None, **_
):
    """Differentiable stand-in for HF flex_attention_forward: evaluates the BlockMask's
    mask_mod densely. Same signature/return as the HF integration."""
    block_mask = attention_mask
    bsz, n_q, seq, dim = query.shape
    idx = torch.arange(seq, device=query.device)
    b = torch.arange(bsz, device=query.device).view(bsz, 1, 1)
    allowed = block_mask.mask_mod(
        b, torch.zeros_like(b), idx.view(1, seq, 1), idx.view(1, 1, seq)
    ).expand(bsz, seq, seq)
    if key.shape[1] != n_q:
        key = key.repeat_interleave(n_q // key.shape[1], dim=1)
        value = value.repeat_interleave(n_q // value.shape[1], dim=1)
    scale = dim**-0.5 if scaling is None else scaling
    scores = (query @ key.transpose(-1, -2)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
    out = scores.softmax(dim=-1) @ value
    return out.transpose(1, 2).contiguous(), None


def _worker(rank, world, kind, port, out_q):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import ringmaster as rm
        from ringmaster.shard import shard_batch, varlen_meta
        from ringmaster.strategies import ulysses as uly

        packed = kind.endswith("_packed")
        model, n_kv = _build(kind)
        model = model.float().train()
        ids, pos, weight = _inputs(packed)
        ref_logits, ref_grads = _reference(model, ids, pos, weight, packed)

        rt = rm.setup(
            rm.RingmasterConfig(size=world, backend=rm.Backend.ULYSSES),
            num_kv_heads=n_kv,
            device_mesh=None,
            device_type="cpu",
            inner_attn="flex_attention",
        )
        model.set_attn_implementation(rt.attn_implementation)
        batch, _ = shard_batch(
            {"input_ids": ids.clone(), "position_ids": pos.clone()},
            cp_rank=rt.cp_rank,
            cp_size=rt.cp_size,
        )
        rt.varlen = varlen_meta(pos, SEQ)
        local = SEQ // world
        sl = slice(rank * local, (rank + 1) * local)

        # (a) real flex_attention forward (CPU flex has no backward). Wrap the
        # registered kernel the way axolotl's flex_attn_kernel_options does, to check
        # the Ulysses path dispatches through the registry with a BlockMask.
        from torch.nn.attention.flex_attention import BlockMask
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        real_flex = ALL_ATTENTION_FUNCTIONS["flex_attention"]
        seen = []

        def recording_flex(module, q, k, v, mask, **kw):
            seen.append(isinstance(mask, BlockMask))
            kw.setdefault("kernel_options", {"num_stages": 1, "bwd_num_stages": 1})
            return real_flex(module, q, k, v, mask, **kw)

        ALL_ATTENTION_FUNCTIONS.register("flex_attention", recording_flex)
        uly._clear_flex_mask_cache()
        try:
            with torch.no_grad():
                flex_logits = model(**batch).logits
        finally:
            ALL_ATTENTION_FUNCTIONS.register("flex_attention", real_flex)
        assert seen and all(seen), f"flex calls / BlockMask: {seen}"
        flex_fwd_err = (flex_logits - ref_logits[:, sl]).abs().max().item()
        n_masks = len(uly._FLEX_MASK_CACHE["masks"])

        # (b) forward + backward through the same BlockMask with a dense kernel
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", _dense_flex_reference)
        try:
            uly._clear_flex_mask_cache()
            model.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            (logits * weight[:, sl]).sum().backward()
        finally:
            ALL_ATTENTION_FUNCTIONS.register("flex_attention", real_flex)

        fwd_err = (logits.detach() - ref_logits[:, sl]).abs().max().item()
        bwd_err = 0.0
        bwd_scale = 0.0
        for name, p in model.named_parameters():
            if name not in ref_grads:
                continue
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            g = g.detach().clone()
            dist.all_reduce(g)
            bwd_err = max(bwd_err, (g - ref_grads[name]).abs().max().item())
            bwd_scale = max(bwd_scale, ref_grads[name].abs().max().item())
        out_q.put((rank, flex_fwd_err, fwd_err, bwd_err, bwd_scale, n_masks, None))
        rm.teardown()
    except Exception:  # noqa: BLE001  # surfaced through the queue
        import traceback

        out_q.put((rank, None, None, None, None, None, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


# kind -> expected number of distinct BlockMasks per step (shared across layers)
CASES = {
    "llama": 1,
    "llama_packed": 1,
    "mistral": 1,
    "mistral_packed": 1,
    "gemma4": 2,  # sliding + global
    "gemma4_packed": 2,
}


@pytest.mark.parametrize("kind", list(CASES))
def test_ulysses_flex_matches_eager(kind):
    import torch.multiprocessing as mp

    world = 2
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, kind, port, out_q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=300) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)

    for rank, flex_err, fwd_err, bwd_err, bwd_scale, n_masks, err in sorted(
        results, key=lambda r: r[0]
    ):
        assert err is None, f"rank {rank}:\n{err}"
        print(
            f"{kind} rank{rank} flex_fwd_max_err={flex_err:.3e} "
            f"dense_fwd_max_err={fwd_err:.3e} "
            f"bwd_max_err={bwd_err:.3e} (max|grad|={bwd_scale:.3e}) masks={n_masks}"
        )
        assert flex_err < 1e-4, f"{kind} rank {rank} flex forward err {flex_err}"
        assert fwd_err < 1e-4, f"{kind} rank {rank} forward err {fwd_err}"
        assert bwd_err < 1e-4 * max(1.0, bwd_scale), (
            f"{kind} rank {rank} grad err {bwd_err}"
        )
        assert n_masks == CASES[kind], f"{kind}: {n_masks} BlockMasks built"
