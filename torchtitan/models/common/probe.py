import torch
try:
    from torchtitan.tools.logging import logger
except ImportError:
    logger = None

class ProbeNaN(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, name):
        assert not x.isnan().any().item(), f"[FWD] NaN found in {name}"
        assert not x.isinf().any().item(), f"[FWD] Inf found in {name}"
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        assert not grad_output.isnan().any().item(), f"[BWD] NaN found in {ctx.name}"
        assert not grad_output.isinf().any().item(), f"[BWD] Inf found in {ctx.name}"
        norm = torch.nn.utils.get_total_norm(grad_output).item()
        if norm > 1e3:
            logger.error(f"[BWD] gradient exploded (norm={norm}) at {ctx.name}")
            raise ValueError(f"[BWD] gradient exploded (norm={norm}) at {ctx.name}")

        return grad_output, None

def probe_nan(x, name):
    return ProbeNaN.apply(x, name)

def check_failed(x: torch.Tensor):
    if x.isnan().any().item():
        return True
    if x.isinf().any().item():
        return True
    norm = torch.nn.utils.get_total_norm(x).item()
    if norm > 1e3:
        return True
    return False


class ProbeVarlenAttn(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seq_q: torch.Tensor,
        cu_seq_k: torch.Tensor | None,
        max_q: int,
        max_k: int,
        scale: float | None = None,
        window_size: tuple[int, int] = (-1, -1),
        enable_gqa: bool = False,
        seqused_k: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_splits: int | None = None,
    ):
        num_heads_q = query.size(1)
        num_heads_k = key.size(2) if block_table is not None else key.size(1)
        if not enable_gqa and num_heads_q != num_heads_k:
            raise ValueError(
                f"Expect query and key/value to have the same number of heads "
                f"but got Hq={num_heads_q} and Hkv={num_heads_k}. "
                f"Try setting enable_gqa=True for GQA."
            )
        if enable_gqa and num_heads_q % num_heads_k != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={num_heads_q} and Hkv={num_heads_k}."
            )

        is_causal = window_size == (-1, 0)
        out, lse, rng_state = torch.ops.torch_attn._varlen_attn(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal,
            scale,
            list(window_size),
            enable_gqa,
            seqused_k,
            block_table,
            num_splits,
        )

        ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)
        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.enable_gqa = enable_gqa
        ctx.seqused_k = seqused_k
        ctx.block_table = block_table
        ctx.num_splits = num_splits

        return out, lse
    
    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors

        max_q = ctx.max_q
        max_k = ctx.max_k
        is_causal = ctx.is_causal
        scale = ctx.scale
        window_size = ctx.window_size
        enable_gqa = ctx.enable_gqa
        seqused_k = ctx.seqused_k
        block_table = ctx.block_table
        num_splits = ctx.num_splits

        dq, dk, dv = torch.ops.torch_attn._varlen_attn_backward(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal,
            rng_state,
            scale,
            window_size,
        )

        if check_failed(dq) or check_failed(dk) or check_failed(dv):
            obj = {
                "query": query,
                "key": key,
                "value": value,
                "out": out,
                "lse": lse,
                "cu_seq_q": cu_seq_q,
                "cu_seq_k": cu_seq_k,
                "max_q": max_q,
                "max_k": max_k,
                "is_causal": is_causal,
                "scale": scale,
                "window_size": window_size,
                "enable_gqa": enable_gqa,
                "seqused_k": seqused_k,
                "block_table": block_table,
                "num_splits": num_splits,
                "grad_out": grad_out,
                "dq": dq,
                "dk": dk,
                "dv": dv,
            }
            torch.save(obj, f"/perf_apps/hanwang2/varlen_attn_io_rank{torch.distributed.get_rank()}.pt")
            logger.error(f"[BWD] varlen attn backward failed!")
            raise ValueError(f"[BWD] varlen attn backward failed!")

        num_params = 10
        return (dq, dk, dv, *((None,) * num_params))


def probe_varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, scale=None, window_size=(-1, -1), enable_gqa=False, seqused_k=None, block_table=None, num_splits=None):
    return ProbeVarlenAttn.apply(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, scale, window_size, enable_gqa, seqused_k, block_table, num_splits)


def _debug():
    from torch.nn.attention.varlen import varlen_attn

    obj = torch.load("/perf_apps/hanwang2/varlen_attn_io_rank0.pt")
    query = obj["query"]
    key = obj["key"]
    value = obj["value"]
    out = obj["out"]
    lse = obj["lse"]
    cu_seq_q = obj["cu_seq_q"]
    cu_seq_k = obj["cu_seq_k"]
    max_q = obj["max_q"]
    max_k = obj["max_k"]
    is_causal = obj["is_causal"]
    scale = obj["scale"]
    window_size = obj["window_size"]
    enable_gqa = obj["enable_gqa"]
    seqused_k = obj["seqused_k"]
    block_table = obj["block_table"]
    num_splits = obj["num_splits"]
    grad_out = obj["grad_out"]
    rng_state = torch.zeros(
        (2,), dtype=torch.uint64, device=query.device
    )

    dq, dk, dv = torch.ops.torch_attn._varlen_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        rng_state,
        scale,
        window_size,
    )
    if check_failed(dq) or check_failed(dk) or check_failed(dv):
        print(f"[BWD] varlen attn backward failed!")
        # raise ValueError(f"[BWD] varlen attn backward failed!")

    print(f"is_causal: {is_causal}, scale: {scale}, window_size: {window_size}, enable_gqa: {enable_gqa}, seqused_k: {seqused_k}, block_table: {block_table}, num_splits: {num_splits}")

    # print(f"dq: {torch.nn.utils.get_total_norm(dq)}, dk: {torch.nn.utils.get_total_norm(dk)}, dv: {torch.nn.utils.get_total_norm(dv)}")
    print(f"query: {torch.nn.utils.get_total_norm(query)}, key: {torch.nn.utils.get_total_norm(key)}, value: {torch.nn.utils.get_total_norm(value)} out: {torch.nn.utils.get_total_norm(out)} grad_out: {torch.nn.utils.get_total_norm(grad_out)}")

    
def _debug_flydsl():
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_varlen_flydsl_backward_impl,
        flash_attn_varlen_flydsl_forward_impl,
    )
    
    obj = torch.load("/perf_apps/hanwang2/varlen_attn_io_rank0.pt")
    query = obj["query"]
    key = obj["key"]
    value = obj["value"]
    out = obj["out"]
    lse = obj["lse"]
    cu_seq_q = obj["cu_seq_q"]
    cu_seq_k = obj["cu_seq_k"]
    max_q = obj["max_q"]
    max_k = obj["max_k"]
    is_causal = obj["is_causal"]
    scale = obj["scale"]
    window_size = obj["window_size"]
    enable_gqa = obj["enable_gqa"]
    seqused_k = obj["seqused_k"]
    block_table = obj["block_table"]
    num_splits = obj["num_splits"]
    grad_out = obj["grad_out"]
    rng_state = torch.zeros(
        (2,), dtype=torch.uint64, device=query.device
    )
    
    grads = flash_attn_varlen_flydsl_backward_impl(
        grad_out,
        query,
        key,
        value,
        out,
        lse.T,  # packed [total_q, Hq]; the impl transposes internally for the uniform path
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        sink=None,
        deterministic=True,
    )
    
    dq, dk, dv = grads
    if check_failed(dq) or check_failed(dk) or check_failed(dv):
        print(f"[BWD] varlen attn backward failed!")
    
    print(f"dq: {torch.nn.utils.get_total_norm(dq)}, dk: {torch.nn.utils.get_total_norm(dk)}, dv: {torch.nn.utils.get_total_norm(dv)}")
    print(dq)

if __name__ == "__main__":
    _debug()
    # _debug_flydsl()
