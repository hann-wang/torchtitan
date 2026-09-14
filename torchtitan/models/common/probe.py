import torch
from torchtitan.tools.logging import logger

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
