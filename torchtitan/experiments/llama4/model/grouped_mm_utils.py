import torch
from grouped_gemm import backend


@torch.library.custom_op("amd::grouped_gemm_cuda_ext_ff", mutates_args=())
def grouped_gemm_cuda_ext_ff(a: torch.Tensor, b: torch.Tensor,
                          batch_sizes: torch.Tensor) -> torch.Tensor:

    return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=False)


@grouped_gemm_cuda_ext_ff.register_fake
def grouped_gemm_cuda_ext_ff_fake(a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    _, _, n = b.shape
    return torch.zeros((m, n), dtype=a.dtype, device=a.device)


@torch.library.custom_op("amd::grouped_gemm_cuda_ext_ft", mutates_args=())
def grouped_gemm_cuda_ext_ft(a: torch.Tensor, b: torch.Tensor,
                             batch_sizes: torch.Tensor) -> torch.Tensor:

    return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=True)


@grouped_gemm_cuda_ext_ft.register_fake
def grouped_gemm_cuda_ext_ft_fake(a: torch.Tensor, b: torch.Tensor,
      batch_sizes: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    _, n, _ = b.shape
    return torch.zeros((m, n), dtype=a.dtype, device=a.device)


@torch.library.custom_op("amd::grouped_gemm_cuda_ext_tf", mutates_args=())
def grouped_gemm_cuda_ext_tf(a: torch.Tensor, b: torch.Tensor,
                             batch_sizes: torch.Tensor) -> torch.Tensor:

    return backend.gmm(a, b, batch_sizes, trans_a=True, trans_b=False)


@grouped_gemm_cuda_ext_tf.register_fake
def grouped_gemm_cuda_ext_tf_fake(a: torch.Tensor, b: torch.Tensor,
      batch_sizes: torch.Tensor) -> torch.Tensor:
    t = batch_sizes.shape[0]
    _, m = a.shape
    _, n = b.shape
    return torch.zeros((t, m, n), dtype=a.dtype, device=a.device)


@torch.library.custom_op("amd::grouped_gemm_cuda_ext", mutates_args=())
def gmm(a: torch.Tensor, b: torch.Tensor,
                             batch_sizes: torch.Tensor) -> torch.Tensor:
    assert torch.count_nonzero(
    batch_sizes) != 0, "Input batch_sizes should not be all zeros!"

    return grouped_gemm_cuda_ext_ff(a, b, batch_sizes)


def backward(ctx, grad):
    grad = grad.contiguous()
    a, b, batch_sizes = ctx.saved_tensors

    agrad = None
    if ctx.needs_input_grad[0]:
        agrad = grouped_gemm_cuda_ext_ft(grad, b, batch_sizes)

    bgrad = None
    if ctx.needs_input_grad[1]:
        lhs, rhs = (a, grad)
        bgrad = grouped_gemm_cuda_ext_tf(lhs, rhs, batch_sizes)
    return agrad, bgrad, None

def setup_context(ctx, inputs, output):
    a, b, batch_sizes = inputs
    ctx.save_for_backward(a, b, batch_sizes)


gmm.register_autograd(backward, setup_context=setup_context)


@gmm.register_fake
def _(a: torch.Tensor, b: torch.Tensor,
                                  batch_sizes: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    _, _, n = b.shape
    return torch.zeros((m, n), dtype=a.dtype, device=a.device)
