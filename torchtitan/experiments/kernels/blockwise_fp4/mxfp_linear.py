import torch
import torch.nn as nn
from torch.library import triton_op, wrap_triton
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import PlacementList
from torch.distributed.tensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)

import triton
import triton.language as tl

from .mxfp_quantization import (
    BLOCK_SIZE_DEFAULT,
)


@triton.jit
def blockwise_mxfp4_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_2DBLOCK_A: tl.constexpr,
    USE_2DBLOCK_B: tl.constexpr,
    K_PACK_A: tl.constexpr,
    K_PACK_B: tl.constexpr,
):
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)

    PACKED_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_M // 2
    PACKED_M: tl.constexpr = M // 2
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    PACKED_N: tl.constexpr = N // 2
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 2
    PACKED_K: tl.constexpr = K // 2
    # number of MXFP blocks inside a tile
    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    # total number of scales
    Ks: tl.constexpr = (K // QUANT_BLOCK_SIZE)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_m_pack = pid_m * PACKED_BLOCK_SIZE_M + tl.arange(
        0, PACKED_BLOCK_SIZE_M)
    mask_m_pack = offs_m_pack < PACKED_M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    offs_n_pack = pid_n * PACKED_BLOCK_SIZE_N + tl.arange(
        0, PACKED_BLOCK_SIZE_N)
    mask_n_pack = offs_n_pack < PACKED_N

    if USE_2DBLOCK_A:
        offs_m_scale = offs_m // QUANT_BLOCK_SIZE
    else:
        offs_m_scale = offs_m
    if USE_2DBLOCK_B:
        offs_n_scale = offs_n // QUANT_BLOCK_SIZE
    else:
        offs_n_scale = offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(num_blocks_k):
        offs_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        offs_k_pack = i * PACKED_BLOCK_SIZE_K + tl.arange(
            0, PACKED_BLOCK_SIZE_K)
        mask_k_pack = offs_k_pack < PACKED_K
        if K_PACK_A:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_pack[
                None, :] * stride_ak
            mask_a = mask_m[:, None] & mask_k_pack[None, :]
        else:
            a_ptrs = a_ptr + offs_m_pack[:, None] * stride_am + offs_k[
                None, :] * stride_ak
            mask_a = mask_m_pack[:, None] & mask_k[None, :]
        if K_PACK_B:
            b_ptrs = b_ptr + offs_k_pack[:, None] * stride_bk + offs_n[
                None, :] * stride_bn
            mask_b = mask_k_pack[:, None] & mask_n[None, :]
        else:
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n_pack[
                None, :] * stride_bn
            mask_b = mask_k[:, None] & mask_n_pack[None, :]
        offs_k_scale = i * n_rep_k + tl.arange(0, n_rep_k)
        mask_k_scale = offs_k_scale < Ks
        a_s_ptrs = a_s_ptr + offs_m_scale[:, None] * stride_asm + offs_k_scale[
            None, :] * stride_ask
        # B scales are N x K even though B operand is K x N.
        b_s_ptrs = b_s_ptr + offs_n_scale[:, None] * stride_bsn + offs_k_scale[
            None, :] * stride_bsk

        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        a_s = tl.load(a_s_ptrs,
                      mask=mask_m[:, None] & mask_k_scale[None, :],
                      other=1)
        b_s = tl.load(b_s_ptrs,
                      mask=mask_n[:, None] & mask_k_scale[None, :],
                      other=1)
        accumulator += tl.dot_scaled(a,
                                     a_s,
                                     "e2m1",
                                     b,
                                     b_s,
                                     "e2m1",
                                     lhs_k_pack=K_PACK_A,
                                     rhs_k_pack=K_PACK_B,
                                     out_dtype=tl.float32)

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


@triton_op("torchtitan::blockwise_mxfp4_gemm", mutates_args={})
def blockwise_mxfp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    use_2dblock_a: bool = False,
    use_2dblock_b: bool = False,
    k_pack_a: bool = True,
    k_pack_b: bool = True,
    trans_a: bool = False,
    trans_b: bool = False,
    block_size: int = BLOCK_SIZE_DEFAULT,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # M, N, K here refers to the shape of unpacked fp4
    assert a.dim() == 2 and b.dim() == 2
    if trans_a:
        K, M = a.shape
        if k_pack_a:
            K *= 2
        else:
            M *= 2
        if use_2dblock_a:
            assert a_s.shape == torch.Size(
                (K // block_size, M // block_size)
            ), f"A scale has shape {a_s.shape}, but {(K // block_size, M // block_size)} is expected."
        else:
            assert a_s.shape == torch.Size(
                (K // block_size, M)
            ), f"A scale has shape {a_s.shape}, but {(K // block_size, M)} is expected."
        stride_ask, stride_asm = a_s.stride()
        stride_ak, stride_am = a.stride()
    else:
        M, K = a.shape
        if k_pack_a:
            K *= 2
        else:
            M *= 2
        if use_2dblock_a:
            assert a_s.shape == torch.Size(
                (M // block_size, K // block_size)
            ), f"A scale has shape {a_s.shape}, but {(M // block_size, K // block_size)} is expected."
        else:
            assert a_s.shape == torch.Size(
                (M, K // block_size)
            ), f"A scale has shape {a_s.shape}, but {(M, K // block_size)} is expected."
        stride_asm, stride_ask = a_s.stride()
        stride_am, stride_ak = a.stride()

    if trans_b:
        N, KB = b.shape
        if k_pack_b:
            KB *= 2
        else:
            N *= 2
        assert KB == K, f"Input B has unpacked reduction shape ({KB}) but {(K)} is expected."
        stride_bn, stride_bk = b.stride()
        if use_2dblock_b:
            assert b_s.shape == torch.Size((N // block_size, K // block_size))
        else:
            assert b_s.shape == torch.Size((N, K // block_size))
        stride_bsn, stride_bsk = b_s.stride()
    else:
        KB, N = b.shape
        if k_pack_b:
            KB *= 2
        else:
            N *= 2
        assert KB == K, f"Input B has unpacked reduction shape ({KB}) but {(K)} is expected."
        stride_bk, stride_bn = b.stride()
        if use_2dblock_b:
            assert b_s.shape == torch.Size((K // block_size, N // block_size))
        else:
            assert b_s.shape == torch.Size((K // block_size, N))
        stride_bsk, stride_bsn = b_s.stride()

    c = a.new_empty((M, N), dtype=output_dtype)
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_M = 64 if M >= 64 else M
    BLOCK_SIZE_N = 64 if N >= 64 else N
    BLOCK_SIZE_K = 64 if K >= 64 else K
    wrap_triton(blockwise_mxfp4_gemm_kernel)[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=block_size,
        USE_2DBLOCK_A=use_2dblock_a,
        USE_2DBLOCK_B=use_2dblock_b,
        K_PACK_A=k_pack_a,
        K_PACK_B=k_pack_b,
    )
    return c


@torch.compiler.allow_in_graph
class MXFP4LinearFunction(torch.autograd.Function):
    """
    Custom autograd function for MXFP linear operations.
    This function handles the forward and backward passes for the linear layer.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
    ):
        """
        Forward pass for the blockwise FP8 linear operation.

        Args:
            ctx: Context object to save information for backward pass.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor, optional): Bias tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        original_shape = x.shape
        original_dtype = x.dtype
        x = x.reshape(-1, original_shape[-1])  # Ensure x is 2D

        x, x_scale = torch.ops.torchtitan.convert_to_mxfp4(
            x,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )

        weight, w_scale = torch.ops.torchtitan.convert_to_mxfp4(
            weight,
            axis=-1,
            is_2d_block=use_2dblock_w,
        )

        y = torch.ops.torchtitan.blockwise_mxfp4_gemm(
            x,
            x_scale,
            weight,
            w_scale,
            use_2dblock_a=use_2dblock_x,
            use_2dblock_b=use_2dblock_w,
            trans_b=True,
            output_dtype=original_dtype,
        )

        ctx.save_for_backward(x, x_scale, weight, w_scale)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.original_dtype = original_dtype
        ctx.use_sr_grad = use_sr_grad

        return y.view(*original_shape[:-1], -1)  # Reshape back to original

    @staticmethod
    def backward(ctx, grad_output):
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(
            -1, original_shape[-1])  # Ensure grad_output is 2D

        x, x_scale, weight, w_scale = ctx.saved_tensors

        if ctx.use_2dblock_w:
            weight_mxfp4 = weight
            weight_scales = w_scale
        else:
            weight_dequant = torch.ops.torchtitan.convert_from_mxfp4(
                weight,
                w_scale,
                axis=-1,
                is_2d_block=False,
            )
            weight_mxfp4, weight_scales = torch.ops.torchtitan.convert_to_mxfp4(
                weight_dequant,
                axis=0,
                is_2d_block=False,
            )

        # dequant-quant as deepseek-v3 paper
        if ctx.use_2dblock_x:
            inputs_mxfp4 = x
            input_scales = x_scale

            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=-1,
                is_2d_block=True,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_mxfp4_m = grad_output_mxfp4
            grad_output_scales_m = grad_output_scales
        else:
            inputs_dequant = torch.ops.torchtitan.convert_from_mxfp4(
                x,
                x_scale,
                axis=-1,
                is_2d_block=False,
            )
            inputs_mxfp4, input_scales = torch.ops.torchtitan.convert_to_mxfp4(
                inputs_dequant,
                axis=0,
                is_2d_block=False,
            )

            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=-1,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
            )
            grad_output_mxfp4_m, grad_output_scales_m = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=0,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
            )

        # Compute gradients
        grad_inputs = torch.ops.torchtitan.blockwise_mxfp4_gemm(
            grad_output_mxfp4,
            grad_output_scales,
            weight_mxfp4,
            weight_scales,
            use_2dblock_a=ctx.use_2dblock_x,
            use_2dblock_b=ctx.use_2dblock_w,
            k_pack_b=not ctx.use_2dblock_w,
            output_dtype=ctx.original_dtype,
        )
        grad_weights = torch.ops.torchtitan.blockwise_mxfp4_gemm(
            grad_output_mxfp4_m,
            grad_output_scales_m,
            inputs_mxfp4,
            input_scales,
            use_2dblock_a=ctx.use_2dblock_x,
            use_2dblock_b=ctx.use_2dblock_x,
            trans_a=True,
            k_pack_a=not ctx.use_2dblock_x,
            k_pack_b=not ctx.use_2dblock_x,
            output_dtype=ctx.original_dtype,
        )

        return grad_inputs.view(*original_shape[:-1],
                                -1), grad_weights, None, None, None


single_mesh_dim_strategies = []
colwise: PlacementList = [
    Shard(-1),
    Replicate(),  # x
    Shard(0),  # w
]
rowwise: PlacementList = [
    Partial(),
    Shard(-1),  # x
    Shard(1),  # w
]
single_mesh_dim_strategies.extend([colwise, rowwise])


class MXFP4Linear(nn.Linear):
    """
    Custom linear layer with support for quantized weights.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 use_2dblock_x: bool = False,
                 use_2dblock_w: bool = True,
                 use_sr_grad: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)

        self.use_2dblock_x = use_2dblock_x
        self.use_2dblock_w = use_2dblock_w
        self.use_sr_grad = use_sr_grad
        assert not self.bias, "Bias is not supported in MXFP4Linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """

        device_mesh = None
        if isinstance(x, DTensor):
            device_mesh = x.device_mesh
            output_placement = None

            rowwise_abs: PlacementList = [
                Partial(),
                Shard(x.dim() - 1),  # x
                Shard(1),  # w
            ]
            for acceptable_placement in single_mesh_dim_strategies + [
                    rowwise_abs
            ]:
                if acceptable_placement[1:] == [
                        x.placements[-1], self.weight.placements[-1]
                ]:
                    output_placement = acceptable_placement[0]
                    break
            assert output_placement is not None, \
                f"Unsupported placement strategy for MXFP4Linear: {x.placements} and {self.weight.placements}"
            x = x.to_local()
            w = self.weight.to_local()
        else:
            w = self.weight
        y = MXFP4LinearFunction.apply(
            x,
            w,
            self.use_2dblock_x,
            self.use_2dblock_w,
            self.use_sr_grad,
        )

        if device_mesh is not None:
            y = DTensor.from_local(y,
                                   device_mesh=device_mesh,
                                   placements=(output_placement, ))

        return y

    @classmethod
    def from_float(
        cls,
        layer: nn.Linear,
        use_2dblock_x: bool = False,
        use_2dblock_w: bool = True,
        use_sr_grad: bool = False,
    ):
        """
        Create a MXFP4Linear layer from a standard nn.Linear layer.

        Args:
            layer (nn.Linear): The linear layer to convert.
            block_size (int): Block size for quantization. Defaults to 128.

        Returns:
            MXFP4Linear: The converted quantized linear layer.
        """
        new_layer = cls(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            use_2dblock_x=use_2dblock_x,
            use_2dblock_w=use_2dblock_w,
            use_sr_grad=use_sr_grad,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
        return new_layer
