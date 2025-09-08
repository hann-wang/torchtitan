from typing import Tuple
import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

BLOCK_SIZE_DEFAULT = 32


@triton.jit
def _convert_to_mxfp4_1dblock_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_xn = pid_n * BLOCK_SIZE
    start_yn = pid_n * BLOCK_SIZE // 2
    offs_n = tl.arange(0, BLOCK_SIZE // 2)

    offs_x0 = pid_m * stride_xm + (start_xn + offs_n * 2) * stride_xn
    offs_x1 = pid_m * stride_xm + (start_xn + offs_n * 2 + 1) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    tl.static_assert(x_ptr.type.element_ty == tl.float32
                     or x_ptr.type.element_ty == tl.bfloat16)
    x0 = tl.load(x_ptr + offs_x0)
    x1 = tl.load(x_ptr + offs_x1)

    if x_ptr.type.element_ty == tl.float32:
        hp_int_dtype = tl.uint32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = tl.uint16
        hp_mbits = 7
        hp_ebits = 8
    mbits = 1
    sbits = 1
    target_max_pow2 = 2

    max_abs = tl.maximum(tl.max(tl.abs(x0)),
                         tl.max(tl.abs(x1))).to(x_ptr.type.element_ty)
    # round even (adaptive)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = ((max_abs + val_to_add) & mask) >> hp_mbits
    scales = max_abs - target_max_pow2

    # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
    # float32 denormal range. For now, manually adjust the fp scale. This is
    # relevant if all of the incoming block values are zeroes.
    # See https://github.com/pytorch/pytorch/issues/125557 for details.
    # Note: it would be more correct to set the minimum to 2**-127, but this
    # does not work in triton either as it looks like subnormal value handling
    # has some gaps.  So, for now just set to the minimum normal value.
    scales = tl.where(scales == 0, 1, scales)

    tl.store(s_ptr + offs_s, scales.to(s_ptr.type.element_ty))

    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if x_ptr.type.element_ty == tl.float32:
        y = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v,v",
            args=[x0, x1, scales_fp32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    else:
        x0 = (x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | x0.to(
            tl.uint16, bitcast=True)
        y = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v",
            args=[x0, scales_fp32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    y = y & 0x00FF

    offs_y = pid_m * stride_ym + (start_yn + offs_n) * stride_yn
    tl.store(y_ptr + offs_y, y.to(y_ptr.type.element_ty))


@triton.jit
def _convert_to_mxfp4_2dblock_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_m = pid_m * BLOCK_SIZE
    start_xn = pid_n * BLOCK_SIZE
    start_yn = pid_n * BLOCK_SIZE // 2
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE // 2)
    offs_xn0 = offs_n * 2
    offs_xn1 = offs_xn0 + 1

    offs_x0 = (start_m + offs_m[:, None]) * stride_xm + (
        start_xn + offs_xn0[None, :]) * stride_xn
    offs_x1 = (start_m + offs_m[:, None]) * stride_xm + (
        start_xn + offs_xn1[None, :]) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    tl.static_assert(x_ptr.type.element_ty == tl.float32
                     or x_ptr.type.element_ty == tl.bfloat16)
    x0 = tl.load(x_ptr + offs_x0)
    x1 = tl.load(x_ptr + offs_x1)

    if x_ptr.type.element_ty == tl.float32:
        hp_int_dtype = tl.uint32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = tl.uint16
        hp_mbits = 7
        hp_ebits = 8
    mbits = 1
    sbits = 1
    target_max_pow2 = 2

    max_abs = tl.maximum(tl.max(tl.abs(x0)),
                         tl.max(tl.abs(x1))).to(x_ptr.type.element_ty)
    # round even (adaptive)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = ((max_abs + val_to_add) & mask) >> hp_mbits
    scales = max_abs - target_max_pow2

    # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
    # float32 denormal range. For now, manually adjust the fp scale. This is
    # relevant if all of the incoming block values are zeroes.
    # See https://github.com/pytorch/pytorch/issues/125557 for details.
    # Note: it would be more correct to set the minimum to 2**-127, but this
    # does not work in triton either as it looks like subnormal value handling
    # has some gaps.  So, for now just set to the minimum normal value.
    scales = tl.where(scales == 0, 1, scales)

    tl.store(s_ptr + offs_s, scales.to(s_ptr.type.element_ty))

    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if x_ptr.type.element_ty == tl.float32:
        y = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v,v",
            args=[x0, x1, scales_fp32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    else:
        x0 = (x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | x0.to(
            tl.uint16, bitcast=True)
        y = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 op_sel:[0,0,0,0];
            """,
            constraints="=&v,v,v",
            args=[x0, scales_fp32],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
    y = y & 0x00FF

    offs_y = (start_m + offs_m[:, None]) * stride_ym + (
        start_yn + offs_n[None, :]) * stride_yn
    tl.store(y_ptr + offs_y, y.to(y_ptr.type.element_ty))


@triton.jit
def _convert_from_mxfp4_1dblock_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantizes the input tensor `x_ptr` with scaling factors in `s_ptr`, and stores the result in `y_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where dequantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the scaling factors.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_xn = pid_n * BLOCK_SIZE // 2
    start_yn = pid_n * BLOCK_SIZE
    offs_n = tl.arange(0, BLOCK_SIZE // 2)

    offs_x = pid_m * stride_xm + (start_xn + offs_n) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    x = tl.load(x_ptr + offs_x).to(tl.uint16)
    s = tl.load(s_ptr + offs_s).to(tl.uint32)

    tl.static_assert(y_ptr.type.element_ty == tl.float32
                     or y_ptr.type.element_ty == tl.bfloat16)

    scales_fp32 = (s << 23).to(tl.float32, bitcast=True)
    if y_ptr.type.element_ty == tl.float32:
        y_packed = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_f32_fp4 $0, $1, $2 op_sel:[0,0];
            """,
            constraints="=&v,v,v",
            args=[x, scales_fp32],
            dtype=tl.uint64,
            is_pure=True,
            pack=1,
        )
        y1 = (y_packed >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        y0 = (y_packed & 0x00000000FFFFFFFF).to(tl.uint32).to(tl.float32,
                                                              bitcast=True)
    else:
        y_packed = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[0,0];
            """,
            constraints="=&v,v,v",
            args=[x, scales_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        y1 = (y_packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        y0 = (y_packed & 0x0000FFFF).to(tl.uint16).to(tl.bfloat16,
                                                      bitcast=True)

    offs_y0 = pid_m * stride_ym + (start_yn + offs_n * 2) * stride_yn
    offs_y1 = pid_m * stride_ym + (start_yn + offs_n * 2 + 1) * stride_yn
    tl.store(y_ptr + offs_y0, y0)
    tl.store(y_ptr + offs_y1, y1)


@triton.jit
def _convert_from_mxfp4_2dblock_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantizes the input tensor `x_ptr` with scaling factors in `s_ptr`, and stores the result in `y_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where dequantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the scaling factors.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_m = pid_m * BLOCK_SIZE
    start_xn = pid_n * BLOCK_SIZE // 2
    start_yn = pid_n * BLOCK_SIZE
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE // 2)

    offs_x = (start_m + offs_m[:, None]) * stride_xm + (
        start_xn + offs_n[None, :]) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    x = tl.load(x_ptr + offs_x).to(tl.uint16)
    s = tl.load(s_ptr + offs_s).to(tl.uint32)

    tl.static_assert(y_ptr.type.element_ty == tl.float32
                     or y_ptr.type.element_ty == tl.bfloat16)

    scales_fp32 = (s << 23).to(tl.float32, bitcast=True)
    if y_ptr.type.element_ty == tl.float32:
        y_packed = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_f32_fp4 $0, $1, $2 op_sel:[0,0];
            """,
            constraints="=&v,v,v",
            args=[x, scales_fp32],
            dtype=tl.uint64,
            is_pure=True,
            pack=1,
        )
        y1 = (y_packed >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        y0 = (y_packed & 0x00000000FFFFFFFF).to(tl.uint32).to(tl.float32,
                                                              bitcast=True)
    else:
        y_packed = tl.inline_asm_elementwise(
            asm="""
            v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[0,0];
            """,
            constraints="=&v,v,v",
            args=[x, scales_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        y1 = (y_packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        y0 = (y_packed & 0x0000FFFF).to(tl.uint16).to(tl.bfloat16,
                                                      bitcast=True)

    offs_y0 = (start_m + offs_m[:, None]) * stride_ym + (
        start_yn + offs_n[None, :] * 2) * stride_yn
    offs_y1 = (start_m + offs_m[:, None]) * stride_ym + (
        start_yn + offs_n[None, :] * 2 + 1) * stride_yn
    tl.store(y_ptr + offs_y0, y0)
    tl.store(y_ptr + offs_y1, y1)


@triton_op("torchtitan::convert_to_mxfp4_1dblock", mutates_args={})
def convert_to_mxfp4_1dblock(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert data_hp.size(axis) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, ori_shape[-1])
    new_shape = (*ori_shape[:-1], ori_shape[-1] // 2)
    data_lp = torch.empty(new_shape, dtype=torch.uint8,
                          device=data_hp.device).reshape(-1, new_shape[-1])

    scales_shape = (*ori_shape[:-1], ori_shape[-1] // block_size)
    scales = torch.empty(scales_shape,
                         dtype=torch.uint8,
                         device=data_hp.device).reshape(-1, scales_shape[-1])
    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape
    num_blocks_n = N // block_size
    grid = (
        M,
        num_blocks_n,
    )
    wrap_triton(_convert_to_mxfp4_1dblock_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )

    return data_lp.reshape(new_shape).transpose(
        axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("torchtitan::convert_to_mxfp4_2dblock", mutates_args={})
def convert_to_mxfp4_2dblock(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert data_hp.size(-2) % block_size == 0 and data_hp.size(
        -1) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, ori_shape[-1])

    new_shape = (*ori_shape[:-1], ori_shape[-1] // 2)
    data_lp = data_hp.new_empty(new_shape, dtype=torch.uint8).reshape(
        -1, ori_shape[-1] // 2)

    scales_shape = (*ori_shape[:-2], ori_shape[-2] // block_size,
                    ori_shape[-1] // block_size)
    scales = data_hp.new_empty(scales_shape,
                               dtype=torch.uint8).reshape(-1, ori_shape[-1] // block_size)

    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape
    num_blocks_m = M // block_size
    num_blocks_n = N // block_size
    grid = (
        num_blocks_m,
        num_blocks_n,
    )
    wrap_triton(_convert_to_mxfp4_2dblock_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )

    return data_lp.reshape(new_shape).transpose(
        axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("torchtitan::convert_from_mxfp4_1dblock", mutates_args={})
def convert_from_mxfp4_1dblock(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, orig_shape[-1])
    orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)

    scales = scales.reshape(-1, orig_shape[-1] // block_size)
    data_hp = data_lp.new_empty(orig_shape,
                                dtype=output_dtype).reshape(-1, orig_shape[-1])

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape
    num_blocks_n = N // block_size
    grid = (
        M,
        num_blocks_n,
    )
    wrap_triton(_convert_from_mxfp4_1dblock_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@triton_op("torchtitan::convert_from_mxfp4_2dblock", mutates_args={})
def convert_from_mxfp4_2dblock(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, orig_shape[-1])
    orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)
    scales = scales.reshape(-1, orig_shape[-1] // block_size)

    data_hp = data_lp.new_empty(orig_shape,
                                dtype=output_dtype).reshape(-1, orig_shape[-1])

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape
    num_blocks_m = M // block_size
    num_blocks_n = N // block_size
    grid = (
        num_blocks_m,
        num_blocks_n,
    )
    wrap_triton(_convert_from_mxfp4_2dblock_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@convert_to_mxfp4_1dblock.register_fake
def _fake_convert_to_mxfp4_1dblock(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    new_shape = (*orig_shape[:-1], orig_shape[-1] // 2)
    data_lp = data_hp.new_empty(new_shape, dtype=torch.uint8)

    scales_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)
    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


@convert_to_mxfp4_2dblock.register_fake
def _fake_convert_to_mxfp4_2dblock(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    new_shape = (*orig_shape[:-1], orig_shape[-1] // 2)
    data_lp = data_hp.new_empty(new_shape, dtype=torch.uint8)

    scales_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                    orig_shape[-1] // block_size)
    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


@convert_from_mxfp4_1dblock.register_fake
def _fake_convert_from_mxfp4_1dblock(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return torch.cat((data_hp, data_hp), dim=axis)


@convert_from_mxfp4_2dblock.register_fake
def _fake_convert_from_mxfp4_2dblock(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return torch.cat((data_hp, data_hp), dim=axis)
