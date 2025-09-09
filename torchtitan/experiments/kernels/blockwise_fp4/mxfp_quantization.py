from typing import Tuple
import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

BLOCK_SIZE_DEFAULT = 32


def is_cdna4():
    # target = triton.runtime.driver.active.get_current_target()
    # return target is not None and target.backend == "hip" and target.arch == "gfx950"
    return False


@triton.jit
def _calculate_scales(x):
    if x.type.element_ty == tl.float32:
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

    max_abs = tl.max(tl.abs(x)).to(x.type.element_ty)
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
    return scales


@triton.jit
def _quantize_fp4(x, scales_fp32):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    qx = x.to(tl.float32) / scales_fp32

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Sset everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = ((EXP_BIAS_FP32 - EXP_BIAS_FP4) +
                                (MBITS_F32 - MBITS_FP4) + 1)
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int,
                                              tl.float32,
                                              bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.zeros_like(qx).to(tl.uint8) + 0x7
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    return e2m1_value


@triton.jit
def _pack_fp4(
    x0,
    x1,
    scales,
    USE_ASM: tl.constexpr = False,
):
    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if USE_ASM:
        if x0.type.element_ty == tl.float32:
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
    else:
        y0 = _quantize_fp4(x0, scales_fp32)
        y1 = _quantize_fp4(x1, scales_fp32)
        y = y0 | (y1 << 4)

    return y


@triton.jit
def _unpack_fp4(x, scales, output_dtype: tl.constexpr):
    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if output_dtype == tl.float32:
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
    return y0, y1


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
    USE_ASM: tl.constexpr,
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
    HALF_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    start_xn = pid_n * BLOCK_SIZE
    start_yn = pid_n * HALF_BLOCK_SIZE
    offs_xn = tl.arange(0, BLOCK_SIZE)
    offs_yn = tl.arange(0, HALF_BLOCK_SIZE)

    offs_x = pid_m * stride_xm + (start_xn + offs_xn) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    tl.static_assert(x_ptr.type.element_ty == tl.float32
                     or x_ptr.type.element_ty == tl.bfloat16)
    x = tl.load(x_ptr + offs_x)
    scales = _calculate_scales(x)
    tl.store(s_ptr + offs_s, scales.to(s_ptr.type.element_ty))

    x = x.reshape(HALF_BLOCK_SIZE, 2)
    x0, x1 = tl.split(x)
    y = _pack_fp4(x0, x1, scales, USE_ASM)
    offs_y = pid_m * stride_ym + (start_yn + offs_yn) * stride_yn
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
    USE_ASM: tl.constexpr,
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
    HALF_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_m = pid_m * BLOCK_SIZE
    start_xn = pid_n * BLOCK_SIZE
    start_yn = pid_n * HALF_BLOCK_SIZE
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_yn = tl.arange(0, HALF_BLOCK_SIZE)
    offs_xn = tl.arange(0, BLOCK_SIZE)

    offs_x = (start_m + offs_m[:, None]) * stride_xm + (
        start_xn + offs_xn[None, :]) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    tl.static_assert(x_ptr.type.element_ty == tl.float32
                     or x_ptr.type.element_ty == tl.bfloat16)
    x = tl.load(x_ptr + offs_x)
    scales = _calculate_scales(x)
    tl.store(s_ptr + offs_s, scales.to(s_ptr.type.element_ty))

    x = x.reshape(BLOCK_SIZE, HALF_BLOCK_SIZE, 2)
    x0, x1 = tl.split(x)
    y = _pack_fp4(x0, x1, scales, USE_ASM)

    offs_y = (start_m + offs_m[:, None]) * stride_ym + (
        start_yn + offs_yn[None, :]) * stride_yn
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
    HALF_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    start_xn = pid_n * HALF_BLOCK_SIZE
    start_yn = pid_n * BLOCK_SIZE
    offs_xn = tl.arange(0, HALF_BLOCK_SIZE)
    offs_yn = tl.arange(0, BLOCK_SIZE)

    offs_x = pid_m * stride_xm + (start_xn + offs_xn) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    x = tl.load(x_ptr + offs_x).to(tl.uint16)
    s = tl.load(s_ptr + offs_s).to(tl.uint32)

    tl.static_assert(y_ptr.type.element_ty == tl.float32
                     or y_ptr.type.element_ty == tl.bfloat16)
    y0, y1 = _unpack_fp4(x, s, y_ptr.type.element_ty)
    y = tl.join(y0, y1).reshape(BLOCK_SIZE)
    offs_y = pid_m * stride_ym + (start_yn + offs_yn) * stride_yn
    tl.store(y_ptr + offs_y, y)


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
    HALF_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    start_m = pid_m * BLOCK_SIZE
    start_xn = pid_n * HALF_BLOCK_SIZE
    start_yn = pid_n * BLOCK_SIZE
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_xn = tl.arange(0, HALF_BLOCK_SIZE)
    offs_yn = tl.arange(0, BLOCK_SIZE)

    offs_x = (start_m + offs_m[:, None]) * stride_xm + (
        start_xn + offs_xn[None, :]) * stride_xn
    offs_s = pid_m * stride_sm + pid_n * stride_sn

    x = tl.load(x_ptr + offs_x).to(tl.uint16)
    s = tl.load(s_ptr + offs_s).to(tl.uint32)

    tl.static_assert(y_ptr.type.element_ty == tl.float32
                     or y_ptr.type.element_ty == tl.bfloat16)
    y0, y1 = _unpack_fp4(x, s, y_ptr.type.element_ty)
    y = tl.join(y0, y1).reshape(BLOCK_SIZE, BLOCK_SIZE)

    offs_y = (start_m + offs_m[:, None]) * stride_ym + (
        start_yn + offs_yn[None, :]) * stride_yn
    tl.store(y_ptr + offs_y, y)


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
        USE_ASM=is_cdna4(),
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
    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8).reshape(
        -1, ori_shape[-1] // block_size)

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
        USE_ASM=is_cdna4(),
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
    data_hp = data_lp.new_empty(orig_shape, dtype=output_dtype).reshape(
        -1, orig_shape[-1])

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

    data_hp = data_lp.new_empty(orig_shape, dtype=output_dtype).reshape(
        -1, orig_shape[-1])

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
