from typing import Tuple, Optional
import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

BLOCK_SIZE_DEFAULT = 32


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


@triton.jit
def _calculate_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
):
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

    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE, NEW_BLOCK_N,
                      QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x), axis=-1)
        max_abs = tl.max(max_abs, axis=-2)
    else:
        x = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

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
    # Set everything to positive, will add sign back at the end
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
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    return e2m1_value


@triton.jit
def _dequantize_fp4(qx, scales_fp32):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    # assume qx is already unpacked
    tl.static_assert(qx.dtype == tl.uint8)

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

    # Extract sign
    s = qx & 0x8
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    zero_mask = qx == 0x0
    denormal_mask = qx == 0x1

    # Normal numbers
    exp_biased_lp = qx >> MBITS_FP4
    exp_biased_f32 = exp_biased_lp - EXP_BIAS_FP4 + EXP_BIAS_FP32
    exp_biased_f32 = exp_biased_f32.to(tl.uint32) << MBITS_F32
    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (qx & 0x1).to(tl.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - MBITS_FP4)
    x_int = exp_biased_f32 | mantissa_f32

    # Zeros
    x_int = tl.where(zero_mask, 0, x_int)
    # Denormal numbers
    x_int = tl.where(denormal_mask, 0x3F000000, x_int)

    # add sign back
    sign_lp = s.to(tl.uint32) << (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    x_int = x_int | sign_lp

    x_fp = x_int.to(tl.float32, bitcast=True)
    return x_fp * scales_fp32


@triton.jit
def _generate_randval(m, n):
    philox_seed: tl.constexpr = 0x1BF52
    philox_offset: tl.constexpr = 0x1D4B42
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    return tl.randint(philox_seed, rng_offsets)


@triton.jit
def _pack_fp4(
    x,
    scales,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if IS_2D_BLOCK:
        # scales_fp32: [SCALE_BLOCK_M, SCALE_BLOCK_N]
        scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
            SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        # scales_fp32: [BLOCK_M, SCALE_BLOCK_N]
        scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)

    if USE_ASM:
        if USE_SR:
            randval = _generate_randval(BLOCK_M, HALF_BLOCK_N)
        if x0.type.element_ty == tl.float32:
            if not USE_SR:
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
                x0 = (x1.to(tl.uint32, bitcast=True).to(tl.uint64) <<
                      32) | x0.to(tl.uint32, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v,v",
                    args=[x0, randval, scales_fp32],
                    dtype=tl.uint16,
                    is_pure=True,
                    pack=1,
                )
        else:
            x0 = (x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | x0.to(
                tl.uint16, bitcast=True)
            if not USE_SR:
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
            else:
                y = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_sr_pk_fp4_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];
                    """,
                    constraints="=&v,v,v,v",
                    args=[x0, randval, scales_fp32],
                    dtype=tl.uint16,
                    is_pure=True,
                    pack=1,
                )
        y = y & 0x00FF
    else:
        tl.static_assert(not USE_SR)
        y0 = _quantize_fp4(x0, scales_fp32)
        y1 = _quantize_fp4(x1, scales_fp32)
        y = y0 | (y1 << 4)

    return y


@triton.jit
def _unpack_fp4(
    x,
    scales,
    output_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr = False,
):
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    scales_fp32 = (scales << 23).to(tl.float32, bitcast=True)
    if IS_2D_BLOCK:
        # scales_fp32: [SCALE_BLOCK_M, SCALE_BLOCK_N]
        scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
            SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        # scales_fp32: [BLOCK_M, SCALE_BLOCK_N]
        scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)

    if USE_ASM:
        x = x.to(tl.uint16)
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
    else:
        x0 = x & 0xF
        x1 = (x & 0xF0) >> 4
        y0 = _dequantize_fp4(x0, scales_fp32)
        y1 = _dequantize_fp4(x1, scales_fp32)

    y = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
    return y


@triton.jit
def _convert_to_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
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
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    tl.static_assert(x_ptr.type.element_ty == tl.float32
                     or x_ptr.type.element_ty == tl.bfloat16)
    x = tl.load(x_ptr + offs_x)
    scales = _calculate_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    tl.store(s_ptr + offs_s, scales.to(s_ptr.type.element_ty))

    y = _pack_fp4(
        x,
        scales,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y.to(y_ptr.type.element_ty))


@triton.jit
def _convert_from_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr,
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
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x)
    s = tl.load(s_ptr + offs_s).to(tl.uint32)

    tl.static_assert(y_ptr.type.element_ty == tl.float32
                     or y_ptr.type.element_ty == tl.bfloat16)

    y = _unpack_fp4(
        x,
        s,
        y_ptr.type.element_ty,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_ASM=USE_ASM,
    )
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    tl.store(y_ptr + offs_y, y)


@triton_op("torchtitan::convert_to_mxfp4", mutates_args={})
def convert_to_mxfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(
        data_hp.shape[axis] % block_size == 0,
        f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}"
    )
    assert not is_2d_block or data_hp.size(-2) % block_size == 0
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4()
    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, ori_shape[-1])
    new_shape = (*ori_shape[:-1], ori_shape[-1] // 2)
    data_lp = torch.empty(new_shape, dtype=torch.uint8,
                          device=data_hp.device).reshape(-1, new_shape[-1])
    if is_2d_block:
        scales_shape = (*ori_shape[:-2], ori_shape[-2] // block_size,
                        ori_shape[-1] // block_size)
    else:
        scales_shape = (*ori_shape[:-1], ori_shape[-1] // block_size)
    scales = torch.zeros(scales_shape,
                         dtype=torch.uint8,
                         device=data_hp.device).reshape(-1, scales_shape[-1])
    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),
                         triton.cdiv(N, META["BLOCK_N"]))
    wrap_triton(_convert_to_mxfp4_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_M=64,
        BLOCK_N=64,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )

    return data_lp.reshape(new_shape).transpose(
        axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("torchtitan::convert_from_mxfp4", mutates_args={})
def convert_from_mxfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    assert output_dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4()

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
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),
                         triton.cdiv(N, META["BLOCK_N"]))
    wrap_triton(_convert_from_mxfp4_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_M=64,
        BLOCK_N=64,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return data_hp.reshape(orig_shape).transpose(axis, -1)


@convert_to_mxfp4.register_fake
def _fake_convert_to_mxfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    new_shape = (*orig_shape[:-1], orig_shape[-1] // 2)
    data_lp = data_hp.new_empty(new_shape, dtype=torch.uint8)
    if is_2d_block:
        scales_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                        orig_shape[-1] // block_size)
    else:
        scales_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)

    scales = data_hp.new_empty(scales_shape, dtype=torch.uint8)
    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


@convert_from_mxfp4.register_fake
def _fake_convert_from_mxfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return torch.cat((data_hp, data_hp), dim=axis)
