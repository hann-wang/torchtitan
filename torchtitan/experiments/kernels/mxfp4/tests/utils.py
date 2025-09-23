# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import Tensor
from torchtitan.experiments.kernels.mxfp4.mxfp_quantization import (
    BLOCK_SIZE_DEFAULT)


def prepare_data(tensor_shape, data_type):
    torch.manual_seed(1234)
    device = torch.device("cuda")
    x = torch.randn(tensor_shape, dtype=data_type, device=device)
    p_mask = torch.bernoulli(torch.ones_like(x) * 0.005)
    x += 100 * torch.randn_like(x) * p_mask
    return x


def calc_snr(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between two tensors.
    SNR = 10 * log10(sum(x^2) / (sum((x - y)^2)))
    """
    signal = torch.sum(x**2)
    noise = torch.sum((x - y)**2)
    return 10 * torch.log10(signal / noise)


def calc_cossim(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the cosine similarity between two tensors.
    Cosine similarity = dot(x, y) / (norm(x) * norm(y))
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    dot_product = torch.sum(x * y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    return dot_product / (norm_x * norm_y)


def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/questions/8981913/how-to-perform-round-to-even-with-floating-point-numbers  # noqa: E501
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2**(_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) /
                                                   (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2**(1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1)
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int,
                                     dtype=torch.int32).view(torch.float32)

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x
                                      < min_normal)
    normal_mask = torch.logical_not(
        torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


# TODO(future): check if LUT for everything is faster than bit shifting,
# especially for fp4 (only 2^4=16 unique values).
def _floatx_unpacked_to_f32(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp -
                                (1 << i)) << (left_shift + MBITS_F32 - mbits)
                exp_biased_f32 = (denormal_exp_biased -
                                  left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s) for |: 'SymInt' and 'int'
                # thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32)

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(
        torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f4_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    assert uint8_data.is_contiguous()

    shape = uint8_data.shape

    second_elements = (uint8_data >> 4).to(torch.uint8)
    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements],
                           dim=-1).view(up_size(shape))
    return unpacked


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def convert_to_mxfp4_pytorch(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
):
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if data_hp.dtype == torch.float32:
        hp_int_dtype = torch.int32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = torch.int16
        hp_mbits = 7
        hp_ebits = 8
    sbits = 1
    mbits = 1
    target_max_pow2 = 2

    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape
    if is_2d_block:
        new_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                     block_size, orig_shape[-1] // block_size, block_size)
        block_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                       orig_shape[-1] // block_size, block_size * block_size)
        data_hp = data_hp.reshape(new_shape)

        max_abs = torch.amax(torch.abs(
            data_hp.transpose(-2, -3).reshape(block_shape)),
                             dim=-1)
    else:
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size,
                     block_size)
        data_hp = data_hp.reshape(new_shape)

        max_abs = torch.amax(torch.abs(data_hp), dim=-1)

    # round even (adaptive)
    max_abs = max_abs.view(hp_int_dtype)
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
    scales = torch.clamp(scales, min=1).to(torch.uint8)

    scales_fp = (scales.to(hp_int_dtype) << hp_mbits).view(
        data_hp.dtype).unsqueeze(-1)
    if is_2d_block:
        scales_fp = scales_fp.unsqueeze(-3)
    data_lp = data_hp / scales_fp

    data_lp = data_lp.reshape(orig_shape)
    data_lp = f32_to_f4_unpacked(data_lp.float())
    data_lp = pack_uint4(data_lp)

    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


def convert_from_mxfp4_pytorch(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
) -> torch.Tensor:
    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)

    orig_shape = data_lp.shape
    f4_unpacked = unpack_uint4(data_lp)
    f32 = f4_unpacked_to_f32(f4_unpacked)
    data_hp = f32.to(output_dtype)
    orig_shape = (*orig_shape[:-1], orig_shape[-1] * 2)

    if is_2d_block:
        new_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                     block_size, orig_shape[-1] // block_size, block_size)
        scale_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, 1,
                       orig_shape[-1] // block_size, 1)
    else:
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size,
                     block_size)
        scale_shape = (*orig_shape[:-1], orig_shape[-1] // block_size, 1)

    data_hp = data_hp.reshape(new_shape)
    s_fp = (scales.to(torch.int32) << 23).view(
        torch.float32).reshape(scale_shape).to(output_dtype)
    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(orig_shape)

    return data_hp.transpose(axis, -1)
