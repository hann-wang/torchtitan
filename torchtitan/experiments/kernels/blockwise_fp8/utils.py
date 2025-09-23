# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
import triton.language as tl


USE_FP8E5M2_BWD = False


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target(
    ).arch in (
        "gfx950",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx90a",
        "gfx908",
    )


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def get_f8_fwd_dtype():
    if is_hip() and not is_cdna4():
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn


def get_f8_bwd_dtype():
    if USE_FP8E5M2_BWD:
        if is_hip() and not is_cdna4():
            return torch.float8_e5m2fnuz
        else:
            return torch.float8_e5m2
    return get_f8_fwd_dtype()


def get_tl_f8_bwd_dtype():
    if USE_FP8E5M2_BWD:
        return tl.float8e5b16 if is_hip() and not is_cdna4() else tl.float8e5
    return tl.float8e4b8 if is_hip() and not is_cdna4() else tl.float8e4nv


F8_FWD_MAX: tl.constexpr = torch.finfo(get_f8_fwd_dtype()).max
F8_BWD_MAX: tl.constexpr = torch.finfo(get_f8_bwd_dtype()).max
