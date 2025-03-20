import os
import time

os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_KERNEL_DUMP'] = '1'
os.environ['TRITON_DUMP_DIR'] = '/workspace/torchtitan/cache'
import triton
import triton.language as tl
import torch

USE_FP8: tl.constexpr = True
USE_B8_F8: tl.constexpr = True


@triton.jit
def matmul_kernel(
    A_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    do0 = tl.load(A_ptr + offs_m[:, None] * stride_am +
                  offs_k[None, :] * stride_ak,
                  mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                  other=0.0)

    v0 = tl.full([BLOCK_K, BLOCK_N], 1., dtype=tl.float32)

    dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if USE_FP8:
        _v0_f8 = v0.to(tl.float8e4b8)
        _v0_b8 = v0.to(tl.float8e5b16)

        # B8@F8 or F8@B8
        if USE_B8_F8:
            _dp = tl.dot(do0, _v0_f8, out_dtype=tl.float32, allow_tf32=False)
        else:
            _dp = tl.trans(
                tl.dot(_v0_f8,
                       tl.trans(do0),
                       out_dtype=tl.float32,
                       allow_tf32=False))
        # it seems that b8 is silently bit-casted to f8 in B8@F8 and F8@B8
        if USE_B8_F8:
            __dp = tl.dot(do0.cast(tl.float8e4b8, bitcast=True),
                          _v0_f8,
                          out_dtype=tl.float32,
                          allow_tf32=False)
        else:
            __dp = tl.trans(
                tl.dot(_v0_f8,
                       tl.trans(do0.cast(tl.float8e4b8, bitcast=True)),
                       out_dtype=tl.float32,
                       allow_tf32=False))
        # B8@B8
        if USE_B8_F8:
            dp = tl.dot(do0, _v0_b8, out_dtype=tl.float32, allow_tf32=False)
        else:
            dp = tl.trans(
                tl.dot(_v0_b8,
                       tl.trans(do0),
                       out_dtype=tl.float32,
                       allow_tf32=False))
        # the expected ratio is 1.0
        tl.device_print('ratio between B8@B8 and B8@F8:',
                        tl.abs(dp) / tl.abs(_dp))
        tl.device_print('ratio between B8.bit_cast(F8)@F8 and B8@F8:',
                        tl.abs(__dp) / tl.abs(_dp))
    else:
        dp = tl.dot(do0, v0, acc=dp, out_dtype=tl.float32, allow_tf32=False)

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             dp,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(A, C):
    M, K = A.shape
    K, N = A.shape

    grid = (1, 1)
    print(A.dtype)
    print(A)
    time.sleep(5)
    matmul_kernel[grid](
        A,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
    )
    return C


if __name__ == "__main__":
    torch.random.manual_seed(0)
    A = torch.ones(128, 128, device='cuda') * 57344.
    # we use tl.full to generate B inside the kernel since B containes only ones
    B = torch.ones(128, 128, device='cuda')

    M, K = A.shape
    K, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    if USE_FP8:
        C = matmul(A.to(torch.float8_e5m2fnuz), C).float()
    else:
        C = matmul(A, C).float()
    print(C)

    print(A @ B)
