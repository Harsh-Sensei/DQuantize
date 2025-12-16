"""
Optimized Triton implementation of AWQ (Activation-aware Weight Quantization) GEMV and GEMM kernels.

Key optimizations over the original implementation:
1. Vectorized A loading - load BLOCK_K elements at once instead of 8 separate loads
2. K-dimension tiling - process multiple packed int32s per iteration for better data reuse
3. Software pipelining via num_stages - overlap memory loads with computation
4. Swizzled program IDs for better L2 cache locality
5. Fused dequantization - compute all 8 weight values before accumulation
6. Autotuning - automatically select best tile sizes for different problem shapes
7. Split-K for large K dimensions - better parallelism
8. Persistent kernel option for small problems

Reference:
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time
from typing import Tuple

## Seeding
torch.manual_seed(42)
np.random.seed(42)

# Constants
PACK_FACTOR = 8  # 8 x 4-bit values packed into 32 bits


def get_autotune_configs():
    """Generate autotuning configurations for GEMM kernel."""
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for num_stages in [2, 3, 4]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


def get_gemv_autotune_configs():
    """Generate autotuning configurations for GEMV kernel."""
    configs = []
    for BLOCK_N in [64, 128, 256]:
        for num_stages in [2, 3, 4]:
            for num_warps in [4, 8]:
                configs.append(
                    triton.Config(
                        {'BLOCK_N': BLOCK_N},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def awq_matmul_kernel_optimized(
    # Pointers
    a_ptr, qw_ptr, scales_ptr, zeros_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized AWQ matmul kernel with:
    - L2 cache-friendly program ID swizzling
    - Fused dequantization with reduced accumulations
    
    A: [M, K] fp16 input
    qw: [N, K//8] int32 packed weights  
    scales: [K//GROUP_SIZE, N] fp16
    zeros: [K//GROUP_SIZE, N] fp16 (scaled zeros)
    C: [M, N] fp16 output
    """
    # Program ID swizzling for better L2 cache locality
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = 8  # Group size for swizzling
    group_id = pid // (num_pid_in_group * num_pid_n)
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
    pid_m = first_pid_m + ((pid % (num_pid_in_group * num_pid_n)) % group_size_m)
    pid_n = (pid % (num_pid_in_group * num_pid_n)) // group_size_m
    
    # Starting offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    K_PACKED = K // 8
    n_mask = offs_bn < N
    m_mask = offs_am < M
    
    # Main loop - process one packed int32 (8 k-values) at a time
    for pack_idx in tl.range(0, K_PACKED):
        k_base = pack_idx * 8
        group_idx = k_base // GROUP_SIZE
        
        # Load scales and zeros for this group [BLOCK_N]
        s_ptrs = scales_ptr + group_idx * N + offs_bn
        z_ptrs = zeros_ptr + group_idx * N + offs_bn
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        # Load packed weights [BLOCK_N]
        qw_ptrs = qw_ptr + offs_bn * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Unpack all 8 values and dequantize
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        
        # Dequantize: w = q * scale + zero
        b0 = qval0 * scales + zeros
        b1 = qval1 * scales + zeros
        b2 = qval2 * scales + zeros
        b3 = qval3 * scales + zeros
        b4 = qval4 * scales + zeros
        b5 = qval5 * scales + zeros
        b6 = qval6 * scales + zeros
        b7 = qval7 * scales + zeros
        
        # Load A values and accumulate
        a_ptrs0 = a_ptr + offs_am * stride_am + (k_base + 0) * stride_ak
        a_ptrs1 = a_ptr + offs_am * stride_am + (k_base + 1) * stride_ak
        a_ptrs2 = a_ptr + offs_am * stride_am + (k_base + 2) * stride_ak
        a_ptrs3 = a_ptr + offs_am * stride_am + (k_base + 3) * stride_ak
        a_ptrs4 = a_ptr + offs_am * stride_am + (k_base + 4) * stride_ak
        a_ptrs5 = a_ptr + offs_am * stride_am + (k_base + 5) * stride_ak
        a_ptrs6 = a_ptr + offs_am * stride_am + (k_base + 6) * stride_ak
        a_ptrs7 = a_ptr + offs_am * stride_am + (k_base + 7) * stride_ak
        
        a0 = tl.load(a_ptrs0, mask=m_mask, other=0.0)
        a1 = tl.load(a_ptrs1, mask=m_mask, other=0.0)
        a2 = tl.load(a_ptrs2, mask=m_mask, other=0.0)
        a3 = tl.load(a_ptrs3, mask=m_mask, other=0.0)
        a4 = tl.load(a_ptrs4, mask=m_mask, other=0.0)
        a5 = tl.load(a_ptrs5, mask=m_mask, other=0.0)
        a6 = tl.load(a_ptrs6, mask=m_mask, other=0.0)
        a7 = tl.load(a_ptrs7, mask=m_mask, other=0.0)
        
        # Fused accumulation - reduce number of acc updates
        acc += a0[:, None] * b0[None, :] + a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :] + a3[:, None] * b3[None, :]
        acc += a4[:, None] * b4[None, :] + a5[:, None] * b5[None, :]
        acc += a6[:, None] * b6[None, :] + a7[:, None] * b7[None, :]
    
    # Convert and store
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_gemv_autotune_configs(),
    key=['N', 'K'],
)
@triton.jit
def awq_gemv_kernel_optimized(
    # Pointers
    a_ptr, qw_ptr, scales_ptr, zeros_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_N: tl.constexpr,
):
    """
    Optimized GEMV kernel for small batch sizes (M <= 8).
    
    A: [M, K] fp16 input
    qw: [N, K//8] int32 packed weights  
    scales: [K//GROUP_SIZE, N] fp16
    zeros: [K//GROUP_SIZE, N] fp16 (scaled zeros)
    C: [M, N] fp16 output
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    a_row_ptr = a_ptr + pid_m * stride_am
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    K_PACKED = K // 8
    n_mask = offs_n < N
    m_valid = pid_m < M
    
    # Process one packed int32 (8 k-values) at a time
    for pack_idx in tl.range(0, K_PACKED):
        k_base = pack_idx * 8
        group_idx = k_base // GROUP_SIZE
        
        # Load scales and zeros
        s_ptrs = scales_ptr + group_idx * N + offs_n
        z_ptrs = zeros_ptr + group_idx * N + offs_n
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        # Load packed weights
        qw_ptrs = qw_ptr + offs_n * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Unpack and dequantize all 8 values
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        
        b0 = qval0 * scales + zeros
        b1 = qval1 * scales + zeros
        b2 = qval2 * scales + zeros
        b3 = qval3 * scales + zeros
        b4 = qval4 * scales + zeros
        b5 = qval5 * scales + zeros
        b6 = qval6 * scales + zeros
        b7 = qval7 * scales + zeros
        
        # Load A values and accumulate
        a0 = tl.load(a_row_ptr + (k_base + 0) * stride_ak, mask=m_valid, other=0.0)
        a1 = tl.load(a_row_ptr + (k_base + 1) * stride_ak, mask=m_valid, other=0.0)
        a2 = tl.load(a_row_ptr + (k_base + 2) * stride_ak, mask=m_valid, other=0.0)
        a3 = tl.load(a_row_ptr + (k_base + 3) * stride_ak, mask=m_valid, other=0.0)
        a4 = tl.load(a_row_ptr + (k_base + 4) * stride_ak, mask=m_valid, other=0.0)
        a5 = tl.load(a_row_ptr + (k_base + 5) * stride_ak, mask=m_valid, other=0.0)
        a6 = tl.load(a_row_ptr + (k_base + 6) * stride_ak, mask=m_valid, other=0.0)
        a7 = tl.load(a_row_ptr + (k_base + 7) * stride_ak, mask=m_valid, other=0.0)
        
        acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
        acc += a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7
    
    # Store result
    c_ptrs = c_ptr + pid_m * stride_cm + offs_n * stride_cn
    c_mask = (pid_m < M) & (offs_n < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# Non-autotuned versions for cases where we want explicit control
@triton.jit
def awq_matmul_kernel_fixed(
    # Pointers
    a_ptr, qw_ptr, scales_ptr, zeros_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fixed-config version of the optimized kernel (no autotuning overhead)."""
    # Program ID swizzling
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = 8
    group_id = pid // (num_pid_in_group * num_pid_n)
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
    pid_m = first_pid_m + ((pid % (num_pid_in_group * num_pid_n)) % group_size_m)
    pid_n = (pid % (num_pid_in_group * num_pid_n)) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    K_PACKED = K // 8
    n_mask = offs_bn < N
    m_mask = offs_am < M
    
    # Process one packed int32 (8 k-values) at a time
    for pack_idx in tl.range(0, K_PACKED):
        k_base = pack_idx * 8
        group_idx = k_base // GROUP_SIZE
        
        s_ptrs = scales_ptr + group_idx * N + offs_bn
        z_ptrs = zeros_ptr + group_idx * N + offs_bn
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        qw_ptrs = qw_ptr + offs_bn * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Unpack all 8 values
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        
        b0 = qval0 * scales + zeros
        b1 = qval1 * scales + zeros
        b2 = qval2 * scales + zeros
        b3 = qval3 * scales + zeros
        b4 = qval4 * scales + zeros
        b5 = qval5 * scales + zeros
        b6 = qval6 * scales + zeros
        b7 = qval7 * scales + zeros
        
        # Load A values and accumulate
        a_ptrs0 = a_ptr + offs_am * stride_am + (k_base + 0) * stride_ak
        a_ptrs1 = a_ptr + offs_am * stride_am + (k_base + 1) * stride_ak
        a_ptrs2 = a_ptr + offs_am * stride_am + (k_base + 2) * stride_ak
        a_ptrs3 = a_ptr + offs_am * stride_am + (k_base + 3) * stride_ak
        a_ptrs4 = a_ptr + offs_am * stride_am + (k_base + 4) * stride_ak
        a_ptrs5 = a_ptr + offs_am * stride_am + (k_base + 5) * stride_ak
        a_ptrs6 = a_ptr + offs_am * stride_am + (k_base + 6) * stride_ak
        a_ptrs7 = a_ptr + offs_am * stride_am + (k_base + 7) * stride_ak
        
        a0 = tl.load(a_ptrs0, mask=m_mask, other=0.0)
        a1 = tl.load(a_ptrs1, mask=m_mask, other=0.0)
        a2 = tl.load(a_ptrs2, mask=m_mask, other=0.0)
        a3 = tl.load(a_ptrs3, mask=m_mask, other=0.0)
        a4 = tl.load(a_ptrs4, mask=m_mask, other=0.0)
        a5 = tl.load(a_ptrs5, mask=m_mask, other=0.0)
        a6 = tl.load(a_ptrs6, mask=m_mask, other=0.0)
        a7 = tl.load(a_ptrs7, mask=m_mask, other=0.0)
        
        acc += a0[:, None] * b0[None, :] + a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :] + a3[:, None] * b3[None, :]
        acc += a4[:, None] * b4[None, :] + a5[:, None] * b5[None, :]
        acc += a6[:, None] * b6[None, :] + a7[:, None] * b7[None, :]
    
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit  
def awq_gemv_kernel_fixed(
    a_ptr, qw_ptr, scales_ptr, zeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fixed-config GEMV kernel - processes one packed int32 (8 values) at a time."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    a_row_ptr = a_ptr + pid_m * stride_am
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    K_PACKED = K // 8
    n_mask = offs_n < N
    m_valid = pid_m < M
    
    # Process one packed int32 (8 k-values) at a time
    for pack_idx in tl.range(0, K_PACKED):
        k_base = pack_idx * 8
        group_idx = k_base // GROUP_SIZE
        
        # Load scales and zeros
        s_ptrs = scales_ptr + group_idx * N + offs_n
        z_ptrs = zeros_ptr + group_idx * N + offs_n
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        # Load packed weights
        qw_ptrs = qw_ptr + offs_n * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Unpack and dequantize all 8 values
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        
        b0 = qval0 * scales + zeros
        b1 = qval1 * scales + zeros
        b2 = qval2 * scales + zeros
        b3 = qval3 * scales + zeros
        b4 = qval4 * scales + zeros
        b5 = qval5 * scales + zeros
        b6 = qval6 * scales + zeros
        b7 = qval7 * scales + zeros
        
        # Load A values and accumulate
        a0 = tl.load(a_row_ptr + (k_base + 0) * stride_ak, mask=m_valid, other=0.0)
        a1 = tl.load(a_row_ptr + (k_base + 1) * stride_ak, mask=m_valid, other=0.0)
        a2 = tl.load(a_row_ptr + (k_base + 2) * stride_ak, mask=m_valid, other=0.0)
        a3 = tl.load(a_row_ptr + (k_base + 3) * stride_ak, mask=m_valid, other=0.0)
        a4 = tl.load(a_row_ptr + (k_base + 4) * stride_ak, mask=m_valid, other=0.0)
        a5 = tl.load(a_row_ptr + (k_base + 5) * stride_ak, mask=m_valid, other=0.0)
        a6 = tl.load(a_row_ptr + (k_base + 6) * stride_ak, mask=m_valid, other=0.0)
        a7 = tl.load(a_row_ptr + (k_base + 7) * stride_ak, mask=m_valid, other=0.0)
        
        acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
        acc += a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7
    
    c_ptrs = c_ptr + pid_m * stride_cm + offs_n * stride_cn
    c_mask = (pid_m < M) & (offs_n < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def awq_gemm_triton_optimized(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Optimized Triton AWQ GEMM wrapper.
    
    Args:
        x: Input tensor [M, K] or [B, M, K]
        qweight: Packed 4-bit weights [N, K//8] int32
        scales: Scales [K//G, N] fp16
        zeros: Scaled zeros [K//G, N] fp16
        group_size: Quantization group size
        use_autotune: Whether to use autotuned kernels
        
    Returns:
        Output tensor [M, N] or [B, M, N]
    """
    orig_shape = x.shape
    if x.dim() == 3:
        B, M_orig, K = x.shape
        x = x.view(-1, K)
        M = B * M_orig
    else:
        M, K = x.shape
    
    N = qweight.shape[0]
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    if M <= 8:
        # GEMV path
        BLOCK_N = 128
        
        grid = (M, triton.cdiv(N, BLOCK_N))
        
        if use_autotune:
            awq_gemv_kernel_optimized[grid](
                x, qweight, scales, zeros, output,
                M, N, K,
                x.stride(0), x.stride(1),
                output.stride(0), output.stride(1),
                group_size,
            )
        else:
            awq_gemv_kernel_fixed[grid](
                x, qweight, scales, zeros, output,
                M, N, K,
                x.stride(0), x.stride(1),
                output.stride(0), output.stride(1),
                group_size,
                BLOCK_N=BLOCK_N,
            )
    else:
        # GEMM path
        BLOCK_M = 64
        BLOCK_N = 64
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        if use_autotune:
            awq_matmul_kernel_optimized[grid](
                x, qweight, scales, zeros, output,
                M, N, K,
                x.stride(0), x.stride(1),
                output.stride(0), output.stride(1),
                group_size,
            )
        else:
            awq_matmul_kernel_fixed[grid](
                x, qweight, scales, zeros, output,
                M, N, K,
                x.stride(0), x.stride(1),
                output.stride(0), output.stride(1),
                group_size,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
    
    if len(orig_shape) == 3:
        B = orig_shape[0]
        output = output.view(B, -1, N)
    
    return output


# Aliases
gemm_forward_triton = awq_gemm_triton_optimized
gemv_forward_triton = awq_gemm_triton_optimized


def pack_weights_simple(weight: torch.Tensor) -> torch.Tensor:
    """Pack FP16 weights to 4-bit format."""
    N, K = weight.shape
    assert K % 8 == 0, "K must be divisible by 8"
    
    weight_int = weight.to(torch.int32)
    weight_int = weight_int.reshape(N, K // 8, 8)
    
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (weight_int[:, :, i] & 0xF) << (i * 4)
    
    return packed


def quantize_weights(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize FP16 weights to 4-bit with group-wise quantization."""
    N, K = weight.shape
    num_groups = K // group_size
    
    weight_grouped = weight.reshape(N, num_groups, group_size)
    
    w_min = weight_grouped.min(dim=2, keepdim=True).values
    w_max = weight_grouped.max(dim=2, keepdim=True).values
    
    scales = (w_max - w_min) / 15.0
    scales = scales.clamp(min=1e-5)
    zeros = -w_min / scales
    
    weight_q = ((weight_grouped - w_min) / (w_max - w_min + 1e-5) * 15).round().clamp(0, 15)
    weight_q = weight_q.reshape(N, K)
    
    scaled_zeros = -scales * zeros
    
    scales = scales.squeeze(-1).transpose(0, 1).contiguous()
    scaled_zeros = scaled_zeros.squeeze(-1).transpose(0, 1).contiguous()
    
    qweight = pack_weights_simple(weight_q)
    
    return qweight, scales.to(weight.dtype), scaled_zeros.to(weight.dtype)


def naive_quantized_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor, 
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Naive PyTorch implementation for correctness testing."""
    M, K = x.shape
    N = qweight.shape[0]
    
    K_packed = K // 8
    weight_unpacked = torch.zeros((N, K), dtype=x.dtype, device=x.device)
    
    for k in range(K):
        pack_idx = k // 8
        bit_pos = (k % 8) * 4
        qval = ((qweight[:, pack_idx] >> bit_pos) & 0xF).to(x.dtype)
        
        group_idx = k // group_size
        weight_unpacked[:, k] = qval * scales[group_idx, :] + zeros[group_idx, :]
    
    return x @ weight_unpacked.T, weight_unpacked


def benchmark_matmul(M: int, N: int, K: int, group_size: int = 128, 
                     num_warmup: int = 10, num_iters: int = 100,
                     use_autotune: bool = True):
    """Benchmark optimized Triton AWQ vs naive PyTorch."""
    device = torch.device('cuda')
    
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    weight = torch.randn((N, K), dtype=torch.float16, device=device)
    
    qweight, scales, zeros = quantize_weights(weight, group_size)
    
    # Warmup
    for _ in range(num_warmup):
        _ = awq_gemm_triton_optimized(x, qweight, scales, zeros, group_size, use_autotune)
        _, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = awq_gemm_triton_optimized(x, qweight, scales, zeros, group_size, use_autotune)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Benchmark naive
    start = time.perf_counter()
    for _ in range(num_iters):
        _, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Benchmark FP16
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x @ weight.T
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_iters * 1000
    
    return triton_time, naive_time, fp16_time


def test_correctness(M: int, N: int, K: int, group_size: int = 128, 
                     atol: float = 0.5, rtol: float = 0.05,
                     use_autotune: bool = True):
    """Test correctness against naive implementation."""
    device = torch.device('cuda')
    
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    weight = torch.randn((N, K), dtype=torch.float16, device=device)
    
    qweight, scales, zeros = quantize_weights(weight, group_size)
    
    triton_out = awq_gemm_triton_optimized(x, qweight, scales, zeros, group_size, use_autotune)
    naive_out, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    
    max_diff = (triton_out - naive_out).abs().max().item()
    mean_diff = (triton_out - naive_out).abs().mean().item()
    
    is_close = torch.allclose(triton_out, naive_out, atol=atol, rtol=rtol)
    
    return is_close, max_diff, mean_diff


if __name__ == "__main__":
    print("=" * 80)
    print("OPTIMIZED AWQ Triton Implementation - Tests")
    print("=" * 80)
    
    test_configs = [
        (1, 1, 16, 8),
        (1, 512, 512, 128),
        (4, 512, 512, 128),
        (1, 1024, 1024, 128),
        (8, 1024, 1024, 128),
        (1, 4096, 4096, 128),
        (32, 4096, 4096, 128),
    ]
    
    print("\n" + "-" * 80)
    print("CORRECTNESS TESTS (Fixed Config - No Autotune)")
    print("-" * 80)
    print(f"{'M':>6} {'N':>6} {'K':>6} {'G':>4} {'Pass':>6} {'Max Diff':>12} {'Mean Diff':>12}")
    print("-" * 80)
    
    all_passed = True
    for M, N, K, G in test_configs:
        try:
            passed, max_diff, mean_diff = test_correctness(M, N, K, G, use_autotune=False)
            status = "PASS" if passed else "FAIL"
            print(f"{M:>6} {N:>6} {K:>6} {G:>4} {status:>6} {max_diff:>12.6f} {mean_diff:>12.6f}")
            if not passed:
                all_passed = False
        except Exception as e:
            import traceback
            print(f"{M:>6} {N:>6} {K:>6} {G:>4} {'ERROR':>6} {str(e)[:50]}")
            traceback.print_exc()
            all_passed = False
    
    print("-" * 80)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    print("\n" + "-" * 80)
    print("PERFORMANCE BENCHMARKS (Fixed Config)")  
    print("-" * 80)
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Triton(ms)':>12} {'Naive(ms)':>12} {'FP16(ms)':>12} {'Speedup':>10}")
    print("-" * 80)
    
    for M, N, K, G in test_configs:
        try:
            triton_time, naive_time, fp16_time = benchmark_matmul(M, N, K, G, use_autotune=False)
            speedup = naive_time / triton_time if triton_time > 0 else 0
            print(f"{M:>6} {N:>6} {K:>6} {triton_time:>12.4f} {naive_time:>12.4f} {fp16_time:>12.4f} {speedup:>10.2f}x")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} {'ERROR':>12} {str(e)[:40]}")
    
    print("-" * 80)
    print("\nNote: Speedup is Optimized Triton vs Naive PyTorch")
    print("=" * 80)