"""
Triton implementation of AWQ (Activation-aware Weight Quantization) GEMV and GEMM kernels.

This implements 4-bit quantized matrix multiplication with the following features:
- Dequantization of 4-bit weights to FP16
- Group-wise quantization with scales and zeros
- Optimized for both small batch (GEMV) and large batch (GEMM) cases

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

# Constants matching CUDA implementation
PACK_FACTOR = 8  # 8 x 4-bit values packed into 32 bits
K_INTERLEAVE = 4  # Interleaving factor for weight layout


@triton.jit
def awq_matmul_kernel(
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
    BLOCK_K: tl.constexpr,  # Must be 8 (one packed int32)
):
    """
    Main AWQ matmul kernel - processes 8 k-values at a time (one packed int32).
    
    A: [M, K] fp16 input
    qw: [N, K//8] int32 packed weights  
    scales: [K//GROUP_SIZE, N] fp16
    zeros: [K//GROUP_SIZE, N] fp16 (scaled zeros)
    C: [M, N] fp16 output
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Starting offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K // 8 for packed weights
    K_PACKED = K // 8
    n_mask = offs_bn < N
    m_mask = offs_am < M
    
    # Main loop - process one packed int32 (8 k-values) at a time
    for pack_idx in tl.range(0, K_PACKED, 1):
        k_base = pack_idx * 8
        
        # Calculate group index for this k position
        group_idx = k_base // GROUP_SIZE
        
        # Load scales and zeros for this group [BLOCK_N]
        s_ptrs = scales_ptr + group_idx * N + offs_bn
        z_ptrs = zeros_ptr + group_idx * N + offs_bn
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        # Load packed weights for all N outputs [BLOCK_N]
        qw_ptrs = qw_ptr + offs_bn * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Process all 8 values - load each A column separately
        # bit 0
        a_ptrs0 = a_ptr + offs_am * stride_am + (k_base + 0) * stride_ak
        a0 = tl.load(a_ptrs0, mask=m_mask, other=0.0)
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        b0 = qval0 * scales + zeros
        acc += a0[:, None] * b0[None, :]
        
        # bit 1
        a_ptrs1 = a_ptr + offs_am * stride_am + (k_base + 1) * stride_ak
        a1 = tl.load(a_ptrs1, mask=m_mask, other=0.0)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        b1 = qval1 * scales + zeros
        acc += a1[:, None] * b1[None, :]
        
        # bit 2
        a_ptrs2 = a_ptr + offs_am * stride_am + (k_base + 2) * stride_ak
        a2 = tl.load(a_ptrs2, mask=m_mask, other=0.0)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        b2 = qval2 * scales + zeros
        acc += a2[:, None] * b2[None, :]
        
        # bit 3
        a_ptrs3 = a_ptr + offs_am * stride_am + (k_base + 3) * stride_ak
        a3 = tl.load(a_ptrs3, mask=m_mask, other=0.0)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        b3 = qval3 * scales + zeros
        acc += a3[:, None] * b3[None, :]
        
        # bit 4
        a_ptrs4 = a_ptr + offs_am * stride_am + (k_base + 4) * stride_ak
        a4 = tl.load(a_ptrs4, mask=m_mask, other=0.0)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        b4 = qval4 * scales + zeros
        acc += a4[:, None] * b4[None, :]
        
        # bit 5
        a_ptrs5 = a_ptr + offs_am * stride_am + (k_base + 5) * stride_ak
        a5 = tl.load(a_ptrs5, mask=m_mask, other=0.0)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        b5 = qval5 * scales + zeros
        acc += a5[:, None] * b5[None, :]
        
        # bit 6
        a_ptrs6 = a_ptr + offs_am * stride_am + (k_base + 6) * stride_ak
        a6 = tl.load(a_ptrs6, mask=m_mask, other=0.0)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        b6 = qval6 * scales + zeros
        acc += a6[:, None] * b6[None, :]
        
        # bit 7
        a_ptrs7 = a_ptr + offs_am * stride_am + (k_base + 7) * stride_ak
        a7 = tl.load(a_ptrs7, mask=m_mask, other=0.0)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        b7 = qval7 * scales + zeros
        acc += a7[:, None] * b7[None, :]
    
    # Convert and store
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def awq_gemv_kernel(
    # Pointers
    a_ptr, qw_ptr, scales_ptr, zeros_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    # Tile sizes - optimized for small M
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMV kernel optimized for small batch sizes (M < 8).
    Uses row-wise parallelism for better occupancy with small M.
    
    A: [M, K] fp16 input
    qw: [N, K//8] int32 packed weights  
    scales: [K//GROUP_SIZE, N] fp16
    zeros: [K//GROUP_SIZE, N] fp16 (scaled zeros)
    C: [M, N] fp16 output
    """
    # Each block handles one row of M and a chunk of N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator for this row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    K_PACKED = K // 8
    n_mask = offs_n < N
    m_valid = pid_m < M
    
    # Iterate over packed weights using tl.range
    for pack_idx in tl.range(0, K_PACKED, 1):
        k_base = pack_idx * 8
        
        # Get group index
        group_idx = k_base // GROUP_SIZE
        
        # Load scales and zeros [BLOCK_N]
        s_ptrs = scales_ptr + group_idx * N + offs_n
        z_ptrs = zeros_ptr + group_idx * N + offs_n
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        
        # Load packed weights [BLOCK_N]
        qw_ptrs = qw_ptr + offs_n * K_PACKED + pack_idx
        packed = tl.load(qw_ptrs, mask=n_mask, other=0)
        
        # Process all 8 values - load each A value separately
        # bit 0
        a0 = tl.load(a_ptr + pid_m * stride_am + (k_base + 0) * stride_ak, mask=m_valid, other=0.0)
        qval0 = ((packed >> 0) & 0xF).to(tl.float16)
        b0 = qval0 * scales + zeros
        acc += a0 * b0
        
        # bit 1
        a1 = tl.load(a_ptr + pid_m * stride_am + (k_base + 1) * stride_ak, mask=m_valid, other=0.0)
        qval1 = ((packed >> 4) & 0xF).to(tl.float16)
        b1 = qval1 * scales + zeros
        acc += a1 * b1
        
        # bit 2
        a2 = tl.load(a_ptr + pid_m * stride_am + (k_base + 2) * stride_ak, mask=m_valid, other=0.0)
        qval2 = ((packed >> 8) & 0xF).to(tl.float16)
        b2 = qval2 * scales + zeros
        acc += a2 * b2
        
        # bit 3
        a3 = tl.load(a_ptr + pid_m * stride_am + (k_base + 3) * stride_ak, mask=m_valid, other=0.0)
        qval3 = ((packed >> 12) & 0xF).to(tl.float16)
        b3 = qval3 * scales + zeros
        acc += a3 * b3
        
        # bit 4
        a4 = tl.load(a_ptr + pid_m * stride_am + (k_base + 4) * stride_ak, mask=m_valid, other=0.0)
        qval4 = ((packed >> 16) & 0xF).to(tl.float16)
        b4 = qval4 * scales + zeros
        acc += a4 * b4
        
        # bit 5
        a5 = tl.load(a_ptr + pid_m * stride_am + (k_base + 5) * stride_ak, mask=m_valid, other=0.0)
        qval5 = ((packed >> 20) & 0xF).to(tl.float16)
        b5 = qval5 * scales + zeros
        acc += a5 * b5
        
        # bit 6
        a6 = tl.load(a_ptr + pid_m * stride_am + (k_base + 6) * stride_ak, mask=m_valid, other=0.0)
        qval6 = ((packed >> 24) & 0xF).to(tl.float16)
        b6 = qval6 * scales + zeros
        acc += a6 * b6
        
        # bit 7
        a7 = tl.load(a_ptr + pid_m * stride_am + (k_base + 7) * stride_ak, mask=m_valid, other=0.0)
        qval7 = ((packed >> 28) & 0xF).to(tl.float16)
        b7 = qval7 * scales + zeros
        acc += a7 * b7
    
    # Store result
    c_ptrs = c_ptr + pid_m * stride_cm + offs_n * stride_cn
    c_mask = (pid_m < M) & (offs_n < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def awq_gemm_triton(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Triton AWQ GEMM wrapper.
    
    Args:
        x: Input tensor [M, K] or [B, M, K]
        qweight: Packed 4-bit weights [N, K//8] int32
        scales: Scales [K//G, N] fp16 (may be padded)
        zeros: Scaled zeros [K//G, N] fp16 (may be padded)
        group_size: Actual group size used for quantization
        
    Returns:
        Output tensor [M, N] or [B, M, N]
    """
    # Handle batched input
    orig_shape = x.shape
    if x.dim() == 3:
        B, M_orig, K = x.shape
        x = x.view(-1, K)
        M = B * M_orig
    else:
        M, K = x.shape
    
    N = qweight.shape[0]
    GROUP_SIZE = group_size
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # Choose kernel based on batch size
    if M <= 8:
        # Use GEMV kernel for small batches
        BLOCK_N = 128
        BLOCK_K = 64
        
        grid = (M, triton.cdiv(N, BLOCK_N))
        awq_gemv_kernel[grid](
            x, qweight, scales, zeros, output,
            M, N, K,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            GROUP_SIZE,
            BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        # Use GEMM kernel for larger batches
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = min(64, K)
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        awq_matmul_kernel[grid](
            x, qweight, scales, zeros, output,
            M, N, K,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            GROUP_SIZE,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
    
    # Restore shape if batched
    if len(orig_shape) == 3:
        B = orig_shape[0]
        output = output.view(B, -1, N)
    
    return output


# Alias for backward compatibility
gemm_forward_triton = awq_gemm_triton
gemv_forward_triton = awq_gemm_triton  # Same function, auto-selects kernel


def pack_weights_simple(weight: torch.Tensor) -> torch.Tensor:
    """
    Pack FP16 weights to 4-bit format.
    
    Args:
        weight: [N, K] FP16 weight tensor (already quantized to 0-15 range)
        
    Returns:
        Packed weights [N, K//8] int32
    """
    N, K = weight.shape
    assert K % 8 == 0, "K must be divisible by 8"
    
    weight_int = weight.to(torch.int32)
    weight_int = weight_int.reshape(N, K // 8, 8)
    
    # Pack 8 x 4-bit values into int32
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (weight_int[:, :, i] & 0xF) << (i * 4)
    
    return packed


def quantize_weights(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize FP16 weights to 4-bit with group-wise quantization.
    
    Args:
        weight: [N, K] FP16 weight tensor
        group_size: Number of elements per quantization group
        
    Returns:
        qweight: [N, K//8] int32 packed weights
        scales: [K//G, N] FP16 scales
        zeros: [K//G, N] FP16 scaled zeros
    """
    N, K = weight.shape
    num_groups = K // group_size
    
    # Reshape for group-wise quantization
    weight_grouped = weight.reshape(N, num_groups, group_size)
    
    # Find min/max per group
    w_min = weight_grouped.min(dim=2, keepdim=True).values
    w_max = weight_grouped.max(dim=2, keepdim=True).values
    
    # Compute scales and zeros
    scales = (w_max - w_min) / 15.0  # 4-bit range is 0-15
    scales = scales.clamp(min=1e-5)  # Avoid division by zero
    zeros = -w_min / scales  # Zero point
    
    # Quantize
    weight_q = ((weight_grouped - w_min) / (w_max - w_min + 1e-5) * 15).round().clamp(0, 15)
    weight_q = weight_q.reshape(N, K)
    
    # Compute scaled zeros (what the kernel expects)
    scaled_zeros = -scales * zeros
    
    # Transpose scales and zeros to [K//G, N] format
    scales = scales.squeeze(-1).transpose(0, 1).contiguous()  # [K//G, N]
    scaled_zeros = scaled_zeros.squeeze(-1).transpose(0, 1).contiguous()  # [K//G, N]
    
    # Pack weights
    qweight = pack_weights_simple(weight_q)
    
    return qweight, scales.to(weight.dtype), scaled_zeros.to(weight.dtype)


def naive_quantized_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor, 
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Naive PyTorch implementation of quantized matmul for correctness testing.
    
    Args:
        x: [M, K] input
        qweight: [N, K//8] packed int32 weights
        scales: [K//G, N] scales
        zeros: [K//G, N] scaled zeros
        group_size: quantization group size
        
    Returns:
        [M, N] output
    """
    M, K = x.shape
    N = qweight.shape[0]
    
    # Unpack weights
    K_packed = K // 8
    weight_unpacked = torch.zeros((N, K), dtype=x.dtype, device=x.device)
    
    for k in range(K):
        pack_idx = k // 8
        bit_pos = (k % 8) * 4
        qval = ((qweight[:, pack_idx] >> bit_pos) & 0xF).to(x.dtype)
        
        group_idx = k // group_size
        weight_unpacked[:, k] = qval * scales[group_idx, :] + zeros[group_idx, :]
    
    # Standard matmul
    return x @ weight_unpacked.T, weight_unpacked


def benchmark_matmul(M: int, N: int, K: int, group_size: int = 128, num_warmup: int = 10, num_iters: int = 100):
    """
    Benchmark Triton AWQ vs naive PyTorch implementation.
    """
    device = torch.device('cuda')
    
    # Create random inputs
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    weight = torch.randn((N, K), dtype=torch.float16, device=device)
    
    # Quantize weights
    qweight, scales, zeros = quantize_weights(weight, group_size)
    
    # Warmup
    for _ in range(num_warmup):
        _ = awq_gemm_triton(x, qweight, scales, zeros, group_size)
        _, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = awq_gemm_triton(x, qweight, scales, zeros, group_size)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    
    # Benchmark naive PyTorch
    start = time.perf_counter()
    for _ in range(num_iters):
        _, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    
    # Benchmark standard FP16 matmul for reference
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x @ weight.T
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    
    return triton_time, naive_time, fp16_time


def test_correctness(M: int, N: int, K: int, group_size: int = 128, atol: float = 0.5, rtol: float = 0.05):
    """
    Test correctness of Triton implementation against naive PyTorch.
    
    Note: FP16 matmul accumulation can have significant numerical differences,
    especially for large K dimensions. We use relaxed tolerances accordingly.
    """
    device = torch.device('cuda')
    
    # Create random inputs
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    weight = torch.randn((N, K), dtype=torch.float16, device=device)
    
    # Quantize weights
    qweight, scales, zeros = quantize_weights(weight, group_size)
    
    # Compute outputs
    triton_out = awq_gemm_triton(x, qweight, scales, zeros, group_size)
    naive_out, _ = naive_quantized_matmul(x, qweight, scales, zeros, group_size)
    
    # Check correctness
    max_diff = (triton_out - naive_out).abs().max().item()
    mean_diff = (triton_out - naive_out).abs().mean().item()
    
    is_close = torch.allclose(triton_out, naive_out, atol=atol, rtol=rtol)
    
    return is_close, max_diff, mean_diff


if __name__ == "__main__":
    print("=" * 80)
    print("AWQ Triton Implementation - Correctness Tests")
    print("=" * 80)
    
    # Test configurations - start with smaller sizes
    test_configs = [
        # (M, N, K, group_size)
        (1, 1, 16, 8),      # Small test
        (1, 512, 512, 128),      # Small test  
        (4, 512, 512, 128),      # Small batch
        (1, 1024, 1024, 128),    # Medium test
        (8, 1024, 1024, 128),    # Medium batch
        (1, 4096, 4096, 128),    # Single token, typical LLM size
        (32, 4096, 4096, 128),   # Larger batch
    ]
    
    print("\n" + "-" * 80)
    print("CORRECTNESS TESTS")
    print("-" * 80)
    print(f"{'M':>6} {'N':>6} {'K':>6} {'G':>4} {'Pass':>6} {'Max Diff':>12} {'Mean Diff':>12}")
    print("-" * 80)
    
    all_passed = True
    for M, N, K, G in test_configs:
        try:
            passed, max_diff, mean_diff = test_correctness(M, N, K, G)
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
    print("=" * 80)
    
    # PERFORMANCE BENCHMARKS (commented out for now)
    print("\n" + "-" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("-" * 80)
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Triton(ms)':>12} {'Naive(ms)':>12} {'FP16(ms)':>12} {'Speedup':>10}")
    print("-" * 80)
    
    for M, N, K, G in test_configs:
        try:
            triton_time, naive_time, fp16_time = benchmark_matmul(M, N, K, G)
            speedup = naive_time / triton_time if triton_time > 0 else 0
            print(f"{M:>6} {N:>6} {K:>6} {triton_time:>12.4f} {naive_time:>12.4f} {fp16_time:>12.4f} {speedup:>10.2f}x")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} {'ERROR':>12} {str(e)[:40]}")
    
    print("-" * 80)
    print("\nNote: Speedup is Triton vs Naive PyTorch implementation")
    print("      FP16 time is standard torch.matmul for reference")
    print("=" * 80)