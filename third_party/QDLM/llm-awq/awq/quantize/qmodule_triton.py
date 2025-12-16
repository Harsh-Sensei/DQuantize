import math
import torch
import torch.nn as nn
try:
    from .awq_inference_trtion_kernels import awq_gemm_triton, pack_weights_simple
except ImportError:
    # Handle case when run as script
    from awq_inference_trtion_kernels import awq_gemm_triton, pack_weights_simple


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


# Using pack_weights_simple from awq_inference_trtion_kernels


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, dtype=torch.float16):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        
        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert self.in_features % 8 == 0, "in_features must be divisible by 8 for packing"
        pack_num = 32 // self.w_bit  # 8 for 4-bit

        # Triton kernel expects:
        # qweight: [N, K//8] int32
        # scales: [K//G, N] fp16
        # zeros: [K//G, N] fp16 (scaled zeros)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // 8),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=dtype,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=dtype,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=dtype, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            dtype=linear.weight.data.dtype
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        dtype = scales.dtype

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=dtype,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        
        # Transpose scales to [K//G, N] format for Triton kernel
        awq_linear.scales = qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().to(dtype)

        # Quantize weights
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.clamp(0, 15).to(dtype=torch.int32)  # Ensure 4-bit range
        
        # Use simple packing for Triton
        awq_linear.qweight = pack_weights_simple(intweight.contiguous())

        zeros = zeros.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(qscales)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(dtype)
        
        # Transpose scaled_zeros to [K//G, N] format for Triton kernel
        awq_linear.scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass using Triton kernels.
        
        Args:
            x: Input tensor [M, K] or [B, M, K]
        
        Returns:
            Output tensor [M, N] or [B, M, N]
        """
        # Triton kernel handles both GEMV and GEMM automatically
        out = awq_gemm_triton(
            x,
            self.qweight,
            self.scales,
            self.scaled_zeros,
            self.group_size,
        )
        out = out + self.bias if self.bias is not None else out
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


def reference_quantized_matmul(x, qweight, scales, scaled_zeros, group_size, bias=None):
    """
    Reference implementation of quantized matmul using PyTorch operations.
    This manually unpacks and dequantizes weights, then performs standard matmul.
    
    Args:
        x: Input tensor [M, K]
        qweight: Packed weights [N, K//8] int32
        scales: Scales [K//G, N] fp16
        scaled_zeros: Scaled zeros [K//G, N] fp16
        group_size: Group size for quantization
        bias: Optional bias tensor [N]
    
    Returns:
        Output tensor [M, N]
    """
    M, K = x.shape
    N = qweight.shape[0]
    
    # Unpack and dequantize weights
    # qweight: [N, K//8] int32, each int32 contains 8 x 4-bit values
    weight_unpacked = torch.zeros((N, K), dtype=x.dtype, device=x.device)
    
    for k in range(K):
        pack_idx = k // 8
        bit_pos = (k % 8) * 4
        # Extract 4-bit value from packed int32
        qval = ((qweight[:, pack_idx] >> bit_pos) & 0xF).to(x.dtype)
        
        # Get group index for this k position
        group_idx = k // group_size
        
        # Dequantize: weight = qval * scale + scaled_zero
        # Note: scaled_zeros is already -scale * zero_point, so:
        # weight = qval * scale + scaled_zero = qval * scale - scale * zero_point
        weight_unpacked[:, k] = qval * scales[group_idx, :] + scaled_zeros[group_idx, :]
    
    # Standard matmul: x @ weight_unpacked.T
    output = x @ weight_unpacked.T
    
    # Add bias if provided
    if bias is not None:
        output = output + bias
    
    return output


def test_forward_correctness():
    """
    Test the correctness of WQLinear Triton kernels by comparing with
    a reference implementation that manually unpacks quantized weights and
    performs standard PyTorch matmul. This tests kernel correctness, not quantization.
    """
    print("=" * 80)
    print("Testing WQLinear Triton Kernel Correctness")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping test")
        return
    
    dtype = torch.float16
    
    # Test configurations: (in_features, out_features, batch_size, group_size)
    test_configs = [
        (128, 256, 1, 128),      # Single sample
        (512, 1024, 8, 128),     # Small batch
        (1024, 2048, 4, 128),    # Medium size
    ]
    
    all_passed = True
    
    for in_features, out_features, batch_size, group_size in test_configs:
        print(f"\nTesting: in_features={in_features}, out_features={out_features}, "
              f"batch_size={batch_size}, group_size={group_size}")
        
        try:
            # Create a standard linear layer
            linear = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)
            
            # Create random input
            x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
            
            # Get gold output (unquantized FP16 forward pass)
            with torch.no_grad():
                gold_output = linear(x)
            
            # Quantize the linear layer
            weight = linear.weight.data  # [out_features, in_features]
            
            # Group-wise quantization
            num_groups = in_features // group_size
            weight_grouped = weight.reshape(out_features, num_groups, group_size)
            
            # Compute scales and zeros per group
            w_min = weight_grouped.min(dim=2, keepdim=True).values
            w_max = weight_grouped.max(dim=2, keepdim=True).values
            scales = (w_max - w_min) / 15.0
            scales = scales.clamp(min=1e-5)
            zeros = -w_min / scales
            
            # Reshape scales and zeros to [out_features, num_groups] as expected by from_linear
            scales = scales.squeeze(-1)  # [out_features, num_groups]
            zeros = zeros.squeeze(-1)    # [out_features, num_groups]
            
            # Create quantized linear layer
            qlinear = WQLinear.from_linear(
                linear,
                w_bit=4,
                group_size=group_size,
                scales=scales,
                zeros=zeros
            )
            
            # Get output from Triton kernels
            with torch.no_grad():
                triton_output = qlinear(x)
            
            # Get reference output using manual unpacking + PyTorch matmul
            with torch.no_grad():
                ref_output = reference_quantized_matmul(
                    x,
                    qlinear.qweight,
                    qlinear.scales,
                    qlinear.scaled_zeros,
                    qlinear.group_size,
                    qlinear.bias
                )
            
            # Compare Triton output vs reference quantized matmul (kernel correctness test)
            triton_vs_ref_diff = (triton_output - ref_output).abs()
            triton_vs_ref_max = triton_vs_ref_diff.max().item()
            triton_vs_ref_mean = triton_vs_ref_diff.mean().item()
            triton_vs_ref_rel = (triton_vs_ref_diff / (ref_output.abs() + 1e-8)).mean().item()
            
            # Use tight tolerances since we're comparing same quantized computation
            # Small differences may occur due to floating point accumulation order
            atol = 0.1
            rtol = 0.01
            
            kernel_correct = torch.allclose(triton_output, ref_output, atol=atol, rtol=rtol)
            
            # Compare quantized outputs vs gold output (quantization accuracy test)
            triton_vs_gold_diff = (triton_output - gold_output).abs()
            triton_vs_gold_max = triton_vs_gold_diff.max().item()
            triton_vs_gold_mean = triton_vs_gold_diff.mean().item()
            triton_vs_gold_rel = (triton_vs_gold_diff / (gold_output.abs() + 1e-8)).mean().item()
            
            ref_vs_gold_diff = (ref_output - gold_output).abs()
            ref_vs_gold_max = ref_vs_gold_diff.max().item()
            ref_vs_gold_mean = ref_vs_gold_diff.mean().item()
            ref_vs_gold_rel = (ref_vs_gold_diff / (gold_output.abs() + 1e-8)).mean().item()
            
            # Print kernel correctness results
            kernel_status = "PASS" if kernel_correct else "FAIL"
            print(f"  Kernel Correctness (Triton vs Reference): {kernel_status}")
            print(f"    Max diff: {triton_vs_ref_max:.6f}")
            print(f"    Mean diff: {triton_vs_ref_mean:.6f}")
            print(f"    Mean rel diff: {triton_vs_ref_rel:.6f}")
            
            # Print quantization accuracy results
            print(f"  Quantization Accuracy (vs Gold FP16):")
            print(f"    Triton vs Gold - Max: {triton_vs_gold_max:.6f}, Mean: {triton_vs_gold_mean:.6f}, Rel: {triton_vs_gold_rel:.6f}")
            print(f"    Reference vs Gold - Max: {ref_vs_gold_max:.6f}, Mean: {ref_vs_gold_mean:.6f}, Rel: {ref_vs_gold_rel:.6f}")
            
            if not kernel_correct:
                all_passed = False
                print(f"  WARNING: Triton kernel output differs from reference!")
                print(f"    Reference output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")
                print(f"    Triton output range: [{triton_output.min():.4f}, {triton_output.max():.4f}]")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED - Triton kernels match reference implementation")
    else:
        print("SOME TESTS FAILED - Triton kernels may have issues")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    test_forward_correctness()

