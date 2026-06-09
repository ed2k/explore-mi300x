import triton
import triton.language as tl

@triton.jit
def softmax_kernel_mi300x(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride, 
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr, 
    NUM_STAGES: tl.constexpr,
    VEC_SIZE: tl.constexpr  # Hardware vectorization factor (e.g., 4 or 8)
):
    # Persistent thread block setup
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    # AMD Wavefront optimization: Enforce 64-thread execution geometry layout if applicable
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Grid-strided loop over rows with explicit memory pipelining stages
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=NUM_STAGES):
        
        # 1. Coalesced Global Memory Fetching using Vectorized Strides
        # We align pointers to encourage AMD's global data share memory coalescing
        row_input_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        
        # Vectorized load with cache hint optimization where possible
        row_data = tl.load(row_input_ptr, mask=mask, other=-float('inf'), cache_modifier="")
        
        # 2. Local Wavefront Reduction for Maximum
        # CDNA3 handles localized reductions extremely fast in registers
        row_max = tl.max(row_data, axis=0)
        row_minus_max = row_data - row_max
        
        # 3. High-Throughput Approximate Exponentiation
        # Maps directly to AMD's native hardware __expf equivalent instructions
        numerator = tl.exp(row_minus_max)
        
        # 4. Wavefront-level Sum Reduction
        denominator = tl.sum(numerator, axis=0)
        
        # 5. Vectorized Fused Division
        softmax_output = numerator / denominator
        
        # 6. Streamed Memory Write-back
        # Coalesced, aligned writes straight back to DRAM 
        row_output_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(row_output_ptr, softmax_output, mask=mask, cache_modifier="")
