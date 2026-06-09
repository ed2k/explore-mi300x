import time
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import Timing, Context

def benchmark_data_path_throughput():
    print("=== 1. Measuring MI300X HBM3 Data Path Throughput ===")
    
    # 2 GB buffer size to saturate the memory controllers and minimize overhead
    num_elements = 512 * 1024 * 1024  # 512M elements
    bytes_per_element = 4             # float32
    total_bytes = num_elements * bytes_per_element
    gb_size = total_bytes / 1e9
    
    # Allocate buffers directly on device VRAM using native AMD backend
    dev = Device["AMD"]
    buf_src = Buffer("AMD", num_elements, dtype=dtypes.float32).allocate()
    buf_dst = Buffer("AMD", num_elements, dtype=dtypes.float32).allocate()
    
    # Create host array to trigger actual device copy
    host_data = np.random.randn(num_elements).astype(np.float32)
    
    # Measure Host to Device (PCIe Gen 5 Data Path)
    with Timing("Host to Device Transfer: "):
        buf_src.copyin(host_data.data)
    
    # Measure Device to Device Copy (SDMA Copy Path - optimized queue submission)
    iterations = 50
    dev.synchronize()
    start = time.perf_counter()
    q = dev.hw_copy_queue_t()
    q.wait(dev.timeline_signal, dev.timeline_value - 1)
    for _ in range(iterations):
        q.copy(buf_dst._buf, buf_src._buf, total_bytes)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize()
    end = time.perf_counter()
    
    elapsed = (end - start) / iterations
    throughput_sdma = ((total_bytes * 2) / elapsed) / 1e12
    print(f"SDMA VRAM-to-VRAM Loop Time: {elapsed*1000:.4f} ms")
    print(f"Measured HBM3 Local Bandwidth (SDMA): {throughput_sdma:.2f} TB/s")
    
    # Measure Device to Device Copy (Compute Shader / Tensor Copy Path)
    # Using Tensors to run on the CDNA 3 Compute cores to saturate the full memory bus
    A = Tensor.rand(num_elements, dtype=dtypes.float32).realize()
    B = Tensor.rand(num_elements, dtype=dtypes.float32).realize()
    dev.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        B.assign(A).realize()
    dev.synchronize()
    end = time.perf_counter()
    
    elapsed = (end - start) / iterations
    throughput_compute = ((total_bytes * 2) / elapsed) / 1e12
    print(f"Compute VRAM-to-VRAM Loop Time: {elapsed*1000:.4f} ms")
    print(f"Measured HBM3 Local Bandwidth (Compute): {throughput_compute:.2f} TB/s")

def benchmark_core_compute_bottleneck():
    print("\n=== 2. Measuring MI300X Core ALU Performance (Matrix/Vector) ===")
    
    dev = Device["AMD"]
    total_cus = dev.cu_cnt * dev.xccs
    print(f"Detected {total_cus} Compute Units across {dev.xccs} XCDs (CU count: {dev.cu_cnt} per XCD).")
    
    # We want Grid_M * Grid_N to be a multiple of total_cus to balance load perfectly
    # 304 CUs * 8 waves per CU = 2432 target workgroups
    # 2432 = 38 * 64.
    Grid_M = 64
    Grid_N = 38
    
    # Standard optimal workgroup tile sizes on CDNA 3 are 256x256
    WG_M = 256
    WG_N = 256
    
    # Calculate matrix dimensions M, N, K
    M = Grid_M * WG_M  # 16384
    N = Grid_N * WG_N  # 9728
    K = 16384          # 16K inner dimension
    
    print(f"Configuring Matrix dimensions to align with CU layout: M={M}, N={N}, K={K}")
    print(f"Grid Layout: {Grid_M}x{Grid_N} = {Grid_M*Grid_N} workgroups (exactly {(Grid_M*Grid_N)/total_cus:.1f} waves per CU)")
    
    # Force creation of un-cached raw matrices
    A = Tensor.rand(M, K, dtype='float16').realize()
    B = Tensor.rand(K, N, dtype='float16').realize()
    
    # Run compiler beam search tuning to find and fill all matrix tiles optimally
    print("Running kernel optimization / beam search to tune matrix tiling...")
    with Context(BEAM=2):
        C = (A @ B).realize()
    Device["AMD"].synchronize()
    
    # Benchmark pure computation core speed using the optimized tuned kernel
    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        C = (A @ B).realize()
    Device["AMD"].synchronize()
    end = time.perf_counter()
    
    # Total FLOPS for MatMul formula: 2 * M * N * K
    total_flops = 2 * M * N * K
    elapsed_per_iter = (end - start) / iterations
    tflops_achieved = (total_flops / elapsed_per_iter) / 1e12
    
    print(f"Matrix Multiplication Time: {elapsed_per_iter*1000:.2f} ms")
    print(f"Achieved CDNA 3 Compute Performance: {tflops_achieved:.2f} TFLOPS")

if __name__ == "__main__":
    # Initialize tinygrad device runtime
    try:
        benchmark_data_path_throughput()
        benchmark_core_compute_bottleneck()
    except Exception as e:
        import traceback
        print("Execution failed:")
        traceback.print_exc()
