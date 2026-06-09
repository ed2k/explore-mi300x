# AMD Instinct MI300X Benchmark & Backend Comparison

The benchmarks were successfully executed on the target hardware using both **tinygrad's native AMD/KFD driver-level backend** and **vLLM (running inside an AMD-optimized ROCm Docker container)**.

## 🖥️ Hardware Specification
*   **Accelerator:** AMD Instinct MI300X (`gfx942`)
*   **Total VRAM:** 205.82 GB HBM3
*   **VRAM Available:** 205.48 GB (99.8% free)
*   **Compute Units:** **304 CUs** (exposing 38 CUs per XCD across 8 Accelerator Complex Dies)

---

## 💾 1. HBM3 Data Path Throughput
We measured data transfer speeds using a **2 GB buffer size** ($512 \text{M}$ float32 elements).

| Data Path / Transfer Type | Time (ms) | Throughput / Bandwidth |
| :--- | :--- | :--- |
| **Host to Device** (PCIe Gen 5) | `284.90 ms` | **~7.54 GB/s** |
| **VRAM-to-VRAM Copy** (SDMA Engine) | `37.55 ms` | **0.11 TB/s** |
| **VRAM-to-VRAM Copy** (Compute Shader) | `2.16 ms` | **1.99 TB/s** |

---

## ⚙️ 2. Core Compute Performance (CDNA 3 Matrix Core ALU)
We benchmarked matrix multiplication performance of half-precision (FP16) matrices dynamically scaled to achieve perfect CU occupancy alignment.

### 📊 Grid & Occupancy Metrics
*   **Matrix Dimensions:** $M=16384, N=9728, K=16384$
*   **Workgroup Tile Size:** $256 \times 256$
*   **Grid Layout:** $64 \times 38 = 2432$ workgroups
*   **CU Occupancy:** **Exactly 8.0 waves** of workgroups per CU (across all 304 CUs)
*   **Matrix Multiplication Time:** `50.68 ms`
*   **Achieved Performance:** **103.05 TFLOPS**

> [!TIP]
> **Why are these dimensions optimal?**
> A grid size that is a multiple of the system's total CUs ($304$) ensures that the workload is distributed completely evenly. When the number of workgroups matches a clean multiple of $304$ (in this case $2432 = 304 \times 8$), all CUs receive exactly the same number of wavefronts (8 waves). This eliminates the **tail effect**, where some CUs finish early and sit idle while waiting for others to finish.

---

## 🚀 3. LLM Backend Speed Comparison (Qwen 2.5 72B Instruct GGUF)
We compared the inference speed of running **Qwen 2.5 72B Instruct Q4_K_M** (47.4 GB) with identical generation constraints (100 tokens maximum, greedy decoding).

### 📊 Performance Summary

| Metric / Phase | tinygrad (Native KFD) | vLLM (ROCm Docker Container) |
| :--- | :--- | :--- |
| **Model Load Time** | `19.00s` | `20.37s` |
| **Warmup & Compilation** | None (JIT runs on first token) | `139.38s` (Warmup + CUDA Graph Capture) |
| **Prefill Latency (Prompt)** | `39.23s` (46-token prompt, JIT compilation) | `< 0.20s` |
| **Steady-state Decode Speed** | **1.60 tokens/sec** | **26.21 tokens/sec** (including prefill) |

---

### 🔍 Architectural Insights

> [!IMPORTANT]
> **Why is vLLM (~26.2 tokens/sec) so much faster than tinygrad (~1.6 tokens/sec) on this model?**
> 
> 1. **CUDA/HIP Graph Capture:** 
>    vLLM pre-captures execution graphs for dynamic sequence shapes during the engine warmup phase. This completely removes the CPU-to-GPU launch overhead during the decode loops. In contrast, tinygrad submits operations sequentially from Python, making host-device queue synchronization a severe bottleneck when looping over 80 transformer layers.
> 
> 2. **ROCm/Triton Optimized Kernels:** 
>    vLLM utilizes highly tuned ROCm-native Triton kernels specifically optimized for PagedAttention, RMSNorm, and GGUF dequantization.
> 
> 3. **JIT Compilation Strategy:** 
>    tinygrad compiles kernels dynamically during the first prefill and decode steps. This leads to high initial latencies (e.g. **39.23s** prefill compilation and **6.64s** first decode step compile), whereas vLLM concentrates compilation overhead upfront during engine startup.
