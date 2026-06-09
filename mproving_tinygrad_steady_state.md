# Improving tinygrad Steady-State Decoding Performance

This document provides a technical blueprint and analysis of steady-state decoding performance for large language models (specifically Qwen 2.5 72B GGUF) running on the **AMD Instinct MI300X** with **tinygrad**.

---

## 📊 1. Memory Bandwidth vs. Current Efficiency

Decoding in large language models is fundamentally memory-bandwidth bound. During generation, every weight parameter must be read from the accelerator's VRAM to compute the activations for the next single token.

### 📐 Theoretical Limits
*   **Model Size (Qwen 2.5 72B Q4_K_M GGUF):** `47.42 GB`
*   **Peak VRAM Bandwidth (MI300X HBM3):** `5,300 GB/s`
*   **Achieved Compute Copy Bandwidth (tinygrad):** `1,990 GB/s` (based on our Compute Shader benchmarks)

Using these numbers, we can calculate the theoretical upper limits of generation speed (assuming 100% memory bandwidth saturation and zero compute/launch latency):

$$\text{Peak Theoretical Speed} = \frac{5300\text{ GB/s}}{47.42\text{ GB}} \approx \mathbf{111.8\text{ tokens/sec}}$$

$$\text{Compute Shader Copy Limit} = \frac{1990\text{ GB/s}}{47.42\text{ GB}} \approx \mathbf{41.9\text{ tokens/sec}}$$

### 📉 Current Measurement
*   **Measured Steady-State Decode Speed:** **1.60 to 1.83 tokens/sec**
*   **Bandwidth Efficiency:** **~4.3%** of our local compute copy bandwidth, and **~1.6%** of peak hardware capability.

This massive delta proves that the hardware is mostly sitting idle. The bottleneck is not the GPU's memory access speed or raw TFLOPS, but host-side scheduling and launch overhead.

---

## 🛠️ 2. Key Optimization Strategies

To bridge this gap and push tinygrad's steady-state generation speed toward the physical limits of the VRAM, the following four optimizations are required:

### 1. HIP/CUDA Graph Capture (Eliminate Kernel Launch Overhead)
*   **The Problem:** Qwen 2.5 72B has 80 layers. Each layer contains attention projection, QKV, rotary embedding, multi-head attention, FFN gating/up/down projections, layer normalization, dequantization, and scaling. This results in **over 1,600 individual GPU kernels** launched per token step. Launching these sequentially from Python takes `~25–40 ms` of CPU time, completely dominating the execution cycle.
*   **The Solution:** Implement graph recording. On the first decode step, record the sequence of kernel launches and memory addresses. For all subsequent steps, execute a single KFD driver call to launch the entire captured graph. This eliminates CPU-GPU roundtrips and reduces launch latency from `40 ms` to `< 0.1 ms`.

### 2. Dequantize-GEMV Kernel Fusion
*   **The Problem:** In GGUF models, weights are quantized (e.g., 4-bit). If the runtime first runs a dequantization kernel to convert 4-bit weights to FP16 in VRAM, and then runs a GEMV (matrix-vector multiplication) kernel, it reads the weights twice and writes them once. This triples the memory bandwidth requirements.
*   **The Solution:** Implement a fused `W4A16` kernel. The GPU reads the 4-bit weights directly, dequantizes them on-the-fly inside the registers (VGPRs) of the Compute Units, and performs the dot product with the 16-bit activations immediately, writing only the final output activation back to VRAM.

### 3. GPU-Resident Autoregressive Loop
*   **The Problem:** At the end of every token step, tinygrad synchronizes the host CPU with the GPU to extract the argmax token index, check if the token is an End-of-Sequence (EOS) token, and determine the next token input. Each host-device synchronization (via PCIe) adds `~30 μs` of latency.
*   **The Solution:** Move the sampling logic (argmax/softmax), EOS verification, and KV-cache pointer offsets into a small GPU helper kernel at the end of the graph. The GPU can loop autonomously, feeding the new token back into the model's input buffers without waking up the host CPU.

### 4. Vectorized KV-Cache Access
*   **The Problem:** The key-value cache contains the historical attention keys and values. If these are read in non-vectorized blocks, the GPU memory controller cannot saturate the wide HBM3 memory bus.
*   **The Solution:** Restructure the KV-cache layout to support 128-bit vectorized global memory loads (`global_load_dwordx4`) to ensure maximum memory channel utilization during attention computation.
