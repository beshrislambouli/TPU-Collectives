# TPU v5p Collectives & Tensor Parallelism (Pallas)

High-performance implementation of **collectives and tensor-parallel neural networks** on a **2×2 TPU v5p mesh** using **Pallas** (low-level TPU programming in JAX).

---

## 🚀 Highlights

- Custom **reduce-scatter** and **all-gather** kernels  
- Asynchronous **RDMA-based communication** over TPU ICI  
- Overlap of **communication and computation**
- **Collective matmuls**:
  - `all_gather + matmul`
  - `matmul + reduce_scatter`
- End-to-end **tensor-parallel neural network **
- Achieves ~200+ TFLOP/s (bf16) per device
- Bandwidth-aware implementation (~92 GB/s per ICI link)

---

### Part 1 – Collectives
- `reduce_scatter`
- `all_gather`

Optimized for high ICI utilization using pipelined RDMAs.

### Part 2 – Collective Matmuls
- `matmul`
- `all_gather_matmul`
- `matmul_reduce_scatter`
- `neural_network` (tensor-parallel MLP)

