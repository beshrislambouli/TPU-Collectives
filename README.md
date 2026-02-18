# TPU v5p Collectives & Tensor Parallelism (Pallas)

High-performance **collectives and tensor-parallel neural networks** on a **2×2 TPU v5p mesh** using **Pallas (JAX)**.


## Kernels

### Collectives
- `reduce_scatter`
- `all_gather`

### Collective Matmuls
- `matmul`
- `all_gather_matmul`
- `matmul_reduce_scatter`
- `neural_network` (tensor-parallel MLP)

---

## Performance (bf16, 2×2 TPU v5p)

**all_gather_matmul**
- rel_rmse: 3.540e-03  
- 209.609 TFLOP/s  

**matmul_reduce_scatter**
- rel_rmse: 2.441e-03  
- 199.660 TFLOP/s  

**neural_network**
- rel_rmse: 7.507e-03  
- 191.334 TFLOP/s  

## Highlights

- Custom `reduce_scatter` and `all_gather`
- Asynchronous **RDMA over ICI**
- Overlap of communication & computation
- Collective matmuls:
  - `all_gather_matmul`
  - `matmul_reduce_scatter`
- End-to-end tensor-parallel MLP
- ~200+ TFLOP/s (bf16) per device
- Bandwidth-aware (~92 GB/s per ICI link)

---
