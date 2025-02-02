# NxBitsAndBytes

Elixir port of [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library for 4-bit quantization, enabling efficient loading and storage of large neural networks. 
Built on top of Nx and featuring CUDA integration.

Currently supports FP4 quantization for Float16 tensors.

TODO:
- [ ] Support FP4/NF4 quantization for Float32 tensors
- [ ] Support FP4/NF4 quantization for BFloat16 tensors 
- [ ] Support dequantization


## Disclaimer

This files were copy&pasted directly from https://github.com/bitsandbytes-foundation/bitsandbytes/tree/main/csrc 
- common.cpp
- common.cuh
- common.h
- cpu_ops.cpp
- cpu_ops.h
- kernels.cu
- kernels.cuh
- mps_kernels.metal
- mps_ops.h
- mps_ops.mm
- ops.cu
- ops.cuh