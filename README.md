# NxBitsAndBytes

Elixir port of [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library for 4-bit quantization, enabling efficient loading and storage of large neural networks. 
Built on top of Nx and featuring CUDA integration.

Currently supports FP4 quantization for Float16 tensors.

TODO:
- [ ] Support FP4/NF4 quantization for Float32 tensors
- [ ] Support FP4/NF4 quantization for BFloat16 tensors 
- [ ] Support dequantization