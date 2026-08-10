# GeForce RTX 5070 Laptop GPU — Reference Spec Sheet (illustrative)

> This is a sample reference document for benchmarking the harness. Figures are
> illustrative placeholders, not official specifications.

## Overview
The RTX 5070 Laptop GPU is a Blackwell-generation mobile graphics processor
targeting thin-and-light gaming and local-AI creator laptops.

## Memory
- VRAM capacity: 8 GB GDDR7
- Memory interface: 128-bit
- Memory bandwidth: 384 GB/s

## Compute
- CUDA cores: 4608
- Tensor cores: 5th-generation, with FP4 acceleration
- Boost clock: 2100 MHz (varies by laptop power profile)

## Power
- Configurable TGP: 65 W to 115 W depending on chassis
- Recommended system power: 180 W adapter

## Local-AI notes
- 5th-gen Tensor Cores add hardware FP4 support, improving throughput for
  quantized LLM inference relative to the prior generation.
- The 8 GB VRAM budget comfortably holds 4-bit quantized 7–8B-class models with
  short context, but larger models or long context will exceed VRAM and force
  partial CPU offload, reducing tokens/sec.
- TensorRT for RTX can accelerate supported ONNX workloads using the Tensor
  Cores on this GPU.
