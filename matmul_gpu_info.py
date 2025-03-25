Test forward type: 2
Tolerance Rate: 0.050000
Open Model /home/kai.wang/MNN/onnx_model/test_matmul_2d/model.mnn
Create CPU Session:
CPU Group: [ 20  21  31  23  25  17  27  19  29  30  22  28  24  18  16  26 ], 800000 - 4300000
CPU Group: [ 14  6  13  1  15  3  4  5  2  7  12  0 ], 800000 - 5500000
CPU Group: [ 10  11  9  8 ], 800000 - 5800000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
c Run on 13
0.000000, before Resize: c - 0
before Resize 2, calling: c - 0 
Resize 1 op for index: 0
Input: 1,1,4,3

=== Input Tensors Information ===
Tensor Name: default
Shape: [3, 4, 1, 1]
ElementSize: 12
DataType: 2
----------------------------

precision=0 in main, 365 
modeNum=1 in main, 370 
stopOp.c_str()=s  in main, 375 

=== Output Tensors Information ===
Op Name: c
Output Tensors:
  - c
----------------------------

Input: a, Shape: [3, 4]
Input: b, Shape: [4, 3]
outputName[0]=c
Start Test 0, opName=c
c Run on 2
0.000000, before Resize: c - 0
before Resize 2, calling: c - 0 

=== MatMul onResize Debug Info ===
Input Tensor Shapes:
- A: [3, 4]
- B: [4, 3]

Computed Parameters:
- Batch Size: 1
- Total Dimensions: 2

Matrix Dimensions:
- E (Output rows): 3
- L (Inner dimension): 4
- H (Output cols): 3

Optimization Flags:
- Large Batch Small GEMM: 0

Padding Information:
- E padded: 8
- L padded: 8
- H padded: 8

Memory Requirements:
- Use RR Layout: 0
- Need A Temp Buffer: 1
- Need B Temp Buffer: 1
- Need Convert Mat AB: 1

=== MatMul Setup Debug Info ===
Matrix Dimensions:
- M (rows of A): 3
- K (cols of A/rows of B): 4
- N (cols of B): 3
Batch size: 1
Precision Mode: FP16/FP32 Mixed
Layout: Row-Column
GPU Compute Capability: 0
Memory Configuration:
- Need Temp Buffer A: 1
- Need Temp Buffer B: 1
- Has Bias: 0

Tensor Addresses:
- Input A: 0x7fffc6c00000
- Input B: 0x7fffc6c00200
- Output: 0x7fffc6c00400
==============================

==============================

Resize 1 op for index: 0
c Run on 13
0.000000, before Resize: c - 0
before Resize 2, calling: c - 0 
Resize 1 op for index: 0
CUDABackend::onExecuteBegin
Group:  c - 0, type=MatMul, inputs: input group: [ 0  0 ], devices: input: [ 140736527859712  140736527860224 ] - output: [ 140736527860736 ]

=== MatMul onExecute Debug Info ===
Execution Configuration:
- Precision: FP16/FP32 Mixed
- Layout: Row-Column
- GPU Compute Cap: 89

Matrix Dimensions:
- M (rows): 3
- K (inner): 4
- N (cols): 3
- Batch Size: 1

Memory Addresses:
- Input A: 0x7fffc6c00000
- Input B: 0x7fffc6c00200
- Output: 0x7fffc6c00400

Conversion Status:
- Need Convert MatAB: 1
- Need A Temp Buffer: 1
- Need B Temp Buffer: 1

Preparing Kernel Execution...
==============================

CUDABackend::onExecuteEnd
Group:  c - 0, type=MatMul, inputs: input group: [ 0  0 ], devices: input: [ 1  1 ] - output: [ 1 ]
Correct ! Run second pass
CUDABackend::onExecuteBegin
Group:  c - 0, type=MatMul, inputs: input group: [ 0  0 ], devices: input: [ 140736527859712  140736527860224 ] - output: [ 140736527860736 ]

=== MatMul onExecute Debug Info ===
Execution Configuration:
- Precision: FP16/FP32 Mixed
- Layout: Row-Column
- GPU Compute Cap: 89

Matrix Dimensions:
- M (rows): 3
- K (inner): 4
- N (cols): 3
- Batch Size: 1

Memory Addresses:
- Input A: 0x7fffc6c00000
- Input B: 0x7fffc6c00200
- Output: 0x7fffc6c00400

Conversion Status:
- Need Convert MatAB: 1
- Need A Temp Buffer: 1
- Need B Temp Buffer: 1

Preparing Kernel Execution...
==============================

CUDABackend::onExecuteEnd
Correct for 0, name=c
Correct !
