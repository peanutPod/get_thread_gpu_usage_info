#include "MatMulExecution.hpp"

namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void PackPadFill(
    const T0* A, const T0* B,
    bool transA, bool transB,
    T1* tempA, T1* tempB, const int batchA, const int batchB,
    const int e, const int l, const int h,
    const int ep, const int lp, const int hp,
    DivModFast d_e, DivModFast d_l, DivModFast d_h,
    DivModFast d_lp, DivModFast d_lp2
) {
    T1 zero = (T1)0.0;

    if((char *)A != (char *)tempA) {
        if(transA) { // l * e , just transpose to e * lp
            const int maxCount = batchA * e * lp;
            for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                int bIndex, lpIndex, eIndex, tmp;
                d_lp.divmod(index, tmp, lpIndex);
                d_e.divmod(tmp, bIndex, eIndex);

                if(lpIndex >= l) {
                    tempA[index] = zero;
                    continue;
                }
                tempA[index] = A[bIndex * e * l + lpIndex * e + eIndex];
            }
        } else { // e * l, just pack for l
            if (l & 1 == 0) {
                const int maxCount = batchA * e * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, eIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_e.divmod(tmp, bIndex, eIndex);

                    if(lp2Index + lp2Index >= l) {
                        tempA[index+index] = zero;
                        tempA[index+index+1] = zero;
                        continue;
                    }
                    tempA[index+index] =  A[bIndex * e * l + eIndex * l + lp2Index + lp2Index];
                    tempA[index+index+1] = A[bIndex * e * l + eIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batchA * e * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, eIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_e.divmod(tmp, bIndex, eIndex);
                    if(lpIndex >= l || eIndex >= e) {
                        tempA[index] = zero;
                        continue;
                    }
                    tempA[index] = A[bIndex * e * l + eIndex * l + lpIndex];
                }
            }
        }
    }
    if((char *)B != (char *)tempB) {
        if(!transB) { // l * h 
            const int maxCount = batchB * lp * h;
            if(h == hp) { // and h already packed, just pack for l -> lp * h
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hpIndex, bIndex, tmp;
                    d_h.divmod(index, tmp, hpIndex);
                    d_lp.divmod(tmp, bIndex, lpIndex);

                    if(lpIndex >= l || hpIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hpIndex];
                }
            } else { // and h not packed, just transpose and pack for l -> h * lp
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lpIndex >= l || hIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hIndex];
                }
            }
        } else { // h * l, just pack for l
            if(l & 1 == 0) {
                const int maxCount = batchB * h * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, hIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lp2Index + lp2Index >= l) {
                        tempB[index+index] = zero;
                        tempB[index+index+1] = zero;
                        continue;
                    }
                    tempB[index+index] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index];
                    tempB[index+index+1] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batchB * h * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);

                    if(lpIndex >= l || hIndex >= h) {
                        tempB[index] = zero;
                        continue;
                    }
                    tempB[index] = B[bIndex * h * l + hIndex * l + lpIndex];
                }
            }
        }
    }

}

template<typename T0, typename T1>
__global__ void GENERAL_BATCH_MATMUL(
    const T0* A, const T0* B, const T0* bias,
    bool transA, bool transB,
    const int coefBatchA, const int coefBatchB,
    const int e, const int l, const int h,
    const int maxCount, T1* C,
    DivModFast d_e, DivModFast d_h
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int bIndex, hIndex, eIndex, tmp;
        d_h.divmod(index, tmp, hIndex);
        d_e.divmod(tmp, bIndex, eIndex);

        float sum = 0.0;
        // [b, e, l] x [b, l, h] -> [b, e, h]
        if(!transA && !transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * e + eIndex) * l;
            const T0* basePtrB = B + (coefBatchB * bIndex * l + 0) * h + hIndex;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i] * (float)basePtrB[i * h];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }

        // [b, l, e] x [b, l, h] -> [b, e, h]
        if(transA && !transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * l + 0) * e + eIndex;
            const T0* basePtrB = B + (coefBatchB * bIndex * l + 0) * h + hIndex;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i * e] * (float)basePtrB[i * h];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }     

        // [b, l, e] x [b, h, l] -> [b, e, h]
        if(transA && transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * l + 0) * e + eIndex;
            const T0* basePtrB = B + (coefBatchB * bIndex * h + hIndex) * l + 0;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i * e] * (float)basePtrB[i];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }    

        // [b, e, l] x [b, h, l] -> [b, e, h]
        if(!transA && transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * e + eIndex) * l + 0;
            const T0* basePtrB = B + (coefBatchB * bIndex * h + hIndex) * l + 0;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i] * (float)basePtrB[i];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }    
    }
}

MatMulExecution::MatMulExecution(bool transposeA, bool transposeB, Backend *backend, int aS, int bS, int cS) : 
    #ifdef ENABLE_CUDA_TUNE_PARAM
    CutlassGemmTuneCommonExecution(backend)
    #else
    Execution(backend)
    #endif
{
    mTransposeA = transposeA;
    mTransposeB = transposeB;
    mBackend = backend;
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
    mAs = aS;
    mBs = bS;
    mCs = cS;
}
MatMulExecution::~ MatMulExecution() {
    // do nothing
}

void MatMulExecution::setArguments(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Add debug prints at the start
    MNN_PRINT("\n=== MatMul Setup Debug Info ===\n");
    MNN_PRINT("Matrix Dimensions:\n");
    MNN_PRINT("- M (rows of A): %d\n", mGemmInfo.elh[0]);
    MNN_PRINT("- K (cols of A/rows of B): %d\n", mGemmInfo.elh[1]);
    MNN_PRINT("- N (cols of B): %d\n", mGemmInfo.elh[2]);
    MNN_PRINT("Batch size: %d\n", mBatch);
    
    MNN_PRINT("Precision Mode: %s\n", 
        mFp16Infer ? "FP16" : 
        mFp32Infer ? "FP32" : 
        "FP16/FP32 Mixed");
    
    MNN_PRINT("Layout: %s\n", mUseRRLayout ? "Row-Row" : "Row-Column");
    MNN_PRINT("GPU Compute Capability: %d\n", mGpuComputeCap);
    
    MNN_PRINT("Memory Configuration:\n");
    MNN_PRINT("- Need Temp Buffer A: %d\n", mNeedATempBuffer);
    MNN_PRINT("- Need Temp Buffer B: %d\n", mNeedBTempBuffer);
    MNN_PRINT("- Has Bias: %d\n", inputs.size() > 2);
    
    // Original code continues...
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    
    // Add debug print for tensor addresses
    MNN_PRINT("\nTensor Addresses:\n");
    MNN_PRINT("- Input A: %p\n", inputs[0]->deviceId());
    MNN_PRINT("- Input B: %p\n", inputs[1]->deviceId());
    MNN_PRINT("- Output: %p\n", outputs[0]->deviceId());
    if (inputs.size() > 2) {
        MNN_PRINT("- Bias: %p\n", inputs[2]->deviceId());
    }
    MNN_PRINT("==============================\n\n");

    // Rest of the original implementation...
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto C = outputs[0];
    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elh[2], mGemmInfo.elhPad[1]);// m n k

    if (inputs.size() > 2) {
        mBiasPtr = (void*)inputs[2]->deviceId();
        beta = ElementComputeEpilogue(1);
    }
    if(mFp32Infer) {
        if(mUseRRLayout) {
            typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                                                {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBatch};                // batch_count

            size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF32F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                                {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBatch};                // batch_count

            size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF32F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status); 
        }
        return;
    }

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);

    if(mGpuComputeCap < 75) {
        if(mFp16Infer) {
            if(mUseRRLayout) {
                typename GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count
    
                size_t workspace_size = GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RR.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedCudaF16F16LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status); 
            } else {
                typename GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RC.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedCudaF16F16LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else {
            if(mUseRRLayout) {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                        {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                        (int64_t)(0), // batch_stride_bias
                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        mBatch};                // batch_count
    
                    size_t workspace_size = GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
    
                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RR.can_implement(arguments);
                    cutlass_check(status);
    
                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF16F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status);
                } else {
                    typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                        (int64_t)(0), // batch_stride_bias
                                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                        {alpha, beta},          // <- tuple of alpha and beta
                                                        mBatch};                // batch_count
    
                    size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Row::get_workspace_size(arguments);
    
                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR.can_implement(arguments);
                    cutlass_check(status);
    
                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF32F32LnAlign1RR.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status);
                }
            } else {
                if(mNeedConvertMatAB) {
                    typename GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RC.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF16F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                } else {
                    typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                        {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                        {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                                        {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                        (int64_t)(0), // batch_stride_bias
                                                        {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                        (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                        {alpha, beta},          // <- tuple of alpha and beta
                                                        mBatch};                // batch_count

                    size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

                    if(workspace_size != 0) {
                        workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                        mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                        mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                    }
                    // Check the problem size is supported or not 
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC.can_implement(arguments);
                    cutlass_check(status);

                    // Initialize CUTLASS kernel with arguments and workspace pointer
                    status = mGemmBatchedCudaF32F32LnAlign1RC.initialize(arguments, (uint8_t *)mWorkspace);
                    cutlass_check(status); 
                }
            }
        }
        return;
    }

    if(mFp16Infer) {
    #ifdef ENABLE_CUDA_TUNE_PARAM
        if(mGpuComputeCap >= 80) {
            mIsTuned = true;
            /*
            // 0 -> Gemm, 1~N -> BatchGemm
            int32_t batchSize = 0;
            // [0]->A, [1]->B, [2]->bias, [3]->output
            std::pair<void *, int32_t> ptrOffset[4]; 
            int32_t batchOffset[4];
            // [0]->alpha, [1]->beta, [2]->splitK
            int32_t coefs[3]; 
            // 0 -> RowColumn, 1 -> RowRow
            int32_t layout;
            bool epilogueVectorize
            */
            mInfo.problemSize[0] = mGemmInfo.elh[0];
            mInfo.problemSize[1] = mGemmInfo.elh[2];
            mInfo.problemSize[2] = mGemmInfo.elhPad[1];

            mInfo.coefs[0] = 1;
            mInfo.coefs[1] = 0;
            if (inputs.size() > 2) {
                mInfo.coefs[1] = 1;
            }
            mInfo.epilogueVectorize = true;
            mInfo.epilogueType = 0;// Linear
            mInfo.precisionType = 2;// FP16_FP16
            mInfo.backend = mBackend;

            if(mUseRRLayout) {
                mInfo.batchSize = mBatch;
                mInfo.layout = 1;

                mInfo.ptrOffset[0] = std::make_pair((void *)mTempMatA, mGemmInfo.elhPad[1]);
                mInfo.ptrOffset[1] = std::make_pair((void *)mTempMatB, mGemmInfo.elhPad[2]);
                mInfo.ptrOffset[2] = std::make_pair((void *)mBiasPtr, 0);
                mInfo.ptrOffset[3] = std::make_pair((void *)C->deviceId(), mGemmInfo.elhPad[2]);

                mInfo.batchOffset[0] = mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs;
                mInfo.batchOffset[1] = mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs;
                mInfo.batchOffset[2] = 0;
                mInfo.batchOffset[3] = mGemmInfo.elh[0] * mGemmInfo.elhPad[2];
            } else {
                if(hAlignment) {
                    mInfo.epilogueVectorize = true;
                } else {
                    mInfo.epilogueVectorize = false;
                }

                if(hAlignment && mConvertGemmSplitK) {
                    mInfo.batchSize = 0;
                    mInfo.layout = 0;
                    mInfo.coefs[2] = 16;

                    mInfo.ptrOffset[0] = std::make_pair((void *)mTempMatA, mGemmInfo.elhPad[1]);
                    mInfo.ptrOffset[1] = std::make_pair((void *)mTempMatB, mGemmInfo.elhPad[1]);
                    mInfo.ptrOffset[2] = std::make_pair((void *)mBiasPtr, 0);
                    mInfo.ptrOffset[3] = std::make_pair((void *)C->deviceId(), mGemmInfo.elh[2]);
                } else {
                    mInfo.batchSize = mBatch;
                    mInfo.layout = 0;
        
                    mInfo.ptrOffset[0] = std::make_pair((void *)mTempMatA, mGemmInfo.elhPad[1]);
                    mInfo.ptrOffset[1] = std::make_pair((void *)mTempMatB, mGemmInfo.elhPad[1]);
                    mInfo.ptrOffset[2] = std::make_pair((void *)mBiasPtr, 0);
                    mInfo.ptrOffset[3] = std::make_pair((void *)C->deviceId(), mGemmInfo.elh[2]);
        
                    mInfo.batchOffset[0] = mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs;
                    mInfo.batchOffset[1] = mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs;
                    mInfo.batchOffset[2] = 0;
                    mInfo.batchOffset[3] = mGemmInfo.elh[0] * mGemmInfo.elh[2];
                }
            }
            getGemmBatchedTensorCoreFloat16Param(&mInfo);

            // set preferd block shape argments
            setGemmBatchedTensorCoreFloat16Argments(&mInfo);
        }
    #endif
        if(!mIsTuned) {
            if(mUseRRLayout) {
                typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                    {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF16F16LnAlign8RRSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF16F16LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status); 
            } else {
                if(hAlignment) {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F16_F16_Linear_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F16_F16_Linear_AlignTensor_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }

                        cutlass::Status status = mGemmF16F16LnAlign8Sm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF16F16LnAlign8Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                            {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                            (int64_t)(0), // batch_stride_bias
                            {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                            {alpha, beta},          // <- tuple of alpha and beta
                            mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF16F16LnAlign8RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF16F16LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F16_F16_Linear_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F16_F16_Linear_AlignCuda_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }

                        cutlass::Status status = mGemmF16F16LnAlign1Sm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF16F16LnAlign1Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                            {(ElementOutput_F16 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                            (int64_t)(0), // batch_stride_bias
                            {(ElementOutput_F16 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                            {alpha, beta},          // <- tuple of alpha and beta
                            mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF16F16LnAlign1RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF16F16LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    }
                }
            }
        }
    } else {
        if(mUseRRLayout) {
            if(mNeedConvertMatAB) {
                typename GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                    {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                    {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                                    {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                    (int64_t)(0), // batch_stride_bias
                                    {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                    {alpha, beta},          // <- tuple of alpha and beta
                                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF16F32LnAlign8RRSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF16F32LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                    {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]* mBs), // batch_stride_B
                                                    {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                    (int64_t)(0), // batch_stride_bias
                                                    {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    mBatch};                // batch_count

                size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Row_Sm75::get_workspace_size(arguments);

                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmBatchedF32F32LnAlign8RRSm75.can_implement(arguments);
                cutlass_check(status);

                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmBatchedF32F32LnAlign8RRSm75.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        } else {
            if(hAlignment) {
                if(mNeedConvertMatAB) {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F16_F32_Linear_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F16_F32_Linear_AlignTensor_Sm75::get_workspace_size(arguments);
    
                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
    
                        cutlass::Status status = mGemmF16F32LnAlign8Sm75.can_implement(arguments);
                        cutlass_check(status);
    
                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF16F32LnAlign8Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF16F32LnAlign8RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF16F32LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F32_F32_Linear_AlignTensor_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F32_F32_Linear_AlignTensor_Sm75::get_workspace_size(arguments);
    
                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
    
                        cutlass::Status status = mGemmF32F32LnAlign8Sm75.can_implement(arguments);
                        cutlass_check(status);
    
                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF32F32LnAlign8Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                            {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF32F32LnAlign8RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF32F32LnAlign8RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status); 
                    }
                }
            } else {
                if(mNeedConvertMatAB) {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F16_F32_Linear_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F16_F32_Linear_AlignCuda_Sm75::get_workspace_size(arguments);
    
                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
    
                        cutlass::Status status = mGemmF16F32LnAlign1Sm75.can_implement(arguments);
                        cutlass_check(status);
    
                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF16F32LnAlign1Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                {(ElementInput_F16 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                                {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF16F32LnAlign1RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF16F32LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status); 
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        int split_k_slices = 16;
                        typename GemmTensor_F32_F32_Linear_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                            {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                            {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                            {alpha, beta},          // <- tuple of alpha and beta
                            split_k_slices};        // <- k-dimension split factor
                        size_t workspace_size = GemmTensor_F32_F32_Linear_AlignCuda_Sm75::get_workspace_size(arguments);
    
                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
    
                        cutlass::Status status = mGemmF32F32LnAlign1Sm75.can_implement(arguments);
                        cutlass_check(status);
    
                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmF32F32LnAlign1Sm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    } else {
                        typename GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                            {(ElementInput_F32 *)mTempMatA, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]* mAs), // batch_stride_A
                                                            {(ElementInput_F32 *)mTempMatB, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elh[2]* mBs), // batch_stride_B
                                                            {(ElementOutput_F32 *)mBiasPtr, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                            (int64_t)(0), // batch_stride_bias
                                                            {(ElementOutput_F32 *)C->deviceId(), mGemmInfo.elh[2]},  //  Ptr + ldm
                                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elh[2]),  // batch_stride_C
                                                            {alpha, beta},          // <- tuple of alpha and beta
                                                            mBatch};                // batch_count

                        size_t workspace_size = GemmBatchedTensor_F32_F32_Linear_AlignCuda_Row_Column_Sm75::get_workspace_size(arguments);

                        if(workspace_size != 0) {
                            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                            mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                        }
                        // Check the problem size is supported or not 
                        cutlass::Status status = mGemmBatchedF32F32LnAlign1RCSm75.can_implement(arguments);
                        cutlass_check(status);

                        // Initialize CUTLASS kernel with arguments and workspace pointer
                        status = mGemmBatchedF32F32LnAlign1RCSm75.initialize(arguments, (uint8_t *)mWorkspace);
                        cutlass_check(status);
                    }
                }
            }
        }
    }
}

ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_PRINT("\n=== MatMul onResize Debug Info ===\n");

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto C = outputs[0];

    // Print input tensor shapes
    MNN_PRINT("Input Tensor Shapes:\n");
    MNN_PRINT("- A: [");
    for (int i = 0; i < A->dimensions(); i++) {
        MNN_PRINT("%d%s", A->length(i), i < A->dimensions()-1 ? ", " : "]\n");
    }
    MNN_PRINT("- B: [");
    for (int i = 0; i < B->dimensions(); i++) {
        MNN_PRINT("%d%s", B->length(i), i < B->dimensions()-1 ? ", " : "]\n");
    }

    auto dimensions = C->dimensions();
    mBatch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        mBatch *= C->length(i);
    }

    // Print basic parameters
    MNN_PRINT("\nComputed Parameters:\n");
    MNN_PRINT("- Batch Size: %d\n", mBatch);
    MNN_PRINT("- Total Dimensions: %d\n", dimensions);

    auto e = C->length(dimensions-2);
    auto h = C->length(dimensions-1);
    auto w0 = inputs[0]->length(dimensions-1);
    auto h0 = inputs[0]->length(dimensions-2);

    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }

    // Print matrix dimensions
    MNN_PRINT("\nMatrix Dimensions:\n");
    MNN_PRINT("- E (Output rows): %d\n", e);
    MNN_PRINT("- L (Inner dimension): %d\n", l);
    MNN_PRINT("- H (Output cols): %d\n", h);

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;

    mLargeBatchSmallGemm = (mBatch > 2048 && l < 8 && e < 8 && h < 8);
    
    MNN_PRINT("\nOptimization Flags:\n");
    MNN_PRINT("- Large Batch Small GEMM: %d\n", mLargeBatchSmallGemm);

    if(mLargeBatchSmallGemm) {
        MNN_PRINT("Early return due to Large Batch Small GEMM optimization\n");
        MNN_PRINT("==============================\n\n");
        return NO_ERROR;
    }

    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;

    // Print padding information
    MNN_PRINT("\nPadding Information:\n");
    MNN_PRINT("- E padded: %d\n", mGemmInfo.elhPad[0]);
    MNN_PRINT("- L padded: %d\n", mGemmInfo.elhPad[1]);
    MNN_PRINT("- H padded: %d\n", mGemmInfo.elhPad[2]);

    bool lAlignment = (mGemmInfo.elhPad[1] == mGemmInfo.elh[1]);
    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);
    bool needBTranspose = (!mTransposeB && !hAlignment);

    mUseRRLayout = (!mTransposeB && hAlignment);
    mNeedATempBuffer = (mTransposeA || !lAlignment);
    mNeedBTempBuffer = (needBTranspose || !lAlignment);
    mNeedConvertMatAB = (mNeedATempBuffer || mNeedBTempBuffer);

    // Print memory requirements
    MNN_PRINT("\nMemory Requirements:\n");
    MNN_PRINT("- Use RR Layout: %d\n", mUseRRLayout);
    MNN_PRINT("- Need A Temp Buffer: %d\n", mNeedATempBuffer);
    MNN_PRINT("- Need B Temp Buffer: %d\n", mNeedBTempBuffer);
    MNN_PRINT("- Need Convert Mat AB: %d\n", mNeedConvertMatAB);


    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    MemChunk bufferAData, bufferBData;
    size_t convertBytes = 2;
    if(mFp32Infer) {
        convertBytes = 4;
    }
    if((mNeedConvertMatAB && mFp16Fp32MixInfer) || mNeedATempBuffer) {
        bufferAData = pool->alloc(convertBytes * mBatch * mAs * mGemmInfo.elh[0] * mGemmInfo.elhPad[1]);
        mTempMatA = (void*)bufferAData.ptr();
    } else {
        mTempMatA = (void *)A->deviceId();
    }

    if((mNeedConvertMatAB && mFp16Fp32MixInfer) || mNeedBTempBuffer) {
        bufferBData = pool->alloc(convertBytes * mBatch * mBs * mGemmInfo.elh[2] * mGemmInfo.elhPad[1]);
        mTempMatB = (void*)bufferBData.ptr();
    } else {
        mTempMatB = (void *)B->deviceId();
    }

    if(bufferAData.first != nullptr) {
        pool->free(bufferAData);
    }
    if(bufferBData.first != nullptr) {
        pool->free(bufferBData);
    }
 
    // inputSize only two, No need Bias, Fake address for mBiasPtr is ok because beta is zero.
    if(inputs.size() == 2) {
    	mBiasPtr = (void*)B->deviceId();
    }
    //printf("MatMulAB:%p-%p-%p-%p\n", A->host<void*>(), A->deviceId(), B->host<void*>(), B->deviceId());

    mConvertGemmSplitK = ((mBatch == 1) && (mGemmInfo.elhPad[1] >= 16384));
    // Set Cutlass Param Arguments
    mResizeSetArgument = (mTempMatA != nullptr && mTempMatB != nullptr && C->deviceId() != 0);
    if(mResizeSetArgument) {
        setArguments(inputs, outputs);
}
    MNN_PRINT("==============================\n\n");
    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_PRINT("\n=== MatMul onExecute Debug Info ===\n");
    
    // Print execution mode and configuration
    MNN_PRINT("Execution Configuration:\n");
    MNN_PRINT("- Precision: %s\n", 
        mFp16Infer ? "FP16" : 
        mFp32Infer ? "FP32" : 
        "FP16/FP32 Mixed");
    MNN_PRINT("- Layout: %s\n", mUseRRLayout ? "Row-Row" : "Row-Column");
    MNN_PRINT("- GPU Compute Cap: %d\n", mGpuComputeCap);
    
    // Print matrix dimensions
    MNN_PRINT("\nMatrix Dimensions:\n");
    MNN_PRINT("- M (rows): %d\n", mGemmInfo.elh[0]);
    MNN_PRINT("- K (inner): %d\n", mGemmInfo.elh[1]);
    MNN_PRINT("- N (cols): %d\n", mGemmInfo.elh[2]);
    MNN_PRINT("- Batch Size: %d\n", mBatch);

    // Get runtime info
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    bool hAlignment = (mGemmInfo.elhPad[2] == mGemmInfo.elh[2]);
    
    // Print memory addresses
    MNN_PRINT("\nMemory Addresses:\n");
    MNN_PRINT("- Input A: %p\n", inputs[0]->deviceId());
    MNN_PRINT("- Input B: %p\n", inputs[1]->deviceId());
    MNN_PRINT("- Output: %p\n", outputs[0]->deviceId());
    if (inputs.size() > 2) {
        MNN_PRINT("- Bias: %p\n", inputs[2]->deviceId());
    }

    // Special case handling for large batch small gemm
    if (mLargeBatchSmallGemm) {
        MNN_PRINT("\nUsing Large Batch Small GEMM optimization\n");
        auto total = mBatch * mGemmInfo.elh[0] * mGemmInfo.elh[2];
        DivModFast eD(mGemmInfo.elh[0]);
        DivModFast hD(mGemmInfo.elh[2]);
        int block_num = runtime->blocks_num(total);
        int block_size = runtime->threads_num();

        void * biasPtr = nullptr;
        if(inputs.size() > 2) {
            biasPtr = (void *)inputs[2]->deviceId();
        }
        if(mFp16Infer) {
            GENERAL_BATCH_MATMUL<<<block_num, block_size>>>((const half*)inputs[0]->deviceId(), \
                    (const half*)inputs[1]->deviceId(), (const half*)biasPtr, \
                    mTransposeA, mTransposeB, mAs, mBs, \
                    mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    total, (half*)outputs[0]->deviceId(), \
                    eD, hD);
            checkKernelErrors;        
        } else {
            GENERAL_BATCH_MATMUL<<<block_num, block_size>>>((const float*)inputs[0]->deviceId(), \
                    (const float*)inputs[1]->deviceId(), (const float*)biasPtr, \
                    mTransposeA, mTransposeB, mAs, mBs, \
                    mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    total, (float*)outputs[0]->deviceId(), \
                    eD, hD);
            checkKernelErrors;   
        }
        MNN_PRINT("==============================\n\n");
        return NO_ERROR;
    }

    // Print conversion info
    MNN_PRINT("\nConversion Status:\n");
    MNN_PRINT("- Need Convert MatAB: %d\n", mNeedConvertMatAB);
    MNN_PRINT("- Need A Temp Buffer: %d\n", mNeedATempBuffer);
    MNN_PRINT("- Need B Temp Buffer: %d\n", mNeedBTempBuffer);

    // Before kernel execution
    MNN_PRINT("\nPreparing Kernel Execution...\n");

    // Add execution timing
    auto startTime = std::chrono::high_resolution_clock::now();

    // PreProcess for Alignment
    if(mNeedConvertMatAB) {
        int aBatch = mBatch;
        int bBatch = mBatch;
        if (mAs == 0) {
            aBatch = 1;
        }
        if (mBs == 0) {
            bBatch = 1;
        }
        DivModFast eD(mGemmInfo.elh[0]);
        DivModFast lD(mGemmInfo.elh[1]);
        DivModFast hD(mGemmInfo.elh[2]);
        DivModFast lpD((mGemmInfo.elhPad[1]));
        DivModFast lp2D((mGemmInfo.elhPad[1]/2));

        auto& prop = runtime->prop();
        int block_num = prop.multiProcessorCount;
        int block_size = prop.maxThreadsPerBlock;
        if(mFp32Infer) {
            PackPadFill<<<block_num, block_size>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (float*)mTempMatA, (float*)mTempMatB,
                    aBatch, bBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2], \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;        
        } else if(mFp16Fp32MixInfer) {
            PackPadFill<<<block_num, block_size>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (half*)mTempMatA, (half*)mTempMatB,
                    aBatch, bBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2], \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;
        } else {
            PackPadFill<<<block_num, block_size>>>((const half*)inputs[0]->deviceId(), (const half*)inputs[1]->deviceId(), \
                    mTransposeA, mTransposeB, (half*)mTempMatA, (half*)mTempMatB,
                    aBatch, bBatch, mGemmInfo.elh[0], mGemmInfo.elh[1], mGemmInfo.elh[2], \
                    mGemmInfo.elhPad[0], mGemmInfo.elhPad[1], mGemmInfo.elhPad[2],  \
                    eD, lD, hD, lpD, lp2D);
            checkKernelErrors;  
        }
    }

    if(!mResizeSetArgument) {
        // Repeat set cutlass argments if possible
        //printf("argment onexecute set\n");

        if(!mNeedConvertMatAB) {
            mTempMatA = (void *)inputs[0]->deviceId();
            mTempMatB = (void *)inputs[1]->deviceId();
        }
        setArguments(inputs, outputs);
    }


    if(mFp32Infer) {
        if(mUseRRLayout) {
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC();
            cutlass_check(status);
        }        
        MNN_PRINT("Execution completed in %.3f ms\n", duration.count() / 1000.0);
        MNN_PRINT("==============================\n\n");
        return NO_ERROR;
    }

    if(mGpuComputeCap < 75) {
        if (mFp16Fp32MixInfer) {
            if(mUseRRLayout) {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RR();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RR();
                    cutlass_check(status);
                }
            } else {
                if(mNeedConvertMatAB) {
                    cutlass::Status status = mGemmBatchedCudaF16F32LnAlign1RC();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmBatchedCudaF32F32LnAlign1RC();
                    cutlass_check(status);
                }
            }
    
        } else {
            if(mUseRRLayout) {
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RR();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedCudaF16F16LnAlign1RC();
                cutlass_check(status);
            }
        }
    
        MNN_PRINT("Execution completed in %.3f ms\n", duration.count() / 1000.0);
        MNN_PRINT("==============================\n\n");
        return NO_ERROR;
    }

    if (mFp16Fp32MixInfer) {
        if(mUseRRLayout) {
            if(mNeedConvertMatAB) {
                cutlass::Status status = mGemmBatchedF16F32LnAlign8RRSm75();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedF32F32LnAlign8RRSm75();
                cutlass_check(status);
            }
        } else {
            if(hAlignment) {
                if(mNeedConvertMatAB) {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF16F32LnAlign8Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF16F32LnAlign8RCSm75();
                        cutlass_check(status);
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF32F32LnAlign8Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF32F32LnAlign8RCSm75();
                        cutlass_check(status);
                    }
                }
            } else {
                if(mNeedConvertMatAB) {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF16F32LnAlign1Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF16F32LnAlign1RCSm75();
                        cutlass_check(status);
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF32F32LnAlign1Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF32F32LnAlign1RCSm75();
                        cutlass_check(status);
                    }
                }
            }
        }

    } else {
        #ifdef ENABLE_CUDA_TUNE_PARAM
        if(mIsTuned) {
            runGemmBatchedTensorCoreFloat16Infer(&mInfo);
        } 
        #endif
        if(!mIsTuned) {
            if(mUseRRLayout) {
                cutlass::Status status = mGemmBatchedF16F16LnAlign8RRSm75();
                cutlass_check(status);
            } else {
                if(hAlignment) {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF16F16LnAlign8Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF16F16LnAlign8RCSm75();
                        cutlass_check(status);
                    }
                } else {
                    if(mConvertGemmSplitK) {
                        cutlass::Status status = mGemmF16F16LnAlign1Sm75();
                        cutlass_check(status);
                    } else {
                        cutlass::Status status = mGemmBatchedF16F16LnAlign1RCSm75();
                        cutlass_check(status);
                    }
                }
            }
        }
    }
    // printf("normal:%d rrlayout:%d convertab:%d halign:%d\n", mFp16Fp32MixInfer, mUseRRLayout, mNeedConvertMatAB, hAlignment);
    
    MNN_PRINT("Execution completed in %.3f ms\n", duration.count() / 1000.0);
    MNN_PRINT("==============================\n\n");
    return NO_ERROR;
}

class MatMulCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulExecution(param->transposeA(), param->transposeB(), backend);
    }
};

static CUDACreatorRegister<MatMulCreator> __init(OpType_MatMul);

}
}
