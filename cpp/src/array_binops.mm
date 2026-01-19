#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../include/array_binops.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::vector<int64_t> broadcast_shapes(std::span<const int64_t>&& a_shape,
                                      std::span<const int64_t>&& b_shape) {
    std::vector<int64_t> out;
    auto it_a = a_shape.rbegin();
    auto it_b = b_shape.rbegin();
    auto end_a = a_shape.rend();
    auto end_b = b_shape.rend();

    while (it_a != end_a || it_b != end_b) {
        int64_t dim_a = (it_a != end_a) ? *it_a : 1;
        int64_t dim_b = (it_b != end_b) ? *it_b : 1;

        if (dim_a == dim_b) {
            out.push_back(dim_a);
        } else if (dim_a == 1) {
            out.push_back(dim_b);
        } else if (dim_b == 1) {
            out.push_back(dim_a);
        } else {
            throw std::runtime_error("broadcast_shapes: shapes cannot be broadcast");
        }

        if (it_a != end_a) ++it_a;
        if (it_b != end_b) ++it_b;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::shared_ptr<ArrayHandle> array_binops(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B,
                                          const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    if (shapeA != shapeB) {
        throw std::runtime_error("array_binops: shape mismatch");
    }

    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device = (__bridge id<MTLDevice>)defaultForgeHandle->device_ptr();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();

    // compile pipeline on first call
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, ELEMENTWISE_METAL_SOURCE);

    // allocate output ArrayHandle
    auto out = std::make_shared<ArrayHandle>(A->shape(), defaultForgeHandle->device_ptr());

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error(
            "Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];

    uint ndim = (uint)out->shape().size();

    [enc setBytes:out->shape().data() length:ndim * 8 atIndex:3];
    size_t current_offsetA = A->offset();
    [enc setBytes:A->strides().data() length:ndim * 8 atIndex:4];
    [enc setBytes:&current_offsetA length:sizeof(size_t) atIndex:5];
    size_t current_offsetB = B->offset();
    [enc setBytes:B->strides().data() length:ndim * 8 atIndex:6];
    [enc setBytes:&current_offsetB length:sizeof(size_t) atIndex:7];

    [enc setBytes:&ndim length:4 atIndex:8];

    MTLSize grid = MTLSizeMake(A->data().size(), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}

std::pair<std::shared_ptr<ArrayHandle>, bool> prepare(const std::shared_ptr<ArrayHandle>& h) {
    int ndim = h->shape().size();
    int64_t R = h->shape()[ndim - 2];
    int64_t C = h->shape()[ndim - 1];
    int64_t sR = h->strides()[ndim - 2];
    int64_t sC = h->strides()[ndim - 1];
    // Contiguous data regular
    if (sR == C && sC == 1) return {h, false};
    // Transposed / Col-Major (Strides: 1, R)
    if (sR == 1 && sC == R) return {h, true};

    auto new_handle = std::make_shared<ArrayHandle>(h->shape());
    new_handle->copy_from(h, new_handle->shape(), new_handle->strides(), 0);
    return {new_handle, false};
}

std::shared_ptr<ArrayHandle> array_matmul(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B) {
    bool squeeze_a = false, squeeze_b = false;
    auto Ashape = A->shape();
    auto Astrides = A->strides();
    auto Bshape = B->shape();
    auto Bstrides = B->strides();
    if (A->shape().size() == 1) {
        squeeze_a = true;
        // Promote (K,) -> (1, K)
        Astrides.insert(Astrides.begin(), Ashape[0] * Astrides[0]);
        Ashape.insert(Ashape.begin(), 1);
    }
    if (B->shape().size() == 1) {
        squeeze_b = true;
        // Promote (K,) -> (K, 1)
        Bshape.push_back(1);
        Bstrides.push_back(1);
    }
    auto [a, trans_a] = prepare(make_shared<ArrayHandle>(A, Ashape, Astrides, A->offset()));
    auto [b, trans_b] = prepare(make_shared<ArrayHandle>(B, Bshape, Bstrides, B->offset()));

    int64_t M = a->shape()[a->shape().size() - 2];
    int64_t K_a = a->shape()[a->shape().size() - 1];
    int64_t K_b = b->shape()[b->shape().size() - 2];
    int64_t N = b->shape()[b->shape().size() - 1];

    if (K_a != K_b) throw std::runtime_error("matmul: dimension mismatch");
    int64_t K = K_a;

    // Compute batch dimensions separately from M, N
    auto batch_shape =
        broadcast_shapes({Ashape.begin(), Ashape.end() - 2}, {Bshape.begin(), Bshape.end() - 2});

    // Build full output shape: batch_dims + M + N
    auto out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    auto c = std::make_shared<ArrayHandle>(out_shape);

    // Compute strides for batch dimensions only (excluding M, N)
    auto get_batch_strides = [&](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides;
        int ndim = shape.size();
        int batch_ndim = ndim - 2;  // Number of batch dims in this tensor
        int offset = batch_shape.size() - batch_ndim;
        auto dense = make_strides(shape);
        for (int i = 0; i < batch_shape.size(); ++i) {
            int idx = i - offset;
            // If dim missing (idx < 0) or dim is 1 -> Stride is 0 (Broadcast)
            if (idx < 0 || shape[idx] == 1)
                strides.push_back(0);
            else
                strides.push_back(dense[idx]);
        }
        return strides;
    };

    auto str_a = get_batch_strides(a->shape());
    auto str_b = get_batch_strides(b->shape());

    id<MTLDevice> device = (__bridge id<MTLDevice>)get_default_forge()->device_ptr();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)get_default_forge()->queue_ptr();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    size_t dsize = sizeof(float);

    auto descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(trans_a ? K : M)
                         columns:(trans_a ? M : K)rowBytes:(trans_a ? M : K) * dsize
                        dataType:MPSDataTypeFloat32];

    auto descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(trans_b ? N : K)
                         columns:(trans_b ? K : N)rowBytes:(trans_b ? K : N) * dsize
                        dataType:MPSDataTypeFloat32];

    auto descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                       columns:N
                                                      rowBytes:N * dsize
                                                      dataType:MPSDataTypeFloat32];

    MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                        transposeLeft:trans_a
                                                                       transposeRight:trans_b
                                                                           resultRows:M
                                                                        resultColumns:N
                                                                      interiorColumns:K
                                                                                alpha:1.0
                                                                                 beta:0.0];

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b->metal_buffer();
    id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c->metal_buffer();

    // total_ops only counts batch dimensions (not M, N)
    size_t total_ops = 1;
    for (auto s : batch_shape) total_ops *= s;

    // For non-batched case (total_ops == 1), use MPS single encoding
    if (total_ops == 1) {
        size_t off_a = a->offset() * dsize;
        size_t off_b = b->offset() * dsize;
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:off_a descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:off_b descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:0 descriptor:descC];
        [kernel encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    } else {
        // Batched case: check if we can use custom tiled kernel
        // Conditions: no transpose, no broadcasting, contiguous batch dimension
        bool can_use_custom_kernel = !trans_a && !trans_b && batch_shape.size() == 1;

        // Check for broadcasting (any zero strides means broadcasting)
        for (auto s : str_a) {
            if (s == 0) can_use_custom_kernel = false;
        }
        for (auto s : str_b) {
            if (s == 0) can_use_custom_kernel = false;
        }

        // Heuristic: use custom kernel when batch overhead dominates
        // For large matrices, MPS is highly optimized and faster
        // For small matrices with many batches, custom kernel reduces overhead
        int64_t matrix_size = M * K + K * N;  // approximate work per batch
        bool small_matrices = matrix_size < 256 * 256;  // threshold
        bool many_batches = total_ops >= 8;
        can_use_custom_kernel = can_use_custom_kernel && small_matrices && many_batches;

        if (can_use_custom_kernel) {
            // Use custom tiled batched GEMM kernel - single GPU dispatch for all batches
            id<MTLComputePipelineState> pipeline =
                get_pipeline("batched_matmul_tiled", ELEMENTWISE_METAL_SOURCE);

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:bufA offset:a->offset() * dsize atIndex:0];
            [enc setBuffer:bufB offset:b->offset() * dsize atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(long) atIndex:3];
            [enc setBytes:&K length:sizeof(long) atIndex:4];
            [enc setBytes:&N length:sizeof(long) atIndex:5];
            long batch_size = total_ops;
            [enc setBytes:&batch_size length:sizeof(long) atIndex:6];
            long stride_a = M * K;
            long stride_b = K * N;
            long stride_c = M * N;
            [enc setBytes:&stride_a length:sizeof(long) atIndex:7];
            [enc setBytes:&stride_b length:sizeof(long) atIndex:8];
            [enc setBytes:&stride_c length:sizeof(long) atIndex:9];

            // Dispatch with tiled threadgroups
            const uint TILE_SIZE = 16;
            MTLSize threadsPerGroup = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
            MTLSize numGroups = MTLSizeMake((N + TILE_SIZE - 1) / TILE_SIZE,
                                            (M + TILE_SIZE - 1) / TILE_SIZE, total_ops);
            [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];
        } else {
            // Fall back to MPS for complex cases (transpose, broadcasting)
            // Pre-compute all offsets first
            std::vector<size_t> offsets_a(total_ops);
            std::vector<size_t> offsets_b(total_ops);
            std::vector<size_t> offsets_c(total_ops);

            std::vector<int> counters(batch_shape.size(), 0);
            size_t off_a = a->offset() * dsize;
            size_t off_b = b->offset() * dsize;
            size_t off_c = 0;

            for (size_t op = 0; op < total_ops; ++op) {
                offsets_a[op] = off_a;
                offsets_b[op] = off_b;
                offsets_c[op] = off_c;

                off_c += M * N * dsize;
                for (int dim = batch_shape.size() - 1; dim >= 0; --dim) {
                    counters[dim]++;
                    if (counters[dim] == batch_shape[dim]) {
                        counters[dim] = 0;
                        off_a -= (batch_shape[dim] - 1) * str_a[dim] * dsize;
                        off_b -= (batch_shape[dim] - 1) * str_b[dim] * dsize;
                    } else {
                        off_a += str_a[dim] * dsize;
                        off_b += str_b[dim] * dsize;
                        break;
                    }
                }
            }

            // Pre-allocate all MPSMatrix objects upfront
            @autoreleasepool {
                std::vector<MPSMatrix*> matricesA(total_ops);
                std::vector<MPSMatrix*> matricesB(total_ops);
                std::vector<MPSMatrix*> matricesC(total_ops);

                for (size_t op = 0; op < total_ops; ++op) {
                    matricesA[op] = [[MPSMatrix alloc] initWithBuffer:bufA
                                                               offset:offsets_a[op]
                                                           descriptor:descA];
                    matricesB[op] = [[MPSMatrix alloc] initWithBuffer:bufB
                                                               offset:offsets_b[op]
                                                           descriptor:descB];
                    matricesC[op] = [[MPSMatrix alloc] initWithBuffer:bufC
                                                               offset:offsets_c[op]
                                                           descriptor:descC];
                }

                for (size_t op = 0; op < total_ops; ++op) {
                    [kernel encodeToCommandBuffer:cmd
                                       leftMatrix:matricesA[op]
                                      rightMatrix:matricesB[op]
                                     resultMatrix:matricesC[op]];
                }
            }
        }
    }

    [cmd commit];
    c->set_event((__bridge void*)cmd);

    std::vector<int64_t> final_shape = c->shape();
    if (squeeze_a && squeeze_b) {
        // Case: (K,) @ (K,) -> Scalar. Current c: (..., 1, 1). Remove last two dims.
        if (final_shape.size() >= 2) {
            final_shape.resize(final_shape.size() - 2);
        }
    } else if (squeeze_a) {
        // Case: (K,) @ (K, N) -> (N,). Current c: (..., 1, N). Remove dim -2.
        auto it = final_shape.end() - 2;
        final_shape.erase(it);
    } else if (squeeze_b) {
        // Case: (M, K) @ (K,) -> (M,). Current c: (..., M, 1). Remove dim -1.
        final_shape.pop_back();
    }

    return std::make_shared<ArrayHandle>(c, final_shape, make_strides(final_shape), c->offset());
}
