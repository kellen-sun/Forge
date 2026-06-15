#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../include/array_matmul.h"
#include "../include/metal_utils.h"

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

    id<MTLBuffer> bufA = a->metal_buffer();
    id<MTLBuffer> bufB = b->metal_buffer();
    id<MTLBuffer> bufC = c->metal_buffer();

    // total_ops only counts batch dimensions (not M, N)
    size_t total_ops = 1;
    for (auto s : batch_shape) total_ops *= s;

    std::vector<int> counters(batch_shape.size(), 0);

    size_t off_a = a->offset() * dsize;
    size_t off_b = b->offset() * dsize;
    size_t off_c = 0;

    // Loop only over batch dimensions
    for (size_t op = 0; op < total_ops; ++op) {
        @autoreleasepool {
            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:off_a descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:off_b descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:off_c descriptor:descC];
            [kernel encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        }

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

    [cmd commit];
    c->set_event(cmd);

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
