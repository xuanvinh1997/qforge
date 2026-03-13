#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qforge/metal_backend.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// Embed shader source
static const char* METAL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cadd(float2 a, float2 b) {
    return float2(a.x + b.x, a.y + b.y);
}

kernel void apply_single_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant float2& m00 [[buffer(3)]],
    constant float2& m01 [[buffer(4)]],
    constant float2& m10 [[buffer(5)]],
    constant float2& m11 [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint mask = 1u << (n_qubits - target - 1);
    uint block = gid / mask;
    uint within = gid % mask;
    uint i = block * 2 * mask + within;
    uint j = i | mask;

    float2 a0 = amp[i];
    float2 a1 = amp[j];
    amp[i] = cadd(cmul(m00, a0), cmul(m01, a1));
    amp[j] = cadd(cmul(m10, a0), cmul(m11, a1));
}

kernel void apply_controlled_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& control [[buffer(2)]],
    constant uint& target [[buffer(3)]],
    constant float2& m00 [[buffer(4)]],
    constant float2& m01 [[buffer(5)]],
    constant float2& m10 [[buffer(6)]],
    constant float2& m11 [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint ctrl_mask = 1u << (n_qubits - control - 1);
    uint tgt_mask  = 1u << (n_qubits - target - 1);
    if ((gid & ctrl_mask) && !(gid & tgt_mask)) {
        uint j = gid | tgt_mask;
        float2 a0 = amp[gid];
        float2 a1 = amp[j];
        amp[gid] = cadd(cmul(m00, a0), cmul(m01, a1));
        amp[j]   = cadd(cmul(m10, a0), cmul(m11, a1));
    }
}

kernel void apply_double_controlled_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& c1 [[buffer(2)]],
    constant uint& c2 [[buffer(3)]],
    constant uint& target [[buffer(4)]],
    constant float2& m00 [[buffer(5)]],
    constant float2& m01 [[buffer(6)]],
    constant float2& m10 [[buffer(7)]],
    constant float2& m11 [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint c1_mask  = 1u << (n_qubits - c1 - 1);
    uint c2_mask  = 1u << (n_qubits - c2 - 1);
    uint tgt_mask = 1u << (n_qubits - target - 1);
    if ((gid & c1_mask) && (gid & c2_mask) && !(gid & tgt_mask)) {
        uint j = gid | tgt_mask;
        float2 a0 = amp[gid];
        float2 a1 = amp[j];
        amp[gid] = cadd(cmul(m00, a0), cmul(m01, a1));
        amp[j]   = cadd(cmul(m10, a0), cmul(m11, a1));
    }
}

kernel void swap_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& t1 [[buffer(2)]],
    constant uint& t2 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint m1 = 1u << (n_qubits - t1 - 1);
    uint m2 = 1u << (n_qubits - t2 - 1);
    if (!(gid & m1) && (gid & m2)) {
        uint j = gid ^ m1 ^ m2;
        float2 tmp = amp[gid];
        amp[gid] = amp[j];
        amp[j] = tmp;
    }
}

kernel void iswap_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& t1 [[buffer(2)]],
    constant uint& t2 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint m1 = 1u << (n_qubits - t1 - 1);
    uint m2 = 1u << (n_qubits - t2 - 1);
    if (!(gid & m1) && (gid & m2)) {
        uint j = gid ^ m1 ^ m2;
        float2 a = amp[gid];
        float2 b = amp[j];
        amp[gid] = float2(-b.y, b.x);
        amp[j]   = float2(-a.y, a.x);
    }
}

kernel void siswap_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& t1 [[buffer(2)]],
    constant uint& t2 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint m1 = 1u << (n_qubits - t1 - 1);
    uint m2 = 1u << (n_qubits - t2 - 1);
    if (!(gid & m1) && (gid & m2)) {
        uint j = gid ^ m1 ^ m2;
        float2 a = amp[gid];
        float2 b = amp[j];
        float c = 0.7071067811865475f;
        amp[gid] = float2(c * a.x - c * b.y, c * a.y + c * b.x);
        amp[j]   = float2(-c * a.y + c * b.x, c * a.x + c * b.y);
    }
}

kernel void cswap_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& control [[buffer(2)]],
    constant uint& t1 [[buffer(3)]],
    constant uint& t2 [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint ctrl_mask = 1u << (n_qubits - control - 1);
    uint m1 = 1u << (n_qubits - t1 - 1);
    uint m2 = 1u << (n_qubits - t2 - 1);
    if ((gid & ctrl_mask) && !(gid & m1) && (gid & m2)) {
        uint j = gid ^ m1 ^ m2;
        float2 tmp = amp[gid];
        amp[gid] = amp[j];
        amp[j] = tmp;
    }
}

kernel void measure_prob0(
    device const float2* amp [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& n_qubits [[buffer(2)]],
    constant uint& qubit [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint mask = 1u << (n_qubits - qubit - 1);
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    if (gid < dim) {
        if (!(gid & mask)) {
            float2 a = amp[gid];
            local_sum = a.x * a.x + a.y * a.y;
        }
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partial_sums[tgid] = shared_sum[0];
    }
}

kernel void compute_probabilities(
    device const float2* amp [[buffer(0)]],
    device float* probs [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float2 a = amp[gid];
    probs[gid] = a.x * a.x + a.y * a.y;
}
)";

namespace qforge {

// ============================================================
// Helper: float2 representation of complex
// ============================================================
struct float2 {
    float x, y;
};

static float2 to_f2(std::complex<double> c) {
    return {static_cast<float>(c.real()), static_cast<float>(c.imag())};
}

// ============================================================
// Construction / Destruction
// ============================================================

MetalBackend::MetalBackend(int n_qubits)
    : n_qubits_(n_qubits),
      dim_(size_t(1) << n_qubits),
      h_cache_(nullptr),
      host_dirty_(false),
      device_dirty_(false),
      device_(nullptr), queue_(nullptr),
      amp_buf_(nullptr), partial_buf_(nullptr), library_(nullptr),
      pso_single_gate_(nullptr), pso_controlled_gate_(nullptr),
      pso_double_controlled_gate_(nullptr),
      pso_swap_(nullptr), pso_iswap_(nullptr), pso_siswap_(nullptr),
      pso_cswap_(nullptr), pso_measure_prob0_(nullptr), pso_probabilities_(nullptr),
      pending_cmd_buf_(nullptr), pending_encoder_(nullptr)
{
    if (n_qubits < 1 || n_qubits > 30)
        throw std::runtime_error("MetalBackend: n_qubits must be 1..30");

    // Allocate host cache
    h_cache_ = new std::complex<double>[dim_];

    init_metal();
    create_pipelines();
    reset();
}

MetalBackend::~MetalBackend() {
    flush();
    delete[] h_cache_;

    // Release Metal objects via ARC — transfer ownership back from void*
    // then let ARC release them when they go out of scope
    if (pso_single_gate_) { (void)(__bridge_transfer id)pso_single_gate_; }
    if (pso_controlled_gate_) { (void)(__bridge_transfer id)pso_controlled_gate_; }
    if (pso_double_controlled_gate_) { (void)(__bridge_transfer id)pso_double_controlled_gate_; }
    if (pso_swap_) { (void)(__bridge_transfer id)pso_swap_; }
    if (pso_iswap_) { (void)(__bridge_transfer id)pso_iswap_; }
    if (pso_siswap_) { (void)(__bridge_transfer id)pso_siswap_; }
    if (pso_cswap_) { (void)(__bridge_transfer id)pso_cswap_; }
    if (pso_measure_prob0_) { (void)(__bridge_transfer id)pso_measure_prob0_; }
    if (pso_probabilities_) { (void)(__bridge_transfer id)pso_probabilities_; }
    if (amp_buf_) { (void)(__bridge_transfer id)amp_buf_; }
    if (partial_buf_) { (void)(__bridge_transfer id)partial_buf_; }
    if (library_) { (void)(__bridge_transfer id)library_; }
    if (queue_) { (void)(__bridge_transfer id)queue_; }
    if (device_) { (void)(__bridge_transfer id)device_; }
}

void MetalBackend::init_metal() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("MetalBackend: no Metal device found");
        device_ = (__bridge_retained void*)device;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            throw std::runtime_error("MetalBackend: failed to create command queue");
        queue_ = (__bridge_retained void*)queue;

        // Compile shader source
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:METAL_SHADER_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }

        id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&error];
        if (!lib) {
            NSString* desc = [error localizedDescription];
            throw std::runtime_error(std::string("MetalBackend: shader compile error: ") +
                                     [desc UTF8String]);
        }
        library_ = (__bridge_retained void*)lib;

        // Create amplitude buffer (float2 = 8 bytes per element)
        size_t amp_bytes = dim_ * 2 * sizeof(float);
        id<MTLBuffer> amp_buf = [device newBufferWithLength:amp_bytes
                                                    options:MTLResourceStorageModeShared];
        if (!amp_buf)
            throw std::runtime_error("MetalBackend: failed to allocate amplitude buffer");
        amp_buf_ = (__bridge_retained void*)amp_buf;

        // Partial sums buffer for reduction (max threadgroups)
        size_t tg_size = 256;
        size_t num_groups = (dim_ + tg_size - 1) / tg_size;
        id<MTLBuffer> partial_buf = [device newBufferWithLength:num_groups * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        partial_buf_ = (__bridge_retained void*)partial_buf;
    }
}

void MetalBackend::create_pipelines() {
    @autoreleasepool {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library_;
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        NSError* error = nil;

        auto make_pso = [&](const char* name) -> void* {
            NSString* fname = [NSString stringWithUTF8String:name];
            id<MTLFunction> fn = [lib newFunctionWithName:fname];
            if (!fn)
                throw std::runtime_error(std::string("MetalBackend: function not found: ") + name);
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) {
                NSString* desc = [error localizedDescription];
                throw std::runtime_error(std::string("MetalBackend: PSO error: ") + [desc UTF8String]);
            }
            return (__bridge_retained void*)pso;
        };

        pso_single_gate_ = make_pso("apply_single_gate");
        pso_controlled_gate_ = make_pso("apply_controlled_gate");
        pso_double_controlled_gate_ = make_pso("apply_double_controlled_gate");
        pso_swap_ = make_pso("swap_gate");
        pso_iswap_ = make_pso("iswap_gate");
        pso_siswap_ = make_pso("siswap_gate");
        pso_cswap_ = make_pso("cswap_gate");
        pso_measure_prob0_ = make_pso("measure_prob0");
        pso_probabilities_ = make_pso("compute_probabilities");
    }
}

// ============================================================
// Data transfer: f64 <-> f32
// ============================================================

void MetalBackend::upload_f64_to_f32() {
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)amp_buf_;
        float* ptr = (float*)[buf contents];
        for (size_t i = 0; i < dim_; ++i) {
            ptr[2 * i]     = static_cast<float>(h_cache_[i].real());
            ptr[2 * i + 1] = static_cast<float>(h_cache_[i].imag());
        }
        device_dirty_ = false;
    }
}

void MetalBackend::download_f32_to_f64() const {
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)amp_buf_;
        const float* ptr = (const float*)[buf contents];
        for (size_t i = 0; i < dim_; ++i) {
            h_cache_[i] = std::complex<double>(ptr[2 * i], ptr[2 * i + 1]);
        }
        host_dirty_ = false;
    }
}

void MetalBackend::sync_to_host() {
    flush();
    if (host_dirty_) {
        download_f32_to_f64();
    }
}

void MetalBackend::sync_to_device() {
    flush();
    if (device_dirty_) {
        upload_f64_to_f32();
    }
}

std::complex<double>* MetalBackend::host_data() {
    sync_to_host();
    device_dirty_ = true;  // host might modify
    return h_cache_;
}

const std::complex<double>* MetalBackend::host_data() const {
    const_cast<MetalBackend*>(this)->sync_to_host();
    return h_cache_;
}

void MetalBackend::reset() {
    std::memset(h_cache_, 0, dim_ * sizeof(std::complex<double>));
    h_cache_[0] = 1.0;
    upload_f64_to_f32();
    host_dirty_ = false;
    device_dirty_ = false;
}

// ============================================================
// Command buffer batching
// ============================================================

void MetalBackend::flush() {
    if (pending_encoder_) {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc =
                (__bridge_transfer id<MTLComputeCommandEncoder>)pending_encoder_;
            pending_encoder_ = nullptr;
            [enc endEncoding];

            id<MTLCommandBuffer> cmdBuf =
                (__bridge_transfer id<MTLCommandBuffer>)pending_cmd_buf_;
            pending_cmd_buf_ = nullptr;
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
    }
}

// Encode a compute command into the pending command buffer (no commit/wait).
// flush() must be called before reading results.
#define METAL_ENCODE(pso_var, thread_count, body) \
    do { \
        @autoreleasepool { \
            if (!pending_encoder_) { \
                id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_; \
                id<MTLCommandBuffer> cmdBuf = [q commandBuffer]; \
                pending_cmd_buf_ = (__bridge_retained void*)cmdBuf; \
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder]; \
                pending_encoder_ = (__bridge_retained void*)enc; \
            } \
            id<MTLComputeCommandEncoder> enc = \
                (__bridge id<MTLComputeCommandEncoder>)pending_encoder_; \
            id<MTLComputePipelineState> pso = \
                (__bridge id<MTLComputePipelineState>)pso_var; \
            [enc setComputePipelineState:pso]; \
            body \
            NSUInteger tc = (thread_count); \
            NSUInteger tgSize = MIN((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup); \
            [enc dispatchThreads:MTLSizeMake(tc, 1, 1) \
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)]; \
        } \
    } while(0)

// ============================================================
// Single-qubit gates
// ============================================================

void MetalBackend::apply_single_gate(int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t tgt = static_cast<uint32_t>(target);
    float2 fm00 = to_f2(m00), fm01 = to_f2(m01), fm10 = to_f2(m10), fm11 = to_f2(m11);

    METAL_ENCODE(pso_single_gate_, dim_ / 2, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&tgt length:sizeof(tgt) atIndex:2];
        [enc setBytes:&fm00 length:sizeof(fm00) atIndex:3];
        [enc setBytes:&fm01 length:sizeof(fm01) atIndex:4];
        [enc setBytes:&fm10 length:sizeof(fm10) atIndex:5];
        [enc setBytes:&fm11 length:sizeof(fm11) atIndex:6];
    });
    host_dirty_ = true;
}

void MetalBackend::H(int target) {
    constexpr double r = 0.7071067811865475244;
    apply_single_gate(target, r, r, r, -r);
}

void MetalBackend::X(int target) {
    apply_single_gate(target, 0, 1, 1, 0);
}

void MetalBackend::Y(int target) {
    using C = std::complex<double>;
    apply_single_gate(target, 0, C(0, -1), C(0, 1), 0);
}

void MetalBackend::Z(int target) {
    apply_single_gate(target, 1, 0, 0, -1);
}

void MetalBackend::RX(int target, double phi) {
    using C = std::complex<double>;
    double c = std::cos(phi / 2), s = std::sin(phi / 2);
    apply_single_gate(target, c, C(0, -s), C(0, -s), c);
}

void MetalBackend::RY(int target, double phi) {
    double c = std::cos(phi / 2), s = std::sin(phi / 2);
    apply_single_gate(target, c, -s, s, c);
}

void MetalBackend::RZ(int target, double phi) {
    using C = std::complex<double>;
    apply_single_gate(target, std::exp(C(0, -phi/2)), 0, 0, std::exp(C(0, phi/2)));
}

void MetalBackend::Phase(int target, double phi) {
    using C = std::complex<double>;
    apply_single_gate(target, 1, 0, 0, std::exp(C(0, phi)));
}

void MetalBackend::S(int target) {
    using C = std::complex<double>;
    apply_single_gate(target, 1, 0, 0, C(0, 1));
}

void MetalBackend::T(int target) {
    using C = std::complex<double>;
    constexpr double r = 0.7071067811865475244;
    apply_single_gate(target, 1, 0, 0, C(r, r));
}

void MetalBackend::Xsquare(int target) {
    using C = std::complex<double>;
    C a(0.5, 0.5), b(0.5, -0.5);
    apply_single_gate(target, a, b, b, a);
}

// ============================================================
// Controlled gates
// ============================================================

void MetalBackend::apply_controlled_gate(int control, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t ctrl = static_cast<uint32_t>(control);
    uint32_t tgt = static_cast<uint32_t>(target);
    float2 fm00 = to_f2(m00), fm01 = to_f2(m01), fm10 = to_f2(m10), fm11 = to_f2(m11);

    METAL_ENCODE(pso_controlled_gate_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&ctrl length:sizeof(ctrl) atIndex:2];
        [enc setBytes:&tgt length:sizeof(tgt) atIndex:3];
        [enc setBytes:&fm00 length:sizeof(fm00) atIndex:4];
        [enc setBytes:&fm01 length:sizeof(fm01) atIndex:5];
        [enc setBytes:&fm10 length:sizeof(fm10) atIndex:6];
        [enc setBytes:&fm11 length:sizeof(fm11) atIndex:7];
    });
    host_dirty_ = true;
}

void MetalBackend::CNOT(int control, int target) {
    apply_controlled_gate(control, target, 0, 1, 1, 0);
}

void MetalBackend::CRX(int control, int target, double phi) {
    using C = std::complex<double>;
    double c = std::cos(phi / 2), s = std::sin(phi / 2);
    apply_controlled_gate(control, target, c, C(0, -s), C(0, -s), c);
}

void MetalBackend::CRY(int control, int target, double phi) {
    double c = std::cos(phi / 2), s = std::sin(phi / 2);
    apply_controlled_gate(control, target, c, -s, s, c);
}

void MetalBackend::CRZ(int control, int target, double phi) {
    using C = std::complex<double>;
    apply_controlled_gate(control, target,
        std::exp(C(0, -phi/2)), 0, 0, std::exp(C(0, phi/2)));
}

void MetalBackend::CPhase(int control, int target, double phi) {
    using C = std::complex<double>;
    apply_controlled_gate(control, target, 1, 0, 0, std::exp(C(0, phi)));
}

void MetalBackend::CP(int control, int target, double phi) {
    CPhase(control, target, phi);
}

// ============================================================
// Double-controlled gates
// ============================================================

void MetalBackend::apply_double_controlled_gate(int c1, int c2, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t uc1 = static_cast<uint32_t>(c1);
    uint32_t uc2 = static_cast<uint32_t>(c2);
    uint32_t tgt = static_cast<uint32_t>(target);
    float2 fm00 = to_f2(m00), fm01 = to_f2(m01), fm10 = to_f2(m10), fm11 = to_f2(m11);

    METAL_ENCODE(pso_double_controlled_gate_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&uc1 length:sizeof(uc1) atIndex:2];
        [enc setBytes:&uc2 length:sizeof(uc2) atIndex:3];
        [enc setBytes:&tgt length:sizeof(tgt) atIndex:4];
        [enc setBytes:&fm00 length:sizeof(fm00) atIndex:5];
        [enc setBytes:&fm01 length:sizeof(fm01) atIndex:6];
        [enc setBytes:&fm10 length:sizeof(fm10) atIndex:7];
        [enc setBytes:&fm11 length:sizeof(fm11) atIndex:8];
    });
    host_dirty_ = true;
}

void MetalBackend::CCNOT(int c1, int c2, int target) {
    apply_double_controlled_gate(c1, c2, target, 0, 1, 1, 0);
}

void MetalBackend::OR(int c1, int c2, int target) {
    // OR = X(c1) X(c2) CCNOT(c1,c2,t) X(c1) X(c2) — but simpler to use
    // the controlled gate approach. OR flips target if c1=1 OR c2=1.
    // Implement via: X target, CCNOT, X target (De Morgan's)
    // Actually, to match original Qsun behavior exactly:
    // For each state where c1=1 OR c2=1, flip target
    // This isn't a standard 2x2 gate on target conditioned on controls.
    // We need to fall back to host for this non-standard gate.
    sync_to_host();
    auto* amp = h_cache_;
    size_t c1_mask = size_t(1) << (n_qubits_ - c1 - 1);
    size_t c2_mask = size_t(1) << (n_qubits_ - c2 - 1);
    size_t tgt_mask = size_t(1) << (n_qubits_ - target - 1);
    for (size_t idx = 0; idx < dim_; ++idx) {
        if (((idx & c1_mask) || (idx & c2_mask)) && !(idx & tgt_mask)) {
            size_t j = idx | tgt_mask;
            std::swap(amp[idx], amp[j]);
        }
    }
    device_dirty_ = true;
    host_dirty_ = false;
}

// ============================================================
// Swap gates
// ============================================================

void MetalBackend::SWAP(int t1, int t2) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t ut1 = static_cast<uint32_t>(t1);
    uint32_t ut2 = static_cast<uint32_t>(t2);

    METAL_ENCODE(pso_swap_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&ut1 length:sizeof(ut1) atIndex:2];
        [enc setBytes:&ut2 length:sizeof(ut2) atIndex:3];
    });
    host_dirty_ = true;
}

void MetalBackend::CSWAP(int control, int t1, int t2) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t ctrl = static_cast<uint32_t>(control);
    uint32_t ut1 = static_cast<uint32_t>(t1);
    uint32_t ut2 = static_cast<uint32_t>(t2);

    METAL_ENCODE(pso_cswap_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&ctrl length:sizeof(ctrl) atIndex:2];
        [enc setBytes:&ut1 length:sizeof(ut1) atIndex:3];
        [enc setBytes:&ut2 length:sizeof(ut2) atIndex:4];
    });
    host_dirty_ = true;
}

void MetalBackend::ISWAP(int t1, int t2) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t ut1 = static_cast<uint32_t>(t1);
    uint32_t ut2 = static_cast<uint32_t>(t2);

    METAL_ENCODE(pso_iswap_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&ut1 length:sizeof(ut1) atIndex:2];
        [enc setBytes:&ut2 length:sizeof(ut2) atIndex:3];
    });
    host_dirty_ = true;
}

void MetalBackend::SISWAP(int t1, int t2) {
    sync_to_device();
    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t ut1 = static_cast<uint32_t>(t1);
    uint32_t ut2 = static_cast<uint32_t>(t2);

    METAL_ENCODE(pso_siswap_, dim_, {
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBytes:&nq length:sizeof(nq) atIndex:1];
        [enc setBytes:&ut1 length:sizeof(ut1) atIndex:2];
        [enc setBytes:&ut2 length:sizeof(ut2) atIndex:3];
    });
    host_dirty_ = true;
}

// ============================================================
// Noise (fall back to host — non-standard operation)
// ============================================================

void MetalBackend::E(double p_noise, int target) {
    sync_to_host();
    auto* amp = h_cache_;
    size_t mask = size_t(1) << (n_qubits_ - target - 1);
    std::vector<double> new_amp(dim_, 0.0);

    for (size_t i = 0; i < dim_; ++i) {
        double prob = std::norm(amp[i]);
        if (!(i & mask)) {
            new_amp[i] += (1.0 - p_noise / 2.0) * prob;
            new_amp[i | mask] += (p_noise / 2.0) * prob;
        } else {
            new_amp[i] += (1.0 - p_noise / 2.0) * prob;
            new_amp[i & ~mask] += (p_noise / 2.0) * prob;
        }
    }
    for (size_t i = 0; i < dim_; ++i) {
        double sign = (amp[i].real() < 0) ? -1.0 : 1.0;
        h_cache_[i] = sign * std::sqrt(new_amp[i]);
    }
    device_dirty_ = true;
    host_dirty_ = false;
}

void MetalBackend::E_all(double p_noise) {
    if (p_noise > 0) {
        for (int i = 0; i < n_qubits_; ++i)
            E(p_noise, i);
    }
}

// ============================================================
// Measurement
// ============================================================

double MetalBackend::measure_one_prob0(int qubit) const {
    const_cast<MetalBackend*>(this)->flush();
    const_cast<MetalBackend*>(this)->sync_to_device();

    uint32_t nq = static_cast<uint32_t>(n_qubits_);
    uint32_t q = static_cast<uint32_t>(qubit);
    uint32_t d = static_cast<uint32_t>(dim_);

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_;
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_measure_prob0_;

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)amp_buf_ offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)partial_buf_ offset:0 atIndex:1];
        [enc setBytes:&nq length:sizeof(nq) atIndex:2];
        [enc setBytes:&q length:sizeof(q) atIndex:3];
        [enc setBytes:&d length:sizeof(d) atIndex:4];

        NSUInteger tgSize = 256;
        NSUInteger numGroups = (dim_ + tgSize - 1) / tgSize;
        [enc dispatchThreads:MTLSizeMake(dim_, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Sum partial results on CPU
        id<MTLBuffer> pbuf = (__bridge id<MTLBuffer>)partial_buf_;
        float* partials = (float*)[pbuf contents];
        double prob0 = 0.0;
        for (NSUInteger i = 0; i < numGroups; ++i) {
            prob0 += partials[i];
        }
        return prob0;
    }
}

void MetalBackend::collapse_one(int qubit, int value) {
    // Fall back to host for collapse (modifies amplitudes non-uniformly)
    sync_to_host();
    size_t mask = size_t(1) << (n_qubits_ - qubit - 1);
    double norm_sq = 0.0;

    for (size_t i = 0; i < dim_; ++i) {
        bool bit_set = (i & mask) != 0;
        if ((value == 0 && bit_set) || (value == 1 && !bit_set)) {
            h_cache_[i] = 0;
        } else {
            norm_sq += std::norm(h_cache_[i]);
        }
    }

    if (norm_sq > 0) {
        double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (size_t i = 0; i < dim_; ++i) {
            h_cache_[i] *= inv_norm;
        }
    }
    device_dirty_ = true;
    host_dirty_ = false;
}

double MetalBackend::pauli_expectation(int qubit, int pauli_type) const {
    // Fall back to host for Pauli expectation
    const_cast<MetalBackend*>(this)->sync_to_host();
    size_t mask = size_t(1) << (n_qubits_ - qubit - 1);

    if (pauli_type == 2) { // Z
        double exp_val = 0.0;
        for (size_t i = 0; i < dim_; ++i) {
            double p = std::norm(h_cache_[i]);
            exp_val += (i & mask) ? -p : p;
        }
        return exp_val;
    } else if (pauli_type == 0) { // X
        double exp_val = 0.0;
        for (size_t i = 0; i < dim_; ++i) {
            if (!(i & mask)) {
                size_t j = i | mask;
                exp_val += 2.0 * (h_cache_[i] * std::conj(h_cache_[j])).real();
            }
        }
        return exp_val;
    } else { // Y
        double exp_val = 0.0;
        for (size_t i = 0; i < dim_; ++i) {
            if (!(i & mask)) {
                size_t j = i | mask;
                // <Y> = 2 * Im(conj(a0) * a1)
                exp_val += 2.0 * (std::conj(h_cache_[i]) * h_cache_[j]).imag();
            }
        }
        return exp_val;
    }
}

void MetalBackend::probabilities(double* out) const {
    const_cast<MetalBackend*>(this)->sync_to_host();
    for (size_t i = 0; i < dim_; ++i) {
        out[i] = std::norm(h_cache_[i]);
    }
}

#undef METAL_ENCODE

} // namespace qforge
