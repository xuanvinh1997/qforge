#include <metal_stdlib>
using namespace metal;

// Complex number as float2: .x = real, .y = imaginary
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cadd(float2 a, float2 b) {
    return float2(a.x + b.x, a.y + b.y);
}

// ============================================================
// Single-qubit gate kernel
// Each thread handles one (i, j) amplitude pair where j = i | mask
// Total threads = dim / 2
// ============================================================
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
    // Map thread index to amplitude pair (i, j)
    uint block = gid / mask;
    uint within = gid % mask;
    uint i = block * 2 * mask + within;
    uint j = i | mask;

    float2 a0 = amp[i];
    float2 a1 = amp[j];
    amp[i] = cadd(cmul(m00, a0), cmul(m01, a1));
    amp[j] = cadd(cmul(m10, a0), cmul(m11, a1));
}

// ============================================================
// Controlled gate kernel
// Each thread checks one index; only acts when control=1, target=0
// Total threads = dim
// ============================================================
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

    // Only process when control bit = 1 and target bit = 0
    if ((gid & ctrl_mask) && !(gid & tgt_mask)) {
        uint j = gid | tgt_mask;
        float2 a0 = amp[gid];
        float2 a1 = amp[j];
        amp[gid] = cadd(cmul(m00, a0), cmul(m01, a1));
        amp[j]   = cadd(cmul(m10, a0), cmul(m11, a1));
    }
}

// ============================================================
// Double-controlled gate kernel
// Acts when both control bits = 1 and target bit = 0
// Total threads = dim
// ============================================================
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

// ============================================================
// SWAP kernel
// Swaps amplitude pairs where bit t1 != bit t2
// Each thread handles one canonical pair (bit t1=0, bit t2=1)
// Total threads = dim
// ============================================================
kernel void swap_gate(
    device float2* amp [[buffer(0)]],
    constant uint& n_qubits [[buffer(1)]],
    constant uint& t1 [[buffer(2)]],
    constant uint& t2 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint m1 = 1u << (n_qubits - t1 - 1);
    uint m2 = 1u << (n_qubits - t2 - 1);

    // Only handle canonical pair: bit t1 = 0, bit t2 = 1
    if (!(gid & m1) && (gid & m2)) {
        uint j = gid ^ m1 ^ m2;
        float2 tmp = amp[gid];
        amp[gid] = amp[j];
        amp[j] = tmp;
    }
}

// ============================================================
// ISWAP kernel: swaps with i phase factor
// |01> -> i|10>, |10> -> i|01>
// ============================================================
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
        // Multiply by i: (x + iy) * i = -y + ix
        amp[gid] = float2(-b.y, b.x);
        amp[j]   = float2(-a.y, a.x);
    }
}

// ============================================================
// SISWAP (sqrt-iSWAP) kernel
// ============================================================
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
        float c = 0.7071067811865475f; // 1/sqrt(2)
        // new_a = c*a + i*c*b
        // new_b = i*c*a + c*b
        amp[gid] = float2(c * a.x - c * b.y, c * a.y + c * b.x);
        amp[j]   = float2(-c * a.y + c * b.x, c * a.x + c * b.y);
    }
}

// ============================================================
// CSWAP kernel
// ============================================================
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

// ============================================================
// Measure probability of |0> for a qubit
// Reduction kernel: each thread accumulates partial sum
// Output: partial sums to be reduced on CPU
// ============================================================
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

    // Reduction within threadgroup
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

// ============================================================
// Compute all probabilities |amp[i]|^2
// ============================================================
kernel void compute_probabilities(
    device const float2* amp [[buffer(0)]],
    device float* probs [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float2 a = amp[gid];
    probs[gid] = a.x * a.x + a.y * a.y;
}
