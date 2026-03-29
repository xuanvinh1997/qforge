#include "qforge/gates.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qforge { namespace gates {

void apply_single_qudit_gate(StateVector& sv, int target,
    const std::complex<double>* gate)
{
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (target < 0 || target >= n)
        throw std::out_of_range("target qudit index out of range");

    const size_t dim = sv.dim();
    const size_t stride = sv.stride(target);
    auto* amp = sv.data();

    // block_size = d * stride (covers all d values of target qudit)
    const size_t block_size = static_cast<size_t>(d) * stride;

    // Temporary buffer for d amplitudes
    std::vector<std::complex<double>> old_amps(d);

    #pragma omp parallel for schedule(static) if(dim > 4096) firstprivate(old_amps)
    for (size_t block = 0; block < dim; block += block_size) {
        for (size_t offset = 0; offset < stride; ++offset) {
            // Collect d amplitudes for this block
            for (int s = 0; s < d; ++s)
                old_amps[s] = amp[block + static_cast<size_t>(s) * stride + offset];

            // Apply d×d matrix
            for (int row = 0; row < d; ++row) {
                std::complex<double> sum = {0.0, 0.0};
                for (int col = 0; col < d; ++col)
                    sum += gate[row * d + col] * old_amps[col];
                amp[block + static_cast<size_t>(row) * stride + offset] = sum;
            }
        }
    }
}

void apply_controlled_qudit_gate(StateVector& sv, int control, int ctrl_val,
    int target, const std::complex<double>* gate)
{
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (control < 0 || control >= n || target < 0 || target >= n)
        throw std::out_of_range("qudit index out of range");
    if (control == target)
        throw std::invalid_argument("control and target must be distinct");
    if (ctrl_val < 0 || ctrl_val >= d)
        throw std::invalid_argument("ctrl_val out of range [0, d)");

    const size_t dim = sv.dim();
    const size_t ctrl_stride = sv.stride(control);
    const size_t tgt_stride = sv.stride(target);
    auto* amp = sv.data();

    // We iterate over all state indices where:
    //   - control qudit == ctrl_val
    //   - target qudit == 0 (canonical representative)
    // Then apply the d×d gate across the d target values.

    std::vector<std::complex<double>> old_amps(d);

    #pragma omp parallel for schedule(static) if(dim > 4096) firstprivate(old_amps)
    for (size_t idx = 0; idx < dim; ++idx) {
        int cv = extract_qudit(idx, ctrl_stride, d);
        int tv = extract_qudit(idx, tgt_stride, d);

        // Only process canonical indices: control matches and target == 0
        if (cv != ctrl_val || tv != 0) continue;

        // Collect d amplitudes across target values
        for (int s = 0; s < d; ++s) {
            size_t state_idx = replace_qudit(idx, tgt_stride, 0, s);
            old_amps[s] = amp[state_idx];
        }

        // Apply d×d matrix
        for (int row = 0; row < d; ++row) {
            std::complex<double> sum = {0.0, 0.0};
            for (int col = 0; col < d; ++col)
                sum += gate[row * d + col] * old_amps[col];
            size_t state_idx = replace_qudit(idx, tgt_stride, 0, row);
            amp[state_idx] = sum;
        }
    }
}

void qudit_swap(StateVector& sv, int t1, int t2) {
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (t1 < 0 || t1 >= n || t2 < 0 || t2 >= n)
        throw std::out_of_range("qudit index out of range");
    if (t1 == t2)
        throw std::invalid_argument("target qudits must be distinct");

    const size_t dim = sv.dim();
    const size_t s1 = sv.stride(t1);
    const size_t s2 = sv.stride(t2);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t idx = 0; idx < dim; ++idx) {
        int v1 = extract_qudit(idx, s1, d);
        int v2 = extract_qudit(idx, s2, d);
        // Only process canonical pairs where v1 < v2 to avoid double-swapping
        if (v1 < v2) {
            size_t j = replace_qudit(replace_qudit(idx, s1, v1, v2), s2, v2, v1);
            std::swap(amp[idx], amp[j]);
        }
    }
}

void csum(StateVector& sv, int control, int target) {
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (control < 0 || control >= n || target < 0 || target >= n)
        throw std::out_of_range("qudit index out of range");
    if (control == target)
        throw std::invalid_argument("control and target must be distinct");

    const size_t dim = sv.dim();
    const size_t ctrl_stride = sv.stride(control);
    const size_t tgt_stride = sv.stride(target);
    auto* amp = sv.data();
    auto* scratch = sv.scratch();

    // CSUM: |c, t> -> |c, (t + c) mod d>
    // Use scratch buffer for safe permutation
    sv.zero_scratch();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t idx = 0; idx < dim; ++idx) {
        int cv = extract_qudit(idx, ctrl_stride, d);
        int tv = extract_qudit(idx, tgt_stride, d);
        int new_tv = (tv + cv) % d;
        size_t new_idx = replace_qudit(idx, tgt_stride, tv, new_tv);
        scratch[new_idx] = amp[idx];
    }

    sv.swap_buffers();
}

}} // namespace qforge::gates
