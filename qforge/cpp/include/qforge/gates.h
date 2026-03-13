#pragma once
#include "state_vector.h"

namespace qforge { namespace gates {

// --- Core kernels ---
void apply_single_qubit_gate(StateVector& sv, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11);

void apply_controlled_gate(StateVector& sv, int control, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11);

void apply_double_controlled_gate(StateVector& sv, int c1, int c2, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11);

// --- Single-qubit gates ---
void H(StateVector& sv, int target);
void X(StateVector& sv, int target);
void Y(StateVector& sv, int target);
void Z(StateVector& sv, int target);
void RX(StateVector& sv, int target, double phi);
void RY(StateVector& sv, int target, double phi);
void RZ(StateVector& sv, int target, double phi);
void Phase(StateVector& sv, int target, double phi);
void S(StateVector& sv, int target);
void T(StateVector& sv, int target);
void Xsquare(StateVector& sv, int target);

// --- Controlled gates ---
void CNOT(StateVector& sv, int control, int target);
void CRX(StateVector& sv, int control, int target, double phi);
void CRY(StateVector& sv, int control, int target, double phi);
void CRZ(StateVector& sv, int control, int target, double phi);
void CPhase(StateVector& sv, int control, int target, double phi);
void CP(StateVector& sv, int control, int target, double phi);
void CCNOT(StateVector& sv, int c1, int c2, int target);
void OR(StateVector& sv, int c1, int c2, int target);

// --- Swap gates ---
void SWAP(StateVector& sv, int t1, int t2);
void CSWAP(StateVector& sv, int control, int t1, int t2);
void ISWAP(StateVector& sv, int t1, int t2);
void SISWAP(StateVector& sv, int t1, int t2);

// --- Noise ---
void E(StateVector& sv, double p_noise, int target);
void E_all(StateVector& sv, double p_noise);

}} // namespace qforge::gates
