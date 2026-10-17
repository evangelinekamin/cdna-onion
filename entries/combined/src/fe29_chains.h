// Multiply-add chains as single inline-asm statements: the F29_CHAIN_ASM=1 build
// of fe29.h's chain functions (see there for why).
//
//   Chain<S, N>::acc(a, b, c)  = c + a[0] b[0] + ... + a[N-1] b[N-1]   (in order)
//   Chain<S, N>::zero(a, b)    =     a[0] b[0] + ... + a[N-1] b[N-1]
//
// as N v_mad_u64_u32 accumulating into one register pair, with every b in an
// SGPR (S) or a VGPR (!S); ChainI<N> is the same with v_mad_i64_i32 (signed b,
// VGPRs). The accumulator is written by the first instruction while later ones
// still read inputs, so it and the dead carry-out are early-clobber outputs.
#pragma once
#include <cstdint>

// F29_Tn(op): terms 2..n of a chain into %0 (term t reads operands 2t and 2t+1).
#define F29_M(op, i, j) "\n\t" op " %0, %1, %" #i ", %" #j ", %0"
#define F29_T1(op)
#define F29_T2(op) F29_T1(op) F29_M(op, 4, 5)
#define F29_T3(op) F29_T2(op) F29_M(op, 6, 7)
#define F29_T4(op) F29_T3(op) F29_M(op, 8, 9)
#define F29_T5(op) F29_T4(op) F29_M(op, 10, 11)
#define F29_T6(op) F29_T5(op) F29_M(op, 12, 13)
#define F29_T7(op) F29_T6(op) F29_M(op, 14, 15)
#define F29_T8(op) F29_T7(op) F29_M(op, 16, 17)
#define F29_T9(op) F29_T8(op) F29_M(op, 18, 19)
#define F29_T10(op) F29_T9(op) F29_M(op, 20, 21)
#define F29_T11(op) F29_T10(op) F29_M(op, 22, 23)
#define F29_T12(op) F29_T11(op) F29_M(op, 24, 25)
#define F29_T13(op) F29_T12(op) F29_M(op, 26, 27)
#define F29_T14(op) F29_T13(op) F29_M(op, 28, 29)
#define F29_T15(op) F29_T14(op) F29_M(op, 30, 31)
#define F29_T16(op) F29_T15(op) F29_M(op, 32, 33)
#define F29_T17(op) F29_T16(op) F29_M(op, 34, 35)
#define F29_T18(op) F29_T17(op) F29_M(op, 36, 37)

// F29_On(B): the inputs of n terms, a[t] in a VGPR and b[t] in register class B.
#define F29_O1(B) "v"(a[0]), B(b[0])
#define F29_O2(B) F29_O1(B), "v"(a[1]), B(b[1])
#define F29_O3(B) F29_O2(B), "v"(a[2]), B(b[2])
#define F29_O4(B) F29_O3(B), "v"(a[3]), B(b[3])
#define F29_O5(B) F29_O4(B), "v"(a[4]), B(b[4])
#define F29_O6(B) F29_O5(B), "v"(a[5]), B(b[5])
#define F29_O7(B) F29_O6(B), "v"(a[6]), B(b[6])
#define F29_O8(B) F29_O7(B), "v"(a[7]), B(b[7])
#define F29_O9(B) F29_O8(B), "v"(a[8]), B(b[8])
#define F29_O10(B) F29_O9(B), "v"(a[9]), B(b[9])
#define F29_O11(B) F29_O10(B), "v"(a[10]), B(b[10])
#define F29_O12(B) F29_O11(B), "v"(a[11]), B(b[11])
#define F29_O13(B) F29_O12(B), "v"(a[12]), B(b[12])
#define F29_O14(B) F29_O13(B), "v"(a[13]), B(b[13])
#define F29_O15(B) F29_O14(B), "v"(a[14]), B(b[14])
#define F29_O16(B) F29_O15(B), "v"(a[15]), B(b[15])
#define F29_O17(B) F29_O16(B), "v"(a[16]), B(b[16])
#define F29_O18(B) F29_O17(B), "v"(a[17]), B(b[17])

namespace f29 {

template <bool S, int N> struct Chain;
template <int N> struct ChainI;

#define F29_CHAIN_BODY(OP, T, BT, B, N)                                                  \
    __device__ __forceinline__ static T acc(const uint32_t *a, const BT *b, T c) {        \
        uint64_t dead;                                                                    \
        asm(OP " %0, %1, %2, %3, %0" F29_T##N(OP) : "+v"(c), "=&s"(dead) : F29_O##N(B));   \
        return c;                                                                         \
    }                                                                                     \
    __device__ __forceinline__ static T zero(const uint32_t *a, const BT *b) {            \
        T r;                                                                              \
        uint64_t dead;                                                                    \
        asm(OP " %0, %1, %2, %3, 0" F29_T##N(OP) : "=&v"(r), "=&s"(dead) : F29_O##N(B));   \
        return r;                                                                         \
    }
#define F29_CHAIN_U(N)                                                                    \
    template <> struct Chain<false, N> {                                                  \
        F29_CHAIN_BODY("v_mad_u64_u32", uint64_t, uint32_t, "v", N)                       \
    };                                                                                    \
    template <> struct Chain<true, N> {                                                   \
        F29_CHAIN_BODY("v_mad_u64_u32", uint64_t, uint32_t, "s", N)                       \
    };
#define F29_CHAIN_I(N)                                                                    \
    template <> struct ChainI<N> {                                                        \
        F29_CHAIN_BODY("v_mad_i64_i32", int64_t, int32_t, "v", N)                         \
    };

// The longest chains: an inner product's column (two products per limb pair)
// and leaf32's signed columns.
static constexpr int CHAIN_MAX = 18, CHAINI_MAX = 9;
F29_CHAIN_U(1) F29_CHAIN_U(2) F29_CHAIN_U(3) F29_CHAIN_U(4) F29_CHAIN_U(5) F29_CHAIN_U(6)
F29_CHAIN_U(7) F29_CHAIN_U(8) F29_CHAIN_U(9) F29_CHAIN_U(10) F29_CHAIN_U(11) F29_CHAIN_U(12)
F29_CHAIN_U(13) F29_CHAIN_U(14) F29_CHAIN_U(15) F29_CHAIN_U(16) F29_CHAIN_U(17) F29_CHAIN_U(18)
F29_CHAIN_I(1) F29_CHAIN_I(2) F29_CHAIN_I(3) F29_CHAIN_I(4) F29_CHAIN_I(5) F29_CHAIN_I(6)
F29_CHAIN_I(7) F29_CHAIN_I(8) F29_CHAIN_I(9)

// Pick the statement for chain length n. Callers' loops are unrolled, so n is a
// constant and this folds to a single statement.
template <bool S, bool Z, int N = 1>
__device__ __forceinline__ uint64_t chain_asm(const uint32_t *a, const uint32_t *b, int n, uint64_t c) {
    if constexpr (N > CHAIN_MAX) {
        __builtin_trap();
        return c;
    } else {
        if (n == N) return Z ? Chain<S, N>::zero(a, b) : Chain<S, N>::acc(a, b, c);
        return chain_asm<S, Z, N + 1>(a, b, n, c);
    }
}

template <bool Z, int N = 1>
__device__ __forceinline__ int64_t chaini_asm(const uint32_t *a, const int32_t *b, int n, int64_t c) {
    if constexpr (N > CHAINI_MAX) {
        __builtin_trap();
        return c;
    } else {
        if (n == N) return Z ? ChainI<N>::zero(a, b) : ChainI<N>::acc(a, b, c);
        return chaini_asm<Z, N + 1>(a, b, n, c);
    }
}

}  // namespace f29
