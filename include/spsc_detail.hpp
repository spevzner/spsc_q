/**
 * @file spsc_detail.hpp
 * @brief Internal utilities shared across spsc headers.
 *
 * Not part of the public API — include via spsc_ring_buffer.hpp or storage.hpp.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <new>  // std::hardware_destructive_interference_size

namespace spsc::detail {

// ---------------------------------------------------------------------------
// Cache-line size
// ---------------------------------------------------------------------------

#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t kCacheLineSize =
    std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64;
#endif

// ---------------------------------------------------------------------------
// CPU spin-hint
//   x86/x86-64 : PAUSE   — drains the speculative load buffer, reduces power
//   AArch64     : YIELD   — hints to the scheduler / SMT sibling
//   fallback    : compiler barrier (prevents loop-hoisting by the optimiser)
// ---------------------------------------------------------------------------
[[gnu::always_inline]] inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// next_power_of_2
// Returns the smallest power-of-2 >= n.  Throws on 0 or overflow.
// Loop body runs at most log2(64) = 6 iterations on 64-bit targets.
// ---------------------------------------------------------------------------
[[nodiscard]] inline constexpr std::size_t next_power_of_2(std::size_t n) {
    if (n == 0)
        throw std::invalid_argument("Ring buffer capacity must be > 0");
    std::size_t p = 1;
    while (p < n) {
        p <<= 1;
        if (p == 0)
            throw std::overflow_error("Requested ring buffer capacity is too large");
    }
    return p;
}

// ---------------------------------------------------------------------------
// Constexpr max for alignment calculations (avoids pulling in <algorithm>)
// ---------------------------------------------------------------------------
[[nodiscard]] inline constexpr std::size_t max_align(std::size_t a,
                                                      std::size_t b) noexcept {
    return a > b ? a : b;
}

}  // namespace spsc::detail
