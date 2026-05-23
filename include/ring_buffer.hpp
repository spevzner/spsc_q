/**
 * @file ring_buffer.hpp
 * @brief Generic RingBuffer concept — the interface every ring buffer
 *        implementation must satisfy.
 *
 * Design rationale
 * ================
 * Rather than an abstract base class (vtable overhead at every push/pop),
 * we use a C++20 concept for zero-cost compile-time polymorphism.  Any type
 * that satisfies `spsc::RingBuffer<RB>` can be plugged into `SPSCQueue` or
 * used directly.
 *
 * Required operations
 * -------------------
 *  - try_push(const T&)   – copy-push; returns false (never blocks) if full
 *  - try_push(T&&)        – move-push; value is NOT moved when false returned
 *  - try_emplace(Args&&…) – in-place construct; returns false if full
 *  - try_pop(T&)          – move-pop into out; returns false if empty
 *  - size() / capacity() / empty() / full()  – approximate state queries
 *
 * Thread-safety contract
 * ----------------------
 * Implementations target SPSC (single-producer / single-consumer) use:
 *   - exactly one thread calls try_push / try_emplace
 *   - exactly one thread calls try_pop
 * State queries may be called from either thread but are inherently
 * approximate (no external lock).
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>  // std::move

namespace spsc {

// ---------------------------------------------------------------------------
// RingBuffer concept
// ---------------------------------------------------------------------------

/**
 * @concept RingBuffer
 * @brief Satisfied by any type that exposes the SPSC ring-buffer interface.
 *
 * @tparam RB  Candidate ring-buffer type.
 */
template <typename RB>
concept RingBuffer =
    // 1. Must expose a value_type alias.
    requires { typename RB::value_type; } &&

    // 2. Core push / pop / query operations.
    requires(
        RB                              rb,
        const typename RB::value_type&  cval,   ///< lvalue  (copy)
        typename RB::value_type         rval,   ///< rvalue  (move)
        typename RB::value_type&        out     ///< pop destination
    ) {
        // --- Producer ----------------------------------------------------------
        /// Copy-push. Returns false without side-effects if the buffer is full.
        { rb.try_push(cval)            } -> std::same_as<bool>;

        /// Move-push. If false is returned the argument has NOT been moved.
        { rb.try_push(std::move(rval)) } -> std::same_as<bool>;

        // --- Consumer ----------------------------------------------------------
        /// Move-pops the oldest element into `out`.
        /// Returns false without side-effects if the buffer is empty.
        { rb.try_pop(out)              } -> std::same_as<bool>;

        // --- Approximate state -------------------------------------------------
        { rb.size()     } -> std::convertible_to<std::size_t>;
        { rb.capacity() } -> std::convertible_to<std::size_t>;
        { rb.empty()    } -> std::same_as<bool>;
        { rb.full()     } -> std::same_as<bool>;
    };

// ---------------------------------------------------------------------------
// SPSCQueueInterface concept (queue adaptor contract)
// ---------------------------------------------------------------------------

/**
 * @concept SPSCQueueInterface
 * @brief Satisfied by types that present a full SPSC-queue surface.
 *
 * Extends RingBuffer with:
 *  - Blocking (spin-wait) push / pop variants.
 *  - In-place construction via try_emplace.
 */
template <typename Q>
concept SPSCQueueInterface =
    RingBuffer<Q> &&
    requires(
        Q                              q,
        const typename Q::value_type&  cval,
        typename Q::value_type         rval,
        typename Q::value_type&        out
    ) {
        /// Spin until push succeeds (useful for latency-sensitive hot paths
        /// where a brief busy-wait beats a context switch).
        { q.push(cval)            } -> std::same_as<void>;
        { q.push(std::move(rval)) } -> std::same_as<void>;

        /// Spin until pop succeeds; returns the element by value.
        { q.pop(out) } -> std::same_as<void>;
    };

}  // namespace spsc
