/**
 * @file spsc_ring_buffer.hpp
 * @brief Default low-latency, lock-free SPSC ring buffer implementation.
 *
 * Architecture & latency decisions
 * =================================
 *
 * 1. Power-of-2 capacity
 *    Index wrap-around: `idx & mask_` (1 cycle) vs `idx % capacity_` (≈20 cycles
 *    on modern μarch).  The constructor silently rounds up to the next power of 2.
 *
 * 2. False-sharing elimination
 *    `head_` (producer-owned) and `tail_` (consumer-owned) live on separate
 *    cache lines via `alignas(kCacheLineSize)`.  Without padding, a write to
 *    either variable would invalidate the other thread's copy, causing a cache
 *    coherence stall on every operation.
 *
 * 3. Minimal memory-order fences
 *    - Producer loads its own `head_` with `relaxed`  (no ordering needed).
 *    - Producer loads `tail_` with `acquire`           (syncs with consumer release).
 *    - Producer stores `head_` with `release`          (consumer sees the element).
 *    - Consumer mirrors these symmetrically for `tail_` / `head_`.
 *    Avoiding `seq_cst` (the default) eliminates MFENCE on x86 / DMB on ARM,
 *    typically saving 20–40 ns per operation on cold paths.
 *
 * 4. Raw storage + placement new
 *    Elements are constructed in-place inside untyped aligned storage.  No
 *    default-construction overhead; no second allocation for a pointer layer.
 *
 * 5. `try_emplace` for zero-copy construction
 *    Lets callers build the element directly in the ring-buffer slot, avoiding
 *    a temporary copy/move when T is not trivially movable.
 *
 * 6. Trivially-destructible fast-path
 *    `if constexpr (!std::is_trivially_destructible_v<T>)` skips explicit
 *    destructor calls for built-in and trivial types (int, float, POD structs).
 *
 * Thread-safety
 * =============
 * Exactly ONE producer thread calls try_push / try_emplace.
 * Exactly ONE consumer thread calls try_pop.
 * No external synchronisation is required between them.
 */

#pragma once

#include "ring_buffer.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>       // std::construct_at, std::destroy_at
#include <new>          // ::operator new, std::align_val_t,
                        // std::hardware_destructive_interference_size
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace spsc {

namespace detail {

// ---------------------------------------------------------------------------
// Cache-line size
// ---------------------------------------------------------------------------
#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t kCacheLineSize =
    std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64;  // safe default
#endif

// ---------------------------------------------------------------------------
// CPU spin-hint
// Signals to the processor that we are in a busy-wait loop:
//   x86 / x86-64 : REP NOP (PAUSE) — prevents pipeline memory-order violations
//   AArch64       : YIELD         — hints to the scheduler / SMT sibling
// ---------------------------------------------------------------------------
[[gnu::always_inline]] inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    // Portable fallback: compiler barrier prevents loop-hoisting
    __asm__ volatile("" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// next_power_of_2: smallest power-of-2 >= n
//
// Implemented with a plain left-shift loop so it compiles cleanly under any
// C++ standard ≥ 17 without pulling in <bit>.  The loop body runs at most
// log2(CHAR_BIT * sizeof(size_t)) = 6 iterations on 64-bit targets, so it
// is fast enough for a constructor path and is fully constexpr.
// ---------------------------------------------------------------------------
[[nodiscard]] inline constexpr std::size_t next_power_of_2(std::size_t n) {
    if (n == 0)
        throw std::invalid_argument("Ring buffer capacity must be > 0");
    std::size_t p = 1;
    while (p < n) {
        p <<= 1;
        if (p == 0)  // left-shift overflow: n was larger than any power of 2
            throw std::overflow_error("Requested ring buffer capacity is too large");
    }
    return p;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// SPSCRingBuffer
// ---------------------------------------------------------------------------

/**
 * @class SPSCRingBuffer<T>
 * @brief Low-latency, lock-free, single-producer / single-consumer ring buffer.
 *
 * @tparam T  Element type.  Must be move-constructible or copy-constructible.
 *
 * @note  Capacity is rounded up to the next power of 2 so that modulo wrapping
 *        is a single bitwise AND.  `capacity()` reflects the *actual* (rounded)
 *        capacity, not the value passed to the constructor.
 */
template <typename T>
class SPSCRingBuffer {
    static_assert(std::is_move_constructible_v<T> || std::is_copy_constructible_v<T>,
                  "T must be move- or copy-constructible");

public:
    using value_type = T;

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct a ring buffer with at least @p capacity slots.
     *
     * The actual capacity is the smallest power of 2 >= @p capacity.
     * Querying `capacity()` returns the rounded value.
     *
     * @throws std::invalid_argument  if capacity == 0
     * @throws std::overflow_error    if capacity is too large
     * @throws std::bad_alloc         if heap allocation fails
     */
    explicit SPSCRingBuffer(std::size_t capacity)
        : capacity_(detail::next_power_of_2(capacity))
        , mask_    (capacity_ - 1)
        , storage_ (alloc(capacity_))
    {}

    ~SPSCRingBuffer() noexcept {
        // Destroy live elements [tail_, head_) without touching consumer state.
        if constexpr (!std::is_trivially_destructible_v<T>) {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            std::size_t       tail = tail_.load(std::memory_order_relaxed);
            while (tail != head) {
                std::destroy_at(slot(tail));
                ++tail;
            }
        }
        ::operator delete(storage_, std::align_val_t{kAlign});
    }

    SPSCRingBuffer(const SPSCRingBuffer&)            = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer(SPSCRingBuffer&&)                 = delete;
    SPSCRingBuffer& operator=(SPSCRingBuffer&&)      = delete;

    // -----------------------------------------------------------------------
    // Producer interface  (call ONLY from the producer thread)
    // -----------------------------------------------------------------------

    /**
     * @brief Try to push a copy of @p value.
     * @return true on success; false (no side-effects) if the buffer is full.
     */
    [[nodiscard]] bool try_push(const T& value)
        noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        return emplace_one(value);
    }

    /**
     * @brief Try to push by moving @p value.
     * @return true on success; false if full — @p value is untouched on false.
     */
    [[nodiscard]] bool try_push(T&& value)
        noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        return emplace_one(std::move(value));
    }

    /**
     * @brief Try to construct an element in-place with @p args.
     *
     * Avoids any temporary copy or move — the element is built directly
     * inside the ring-buffer slot.
     *
     * @return true on success; false if full — no arguments are forwarded
     *         when false is returned.
     */
    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    [[nodiscard]] bool try_emplace(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>)
    {
        return emplace_one(std::forward<Args>(args)...);
    }

    // -----------------------------------------------------------------------
    // Consumer interface  (call ONLY from the consumer thread)
    // -----------------------------------------------------------------------

    /**
     * @brief Try to move the oldest element into @p out.
     * @return true on success; false (no side-effects) if the buffer is empty.
     */
    [[nodiscard]] bool try_pop(T& out)
        noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        // Own index: relaxed — no ordering with respect to self.
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        // Acquire: synchronise with the producer's release-store of head_.
        // Ensures we see the element written before head_ was incremented.
        const std::size_t head = head_.load(std::memory_order_acquire);

        if (head == tail) [[unlikely]]
            return false;  // empty

        T* p = slot(tail);
        out  = std::move(*p);

        if constexpr (!std::is_trivially_destructible_v<T>)
            std::destroy_at(p);

        // Release: the producer's next acquire-load of tail_ will see this.
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // State queries  (approximate; safe from either thread)
    // -----------------------------------------------------------------------

    /**
     * @brief Approximate number of elements currently in the buffer.
     *
     * The value may be stale by at most one operation; use only for
     * monitoring, not for synchronisation.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    /** @brief Rounded-up power-of-2 capacity chosen at construction. */
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /** @brief Returns true if no elements are present (may be stale). */
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /** @brief Returns true if the buffer cannot accept more pushes (may be stale). */
    [[nodiscard]] bool full() const noexcept {
        return size() == capacity_;
    }

private:
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Alignment used for the storage block.
    // Take the larger of T's natural alignment and a cache-line so that
    // &storage_[0] always starts on a cache-line boundary.
    // Constexpr ternary avoids pulling in <algorithm> for std::max.
    static constexpr std::size_t kAlign =
        (alignof(T) > detail::kCacheLineSize)
            ? alignof(T)
            : detail::kCacheLineSize;

    [[nodiscard]] static T* alloc(std::size_t n) {
        return static_cast<T*>(
            ::operator new(n * sizeof(T), std::align_val_t{kAlign}));
    }

    // Returns a pointer to the live slot for absolute index `idx`.
    [[nodiscard]] T* slot(std::size_t idx) noexcept {
        return std::launder(reinterpret_cast<T*>(storage_) + (idx & mask_));
    }

    template <typename... Args>
    [[nodiscard]] bool emplace_one(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>)
    {
        // Own index: relaxed.
        const std::size_t head = head_.load(std::memory_order_relaxed);
        // Acquire: synchronise with the consumer's release-store of tail_.
        // Ensures we see the slot as "freed" after the consumer popped it.
        const std::size_t tail = tail_.load(std::memory_order_acquire);

        if ((head - tail) == capacity_) [[unlikely]]
            return false;  // full — args NOT touched

        std::construct_at(slot(head), std::forward<Args>(args)...);

        // Release: consumer's next acquire-load of head_ will see the element.
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Data layout
    // -----------------------------------------------------------------------
    //
    // Intentional ordering:
    //   1. Immutable fields (capacity_, mask_, storage_) — read by both
    //      threads but never written after construction; no coherence traffic.
    //   2. head_ on its own cache line — written by producer, read by consumer.
    //   3. tail_ on its own cache line — written by consumer, read by producer.
    //
    // Placing head_ and tail_ on separate lines prevents the false-sharing
    // "ping-pong" that would otherwise stall every push/pop pair.

    const std::size_t capacity_;    ///< Rounded-up power-of-2 capacity
    const std::size_t mask_;        ///< capacity_ - 1  (fast wrap bitmask)
    T* const          storage_;     ///< Raw aligned storage (placement-new arena)

    alignas(detail::kCacheLineSize)
    std::atomic<std::size_t> head_{0};  ///< Next write slot  (producer-owned)

    alignas(detail::kCacheLineSize)
    std::atomic<std::size_t> tail_{0};  ///< Next read  slot  (consumer-owned)
};

// Verify the concept at compile time with a trivial type.
// (The std::string check lives in the test file where <string> is already
//  included and forces no extra dependency here.)
static_assert(RingBuffer<SPSCRingBuffer<int>>);

}  // namespace spsc
