/**
 * @file spsc_ring_buffer.hpp
 * @brief Default low-latency, lock-free SPSC ring buffer implementation.
 *
 * Template signature
 * ==================
 *
 *   SPSCRingBuffer<T, Storage = HeapStorage<T>>
 *
 * `Storage` must satisfy `spsc::StoragePolicy` and have `value_type == T`.
 * Swap in any of the three built-in policies — or your own — to change only
 * the memory layer while keeping all the lock-free logic intact.
 *
 * Example
 * -------
 *
 *   // Default: cache-line aligned heap, runtime capacity
 *   spsc::SPSCRingBuffer<int> a(1024);
 *
 *   // Zero-heap: all slots live inside the object itself (stack / member)
 *   spsc::SPSCRingBuffer<int, spsc::InlineStorage<int, 64>> b;
 *
 *   // std::vector backend (no custom alignment)
 *   spsc::SPSCRingBuffer<int, spsc::VectorStorage<int>> c(1024);
 *
 *   // Your own policy
 *   spsc::SPSCRingBuffer<MyEvent, MyStorage<MyEvent>> d(512);
 *
 * Latency decisions (unchanged from the single-storage version)
 * =============================================================
 *  1. Power-of-2 capacity   → index masking instead of modulo
 *  2. Cache-line padding     → head_ / tail_ on separate lines (no false sharing)
 *  3. acquire / release only → no MFENCE / DMB on critical path
 *  4. Placement new          → no default-construction overhead
 *  5. try_emplace            → in-place construction, zero temporaries
 *  6. trivial-dtor fast-path → skips destroy_at loop for POD element types
 *
 * Thread-safety
 * =============
 * Exactly ONE producer thread calls try_push / try_emplace.
 * Exactly ONE consumer thread calls try_pop.
 */

#pragma once

#include "ring_buffer.hpp"
#include "storage.hpp"       // StoragePolicy + HeapStorage (default)

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>        // std::construct_at, std::destroy_at
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace spsc {

// ---------------------------------------------------------------------------
// SPSCRingBuffer
// ---------------------------------------------------------------------------

/**
 * @class SPSCRingBuffer<T, Storage>
 * @brief Low-latency, lock-free SPSC ring buffer with pluggable storage.
 *
 * @tparam T        Element type.
 * @tparam Storage  Storage policy (default: HeapStorage<T>).
 *                  Must satisfy `spsc::StoragePolicy<Storage>` and have
 *                  `Storage::value_type == T`.
 */
template <typename T, typename Storage = HeapStorage<T>>
    requires StoragePolicy<Storage> && std::same_as<typename Storage::value_type, T>
class SPSCRingBuffer {
    static_assert(std::is_move_constructible_v<T> || std::is_copy_constructible_v<T>,
                  "T must be move- or copy-constructible");

public:
    using value_type       = T;
    using storage_type     = Storage;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct with a runtime capacity (rounded up to the next power of 2).
     *
     * Available for storage types that accept a `std::size_t` argument
     * (HeapStorage, VectorStorage, InlineStorage — all built-in types).
     *
     * @throws std::invalid_argument if capacity == 0.
     * @throws std::overflow_error   if capacity is too large.
     * @throws std::bad_alloc        if the storage allocation fails.
     */
    explicit SPSCRingBuffer(std::size_t capacity)
        requires std::constructible_from<Storage, std::size_t>
        : storage_ (detail::next_power_of_2(capacity))
        , capacity_(storage_.capacity())
        , mask_    (capacity_ - 1)
    {}

    /**
     * @brief Construct using the storage type's default constructor.
     *
     * Available for storage types that are default-constructible, i.e.
     * `InlineStorage<T, N>`.  The capacity is taken from `storage_.capacity()`
     * (which returns N for InlineStorage).
     *
     * Usage:
     *   SPSCRingBuffer<int, InlineStorage<int, 64>> rb;
     */
    SPSCRingBuffer()
        requires std::default_initializable<Storage>
        : storage_ ()
        , capacity_(storage_.capacity())
        , mask_    (capacity_ - 1)
    {}

    // -----------------------------------------------------------------------
    // Destructor
    // -----------------------------------------------------------------------

    ~SPSCRingBuffer() noexcept {
        if constexpr (Storage::is_uninitialized) {
            // Slots outside [tail_, head_) are raw uninitialised bytes — only
            // destroy elements that were actually pushed but not yet popped.
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::size_t       tail = tail_.load(std::memory_order_relaxed);
                const std::size_t head = head_.load(std::memory_order_relaxed);
                while (tail != head) {
                    std::destroy_at(storage_.slot(tail & mask_));
                    ++tail;
                }
            }
        }
        // For initialised storage (ArrayStorage): all elements are owned by
        // the storage object's own destructor — nothing extra to do here.
        // storage_'s destructor runs automatically after this body.
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
     * @return true on success; @p value is untouched on false.
     */
    [[nodiscard]] bool try_push(T&& value)
        noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        return emplace_one(std::move(value));
    }

    /**
     * @brief Try to construct an element in-place with @p args.
     *
     * The element is built directly inside the ring-buffer slot — no
     * temporary is created.  Returns false if the buffer is full; in that
     * case no arguments are forwarded.
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
     * @return true on success; @p out is untouched on false.
     */
    [[nodiscard]] bool try_pop(T& out)
        noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        // Relaxed: no ordering needed with respect to our own tail_.
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        // Acquire: synchronise with the producer's release-store of head_.
        // Guarantees we see the element written before head_ was incremented.
        const std::size_t head = head_.load(std::memory_order_acquire);

        if (head == tail) [[unlikely]]
            return false;  // empty

        T* p = storage_.slot(tail & mask_);
        out  = std::move(*p);

        if constexpr (Storage::is_uninitialized) {
            // Raw bytes: explicitly destroy the moved-from object so the slot
            // reverts to uninitialised storage ready for the next push.
            if constexpr (!std::is_trivially_destructible_v<T>)
                std::destroy_at(p);
        }
        // Initialised storage (ArrayStorage): leave the slot in its moved-from
        // state (valid per the standard); the next push will overwrite it via
        // assignment, and the storage destructor will destroy it eventually.

        // Release: the producer's next acquire-load of tail_ will see this,
        // marking the slot as free.
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // State queries  (approximate; safe to call from either thread)
    // -----------------------------------------------------------------------

    /** @brief Approximate element count (may lag by ≤ 1 operation). */
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    /** @brief Actual (possibly rounded-up) capacity reported by the storage. */
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /** @brief Returns true if the buffer appears empty (may be stale). */
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /** @brief Returns true if the buffer appears full (may be stale). */
    [[nodiscard]] bool full() const noexcept {
        return size() == capacity_;
    }

    // -----------------------------------------------------------------------
    // Storage access
    // -----------------------------------------------------------------------

    /** @brief Direct access to the underlying storage object. */
    [[nodiscard]] Storage&       storage()       noexcept { return storage_; }
    [[nodiscard]] const Storage& storage() const noexcept { return storage_; }

private:
    // -----------------------------------------------------------------------
    // Core push implementation
    // -----------------------------------------------------------------------

    template <typename... Args>
    [[nodiscard]] bool emplace_one(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>)
    {
        // Relaxed: no ordering needed with respect to our own head_.
        const std::size_t head = head_.load(std::memory_order_relaxed);
        // Acquire: synchronise with the consumer's release-store of tail_.
        // Ensures we see the slot as free before we write into it.
        const std::size_t tail = tail_.load(std::memory_order_acquire);

        if ((head - tail) == capacity_) [[unlikely]]
            return false;  // full — args NOT touched

        if constexpr (Storage::is_uninitialized) {
            // Raw bytes: construct directly in the slot (zero copies).
            std::construct_at(storage_.slot(head & mask_), std::forward<Args>(args)...);
        } else {
            // Live slot: assign into the already-constructed object.
            // For try_push(T&&) this is a move-assign; for try_push(const T&)
            // a copy-assign; for try_emplace a construct-then-move-assign.
            *storage_.slot(head & mask_) = T(std::forward<Args>(args)...);
        }

        // Release: consumer's next acquire-load of head_ will see the element.
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // -----------------------------------------------------------------------
    // Data layout
    // -----------------------------------------------------------------------
    //
    // storage_   — first, because InlineStorage<T,N> can be large and we want
    //              its alignas to take effect at the start of the object.
    // capacity_  — immutable after construction; cached to avoid a virtual call.
    // mask_      — immutable after construction; hot in emplace_one / try_pop.
    // head_      — producer-owned; on its own cache line (no false sharing).
    // tail_      — consumer-owned; on its own cache line.

    Storage           storage_;
    const std::size_t capacity_;
    const std::size_t mask_;

    alignas(detail::kCacheLineSize)
    std::atomic<std::size_t> head_{0};

    alignas(detail::kCacheLineSize)
    std::atomic<std::size_t> tail_{0};
};

// Compile-time concept checks.
static_assert(RingBuffer<SPSCRingBuffer<int>>);
static_assert(RingBuffer<SPSCRingBuffer<int, InlineStorage<int, 8>>>);
static_assert(RingBuffer<SPSCRingBuffer<int, VectorStorage<int>>>);
static_assert(RingBuffer<SPSCRingBuffer<int, ArrayStorage<int, 8>>>);

}  // namespace spsc
