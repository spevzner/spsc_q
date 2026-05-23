/**
 * @file spsc_queue.hpp
 * @brief SPSC queue adaptor — wraps any RingBuffer implementation and adds
 *        spinning push/pop, try_emplace, and the full SPSCQueueInterface.
 *
 * Usage
 * =====
 *
 *   // Default: SPSCRingBuffer<int> under the hood
 *   spsc::SPSCQueue<int> q(1024);
 *   q.push(42);
 *   int v;
 *   q.pop(v);  // v == 42
 *
 *   // Custom ring buffer:
 *   spsc::SPSCQueue<MyEvent, MyCustomRingBuffer<MyEvent>> q2(512);
 *
 * Blocking vs non-blocking
 * ========================
 * `try_push` / `try_pop`  — non-blocking; return bool.
 * `push` / `pop`          — spin-wait with CPU yield hint (`pause` / `yield`).
 *                           These are appropriate for hot paths where a
 *                           predictable, bounded busy-wait is cheaper than a
 *                           context switch.  Do NOT use them if the producer /
 *                           consumer can be indefinitely stalled.
 */

#pragma once

#include "ring_buffer.hpp"
#include "spsc_ring_buffer.hpp"  // default implementation

#include <cstddef>
#include <type_traits>
#include <utility>

namespace spsc {

/**
 * @class SPSCQueue<T, RB>
 * @brief SPSC queue with configurable ring-buffer backend.
 *
 * @tparam T   Element type.
 * @tparam RB  Ring buffer implementation.  Must satisfy `spsc::RingBuffer<RB>`.
 *             Defaults to `SPSCRingBuffer<T>` — the best built-in low-latency
 *             implementation.
 */
template <typename T, typename RB = SPSCRingBuffer<T>>
    requires RingBuffer<RB> && std::same_as<typename RB::value_type, T>
class SPSCQueue {
public:
    using value_type       = T;
    using ring_buffer_type = RB;

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct a queue whose ring buffer has at least @p capacity slots.
     *
     * The ring buffer implementation may round the value up (e.g.
     * `SPSCRingBuffer` rounds to the next power of 2).  Query `capacity()` to
     * learn the actual allocated size.
     */
    explicit SPSCQueue(std::size_t capacity) : rb_(capacity) {}

    ~SPSCQueue()                             = default;
    SPSCQueue(const SPSCQueue&)              = delete;
    SPSCQueue& operator=(const SPSCQueue&)   = delete;
    SPSCQueue(SPSCQueue&&)                   = delete;
    SPSCQueue& operator=(SPSCQueue&&)        = delete;

    // -----------------------------------------------------------------------
    // Non-blocking interface  (return bool; never block)
    // -----------------------------------------------------------------------

    /**
     * @brief Try to push a copy of @p value.  Returns false if the queue is full.
     */
    [[nodiscard]] bool try_push(const T& value)
        noexcept(noexcept(rb_.try_push(value)))
    {
        return rb_.try_push(value);
    }

    /**
     * @brief Try to push by moving @p value.
     * @p value is untouched when false is returned.
     */
    [[nodiscard]] bool try_push(T&& value)
        noexcept(noexcept(rb_.try_push(std::move(value))))
    {
        return rb_.try_push(std::move(value));
    }

    /**
     * @brief Try to construct an element in-place with @p args.
     *
     * Builds the element directly in the ring-buffer slot — no temporary
     * object is created.  Returns false if the queue is full.
     */
    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    [[nodiscard]] bool try_emplace(Args&&... args)
        noexcept(noexcept(rb_.try_emplace(std::forward<Args>(args)...)))
    {
        return rb_.try_emplace(std::forward<Args>(args)...);
    }

    /**
     * @brief Try to pop the oldest element into @p out.
     * @p out is untouched when false is returned.
     */
    [[nodiscard]] bool try_pop(T& out)
        noexcept(noexcept(rb_.try_pop(out)))
    {
        return rb_.try_pop(out);
    }

    // -----------------------------------------------------------------------
    // Spin-wait interface  (block until success with CPU yield hint)
    // -----------------------------------------------------------------------

    /**
     * @brief Spin until a copy of @p value is pushed.
     *
     * Emits a CPU yield hint (`PAUSE` / `YIELD`) on each failed attempt to
     * reduce power consumption and avoid memory-ordering hazards in a
     * tight spin loop.
     *
     * @warning  Only use when the consumer is guaranteed to drain the queue
     *           eventually.  An infinitely-full queue will spin forever.
     */
    void push(const T& value)
        noexcept(noexcept(rb_.try_push(value)))
    {
        while (!rb_.try_push(value)) [[unlikely]]
            detail::cpu_relax();
    }

    /**
     * @brief Spin until @p value is moved into the queue.
     * @p value is only moved on the successful attempt.
     */
    void push(T&& value)
        noexcept(noexcept(rb_.try_push(std::move(value))))
    {
        // `std::move(value)` yields an rvalue-ref; the argument is NOT consumed
        // until the placement-new inside try_push succeeds — so re-trying with
        // the same expression is safe.
        while (!rb_.try_push(std::move(value))) [[unlikely]]
            detail::cpu_relax();
    }

    /**
     * @brief Spin until an element is popped into @p out.
     *
     * @warning  Only use when the producer is guaranteed to push eventually.
     */
    void pop(T& out)
        noexcept(noexcept(rb_.try_pop(out)))
    {
        while (!rb_.try_pop(out)) [[unlikely]]
            detail::cpu_relax();
    }

    // -----------------------------------------------------------------------
    // State queries  (approximate; safe from either thread)
    // -----------------------------------------------------------------------

    /** @brief Approximate element count (may lag by ≤1 operation). */
    [[nodiscard]] std::size_t size()     const noexcept { return rb_.size();     }
    /** @brief Actual (possibly rounded-up) capacity of the ring buffer. */
    [[nodiscard]] std::size_t capacity() const noexcept { return rb_.capacity(); }
    /** @brief Returns true if the queue appears empty (may be stale). */
    [[nodiscard]] bool        empty()    const noexcept { return rb_.empty();    }
    /** @brief Returns true if the queue appears full (may be stale). */
    [[nodiscard]] bool        full()     const noexcept { return rb_.full();     }

    // -----------------------------------------------------------------------
    // Ring-buffer access
    // -----------------------------------------------------------------------

    /** @brief Direct access to the underlying ring buffer. */
    [[nodiscard]] RB&       ring_buffer()       noexcept { return rb_; }
    [[nodiscard]] const RB& ring_buffer() const noexcept { return rb_; }

private:
    RB rb_;
};

// Compile-time interface verification with a trivial type.
// The std::string variant lives in the test file where <string> is in scope.
static_assert(RingBuffer<SPSCQueue<int>>);

}  // namespace spsc
