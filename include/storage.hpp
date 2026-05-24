/**
 * @file storage.hpp
 * @brief StoragePolicy concept and built-in storage implementations.
 *
 * Overview
 * ========
 *
 * A storage policy owns a contiguous block of raw (uninitialised) memory and
 * hands out typed pointers to individual slots.  It knows nothing about
 * producers, consumers, or element lifetimes — that is entirely the ring
 * buffer's responsibility.
 *
 * Implementing your own storage
 * ==============================
 * Any class T that satisfies `spsc::StoragePolicy<T>` can be passed as the
 * second template argument to `SPSCRingBuffer` or `SPSCQueue`:
 *
 *   struct MyStorage {
 *       using value_type = int;
 *       MyStorage(std::size_t n);            // construct with n slots
 *       int*        slot(std::size_t i);     // raw pointer to slot i (0-based)
 *       std::size_t capacity() const;        // number of available slots
 *   };
 *   static_assert(spsc::StoragePolicy<MyStorage>);
 *
 * Built-in policies
 * =================
 *
 *  ┌──────────────────────────────┬────────┬──────────────────────────────────┐
 *  │ Type                         │ Init?  │ Best for                         │
 *  ├──────────────────────────────┼────────┼──────────────────────────────────┤
 *  │ HeapStorage<T>   (default)   │ raw    │ General; cache-line aligned heap │
 *  │ InlineStorage<T, N>          │ raw    │ Hot paths; zero heap, inline     │
 *  │ VectorStorage<T>             │ raw    │ Convenience; portable vector     │
 *  │ ArrayStorage<T, N>           │ live   │ Simple; std::array<T,N> backend; │
 *  │                              │        │ push=assign, no placement new    │
 *  └──────────────────────────────┴────────┴──────────────────────────────────┘
 *
 * "Init?" column
 * ==============
 * raw  — slots are uninitialised bytes; the ring buffer uses placement new on
 *         push and calls the destructor on pop (is_uninitialized = true).
 * live — slots hold default-constructed T objects from the start; the ring
 *         buffer uses assignment on push and skips destroy_at on pop
 *         (is_uninitialized = false).
 */

#pragma once

#include "spsc_detail.hpp"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>   // std::launder
#include <new>
#include <type_traits>
#include <vector>

namespace spsc {

// ---------------------------------------------------------------------------
// StoragePolicy concept
// ---------------------------------------------------------------------------

/**
 * @concept StoragePolicy
 * @brief Satisfied by any type that provides raw slot access and a capacity.
 *
 * @tparam S  Candidate storage type.
 *
 * Contract:
 *  - `slot(i)` returns a pointer to the element slot at index `i`.
 *    The index is raw (no modulo) — the ring buffer applies masking.
 *  - `capacity()` returns the number of slots (not bytes).
 *  - `is_uninitialized` (static constexpr bool) tells the ring buffer how to
 *    manage element lifetimes:
 *      true  — slots are raw uninitialised bytes; ring buffer uses placement
 *              new on push and calls the destructor on pop / queue destruction.
 *      false — slots hold live, default-constructed objects; ring buffer uses
 *              assignment on push and skips destroy_at (the storage destructor
 *              handles all element lifetimes).
 */
template <typename S>
concept StoragePolicy =
    requires { typename S::value_type; } &&
    requires(S& s, const S& cs, std::size_t i) {
        { s.slot(i)         } -> std::same_as<typename S::value_type*>;
        { cs.capacity()     } -> std::convertible_to<std::size_t>;
        { S::is_uninitialized } -> std::convertible_to<bool>;
    };

// ---------------------------------------------------------------------------
// HeapStorage<T>  — default, cache-line aligned heap allocation
// ---------------------------------------------------------------------------

/**
 * @class HeapStorage<T>
 * @brief Allocates raw, cache-line-aligned memory on the heap.
 *
 * The alignment guarantees that `slot(0)` lies on a cache-line boundary,
 * which minimises the chance that the first element shares a line with
 * unrelated data in the ring-buffer header.
 *
 * The allocation uses `::operator new` with an explicit `std::align_val_t`
 * so the same `delete` overload is called in the destructor.
 *
 * @note  The ring buffer passes an already-rounded power-of-2 capacity;
 *        HeapStorage does no rounding of its own.
 */
template <typename T>
class HeapStorage {
public:
    using value_type = T;

    explicit HeapStorage(std::size_t n)
        : n_    (n)
        , bytes_(static_cast<std::byte*>(
              ::operator new(n * sizeof(T), std::align_val_t{kAlign})))
    {
        assert(n > 0);
    }

    ~HeapStorage() noexcept {
        ::operator delete(bytes_, std::align_val_t{kAlign});
    }

    HeapStorage(const HeapStorage&)            = delete;
    HeapStorage& operator=(const HeapStorage&) = delete;
    HeapStorage(HeapStorage&&)                 = delete;
    HeapStorage& operator=(HeapStorage&&)      = delete;

    /// Slots are raw uninitialised bytes — ring buffer uses placement new.
    static constexpr bool is_uninitialized = true;

    /** @brief Pointer to the raw slot at index @p i (no bounds checking). */
    [[nodiscard]] T* slot(std::size_t i) noexcept {
        return std::launder(reinterpret_cast<T*>(bytes_) + i);
    }

    /** @brief Number of allocated slots. */
    [[nodiscard]] std::size_t capacity() const noexcept { return n_; }

private:
    // Storage block aligned to max(T's alignment, one cache line).
    static constexpr std::size_t kAlign =
        detail::max_align(alignof(T), detail::kCacheLineSize);

    std::size_t n_;
    std::byte*  bytes_;
};

static_assert(StoragePolicy<HeapStorage<int>>);

// ---------------------------------------------------------------------------
// InlineStorage<T, N>  — zero heap, all slots embedded inline
// ---------------------------------------------------------------------------

/**
 * @class InlineStorage<T, N>
 * @brief Embeds all N element slots directly inside the storage object itself.
 *
 * No heap allocation occurs at any point in the object's lifetime.  When
 * `InlineStorage` is used as the backend for an `SPSCRingBuffer`, the entire
 * queue (slots + control state) can fit inside a single struct, making it
 * fully cache-resident for small capacities.
 *
 * Constraints:
 *  - @p N must be a power of 2 (required by the ring buffer's bitmask logic).
 *  - @p N must be > 0.
 *
 * Construction:
 *  - Default-construct for zero-argument queues:
 *      `SPSCQueue<int, InlineStorage<int, 64>> q;`
 *  - Constructible from `std::size_t` for API uniformity; the value must
 *    equal N (checked by assertion in debug builds).
 *
 * @tparam T  Element type.
 * @tparam N  Compile-time slot count (power of 2).
 */
template <typename T, std::size_t N>
class InlineStorage {
    static_assert(N > 0,
        "InlineStorage<T, N>: N must be > 0");
    static_assert((N & (N - 1)) == 0,
        "InlineStorage<T, N>: N must be a power of 2");

public:
    using value_type = T;

    /// Slots are raw uninitialised bytes — ring buffer uses placement new.
    static constexpr bool is_uninitialized = true;

    /// Default-construct; capacity is always N.
    InlineStorage() noexcept = default;

    /// Construct from a runtime size; must equal N.
    explicit InlineStorage(std::size_t n) noexcept {
        assert(n == N && "InlineStorage: requested capacity must equal compile-time N");
        (void)n;
    }

    /** @brief Pointer to the raw slot at index @p i. */
    [[nodiscard]] T* slot(std::size_t i) noexcept {
        return std::launder(reinterpret_cast<T*>(bytes_.data()) + i);
    }

    /** @brief Always returns N. */
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return N; }

private:
    // Align the byte array to a cache-line boundary so that slot(0) never
    // shares a line with unrelated fields in the enclosing struct.
    static constexpr std::size_t kAlign =
        detail::max_align(alignof(T), detail::kCacheLineSize);

    alignas(kAlign) std::array<std::byte, N * sizeof(T)> bytes_{};
};

static_assert(StoragePolicy<InlineStorage<int, 8>>);

// ---------------------------------------------------------------------------
// VectorStorage<T>  — std::vector<std::byte> backend
// ---------------------------------------------------------------------------

/**
 * @class VectorStorage<T>
 * @brief Heap storage backed by `std::vector<std::byte>`.
 *
 * Trade-offs compared to `HeapStorage`:
 *  + Simpler; delegates all lifecycle to the vector.
 *  + No manual `operator delete` needed in the destructor.
 *  − No guaranteed cache-line alignment (system allocator decides).
 *  − Slight overhead from `std::vector`'s size/capacity bookkeeping.
 *
 * Suitable for:
 *  - Non-latency-critical queues where code simplicity matters more.
 *  - Testing custom ring-buffer logic with a well-known allocator.
 *  - Environments where aligned allocation is unavailable.
 */
template <typename T>
class VectorStorage {
public:
    using value_type = T;

    /// Slots are raw uninitialised bytes — ring buffer uses placement new.
    static constexpr bool is_uninitialized = true;

    explicit VectorStorage(std::size_t n)
        : bytes_(n * sizeof(T))
    {
        assert(n > 0);
    }

    /** @brief Pointer to the raw slot at index @p i. */
    [[nodiscard]] T* slot(std::size_t i) noexcept {
        return std::launder(reinterpret_cast<T*>(bytes_.data()) + i);
    }

    /** @brief Number of allocated slots. */
    [[nodiscard]] std::size_t capacity() const noexcept {
        return bytes_.size() / sizeof(T);
    }

private:
    std::vector<std::byte> bytes_;
};

static_assert(StoragePolicy<VectorStorage<int>>);

// ---------------------------------------------------------------------------
// ArrayStorage<T, N>  — std::array<T, N> backend  (slots are live objects)
// ---------------------------------------------------------------------------

/**
 * @class ArrayStorage<T, N>
 * @brief Compile-time-sized storage backed by a plain `std::array<T, N>`.
 *
 * Unlike the other storage policies, slots hold **fully constructed** T
 * objects from the moment the storage is created (default-initialised by
 * `std::array`).  The ring buffer therefore uses *assignment* on push and
 * skips `destroy_at` on pop — the array's own destructor handles the full
 * element lifetime.
 *
 * Trade-offs vs InlineStorage
 * ---------------------------
 *  + Simpler element access: no `reinterpret_cast` or `std::launder`.
 *  + Works naturally with any default-constructible T.
 *  − All N elements are default-constructed up front (one-time cost).
 *  − Push calls `operator=` rather than a constructor; for non-trivial T
 *    this means one extra move compared to placement new.
 *  − `T` must be default-constructible (required by `std::array`).
 *
 * @tparam T  Element type — must be default-constructible.
 * @tparam N  Compile-time slot count (power of 2).
 */
template <typename T, std::size_t N>
class ArrayStorage {
    static_assert(std::is_default_constructible_v<T>,
        "ArrayStorage<T, N>: T must be default-constructible");
    static_assert(N > 0,
        "ArrayStorage<T, N>: N must be > 0");
    static_assert((N & (N - 1)) == 0,
        "ArrayStorage<T, N>: N must be a power of 2");

public:
    using value_type = T;

    /// Slots are already-constructed live objects — ring buffer uses assignment.
    static constexpr bool is_uninitialized = false;

    /// Default-construct; all N elements are value-initialised by std::array.
    ArrayStorage() = default;

    /// Construct from a runtime size; must equal N (debug assertion).
    explicit ArrayStorage(std::size_t n) noexcept {
        assert(n == N && "ArrayStorage: requested capacity must equal compile-time N");
        (void)n;
    }

    /** @brief Pointer to the live element at index @p i. */
    [[nodiscard]] T* slot(std::size_t i) noexcept {
        return arr_.data() + i;
    }

    /** @brief Always returns N. */
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return N; }

private:
    std::array<T, N> arr_{};
};

static_assert(StoragePolicy<ArrayStorage<int, 8>>);

}  // namespace spsc
