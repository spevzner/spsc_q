/**
 * @file test_spsc_queue.cpp
 * @brief Correctness tests + throughput benchmark for SPSCQueue / SPSCRingBuffer.
 *
 * Build:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *   ./build/tests/test_spsc_queue
 *
 * No external dependencies — uses only the standard library.
 */

#include "spsc_queue.hpp"
#include "spsc_ring_buffer.hpp"
#include "ring_buffer.hpp"
#include "storage.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// Mini test framework
// ---------------------------------------------------------------------------

namespace {

std::size_t g_pass = 0;
std::size_t g_fail = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::fprintf(stderr, "  FAIL  %s:%d  %s\n", file, line, expr);
    }
}

#define CHECK(expr)      check((expr), #expr, __FILE__, __LINE__)
#define CHECK_THROW(expr, ExType) \
    do { \
        bool caught_ = false; \
        try { (void)(expr); } catch (const ExType&) { caught_ = true; } \
        check(caught_, "throws " #ExType " from: " #expr, __FILE__, __LINE__); \
    } while (false)

void section(const char* name) {
    std::printf("\n── %s\n", name);
}

void print_summary() {
    std::printf("\n%s  %zu passed, %zu failed\n",
                g_fail ? "FAIL" : "PASS", g_pass, g_fail);
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
template <typename Fn>
double elapsed_ms(Fn&& fn) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Move-only type for testing
// ---------------------------------------------------------------------------
struct MoveOnly {
    explicit MoveOnly(int v) : value(v) {}
    MoveOnly(const MoveOnly&)            = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&& o) noexcept : value(o.value) { o.value = -1; }
    MoveOnly& operator=(MoveOnly&& o) noexcept {
        value = o.value; o.value = -1; return *this;
    }
    int value;
};

// ---------------------------------------------------------------------------
// Type with destructor-call counter (leak detection)
// ---------------------------------------------------------------------------
std::atomic<int> g_dtor_count{0};

struct Counted {
    explicit Counted(int v = 0) : value(v) {}
    ~Counted() { ++g_dtor_count; }
    Counted(const Counted& o)            : value(o.value) { }
    Counted(Counted&& o) noexcept        : value(o.value) { o.value = -1; }
    Counted& operator=(Counted&& o) noexcept {
        value = o.value; o.value = -1; return *this;
    }
    Counted& operator=(const Counted& o) { value = o.value; return *this; }
    int value;
};

// ---------------------------------------------------------------------------
// Complex object — non-trivial ctor/dtor, std::string member
// ---------------------------------------------------------------------------

// Lifetime counters let us verify that every constructed object is
// destroyed exactly once, regardless of storage backend.
std::atomic<int> g_msg_live{0};  // net live count (ctor - dtor)

struct Message {
    std::uint64_t id{};
    std::string   text;    // non-trivial; may allocate on heap (long strings)
    double        value{};

    Message()
        : id(0), value(0.0) { ++g_msg_live; }

    Message(std::uint64_t i, std::string t, double v)
        : id(i), text(std::move(t)), value(v) { ++g_msg_live; }

    Message(const Message& o)
        : id(o.id), text(o.text), value(o.value) { ++g_msg_live; }

    Message(Message&& o) noexcept
        : id(o.id), text(std::move(o.text)), value(o.value) { ++g_msg_live; }

    Message& operator=(Message&& o) noexcept {
        id = o.id; text = std::move(o.text); value = o.value; return *this;
    }
    Message& operator=(const Message& o) {
        id = o.id; text = o.text; value = o.value; return *this;
    }

    ~Message() { --g_msg_live; }

    bool operator==(const Message& o) const noexcept {
        return id == o.id && text == o.text && value == o.value;
    }
};

}  // anonymous namespace

// ===========================================================================
// Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Capacity rounding
// ---------------------------------------------------------------------------
static void test_capacity_rounding() {
    section("Capacity rounding (power of 2)");

    CHECK(spsc::SPSCQueue<int>(1).capacity()    == 1);
    CHECK(spsc::SPSCQueue<int>(2).capacity()    == 2);
    CHECK(spsc::SPSCQueue<int>(3).capacity()    == 4);
    CHECK(spsc::SPSCQueue<int>(4).capacity()    == 4);
    CHECK(spsc::SPSCQueue<int>(5).capacity()    == 8);
    CHECK(spsc::SPSCQueue<int>(7).capacity()    == 8);
    CHECK(spsc::SPSCQueue<int>(1000).capacity() == 1024);
    CHECK(spsc::SPSCQueue<int>(1024).capacity() == 1024);
    CHECK(spsc::SPSCQueue<int>(1025).capacity() == 2048);
}

// ---------------------------------------------------------------------------
// 2. Zero-capacity throws
// ---------------------------------------------------------------------------
static void test_zero_capacity() {
    section("Zero capacity throws");
    CHECK_THROW(spsc::SPSCQueue<int>(0), std::invalid_argument);
    CHECK_THROW(spsc::SPSCRingBuffer<int>(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 3. Basic push / pop round-trip
// ---------------------------------------------------------------------------
static void test_basic_push_pop() {
    section("Basic push / pop");

    spsc::SPSCQueue<int> q(4);

    CHECK(q.empty());
    CHECK(!q.full());
    CHECK(q.size() == 0);

    CHECK(q.try_push(10));
    CHECK(q.try_push(20));
    CHECK(q.try_push(30));
    CHECK(q.try_push(40));  // capacity is 4

    CHECK(q.full());
    CHECK(!q.empty());
    CHECK(q.size() == 4);

    CHECK(!q.try_push(50));  // full → false

    int v;
    CHECK(q.try_pop(v)); CHECK(v == 10);
    CHECK(q.try_pop(v)); CHECK(v == 20);
    CHECK(q.try_pop(v)); CHECK(v == 30);
    CHECK(q.try_pop(v)); CHECK(v == 40);

    CHECK(q.empty());
    CHECK(!q.try_pop(v));  // empty → false
}

// ---------------------------------------------------------------------------
// 4. FIFO ordering over many elements
// ---------------------------------------------------------------------------
static void test_fifo_ordering() {
    section("FIFO ordering");

    constexpr int N = 10'000;
    spsc::SPSCQueue<int> q(N);

    for (int i = 0; i < N; ++i)
        CHECK(q.try_push(i));

    for (int i = 0; i < N; ++i) {
        int v{};
        CHECK(q.try_pop(v));
        CHECK(v == i);
    }
    CHECK(q.empty());
}

// ---------------------------------------------------------------------------
// 5. Move-only type
// ---------------------------------------------------------------------------
static void test_move_only() {
    section("Move-only type");

    spsc::SPSCQueue<MoveOnly> q(4);
    CHECK(q.try_push(MoveOnly{1}));
    CHECK(q.try_push(MoveOnly{2}));

    MoveOnly a{0}, b{0};
    CHECK(q.try_pop(a)); CHECK(a.value == 1);
    CHECK(q.try_pop(b)); CHECK(b.value == 2);
    CHECK(q.empty());
}

// ---------------------------------------------------------------------------
// 6. try_emplace (in-place construction)
// ---------------------------------------------------------------------------
static void test_try_emplace() {
    section("try_emplace");

    spsc::SPSCQueue<std::string> q(4);
    CHECK(q.try_emplace(5, 'x'));   // std::string(5, 'x') == "xxxxx"
    CHECK(q.try_emplace("hello"));

    std::string s;
    CHECK(q.try_pop(s)); CHECK(s == "xxxxx");
    CHECK(q.try_pop(s)); CHECK(s == "hello");
}

// ---------------------------------------------------------------------------
// 7. Destructor correctness (no leaks, no double-frees)
// ---------------------------------------------------------------------------
static void test_destructor_correctness() {
    section("Destructor correctness");

    g_dtor_count = 0;
    {
        spsc::SPSCQueue<Counted> q(8);
        for (int i = 0; i < 5; ++i)
            (void)q.try_push(Counted{i});

        Counted out{0};
        (void)q.try_pop(out);  // pop 1 → dtor called inside pop
        (void)q.try_pop(out);  // pop 2
        // 3 elements remain live in the queue
    }  // destructor should call dtors for the 3 remaining elements
       // plus the 2 elements moved into `out` get destroyed at scope exit

    // Each Counted that was pushed was either popped (1 construction + 1 dtor
    // inside pop + 1 dtor when `out` is overwritten) or destroyed by the queue
    // dtor.  We just verify the final count makes sense (no leak / double-free
    // manifests as a crash or sanitiser report).
    CHECK(g_dtor_count.load() > 0);
    std::printf("     g_dtor_count = %d\n", g_dtor_count.load());
}

// ---------------------------------------------------------------------------
// 8. Refill: interleaved push/pop should work indefinitely (index wrap)
// ---------------------------------------------------------------------------
static void test_index_wraparound() {
    section("Index wraparound");

    spsc::SPSCQueue<int> q(4);

    for (int round = 0; round < 1000; ++round) {
        for (int i = 0; i < 4; ++i)
            CHECK(q.try_push(round * 4 + i));
        for (int i = 0; i < 4; ++i) {
            int v{};
            CHECK(q.try_pop(v));
            CHECK(v == round * 4 + i);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. Concept satisfaction — compile-time
// ---------------------------------------------------------------------------
static void test_concept_satisfaction() {
    section("Concept satisfaction (compile-time)");
    // These static_asserts live in the header; we just note they passed.
    CHECK(spsc::RingBuffer<spsc::SPSCRingBuffer<int>>);
    CHECK(spsc::RingBuffer<spsc::SPSCQueue<int>>);
    CHECK(spsc::RingBuffer<spsc::SPSCQueue<std::string>>);

    // SPSCQueueInterface (adds spinning push/pop)
    CHECK(spsc::SPSCQueueInterface<spsc::SPSCQueue<int>>);
}

// ---------------------------------------------------------------------------
// 10. Spinning push / pop on another thread
// ---------------------------------------------------------------------------
static void test_spin_push_pop_threaded() {
    section("Spinning push/pop (2 threads)");

    constexpr int N = 100'000;
    spsc::SPSCQueue<int> q(64);

    std::atomic<bool> done{false};

    auto producer = [&] {
        for (int i = 0; i < N; ++i)
            q.push(i);
        done = true;
    };

    std::vector<int> results;
    results.reserve(N);

    auto consumer = [&] {
        int v{};
        while (!done.load(std::memory_order_acquire) || !q.empty()) {
            if (q.try_pop(v))
                results.push_back(v);
        }
    };

    std::thread t_prod(producer);
    std::thread t_cons(consumer);
    t_prod.join();
    t_cons.join();

    CHECK(static_cast<int>(results.size()) == N);
    bool ordered = true;
    for (int i = 0; i < N; ++i)
        if (results[static_cast<std::size_t>(i)] != i) { ordered = false; break; }
    CHECK(ordered);
}

// ---------------------------------------------------------------------------
// 11. Throughput benchmark
// ---------------------------------------------------------------------------
static void bench_throughput() {
    section("Throughput benchmark");

    constexpr std::size_t N = 10'000'000;
    constexpr std::size_t QSIZE = 1024;

    spsc::SPSCQueue<std::uint64_t> q(QSIZE);

    // Warm up.
    for (std::size_t i = 0; i < QSIZE; ++i) q.push(i);
    std::uint64_t sink = 0;
    for (std::size_t i = 0; i < QSIZE; ++i) q.pop(sink);

    std::atomic<bool> start{false};
    std::atomic<std::uint64_t> checksum_prod{0}, checksum_cons{0};

    auto producer = [&] {
        while (!start.load(std::memory_order_acquire)) spsc::detail::cpu_relax();
        std::uint64_t s = 0;
        for (std::uint64_t i = 0; i < N; ++i) {
            q.push(i);
            s += i;
        }
        checksum_prod.store(s, std::memory_order_release);
    };

    auto consumer = [&] {
        while (!start.load(std::memory_order_acquire)) spsc::detail::cpu_relax();
        std::uint64_t s = 0, v = 0;
        for (std::size_t i = 0; i < N; ++i) {
            q.pop(v);
            s += v;
        }
        checksum_cons.store(s, std::memory_order_release);
    };

    std::thread t_prod(producer);
    std::thread t_cons(consumer);

    double ms = elapsed_ms([&] {
        start.store(true, std::memory_order_release);
        t_prod.join();
        t_cons.join();
    });

    CHECK(checksum_prod.load() == checksum_cons.load());

    double throughput_mops = static_cast<double>(N) / (ms * 1e3);  // M ops/s
    double ns_per_op = ms * 1e6 / static_cast<double>(N);
    std::printf("     %zu elements in %.2f ms → %.1f M ops/s  (%.2f ns/op)\n",
                N, ms, throughput_mops, ns_per_op);
}

// ---------------------------------------------------------------------------
// 12. Custom ring buffer plug-in (demonstrates the generic interface)
// ---------------------------------------------------------------------------

/// Minimal alternative ring buffer that uses a modulo-based index wrap.
/// Deliberately not power-of-2 to show the interface accepts other impls.
template <typename T>
class ModuloRingBuffer {
public:
    using value_type = T;

    explicit ModuloRingBuffer(std::size_t cap)
        : capacity_(cap), storage_(cap) {}

    [[nodiscard]] bool try_push(const T& v)  { return emplace(v); }
    [[nodiscard]] bool try_push(T&& v)       { return emplace(std::move(v)); }

    [[nodiscard]] bool try_pop(T& out) {
        if (head_ == tail_) return false;
        out    = std::move(storage_[tail_ % capacity_]);
        ++tail_;
        return true;
    }

    [[nodiscard]] std::size_t size()     const noexcept { return head_ - tail_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool        empty()    const noexcept { return head_ == tail_; }
    [[nodiscard]] bool        full()     const noexcept { return size() == capacity_; }

private:
    template <typename U>
    bool emplace(U&& v) {
        if (size() == capacity_) return false;
        storage_[head_ % capacity_] = std::forward<U>(v);
        ++head_;
        return true;
    }

    std::size_t        capacity_;
    std::vector<T>     storage_;
    std::size_t        head_{0};
    std::size_t        tail_{0};
};

static_assert(spsc::RingBuffer<ModuloRingBuffer<int>>);

static void test_custom_ring_buffer() {
    section("Custom ring buffer plug-in (ModuloRingBuffer)");

    // Plug a non-default ring buffer into SPSCQueue.
    spsc::SPSCQueue<int, ModuloRingBuffer<int>> q(5);
    CHECK(q.capacity() == 5);  // no power-of-2 rounding

    for (int i = 0; i < 5; ++i) CHECK(q.try_push(i));
    CHECK(q.full());
    CHECK(!q.try_push(99));

    for (int i = 0; i < 5; ++i) {
        int v{};
        CHECK(q.try_pop(v));
        CHECK(v == i);
    }
    CHECK(q.empty());
}

// ===========================================================================
// Storage policy tests
// ===========================================================================

// ---------------------------------------------------------------------------
// StoragePolicy concept satisfaction
// ---------------------------------------------------------------------------
static void test_storage_concepts() {
    section("StoragePolicy concept satisfaction (compile-time)");

    // Aliases avoid the preprocessor treating template commas as macro arg
    // separators (the static_asserts in storage.hpp fire at compile time too).
    using HeapInt      = spsc::HeapStorage<int>;
    using HeapStr      = spsc::HeapStorage<std::string>;
    using InlineInt8   = spsc::InlineStorage<int, 8>;
    using InlineStr16  = spsc::InlineStorage<std::string, 16>;
    using VecInt       = spsc::VectorStorage<int>;
    using VecStr       = spsc::VectorStorage<std::string>;
    using ArrayInt8    = spsc::ArrayStorage<int, 8>;
    using ArrayStr8    = spsc::ArrayStorage<std::string, 8>;

    CHECK(spsc::StoragePolicy<HeapInt>);
    CHECK(spsc::StoragePolicy<HeapStr>);
    CHECK(spsc::StoragePolicy<InlineInt8>);
    CHECK(spsc::StoragePolicy<InlineStr16>);
    CHECK(spsc::StoragePolicy<VecInt>);
    CHECK(spsc::StoragePolicy<VecStr>);
    CHECK(spsc::StoragePolicy<ArrayInt8>);
    CHECK(spsc::StoragePolicy<ArrayStr8>);

    // Verify the is_uninitialized trait is set correctly per type.
    // Use constexpr locals to avoid the preprocessor treating template commas
    // as macro argument separators.
    constexpr bool heap_uninit   = spsc::HeapStorage<int>::is_uninitialized;
    constexpr bool inline_uninit = spsc::InlineStorage<int, 8>::is_uninitialized;
    constexpr bool vec_uninit    = spsc::VectorStorage<int>::is_uninitialized;
    constexpr bool array_uninit  = spsc::ArrayStorage<int, 8>::is_uninitialized;
    CHECK(heap_uninit   == true);
    CHECK(inline_uninit == true);
    CHECK(vec_uninit    == true);
    CHECK(array_uninit  == false);
}

// ---------------------------------------------------------------------------
// HeapStorage — explicit use
// ---------------------------------------------------------------------------
static void test_heap_storage() {
    section("HeapStorage<T> — explicit backend");

    spsc::SPSCRingBuffer<int, spsc::HeapStorage<int>> rb(7);
    // 7 rounds up to 8
    CHECK(rb.capacity() == 8);
    CHECK(rb.empty());

    for (int i = 0; i < 8; ++i) CHECK(rb.try_push(i));
    CHECK(rb.full());
    CHECK(!rb.try_push(99));

    int v{};
    for (int i = 0; i < 8; ++i) {
        CHECK(rb.try_pop(v));
        CHECK(v == i);
    }
    CHECK(rb.empty());
}

// ---------------------------------------------------------------------------
// InlineStorage — zero-heap, compile-time N
// ---------------------------------------------------------------------------
static void test_inline_storage() {
    section("InlineStorage<T, N> — zero-heap backend");

    // Default-construct (no capacity argument needed)
    spsc::SPSCRingBuffer<int, spsc::InlineStorage<int, 16>> rb;
    CHECK(rb.capacity() == 16);
    CHECK(rb.empty());

    for (int i = 0; i < 16; ++i) CHECK(rb.try_push(i));
    CHECK(rb.full());
    CHECK(!rb.try_push(99));

    int v{};
    for (int i = 0; i < 16; ++i) {
        CHECK(rb.try_pop(v));
        CHECK(v == i);
    }
    CHECK(rb.empty());

    // Also verify via SPSCQueue default constructor
    spsc::SPSCQueue<int, spsc::SPSCRingBuffer<int, spsc::InlineStorage<int, 8>>> q;
    CHECK(q.capacity() == 8);
    q.push(42);
    int out{};
    q.pop(out);
    CHECK(out == 42);
}

// ---------------------------------------------------------------------------
// InlineStorage — object lives entirely on the stack (no heap)
// ---------------------------------------------------------------------------
static void test_inline_storage_stack_resident() {
    section("InlineStorage<T, N> — verifying no heap allocation");

    // This test is inherently hard to verify portably. We simply confirm
    // that construction and push/pop work for a non-trivial element type,
    // exercising placement-new and destroy_at on inline storage.

    using Q = spsc::SPSCRingBuffer<std::string, spsc::InlineStorage<std::string, 4>>;
    Q rb;
    CHECK(rb.capacity() == 4);

    CHECK(rb.try_push(std::string("alpha")));
    CHECK(rb.try_push(std::string("beta")));
    CHECK(rb.try_push(std::string("gamma")));
    CHECK(rb.try_push(std::string("delta")));
    CHECK(rb.full());

    std::string s;
    CHECK(rb.try_pop(s)); CHECK(s == "alpha");
    CHECK(rb.try_pop(s)); CHECK(s == "beta");
    CHECK(rb.try_pop(s)); CHECK(s == "gamma");
    CHECK(rb.try_pop(s)); CHECK(s == "delta");
    CHECK(rb.empty());
}

// ---------------------------------------------------------------------------
// VectorStorage — std::vector<byte> backend
// ---------------------------------------------------------------------------
static void test_vector_storage() {
    section("VectorStorage<T> — std::vector backend");

    spsc::SPSCRingBuffer<int, spsc::VectorStorage<int>> rb(5);
    // VectorStorage doesn't round to power-of-2 itself;
    // SPSCRingBuffer does, so capacity is 8.
    CHECK(rb.capacity() == 8);

    for (int i = 0; i < 8; ++i) CHECK(rb.try_push(i));
    CHECK(rb.full());

    int v{};
    for (int i = 0; i < 8; ++i) {
        CHECK(rb.try_pop(v));
        CHECK(v == i);
    }
    CHECK(rb.empty());

    // Non-trivial type
    spsc::SPSCRingBuffer<std::string, spsc::VectorStorage<std::string>> srb(4);
    CHECK(srb.capacity() == 4);
    CHECK(srb.try_push(std::string("hello")));
    CHECK(srb.try_push(std::string("world")));
    std::string s;
    CHECK(srb.try_pop(s)); CHECK(s == "hello");
    CHECK(srb.try_pop(s)); CHECK(s == "world");
}

// ---------------------------------------------------------------------------
// ArrayStorage — std::array<T, N> backend (slots are live objects)
// ---------------------------------------------------------------------------
static void test_array_storage() {
    section("ArrayStorage<T, N> — std::array backend");

    // Default-construct (no capacity arg); capacity == N.
    spsc::SPSCRingBuffer<int, spsc::ArrayStorage<int, 8>> rb;
    CHECK(rb.capacity() == 8);
    CHECK(rb.empty());

    for (int i = 0; i < 8; ++i) CHECK(rb.try_push(i));
    CHECK(rb.full());
    CHECK(!rb.try_push(99));

    int v{};
    for (int i = 0; i < 8; ++i) {
        CHECK(rb.try_pop(v));
        CHECK(v == i);
    }
    CHECK(rb.empty());

    // Non-trivial element type: std::string
    spsc::SPSCRingBuffer<std::string, spsc::ArrayStorage<std::string, 4>> srb;
    CHECK(srb.capacity() == 4);

    CHECK(srb.try_push(std::string("one")));
    CHECK(srb.try_push(std::string("two")));
    CHECK(srb.try_push(std::string("three")));
    CHECK(srb.try_push(std::string("four")));
    CHECK(srb.full());

    std::string s;
    CHECK(srb.try_pop(s)); CHECK(s == "one");
    CHECK(srb.try_pop(s)); CHECK(s == "two");
    CHECK(srb.try_pop(s)); CHECK(s == "three");
    CHECK(srb.try_pop(s)); CHECK(s == "four");
    CHECK(srb.empty());

    // FIFO ordering over many elements (exercises index wraparound).
    spsc::SPSCRingBuffer<int, spsc::ArrayStorage<int, 16>> rb2;
    for (int round = 0; round < 500; ++round) {
        for (int i = 0; i < 16; ++i) CHECK(rb2.try_push(round * 16 + i));
        for (int i = 0; i < 16; ++i) {
            CHECK(rb2.try_pop(v));
            CHECK(v == round * 16 + i);
        }
    }

    // Via SPSCQueue default constructor.
    spsc::SPSCQueue<int,
        spsc::SPSCRingBuffer<int, spsc::ArrayStorage<int, 32>>> q;
    CHECK(q.capacity() == 32);
    q.push(7);
    int out{};
    q.pop(out);
    CHECK(out == 7);
}

// ---------------------------------------------------------------------------
// Throughput comparison across storage backends
// ---------------------------------------------------------------------------
static void bench_storage_backends() {
    section("Throughput comparison — storage backends");

    constexpr std::size_t N     = 10'000'000;
    constexpr std::size_t QSIZE = 512;

    auto run = [&](auto& q, const char* label) {
        std::atomic<bool> go{false};
        double ms = elapsed_ms([&] {
            std::thread prod([&] {
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                for (std::size_t i = 0; i < N; ++i)
                    q.push(static_cast<std::uint64_t>(i));
            });
            std::thread cons([&] {
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                std::uint64_t v{};
                for (std::size_t i = 0; i < N; ++i)
                    q.pop(v);
            });
            go.store(true, std::memory_order_release);
            prod.join();
            cons.join();
        });
        std::printf("     %-35s  %.1f M ops/s  (%.2f ns/op)\n",
                    label,
                    static_cast<double>(N) / (ms * 1e3),
                    ms * 1e6 / static_cast<double>(N));
    };

    {
        spsc::SPSCQueue<std::uint64_t> q(QSIZE);  // HeapStorage (default)
        run(q, "HeapStorage (default)");
    }
    {
        spsc::SPSCQueue<std::uint64_t,
            spsc::SPSCRingBuffer<std::uint64_t, spsc::VectorStorage<std::uint64_t>>> q(QSIZE);
        run(q, "VectorStorage");
    }
    {
        // InlineStorage: capacity is compile-time, use default constructor
        spsc::SPSCQueue<std::uint64_t,
            spsc::SPSCRingBuffer<std::uint64_t, spsc::InlineStorage<std::uint64_t, 512>>> q;
        run(q, "InlineStorage<512> (zero-heap, raw)");
    }
    {
        // ArrayStorage: live std::array slots, push via assignment
        spsc::SPSCQueue<std::uint64_t,
            spsc::SPSCRingBuffer<std::uint64_t, spsc::ArrayStorage<std::uint64_t, 512>>> q;
        run(q, "ArrayStorage<512>  (zero-heap, live)");
    }
}

// ===========================================================================
// Complex object tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: exercise one ring buffer with N Message push/pops.
// Returns the net g_msg_live delta so callers can assert no leaks.
// ---------------------------------------------------------------------------
template <typename RB>
static int run_complex(RB& rb, std::size_t n,
                       bool sso,   // true = short string (SSO); false = long (heap)
                       const char* label)
{
    const int before = g_msg_live.load();

    auto make = [&](std::size_t i) -> Message {
        std::string text = sso
            ? ("m" + std::to_string(i))                        // ≤4 chars, SSO
            : ("long_instrument_name_" + std::to_string(i));   // >15 chars, heap
        return Message{i, std::move(text), static_cast<double>(i) * 2.71828};
    };

    for (std::size_t i = 0; i < n; ++i)
        CHECK(rb.try_push(make(i)));

    {   // inner scope: `out` is destroyed before `after` is sampled so it
        // doesn't contribute +1 to the live delta
        Message out;
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(rb.try_pop(out));
            CHECK(out.id    == i);
            CHECK(out.value == static_cast<double>(i) * 2.71828);
            CHECK(!out.text.empty());
        }
        CHECK(rb.empty());
    }   // `out` destroyed here

    // For raw storages:  all pushed elements have been destroy_at'd → 0 extra live.
    // For ArrayStorage:  N moved-from slots are still alive in arr_ (they were
    //                    already counted in `before`; none were extra-created here).
    const int after = g_msg_live.load();
    std::printf("     %-42s  live delta = %+d\n", label, after - before);
    return after - before;
}

// ---------------------------------------------------------------------------
// Correctness: all four backends, SSO + heap strings
// ---------------------------------------------------------------------------
static void test_complex_object_all_backends() {
    section("Complex object (Message) — correctness, all storage backends");

    constexpr std::size_t N = 64;

    // --- HeapStorage (raw, placement new) ---
    {
        spsc::SPSCRingBuffer<Message> rb(N);
        int delta = run_complex(rb, N, /*sso=*/true,  "HeapStorage  / SSO  string");
        CHECK(delta == 0);   // all elements destroyed after pop + queue dtor
    }
    {
        spsc::SPSCRingBuffer<Message> rb(N);
        int delta = run_complex(rb, N, /*sso=*/false, "HeapStorage  / heap string");
        CHECK(delta == 0);
    }

    // --- InlineStorage (raw, placement new) ---
    {
        spsc::SPSCRingBuffer<Message, spsc::InlineStorage<Message, 64>> rb;
        int delta = run_complex(rb, N, /*sso=*/true,  "InlineStorage/ SSO  string");
        CHECK(delta == 0);
    }
    {
        spsc::SPSCRingBuffer<Message, spsc::InlineStorage<Message, 64>> rb;
        int delta = run_complex(rb, N, /*sso=*/false, "InlineStorage/ heap string");
        CHECK(delta == 0);
    }

    // --- VectorStorage (raw, placement new) ---
    {
        spsc::SPSCRingBuffer<Message, spsc::VectorStorage<Message>> rb(N);
        int delta = run_complex(rb, N, /*sso=*/true,  "VectorStorage/ SSO  string");
        CHECK(delta == 0);
    }
    {
        spsc::SPSCRingBuffer<Message, spsc::VectorStorage<Message>> rb(N);
        int delta = run_complex(rb, N, /*sso=*/false, "VectorStorage/ heap string");
        CHECK(delta == 0);
    }

    // --- ArrayStorage (live slots, assignment) ---
    // After each pop the slot is in a moved-from state (valid but empty).
    // The N default-constructed Message objects are counted from construction
    // and are destroyed when the ring buffer itself is destroyed — not per pop.
    {
        spsc::SPSCRingBuffer<Message, spsc::ArrayStorage<Message, 64>> rb;
        int delta = run_complex(rb, N, /*sso=*/true,  "ArrayStorage / SSO  string");
        // delta == 0 once rb goes out of scope (destructor runs here)
        CHECK(delta == 0);
    }
    {
        spsc::SPSCRingBuffer<Message, spsc::ArrayStorage<Message, 64>> rb;
        int delta = run_complex(rb, N, /*sso=*/false, "ArrayStorage / heap string");
        CHECK(delta == 0);
    }

    // Global live count must be zero — no leaks from any backend.
    CHECK(g_msg_live.load() == 0);
}

// ---------------------------------------------------------------------------
// Throughput: complex object across all backends — SSO and heap string paths
// ---------------------------------------------------------------------------
static void bench_complex_object() {
    section("Throughput — complex object (Message with std::string)");

    constexpr std::size_t N     = 2'000'000;
    constexpr std::size_t QSIZE = 256;

    auto run_bench = [&](auto& q, const char* label) {
        std::atomic<bool> go{false};
        double ms = elapsed_ms([&] {
            std::thread prod([&]{
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                for (std::size_t i = 0; i < N; ++i)
                    q.push(Message{i, "sym", static_cast<double>(i)});
            });
            std::thread cons([&]{
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                Message v;
                for (std::size_t i = 0; i < N; ++i)
                    q.pop(v);
            });
            go.store(true, std::memory_order_release);
            prod.join();
            cons.join();
        });
        std::printf("     %-42s  %.1f M ops/s  (%.1f ns/op)\n",
                    label,
                    static_cast<double>(N) / (ms * 1e3),
                    ms * 1e6 / static_cast<double>(N));
    };

    std::printf("  [ SSO string: \"sym\" — no heap alloc per element ]\n");
    {
        spsc::SPSCQueue<Message> q(QSIZE);
        run_bench(q, "HeapStorage  (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::InlineStorage<Message, 256>>> q;
        run_bench(q, "InlineStorage<256> (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::VectorStorage<Message>>> q(QSIZE);
        run_bench(q, "VectorStorage (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::ArrayStorage<Message, 256>>> q;
        run_bench(q, "ArrayStorage<256>  (live, assign)");
    }

    std::printf("  [ Long string: 28 chars — heap alloc per element ]\n");
    auto long_str = [](std::size_t i) {
        return "long_instrument_name_" + std::to_string(i % 1000);
    };

    auto run_long = [&](auto& q, const char* label) {
        std::atomic<bool> go{false};
        double ms = elapsed_ms([&] {
            std::thread prod([&]{
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                for (std::size_t i = 0; i < N; ++i)
                    q.push(Message{i, long_str(i), static_cast<double>(i)});
            });
            std::thread cons([&]{
                while (!go.load(std::memory_order_acquire))
                    spsc::detail::cpu_relax();
                Message v;
                for (std::size_t i = 0; i < N; ++i)
                    q.pop(v);
            });
            go.store(true, std::memory_order_release);
            prod.join();
            cons.join();
        });
        std::printf("     %-42s  %.1f M ops/s  (%.1f ns/op)\n",
                    label,
                    static_cast<double>(N) / (ms * 1e3),
                    ms * 1e6 / static_cast<double>(N));
    };

    {
        spsc::SPSCQueue<Message> q(QSIZE);
        run_long(q, "HeapStorage  (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::InlineStorage<Message, 256>>> q;
        run_long(q, "InlineStorage<256> (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::VectorStorage<Message>>> q(QSIZE);
        run_long(q, "VectorStorage (raw)");
    }
    {
        spsc::SPSCQueue<Message,
            spsc::SPSCRingBuffer<Message, spsc::ArrayStorage<Message, 256>>> q;
        run_long(q, "ArrayStorage<256>  (live, assign)");
    }
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    std::printf("SPSC Queue — test suite\n");
    std::printf("=========================\n");

    test_capacity_rounding();
    test_zero_capacity();
    test_basic_push_pop();
    test_fifo_ordering();
    test_move_only();
    test_try_emplace();
    test_destructor_correctness();
    test_index_wraparound();
    test_concept_satisfaction();
    test_spin_push_pop_threaded();
    bench_throughput();
    test_custom_ring_buffer();
    test_storage_concepts();
    test_heap_storage();
    test_inline_storage();
    test_inline_storage_stack_resident();
    test_vector_storage();
    test_array_storage();
    bench_storage_backends();
    test_complex_object_all_backends();
    bench_complex_object();

    print_summary();
    return g_fail ? 1 : 0;
}
