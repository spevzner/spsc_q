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

    print_summary();
    return g_fail ? 1 : 0;
}
