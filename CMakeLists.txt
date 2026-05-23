cmake_minimum_required(VERSION 3.25)
project(spsc_q VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall -Wextra -Wpedantic
)

# ---------------------------------------------------------------------------
# Header-only library target
# ---------------------------------------------------------------------------
add_library(spsc_q INTERFACE)
target_include_directories(spsc_q INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---------------------------------------------------------------------------
# Tests + benchmark
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_spsc_queue tests/test_spsc_queue.cpp)
target_link_libraries(test_spsc_queue PRIVATE spsc_q)
target_compile_options(test_spsc_queue PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fsanitize=address>
)
target_link_options(test_spsc_queue PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)

# Apple Clang doesn't need -pthread explicitly but it's harmless.
find_package(Threads REQUIRED)
target_link_libraries(test_spsc_queue PRIVATE Threads::Threads)

add_test(NAME spsc_correctness COMMAND test_spsc_queue)
