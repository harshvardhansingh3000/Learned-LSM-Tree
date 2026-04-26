cmake_minimum_required(VERSION 3.14)
project(LearnedLSMTree VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Include directories
# This tells the compiler: "When you see `#include "bloom/bloom_filter.h"`, look in `/Users/hskhurmi/LSM/include/`"
# without this the compiler would not know where to find our header files
include_directories(${CMAKE_SOURCE_DIR}/include) # ${CMAKE_SOURCE_DIR} = the root of our project (/Users/hskhurmi/LSM)

# ─── Google Test ───────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ─── Library: lsm_core ────────────────────────────────────
# All source files for the core LSM library
file(GLOB_RECURSE LSM_SOURCES
    "src/memtable/*.cpp"
    "src/wal/*.cpp"
    "src/sstable/*.cpp"
    "src/common/*.cpp"
    "src/bloom/*.cpp"
    "src/tree/*.cpp"
)

add_library(lsm_core STATIC ${LSM_SOURCES})
target_include_directories(lsm_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ─── Tests ─────────────────────────────────────────────────
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

add_executable(lsm_tests ${TEST_SOURCES})
target_link_libraries(lsm_tests PRIVATE lsm_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lsm_tests)

# ─── Main executable (placeholder) ────────────────────────
# add_executable(lsm_db src/main.cpp)
# target_link_libraries(lsm_db PRIVATE lsm_core)
