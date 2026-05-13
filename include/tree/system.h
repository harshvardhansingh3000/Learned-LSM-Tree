#pragma once

#include <string>
#include "common/types.h"

namespace lsm {

// ─── Abstract System Interface ────────────────────────────
// Base class for different storage systems (LSM-Tree, B+ Tree, etc.)
// Provides a common interface for benchmarking

class System {
public:
    virtual ~System() = default;

    // Insert or update a key-value pair
    virtual void put(const Key& key, const Value& value) = 0;

    // Look up a key. Returns the value if found, empty if not
    virtual GetResult get(const Key& key) = 0;

    // Force flush (for LSM-Tree, no-op for others)
    virtual void flush() {}

    // Get system name for reporting
    virtual std::string name() const = 0;
};

} // namespace lsm