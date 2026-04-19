#pragma once

#include <string>
#include <optional>
#include <vector>
#include <cstdint>

namespace lsm {

// ─── Key-Value Types ──────────────────────────────────────
// Keys and values are both strings (byte sequences).
// Keys are 16 bytes in the paper's configuration, but we allow variable length.
// Values are 100 bytes in the paper's configuration, but we allow variable length.

using Key = std::string;
using Value = std::string;

// ─── Entry Types ──────────────────────────────────────────

enum class EntryType : uint8_t {
    PUT = 0,
    DELETE = 1   // Tombstone — marks a key as deleted
};

// A single key-value entry in the database.
// If type == DELETE, the value is empty (tombstone).
struct Entry {
    Key key;
    Value value;
    EntryType type;
    uint64_t sequence_number;  // For ordering — newer entries win

    Entry() : type(EntryType::PUT), sequence_number(0) {}

    Entry(Key k, Value v, EntryType t, uint64_t seq)
        : key(std::move(k)), value(std::move(v)), type(t), sequence_number(seq) {}

    // Entries are ordered by key first, then by sequence number (descending — newer first)
    bool operator<(const Entry& other) const {
        if (key != other.key) return key < other.key;
        return sequence_number > other.sequence_number;  // Newer first
    }

    bool operator==(const Entry& other) const {
        return key == other.key && sequence_number == other.sequence_number;
    }

    // Approximate size in bytes (for tracking MemTable size)
    size_t size() const {
        return key.size() + value.size() + sizeof(EntryType) + sizeof(uint64_t);
    }
};

// Result of a Get operation
struct GetResult {
    bool found;
    Value value;
    bool is_deleted;  // True if we found a tombstone

    static GetResult Found(Value v) {
        return {true, std::move(v), false};
    }

    static GetResult Deleted() {
        return {true, "", true};
    }

    static GetResult NotFound() {
        return {false, "", false};
    }
};

} // namespace lsm
