#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include "common/types.h"
#include "common/config.h"

namespace lsm {

// ─── Write-Ahead Log (WAL) ───────────────────────────────
//
// The WAL ensures DURABILITY — no data is lost if the program crashes.
//
// Problem without WAL:
//   1. Client calls PUT("name", "Alice")
//   2. We write to MemTable (in memory)
//   3. CRASH! Power goes out.
//   4. MemTable is gone — it was only in RAM.
//   5. "Alice" is lost forever. ❌
//
// Solution with WAL:
//   1. Client calls PUT("name", "Alice")
//   2. We FIRST append to WAL file on disk  ← this survives crashes
//   3. Then we write to MemTable (in memory)
//   4. CRASH! Power goes out.
//   5. On restart: read WAL file → replay all entries → MemTable is restored ✓
//
// The WAL is an APPEND-ONLY file. We never modify existing data in it.
// This makes writes very fast (sequential I/O, no seeking).
//
// WAL Entry Format (binary):
// ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
// │ checksum │  type    │ key_len  │ val_len  │   key    │  value   │
// │ (4 bytes)│ (1 byte) │ (4 bytes)│ (4 bytes)│ (varies) │ (varies) │
// └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
//
// - checksum: CRC32 of everything after it (type + key_len + val_len + key + value)
//   Used to detect corruption. If checksum doesn't match, the entry is corrupted.
// - type: 0 = PUT, 1 = DELETE
// - key_len: length of the key in bytes
// - val_len: length of the value in bytes (0 for DELETE)
// - key: the raw key bytes
// - value: the raw value bytes (empty for DELETE)
//
// Lifecycle:
//   1. Created when database starts (or opens existing WAL)
//   2. Every Put/Delete appends an entry
//   3. When MemTable is flushed to SSTable, WAL is cleared (data is safe on disk now)
//   4. On crash recovery: replay WAL to rebuild MemTable

class WAL {
public:
    // Create/open a WAL at the given file path.
    // If the file exists, it can be replayed. If not, a new one is created.
    explicit WAL(const std::string& file_path);

    // Destructor: closes the file
    ~WAL();

    // ── Write Operations ──────────────────────────────────

    // Append a PUT entry to the WAL.
    // Must be called BEFORE writing to MemTable.
    // The entry is flushed to disk immediately (fsync) for durability.
    void append_put(const Key& key, const Value& value);

    // Append a DELETE entry to the WAL.
    // Must be called BEFORE writing to MemTable.
    void append_delete(const Key& key);

    // ── Recovery ──────────────────────────────────────────

    // Read all valid entries from the WAL file.
    // Used on startup to rebuild the MemTable after a crash.
    // Corrupted entries (bad checksum) are skipped.
    // Returns entries in the order they were written.
    std::vector<Entry> replay();

    // ── Lifecycle ─────────────────────────────────────────

    // Clear the WAL (delete all contents).
    // Called after a successful MemTable flush to SSTable.
    // The data is now safely on disk in the SSTable, so the WAL is no longer needed.
    void clear();

    // Sync the WAL to disk (fsync).
    // Ensures all buffered writes are physically on disk.
    void sync();

    // ── Statistics ─────────────────────────────────────────

    // Get the file path of this WAL
    const std::string& file_path() const { return file_path_; }

    // Get the number of entries written since last clear
    size_t entry_count() const { return entry_count_; }

    // ── No Copy ───────────────────────────────────────────
    WAL(const WAL&) = delete;
    WAL& operator=(const WAL&) = delete;

private:
    // Write a single entry to the WAL file (internal helper)
    void append_entry(EntryType type, const Key& key, const Value& value);

    // Compute CRC32 checksum of a byte buffer
    static uint32_t compute_crc32(const std::vector<uint8_t>& data);

    // Write raw bytes to the file
    void write_bytes(const void* data, size_t size);

    // Read raw bytes from the file
    bool read_bytes(void* data, size_t size, std::ifstream& in);

    std::string file_path_;        // Path to the WAL file
    std::ofstream write_stream_;   // File handle for writing (append mode)
    size_t entry_count_;           // Number of entries written
};

} // namespace lsm
