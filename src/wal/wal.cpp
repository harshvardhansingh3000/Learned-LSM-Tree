#include "wal/wal.h"
#include <filesystem>
#include <stdexcept>
#include <cstring>

// For fsync on macOS/Linux
#include <unistd.h>
#include <fcntl.h>

namespace lsm {

// ─── CRC32 Lookup Table ──────────────────────────────────
// CRC32 uses a precomputed table of 256 entries for fast checksum calculation.
// Each entry is computed from the CRC32 polynomial (0xEDB88320).
//
// How CRC32 works (simplified):
//   1. Start with checksum = 0xFFFFFFFF
//   2. For each byte of data:
//      a. XOR the byte with the low 8 bits of checksum
//      b. Look up the result in this table
//      c. XOR with checksum shifted right by 8
//   3. Final XOR with 0xFFFFFFFF
//
// This detects single-bit errors, burst errors, and most multi-bit errors.
// If even ONE bit of the data changes, the CRC32 will be completely different.

static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

static void init_crc32_table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;  // CRC32 polynomial
            } else {
                crc = crc >> 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = true;
}

// ─── Constructor ──────────────────────────────────────────
// Opens or creates the WAL file.
//
// Steps:
//   1. Create parent directories if they don't exist
//   2. Open the file in append + binary mode
//   3. Initialize the CRC32 lookup table (once)
//
// std::ios::app    = append mode (write at end of file)
// std::ios::binary = binary mode (don't translate newlines)

WAL::WAL(const std::string& file_path)
    : file_path_(file_path)
    , entry_count_(0)
{
    // Create parent directories (e.g., "data/wal/")
    std::filesystem::path dir = std::filesystem::path(file_path).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }

    // Open file for writing in append + binary mode
    write_stream_.open(file_path_, std::ios::app | std::ios::binary);
    if (!write_stream_.is_open()) {
        throw std::runtime_error("Failed to open WAL file: " + file_path_);
    }

    // Initialize CRC32 table (only once, even if multiple WALs are created)
    if (!crc32_table_initialized) {
        init_crc32_table();
    }
}

// ─── Destructor ───────────────────────────────────────────
// Closes the write stream. Any buffered data is flushed.

WAL::~WAL() {
    if (write_stream_.is_open()) {
        write_stream_.close();
    }
}

// ─── Compute CRC32 ───────────────────────────────────────
// Calculates the CRC32 checksum of a byte buffer.
//
// Example:
//   Data: [0x00, 0x04, 0x00, 0x00, 0x00, 'n', 'a', 'm', 'e']
//   CRC32: 0xA3B2C1D0 (some 32-bit value)
//
// If we change even one byte (e.g., 'n' → 'o'):
//   CRC32: 0x7F8E9A12 (completely different!)

uint32_t WAL::compute_crc32(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xFFFFFFFF;
    for (uint8_t byte : data) {
        uint8_t index = (crc ^ byte) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return crc ^ 0xFFFFFFFF;
}

// ─── Write Bytes ──────────────────────────────────────────
// Writes raw bytes to the WAL file.
// static_cast<const char*> converts the void* to char* (required by ofstream).

void WAL::write_bytes(const void* data, size_t size) {
    write_stream_.write(static_cast<const char*>(data), size);
}

// ─── Read Bytes ───────────────────────────────────────────
// Reads raw bytes from the WAL file.
// Returns false if we couldn't read enough bytes (EOF or error).

bool WAL::read_bytes(void* data, size_t size, std::ifstream& in) {
    in.read(static_cast<char*>(data), size);
    return in.good() && (static_cast<size_t>(in.gcount()) == size);
}

// ─── Append Entry ─────────────────────────────────────────
// The core write function. Both append_put and append_delete call this.
//
// Steps:
//   1. Build the payload (everything after the checksum):
//      [type (1 byte)] [key_len (4 bytes)] [val_len (4 bytes)] [key] [value]
//   2. Compute CRC32 of the payload
//   3. Write checksum + payload to file
//   4. Flush to disk (sync)
//
// Why build payload first, then checksum?
//   Because the checksum covers the payload. We need the complete payload
//   to compute the checksum, then we write checksum first so during replay
//   we can read the checksum, read the payload, and verify.

void WAL::append_entry(EntryType type, const Key& key, const Value& value) {
    // Step 1: Build the payload bytes
    uint8_t type_byte = static_cast<uint8_t>(type);
    uint32_t key_len = static_cast<uint32_t>(key.size());
    uint32_t val_len = static_cast<uint32_t>(value.size());

    // Calculate total payload size: type(1) + key_len(4) + val_len(4) + key + value
    size_t payload_size = 1 + 4 + 4 + key.size() + value.size();
    std::vector<uint8_t> payload(payload_size);

    // Fill the payload buffer
    size_t offset = 0;

    // Type (1 byte)
    payload[offset] = type_byte;
    offset += 1;

    // Key length (4 bytes)
    std::memcpy(payload.data() + offset, &key_len, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Value length (4 bytes)
    std::memcpy(payload.data() + offset, &val_len, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Key bytes
    std::memcpy(payload.data() + offset, key.data(), key.size());
    offset += key.size();

    // Value bytes
    if (!value.empty()) {
        std::memcpy(payload.data() + offset, value.data(), value.size());
    }

    // Step 2: Compute CRC32 checksum of the payload
    uint32_t checksum = compute_crc32(payload);

    // Step 3: Write to file — checksum first, then payload
    write_bytes(&checksum, sizeof(uint32_t));
    write_bytes(payload.data(), payload.size());

    // Step 4: Flush to disk
    write_stream_.flush();

    entry_count_++;
}

// ─── Append Put ───────────────────────────────────────────
// Public API: append a PUT entry.

void WAL::append_put(const Key& key, const Value& value) {
    append_entry(EntryType::PUT, key, value);
}

// ─── Append Delete ────────────────────────────────────────
// Public API: append a DELETE entry (tombstone, empty value).

void WAL::append_delete(const Key& key) {
    append_entry(EntryType::DELETE, key, "");
}

// ─── Sync ─────────────────────────────────────────────────
// Forces all buffered data to be physically written to disk.
//
// write_stream_.flush() flushes C++ buffers to the OS.
// But the OS might still hold data in its own buffer!
// fsync() tells the OS: "Write to the physical disk NOW."
//
// On macOS, we use fcntl(F_FULLFSYNC) which is even stronger than fsync —
// it ensures data reaches the physical platters/flash cells.

void WAL::sync() {
    write_stream_.flush();

    // Get the file descriptor from the path and fsync it
    int fd = open(file_path_.c_str(), O_WRONLY | O_APPEND);
    if (fd >= 0) {
#ifdef __APPLE__
        fcntl(fd, F_FULLFSYNC);  // macOS: stronger than fsync
#else
        fsync(fd);               // Linux: standard fsync
#endif
        close(fd);
    }
}

// ─── Replay ──────────────────────────────────────────────
// Reads all valid entries from the WAL file for crash recovery.
//
// Algorithm:
//   1. Open the WAL file for reading
//   2. For each entry:
//      a. Read checksum (4 bytes)
//      b. Read payload (type + key_len + val_len + key + value)
//      c. Compute CRC32 of the payload
//      d. If computed CRC32 == stored checksum → valid entry, add to result
//      e. If mismatch → corrupted entry, stop reading
//   3. Return all valid entries
//
// Entries get sequential sequence numbers (1, 2, 3, ...) during replay
// since the WAL doesn't store sequence numbers (they're assigned by MemTable).

std::vector<Entry> WAL::replay() {
    std::vector<Entry> entries;

    // Open file for reading
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        return entries;  // File doesn't exist — nothing to replay
    }

    uint64_t seq = 0;  // Assign sequence numbers during replay

    while (true) {
        // Step 2a: Read checksum
        uint32_t stored_checksum;
        if (!read_bytes(&stored_checksum, sizeof(uint32_t), in)) {
            break;  // End of file — done
        }

        // Step 2b: Read type
        uint8_t type_byte;
        if (!read_bytes(&type_byte, 1, in)) {
            break;  // Truncated entry
        }

        // Read key length
        uint32_t key_len;
        if (!read_bytes(&key_len, sizeof(uint32_t), in)) {
            break;
        }

        // Read value length
        uint32_t val_len;
        if (!read_bytes(&val_len, sizeof(uint32_t), in)) {
            break;
        }

        // Sanity check: key and value shouldn't be absurdly large
        // (protects against reading garbage data)
        if (key_len > 10 * 1024 * 1024 || val_len > 10 * 1024 * 1024) {
            break;  // Corrupted — unreasonable sizes
        }

        // Read key
        std::string key(key_len, '\0');
        if (key_len > 0 && !read_bytes(key.data(), key_len, in)) {
            break;
        }

        // Read value
        std::string value(val_len, '\0');
        if (val_len > 0 && !read_bytes(value.data(), val_len, in)) {
            break;
        }

        // Step 2c: Rebuild payload and compute checksum
        size_t payload_size = 1 + 4 + 4 + key_len + val_len;
        std::vector<uint8_t> payload(payload_size);
        size_t offset = 0;

        payload[offset] = type_byte;
        offset += 1;
        std::memcpy(payload.data() + offset, &key_len, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(payload.data() + offset, &val_len, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(payload.data() + offset, key.data(), key_len);
        offset += key_len;
        if (val_len > 0) {
            std::memcpy(payload.data() + offset, value.data(), val_len);
        }

        uint32_t computed_checksum = compute_crc32(payload);

        // Step 2d: Verify checksum
        if (computed_checksum != stored_checksum) {
            break;  // Corrupted entry — stop here
        }

        // Step 2e: Valid entry! Add to results
        seq++;
        EntryType type = static_cast<EntryType>(type_byte);
        entries.emplace_back(std::move(key), std::move(value), type, seq);
    }

    return entries;
}

// ─── Clear ────────────────────────────────────────────────
// Deletes all WAL contents by closing and reopening the file in truncate mode.
//
// std::ios::trunc = truncate mode (delete all existing content)
//
// Called after a successful MemTable flush to SSTable.
// The data is now safely on disk in the SSTable, so the WAL is no longer needed.

void WAL::clear() {
    // Close the current write stream
    if (write_stream_.is_open()) {
        write_stream_.close();
    }

    // Reopen in truncate mode (deletes all content) + append + binary
    write_stream_.open(file_path_, std::ios::trunc | std::ios::binary);
    if (!write_stream_.is_open()) {
        throw std::runtime_error("Failed to reopen WAL file: " + file_path_);
    }

    entry_count_ = 0;
}

} // namespace lsm
