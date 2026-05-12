#pragma once

#include <string>
#include <optional>
#include <memory>

namespace lsm {
namespace bench {

// Simple storage engine abstraction for benchmarking
class StorageEngine {
public:
    virtual ~StorageEngine() = default;
    
    virtual bool put(const std::string& key, const std::string& value) = 0;
    virtual std::optional<std::string> get(const std::string& key) = 0;
    virtual size_t size() const = 0;
};

}  // namespace bench
}  // namespace lsm
