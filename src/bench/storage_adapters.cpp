#include "bench/storage_engine.h"
#include "tree/lsm_tree.h"
#include "bptree/bptree.h"

namespace lsm {
namespace bench {

// ─── LSM Tree Adapter ──────────────────────────────────────
class LSMAdapter : public StorageEngine {
public:
    LSMAdapter(LSMTree* db) : db_(db) {}
    
    bool put(const std::string& key, const std::string& value) override {
        try {
            db_->put(key, value);
            return true;
        } catch (...) {
            return false;
        }
    }
    
    std::optional<std::string> get(const std::string& key) override {
        try {
            auto result = db_->get(key);
            if (result) {
                return result;
            }
            return std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    
    size_t size() const override {
        return 0;  // Not used in benchmark
    }
    
private:
    LSMTree* db_;
};

// ─── B+ Tree Adapter ──────────────────────────────────────
class BPTreeAdapter : public StorageEngine {
public:
    BPTreeAdapter() : tree_(std::make_unique<bptree::BPlusTree>()) {}
    
    bool put(const std::string& key, const std::string& value) override {
        return tree_->put(key, value);
    }
    
    std::optional<std::string> get(const std::string& key) override {
        return tree_->get(key);
    }
    
    size_t size() const override {
        return tree_->size();
    }
    
private:
    std::unique_ptr<bptree::BPlusTree> tree_;
};

}  // namespace bench
}  // namespace lsm
