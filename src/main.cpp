#include "tree/lsm_tree.h"
#include "ml/classifier.h"
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

using namespace lsm;

// Read modes
enum class ReadMode { TRADITIONAL, CLASSIFIER };
static ReadMode current_mode = ReadMode::TRADITIONAL;
static LevelClassifier level_classifier;
static size_t bloom_checks_skipped = 0;
static size_t bloom_checks_total = 0;

// Classifier-augmented GET
static GetResult classifier_get(LSMTree& db, const Key& key) {
    // Step 1: Check MemTable (same as traditional)
    auto result = db.get(key);  // This always checks MemTable first
    
    // For now, we use the traditional get but track what the classifier would do
    // In a full implementation, we'd modify the Level::get() to skip levels
    auto predictions = level_classifier.predict_levels(key);
    
    for (size_t i = 0; i < predictions.size(); i++) {
        bloom_checks_total++;
        if (!predictions[i]) {
            bloom_checks_skipped++;
        }
    }
    
    return result;
}

// ─── CLI for LSM-Tree Database ────────────────────────────
//
// Interactive command-line interface to interact with the database.
//
// Commands:
//   put <key> <value>     — Insert or update a key-value pair
//   get <key>             — Look up a key
//   delete <key>          — Delete a key
//   scan <start> <end>    — Range scan
//   flush                 — Force flush MemTable to disk
//   compact               — Force compaction
//   stats                 — Show database statistics
//   help                  — Show available commands
//   quit / exit           — Exit the program

static void print_help() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║         LSM-Tree Database — Commands             ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣\n";
    std::cout << "║  put <key> <value>   Insert/update a key-value  ║\n";
    std::cout << "║  get <key>           Look up a key              ║\n";
    std::cout << "║  delete <key>        Delete a key               ║\n";
    std::cout << "║  scan <start> <end>  Range scan (inclusive)     ║\n";
    std::cout << "║  flush               Force flush to disk        ║\n";
    std::cout << "║  compact             Force compaction           ║\n";
    std::cout << "║  stats               Show database statistics   ║\n";
    std::cout << "║  load <count>        Load N random key-values   ║\n";
    std::cout << "║  predict <key>       ML: predict which level    ║\n";
    std::cout << "║  train               Train ML classifier        ║\n";
    std::cout << "║  help                Show this help             ║\n";
    std::cout << "║  quit / exit         Exit the program           ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

static void print_stats(LSMTree& db) {
    std::cout << "\n";
    std::cout << "┌─────────────────────────────────┐\n";
    std::cout << "│      Database Statistics         │\n";
    std::cout << "├─────────────────────────────────┤\n";
    std::cout << "│  Levels:     " << std::setw(18) << db.num_levels() << " │\n";
    std::cout << "│  SSTables:   " << std::setw(18) << db.total_sstables() << " │\n";
    std::cout << "│  Entries:    " << std::setw(18) << db.total_entries() << " │\n";
    std::cout << "└─────────────────────────────────┘\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    // Parse optional data directory from command line
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    // Configure the database
    Config config;
    config.data_dir = data_dir;
    config.wal_dir = data_dir + "/wal";
    config.sstable_dir = data_dir + "/sstables";

    std::cout << "\n";
    std::cout << "🌲 Learned LSM-Tree Database v1.0\n";
    std::cout << "   Data directory: " << data_dir << "\n";
    std::cout << "   MemTable size:  " << config.memtable_size_bytes / 1024 << " KB\n";
    std::cout << "   Size ratio:     " << config.size_ratio << "x\n";
    std::cout << "   Type 'help' for commands.\n";
    std::cout << "\n";

    // Create the database
    LSMTree db(config);

    std::string line;
    while (true) {
        std::cout << "lsm> ";
        if (!std::getline(std::cin, line)) {
            break;  // EOF (Ctrl+D)
        }

        // Parse the command
        std::istringstream iss(line);
        std::string command;
        if (!(iss >> command)) {
            continue;  // Empty line
        }

        // ── PUT ───────────────────────────────────────────
        if (command == "put") {
            std::string key, value;
            if (!(iss >> key)) {
                std::cout << "  Error: usage: put <key> <value>\n";
                continue;
            }
            // Value is the rest of the line (may contain spaces)
            std::getline(iss >> std::ws, value);
            if (value.empty()) {
                std::cout << "  Error: usage: put <key> <value>\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            db.put(key, value);
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "  OK (" << us << " μs)\n";
        }
        // ── GET ───────────────────────────────────────────
        else if (command == "get") {
            std::string key;
            if (!(iss >> key)) {
                std::cout << "  Error: usage: get <key>\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto result = db.get(key);
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            if (result.found) {
                std::cout << "  " << result.value << " (" << us << " μs)\n";
            } else {
                std::cout << "  (not found) (" << us << " μs)\n";
            }
        }
        // ── DELETE ────────────────────────────────────────
        else if (command == "delete" || command == "del" || command == "remove") {
            std::string key;
            if (!(iss >> key)) {
                std::cout << "  Error: usage: delete <key>\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            db.remove(key);
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "  OK (" << us << " μs)\n";
        }
        // ── SCAN ──────────────────────────────────────────
        else if (command == "scan") {
            std::string start_key, end_key;
            if (!(iss >> start_key >> end_key)) {
                std::cout << "  Error: usage: scan <start_key> <end_key>\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto results = db.scan(start_key, end_key);
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            if (results.empty()) {
                std::cout << "  (no results)\n";
            } else {
                for (const auto& [k, v] : results) {
                    std::cout << "  " << k << " → " << v << "\n";
                }
                std::cout << "  (" << results.size() << " results, " << us << " μs)\n";
            }
        }
        // ── FLUSH ─────────────────────────────────────────
        else if (command == "flush") {
            auto start = std::chrono::high_resolution_clock::now();
            db.flush();
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "  Flushed (" << us << " μs)\n";
        }
        // ── COMPACT ───────────────────────────────────────
        else if (command == "compact") {
            auto start = std::chrono::high_resolution_clock::now();
            db.compact();
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "  Compacted (" << us << " μs)\n";
        }
        // ── STATS ─────────────────────────────────────────
        else if (command == "stats") {
            print_stats(db);
        }
        // ── LOAD (bulk insert) ────────────────────────────
        else if (command == "load") {
            int count = 100;
            iss >> count;

            std::cout << "  Loading " << count << " key-value pairs...\n";
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < count; i++) {
                std::string num = std::to_string(i);
                std::string key = "key_" + std::string(6 - std::min(num.length(), static_cast<size_t>(6)), '0') + num;
                std::string value = "value_" + num + "_" + std::string(80, 'x');  // ~100 byte values
                db.put(key, value);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            double ops_per_sec = (count * 1000.0) / std::max(ms, static_cast<long long>(1));

            std::cout << "  Loaded " << count << " entries in " << ms << " ms";
            std::cout << " (" << static_cast<int>(ops_per_sec) << " ops/sec)\n";
        }
        // ── HELP ──────────────────────────────────────────
        else if (command == "help" || command == "?") {
            print_help();
        }
        // ── QUIT ──────────────────────────────────────────
        else if (command == "quit" || command == "exit" || command == "q") {
            std::cout << "  Goodbye! 👋\n";
            break;
        }
        // ── PREDICT (ML classifier) ───────────────────────
        else if (command == "predict") {
            std::string key;
            if (!(iss >> key)) {
                std::cout << "  Error: usage: predict <key>\n";
                continue;
            }

            if (!level_classifier.is_loaded()) {
                // Try to load models from multiple locations
                level_classifier.load(data_dir + "/models", 3);
                if (!level_classifier.is_loaded()) {
                    level_classifier.load("../" + data_dir + "/models", 3);  // from build/
                }
            }

            if (!level_classifier.is_loaded()) {
                std::cout << "  ML models not loaded! Run: python3 ml/train_classifier.py\n";
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto predictions = level_classifier.predict_levels(key);
            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "  ML Prediction for '" << key << "': ";
            bool any_check = false;
            for (size_t i = 0; i < predictions.size(); i++) {
                if (predictions[i]) {
                    std::cout << "L" << i << "=CHECK ";
                    any_check = true;
                } else {
                    std::cout << "L" << i << "=SKIP ";
                }
            }
            if (!any_check) {
                std::cout << " → Key likely NOT in database";
            }
            std::cout << " (" << us << " μs)\n";
        }
        // ── TRAIN (run Python training) ───────────────────
        else if (command == "train") {
            std::cout << "  Training ML classifier...\n";
            int ret = system("python3 ml/train_classifier.py");
            if (ret == 0) {
                // Reload models
                level_classifier.load(data_dir + "/models", 3);
                if (level_classifier.is_loaded()) {
                    std::cout << "  Models loaded! Use 'predict <key>' to test.\n";
                }
            } else {
                std::cout << "  Training failed. Make sure Python + sklearn are installed.\n";
            }
        }
        // ── UNKNOWN ───────────────────────────────────────
        else {
            std::cout << "  Unknown command: " << command << ". Type 'help' for commands.\n";
        }
    }

    return 0;
}
