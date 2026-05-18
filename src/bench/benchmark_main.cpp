#include "bench/benchmark.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>
#include <iomanip>
#include <ctime>
#include <sstream>
#include "tree/btree.h"
#include "tree/lsm_tree.h"

using namespace lsm;
using namespace lsm::bench;

// Helper to clean up data directory
void cleanup_data_dir(const std::string& dir) {
    try {
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
        std::filesystem::create_directories(dir);
        std::filesystem::create_directories(dir + "/sstables");
        std::filesystem::create_directories(dir + "/wal");
        std::filesystem::create_directories(dir + "/models");
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning data dir: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "   LSM-Tree Benchmark Suite — Comprehensive Multi-System Test\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "\n  Systems under test:\n";
    std::cout << "    1. LSM (Baseline)      — Traditional LSM-tree with Bloom filters\n";
    std::cout << "    2. LSM+Classifier      — Approach 1: ML classifier skips levels\n";
    std::cout << "    3. LSM+LearnedBF       — Approach 2: Learned Bloom Filter replaces BF\n";
    std::cout << "    4. B+ Tree (Disk)      — B+ Tree with simulated SSD I/O\n";
    std::cout << "\n";
    
    BenchmarkHarness harness;
    SystemComparator comparator;
    std::vector<WorkloadMetrics> all_results;
    
    // Dataset sizes to test
    const std::vector<size_t> dataset_sizes = {1000, 5000, 10000};
    
    // Workloads to benchmark
    const std::vector<Workload> workloads = {
        Workload::point_query_random(5000),
        Workload::write_heavy(5000),
        Workload::temporal_skewed(5000)
    };
    
    // Test configurations
    // type: 0 = LSM baseline, 1 = LSM+Classifier, 2 = LSM+LearnedBF, 3 = B+ Tree (Disk)
    struct TestConfig {
        std::string name;
        int type;  // 0=baseline, 1=classifier, 2=learned_bf, 3=btree_disk
    };
    
    const std::vector<TestConfig> configs = {
        {"LSM (Baseline)",  0},
        {"LSM+Classifier",  1},
        {"LSM+LearnedBF",   2},
        {"B+ Tree (Disk)",  3},
    };
    
    // ─── Main Benchmark Loop ──────────────────────────────
    for (const auto& dataset_size : dataset_sizes) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Dataset Size: " << std::setw(48) << std::left << (std::to_string(dataset_size) + " keys")
                  << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        
        for (const auto& config : configs) {
            std::cout << "┌─ Configuration: " << config.name << " ─────────────────────────────────┐\n\n";
            
            for (const auto& workload : workloads) {
                // Setup fresh database
                std::string data_dir = "data_bench_" + std::to_string(dataset_size) + "_" + config.name;
                cleanup_data_dir(data_dir);
                
                std::unique_ptr<System> db;
                bool use_classifier = false;
                
                if (config.type == 3) {
                    // B+ Tree with simulated disk I/O
                    db = std::make_unique<BTree>(true);  // simulate_disk_io = true
                } else {
                    // LSM-tree variants
                    Config db_config;
                    db_config.data_dir = data_dir;
                    db_config.wal_dir = data_dir + "/wal";
                    db_config.sstable_dir = data_dir + "/sstables";
                    db = std::make_unique<LSMTree>(db_config);
                }
                
                // Load data
                std::cout << "  Loading data for " << config.name << ":\n";
                harness.load_data(db.get(), dataset_size);
                
                // Load ML models if needed
                if (config.type == 1) {
                    harness.load_classifier("data/models", 3);
                    use_classifier = true;
                } else if (config.type == 2) {
                    harness.load_learned_bloom_filters("data/models", 3);
                }
                
                // Warmup
                std::cout << "  Warming up...\n";
                harness.warmup(db.get(), workload);
                
                // Run benchmark — include dataset size in system name for clarity
                std::string full_name = config.name + " (" + std::to_string(dataset_size) + "k)";
                auto result = harness.run_benchmark(db.get(), full_name, workload, use_classifier);
                all_results.push_back(result);
                comparator.add_result(result);
                
                db.reset();
                
                // Small delay between tests
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            std::cout << "\n└──────────────────────────────────────────────────────────────┘\n\n";
        }
    }
    
    // ─── Print Results Summary ────────────────────────────
    std::cout << "\n\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "                        RESULTS SUMMARY\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    
    harness.print_results(all_results);
    comparator.print_comparison_table();
    comparator.generate_comparison_chart();
    
    // Export to CSV with datetime suffix
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    std::ostringstream ts;
    ts << std::put_time(tm, "%Y%m%d_%H%M%S");
    std::string csv_filename = "benchmark_results_" + ts.str() + ".csv";
    
    std::cout << "Exporting detailed results...\n";
    harness.export_csv(all_results, csv_filename);
    
    std::cout << "\n✓ Benchmark completed successfully!\n";
    std::cout << "  Results saved to: " << csv_filename << "\n\n";
    
    return 0;
}
