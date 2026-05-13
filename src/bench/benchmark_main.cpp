#include "bench/benchmark.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>
#include <iomanip>
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
    std::cout << "          LSM-Tree Benchmark Suite - Comprehensive Test\n";
    std::cout << "════════════════════════════════════════════════════════════════\n\n";
    
    BenchmarkHarness harness;
    SystemComparator comparator;
    std::vector<WorkloadMetrics> all_results;
    
    // Dataset sizes to test (small for quick iteration)
    const std::vector<size_t> dataset_sizes = {1000, 5000, 10000};
    
    // Workloads to benchmark
    const std::vector<Workload> workloads = {
        Workload::point_query_random(5000),
        Workload::point_query_sequential(5000),
        Workload::write_heavy(5000),
        Workload::temporal_skewed(5000)
    };
    
    // Test configurations
    struct TestConfig {
        std::string name;
        bool use_classifier;
    };
    
    const std::vector<TestConfig> configs = {
        {"LSM (Baseline)", false},
        {"LSM+Classifier", true},
        {"B+ Tree", false}
    };
    
    // ─── Main Benchmark Loop ──────────────────────────────
    for (const auto& dataset_size : dataset_sizes) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Dataset Size: " << std::setw(48) << std::left << (std::to_string(dataset_size) + " keys")
                  << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        
        for (const auto& config : configs) {
            if (config.use_classifier) {
                harness.load_classifier("data/models", 3);
            }

            std::cout << "┌─ Configuration: " << config.name << " ─────────────────────────────────┐\n\n";
            
            for (const auto& workload : workloads) {
                // Setup fresh database
                std::string data_dir = "data_bench_" + std::to_string(dataset_size) + "_" + config.name;
                cleanup_data_dir(data_dir);
                
                std::unique_ptr<System> db;
                
                if (config.name == "B+ Tree") {
                    db = std::make_unique<BTree>();
                } else {
                    Config db_config;
                    db_config.data_dir = data_dir;
                    db_config.wal_dir = data_dir + "/wal";
                    db_config.sstable_dir = data_dir + "/sstables";
                    db = std::make_unique<LSMTree>(db_config);
                }
                
                // Load data
                std::cout << "  Loading data for " << config.name << ":\n";
                harness.load_data(db.get(), dataset_size);
                
                // Warmup
                std::cout << "  Warming up...\n";
                harness.warmup(db.get(), workload);
                
                // Run benchmark
                auto result = harness.run_benchmark(db.get(), config.name, workload, config.use_classifier);
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
    
    // Export to CSV
    std::cout << "Exporting detailed results...\n";
    harness.export_csv(all_results, "benchmark_results.csv");
    
    std::cout << "\n✓ Benchmark completed successfully!\n";
    std::cout << "  Results saved to: benchmark_results.csv\n\n";
    
    return 0;
}
