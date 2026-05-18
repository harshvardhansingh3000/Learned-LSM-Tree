#include "bench/benchmark.h"
#include "tree/lsm_tree.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <filesystem>

namespace lsm {
namespace bench {

// ─── Workload Factory Functions ────────────────────────────

Workload Workload::point_query_random(size_t ops) {
    return {
        WorkloadType::POINT_QUERY_RANDOM,
        ops,
        90,  // 90% reads
        "Point Query (Random)"
    };
}

Workload Workload::point_query_sequential(size_t ops) {
    return {
        WorkloadType::POINT_QUERY_SEQUENTIAL,
        ops,
        90,  // 90% reads
        "Point Query (Sequential)"
    };
}

Workload Workload::write_heavy(size_t ops) {
    return {
        WorkloadType::WRITE_HEAVY,
        ops,
        10,  // 10% reads
        "Write Heavy"
    };
}

Workload Workload::range_query(size_t ops) {
    return {
        WorkloadType::RANGE_QUERY,
        ops,
        5,  // 5% reads (range)
        "Range Query"
    };
}

Workload Workload::temporal_skewed(size_t ops) {
    return {
        WorkloadType::TEMPORAL_SKEWED,
        ops,
        90,  // 90% reads
        "Temporal Skewed"
    };
}

// ─── Benchmark Harness Implementation ──────────────────────

BenchmarkHarness::BenchmarkHarness() {}

void BenchmarkHarness::load_data(System* db, size_t num_keys) {
    std::cout << "  Loading " << num_keys << " keys..." << std::flush;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    current_num_keys_ = num_keys;
    for (size_t i = 0; i < num_keys; i++) {
        std::string key = generate_key(i, WorkloadType::POINT_QUERY_SEQUENTIAL);
        std::string value = generate_value(i);
        db->put(key, value);
    }
    
    // Force flush to ensure data is on disk
    db->flush();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << " done (" << ms << " ms)\n";
}

void BenchmarkHarness::warmup(System* db, const Workload& workload, size_t warmup_ops) {
    for (size_t i = 0; i < warmup_ops; i++) {
        size_t key_idx = random_key_index(current_num_keys_, workload.type, i);
        std::string key = generate_key(key_idx, workload.type);
        
        if (i % 100 < workload.read_ratio) {
            db->get(key);
        } else {
            db->put(key, generate_value(key_idx));
        }
    }
}

WorkloadMetrics BenchmarkHarness::run_benchmark(
    System* db,
    const std::string& system_name,
    const Workload& workload,
    bool use_classifier
) {
    std::cout << "  Running " << workload.name << "..." << std::flush;
    
    std::vector<uint64_t> latencies;
    latencies.reserve(workload.num_operations);
    
    int false_positives = 0;
    int false_negatives = 0;
    int total_bloom_checks = 0;
    int total_bloom_skips = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < workload.num_operations; i++) {
        size_t key_idx = random_key_index(current_num_keys_, workload.type, i);
        std::string key = generate_key(key_idx, workload.type);
        
        bool is_read = (i % 100) < workload.read_ratio;
        
        auto op_start = std::chrono::high_resolution_clock::now();
        
        if (is_read) {
            if (use_classifier) {
                if (auto* lsm = dynamic_cast<LSMTree*>(db)) {
                    auto predictions = level_classifier_.predict_levels(key);
                    auto result = lsm->get(key, predictions);
                    for (size_t level = 0; level < predictions.size(); level++) {
                        total_bloom_checks++;
                        if (!predictions[level]) {
                            total_bloom_skips++;
                        }
                    }
                    (void)result;
                } else {
                    db->get(key);
                }
            } else {
                db->get(key);
            }
        } else {
            std::string value = generate_value(key_idx);
            db->put(key, value);
        }
        
        auto op_end = std::chrono::high_resolution_clock::now();
        uint64_t op_us = std::chrono::duration_cast<std::chrono::microseconds>(op_end - op_start).count();
        latencies.push_back(op_us);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double median = (latencies.size() % 2 == 0)
        ? (latencies[latencies.size() / 2 - 1] + latencies[latencies.size() / 2]) / 2.0
        : latencies[latencies.size() / 2];
    
    double p95 = calculate_percentile(latencies, 0.95);
    double p99 = calculate_percentile(latencies, 0.99);
    
    double throughput = (workload.num_operations * 1e6) / total_us;

    if (use_classifier && !classifier_loaded_) {
        std::cerr << "Warning: classifier enabled but models not loaded; falling back to traditional reads.\n";
    }
    
    std::cout << " done (avg: " << static_cast<int>(mean) << " µs, p99: " 
              << static_cast<int>(p99) << " µs)\n";
    
    // Calculate Bloom filter memory usage for this system
    // Traditional BF: 10 bits/key * num_keys * num_levels (levels with data)
    // Classifier: model files loaded (~96 KB total for 3 levels) + traditional BF still present
    // LearnedBF: model files + tiny backup BFs (much smaller than traditional)
    uint64_t bf_memory = 0;
    size_t num_keys = current_num_keys_;
    int num_levels = 3;
    
    if (system_name.find("LearnedBF") != std::string::npos) {
        // Learned BF: sum of model sizes + backup BF sizes
        for (const auto& lbf : learned_bloom_filters_) {
            if (lbf) {
                bf_memory += lbf->total_size_bytes();
            }
        }
    } else if (system_name.find("Classifier") != std::string::npos) {
        // Classifier approach: traditional BF still used + classifier model overhead
        // Traditional BF at each level + ~96 KB for classifier models
        for (int i = 0; i < num_levels; i++) {
            size_t level_keys = num_keys;  // approximate
            bf_memory += (level_keys * 10) / 8;  // 10 bits per key, convert to bytes
        }
        bf_memory += 96 * 1024;  // ~96 KB for 3 level classifier models
    } else if (system_name.find("LSM") != std::string::npos) {
        // Traditional LSM: Bloom filters at each level
        for (int i = 0; i < num_levels; i++) {
            size_t level_keys = num_keys;  // approximate
            bf_memory += (level_keys * 10) / 8;  // 10 bits per key
        }
    }
    // B+ Tree: no bloom filters, memory = 0
    
    return WorkloadMetrics{
        system_name,
        workload.name,
        workload.num_operations,
        static_cast<double>(latencies.front()),
        static_cast<double>(latencies.back()),
        mean,
        median,
        p95,
        p99,
        throughput,
        false_positives,
        false_negatives,
        false_positives > 0 ? (100.0 * false_positives / workload.num_operations) : 0.0,
        false_negatives > 0 ? (100.0 * false_negatives / workload.num_operations) : 0.0,
        bf_memory,
        total_bloom_checks,
        total_bloom_skips,
        total_bloom_checks > 0 ? (100.0 * total_bloom_skips / total_bloom_checks) : 0.0
    };
}

void BenchmarkHarness::print_results(const std::vector<WorkloadMetrics>& results) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Benchmark Results Summary                        ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════╣\n";
    
    // Group by workload
    std::map<std::string, std::vector<const WorkloadMetrics*>> by_workload;
    for (const auto& result : results) {
        by_workload[result.workload_name].push_back(&result);
    }
    
    for (const auto& [workload, metrics_list] : by_workload) {
        std::cout << "║ " << std::setw(66) << std::left << workload << "║\n";
        std::cout << "├─────────────────────┬─────────────────┬──────────────┬──────────────┤\n";
        std::cout << "│ System              │ Latency (µs)    │ Throughput   │ BF Mem (KB)  │\n";
        std::cout << "├─────────────────────┼─────────────────┼──────────────┼──────────────┤\n";
        
        for (const auto* metrics : metrics_list) {
            std::cout << "│ " << std::setw(19) << std::left << metrics->system_name
                      << "│ " << std::setw(15) << std::right 
                      << std::fixed << std::setprecision(0) << metrics->latency_mean
                      << "│ " << std::setw(12) << std::right
                      << std::fixed << std::setprecision(0) << metrics->throughput_ops_per_sec
                      << "│ " << std::setw(12) << std::right
                      << std::fixed << std::setprecision(1) << (metrics->memory_used_bytes / 1024.0)
                      << "│\n";
        }
        std::cout << "└─────────────────────┴─────────────────┴──────────────┴─────────────┘\n\n";
    }
    
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
}

void BenchmarkHarness::export_csv(const std::vector<WorkloadMetrics>& results, const std::string& filename) {
    std::ofstream csv(filename);
    
    csv << "System,Workload,Operations,"
        << "Latency_Mean_us,Latency_Median_us,Throughput_ops_sec,"
        << "BloomFilter_Memory_KB,BloomChecks,BloomSkips,BypassRate_pct\n";
    
    for (const auto& r : results) {
        csv << r.system_name << ","
            << r.workload_name << ","
            << r.num_operations << ","
            << std::fixed << std::setprecision(2)
            << r.latency_mean << ","
            << r.latency_median << ","
            << r.throughput_ops_per_sec << ","
            << std::setprecision(2) << (r.memory_used_bytes / 1024.0) << ","
            << r.total_bloom_checks << ","
            << r.total_bloom_skips << ","
            << r.bypass_rate << "\n";
    }
    
    csv.close();
    std::cout << "Results exported to " << filename << "\n";
}

std::string BenchmarkHarness::generate_key(size_t index, WorkloadType /*type*/) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "key_%08zu", index);
    return std::string(buffer);
}

std::string BenchmarkHarness::generate_value(size_t index) {
    std::string value = "value_" + std::to_string(index) + "_";
    // Pad to ~100 bytes like in the paper
    value.append(80, 'x');
    return value;
}

size_t BenchmarkHarness::random_key_index(size_t num_keys, [[maybe_unused]] WorkloadType type, size_t op_index) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (num_keys == 0) return 0;

    switch (type) {
        case WorkloadType::POINT_QUERY_SEQUENTIAL:
            return op_index % num_keys;

        case WorkloadType::TEMPORAL_SKEWED: {
            size_t hot_size = std::max<size_t>(1, num_keys / 5);
            std::uniform_int_distribution<> hot_dist(0, static_cast<int>(hot_size - 1));
            std::uniform_int_distribution<> cold_dist(0, static_cast<int>(num_keys - hot_size - 1));
            std::uniform_int_distribution<> choice(0, 99);
            if (choice(gen) < 80) {
                return num_keys - hot_size + hot_dist(gen);
            }
            return cold_dist(gen);
        }

        case WorkloadType::POINT_QUERY_RANDOM:
        case WorkloadType::WRITE_HEAVY:
        case WorkloadType::RANGE_QUERY:
        default: {
            std::uniform_int_distribution<> dis(0, static_cast<int>(num_keys - 1));
            return dis(gen);
        }
    }
}

void BenchmarkHarness::load_classifier(const std::string& model_dir, int num_levels) {
    classifier_loaded_ = level_classifier_.load(model_dir, num_levels);
    if (!classifier_loaded_) {
        std::cerr << "Warning: failed to load classifier models from " << model_dir << "\n";
    }
}

void BenchmarkHarness::load_learned_bloom_filters(const std::string& model_dir, int num_levels) {
    learned_bloom_filters_.clear();
    learned_bf_loaded_ = false;
    
    for (int level = 0; level < num_levels; level++) {
        std::string path = model_dir + "/level_" + std::to_string(level) + "_classifier.json";
        auto lbf = std::make_unique<LearnedBloomFilter>();
        
        if (lbf->load_model(path)) {
            // Initialize backup BF — assume ~5% FN rate, sized for current dataset
            size_t expected_keys = current_num_keys_ > 0 ? current_num_keys_ : 10000;
            lbf->init_backup(expected_keys, 0.05);
            
            // Add all known keys to build the backup filter
            for (size_t i = 0; i < (current_num_keys_ > 0 ? current_num_keys_ : 10000); i++) {
                std::string key = generate_key(i, WorkloadType::POINT_QUERY_SEQUENTIAL);
                lbf->add(key);
            }
            
            std::cout << "  Learned BF Level " << level << ": " << lbf->classifier_type()
                      << " (FN caught: " << lbf->false_negatives_caught()
                      << "/" << lbf->total_keys_added()
                      << ", backup: " << lbf->backup_size_bytes() << " bytes"
                      << ", model: " << lbf->model_size_bytes() << " bytes)\n";
            
            learned_bloom_filters_.push_back(std::move(lbf));
        } else {
            learned_bloom_filters_.push_back(nullptr);
        }
    }
    
    learned_bf_loaded_ = !learned_bloom_filters_.empty();
}

double BenchmarkHarness::calculate_percentile(const std::vector<uint64_t>& latencies, double percentile) {
    if (latencies.empty()) return 0.0;
    
    size_t index = static_cast<size_t>(latencies.size() * percentile);
    if (index >= latencies.size()) index = latencies.size() - 1;
    
    return latencies[index];
}

// ─── System Comparator Implementation ──────────────────────

void SystemComparator::add_result(const WorkloadMetrics& metrics) {
    results_by_system[metrics.system_name].push_back(metrics);
}

void SystemComparator::print_comparison_table() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    System Comparison Table                            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    
    for (const auto& [system_name, metrics_vec] : results_by_system) {
        double avg_latency = 0, avg_throughput = 0;
        for (const auto& m : metrics_vec) {
            avg_latency += m.latency_mean;
            avg_throughput += m.throughput_ops_per_sec;
        }
        avg_latency /= metrics_vec.size();
        avg_throughput /= metrics_vec.size();
        
        std::cout << "║ " << std::setw(20) << std::left << system_name
                  << " │ Avg Latency: " << std::setw(10) << std::right
                  << std::fixed << std::setprecision(0) << avg_latency << " µs"
                  << " │ Throughput: " << std::setw(10) << std::right
                  << std::fixed << std::setprecision(0) << avg_throughput << " ops/s │\n";
    }
    
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n\n";
}

void SystemComparator::generate_comparison_chart() {
    std::cout << "\nLatency Comparison (lower is better):\n";
    
    std::map<std::string, double> avg_latencies;
    
    for (const auto& [system_name, metrics_vec] : results_by_system) {
        double avg = 0;
        for (const auto& m : metrics_vec) {
            avg += m.latency_mean;
        }
        avg_latencies[system_name] = avg / metrics_vec.size();
    }
    
    // Find baseline (LSM)
    double baseline = 0;
    if (avg_latencies.count("LSM")) {
        baseline = avg_latencies["LSM"];
    }
    
    for (const auto& [system, latency] : avg_latencies) {
        double speedup = baseline > 0 ? baseline / latency : 1.0;
        int bars = static_cast<int>(speedup * 20);
        
        std::cout << std::setw(20) << std::left << system << " │";
        for (int i = 0; i < bars; ++i) std::cout << "█";
        std::cout << " " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    std::cout << "\n";
}

} // namespace bench
} // namespace lsm
