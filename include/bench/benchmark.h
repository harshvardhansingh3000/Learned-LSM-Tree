#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <cstdint>
#include <functional>
#include "tree/system.h"
#include "ml/classifier.h"

namespace lsm {
namespace bench {

// ─── Workload Types ────────────────────────────────────────
enum class WorkloadType {
    POINT_QUERY_RANDOM,      // 90% reads, 10% writes, random keys
    POINT_QUERY_SEQUENTIAL,  // 90% reads, 10% writes, sequential keys
    WRITE_HEAVY,             // 10% reads, 90% writes
    RANGE_QUERY,             // 5% reads (range), 95% writes
    TEMPORAL_SKEWED          // Hot recent data, cold old data
};

// ─── Measurement Results ───────────────────────────────────
struct OperationMetrics {
    uint64_t latency_us;           // Operation latency in microseconds
    bool found;                     // Whether key was found
    int bloom_checks;              // Number of Bloom filter checks
    int bloom_skips;               // Number of levels skipped
};

struct WorkloadMetrics {
    std::string system_name;
    std::string workload_name;
    size_t num_operations;
    
    // Latency statistics (microseconds)
    double latency_min;
    double latency_max;
    double latency_mean;
    double latency_median;
    double latency_p95;
    double latency_p99;
    
    // Throughput
    double throughput_ops_per_sec;
    
    // Accuracy metrics
    int false_positives;
    int false_negatives;
    double false_positive_rate;
    double false_negative_rate;
    
    // Memory metrics
    uint64_t memory_used_bytes;
    
    // Filter metrics
    int total_bloom_checks;
    int total_bloom_skips;
    double bypass_rate;
};

// ─── Workload Definition ───────────────────────────────────
struct Workload {
    WorkloadType type;
    size_t num_operations;
    size_t read_ratio;              // Percentage of reads (0-100)
    std::string name;
    
    static Workload point_query_random(size_t ops = 10000);
    static Workload point_query_sequential(size_t ops = 10000);
    static Workload write_heavy(size_t ops = 10000);
    static Workload range_query(size_t ops = 10000);
    static Workload temporal_skewed(size_t ops = 10000);
};

// ─── Benchmark Harness ─────────────────────────────────────
class BenchmarkHarness {
public:
    BenchmarkHarness();
    
    // Data loading
    void load_data(System* db, size_t num_keys);
    
    // Warmup phase (discard results)
    void warmup(System* db, const Workload& workload, size_t warmup_ops = 1000);
    
    // Main benchmark execution
    WorkloadMetrics run_benchmark(
        System* db,
        const std::string& system_name,
        const Workload& workload,
        bool use_classifier = false
    );

    void load_classifier(const std::string& model_dir, int num_levels = 3);
    
    // Analysis
    void print_results(const std::vector<WorkloadMetrics>& results);
    void export_csv(const std::vector<WorkloadMetrics>& results, const std::string& filename);
    
private:
    // Helper functions
    std::string generate_key(size_t index, WorkloadType type);
    std::string generate_value(size_t index);
    uint64_t measure_time(std::function<void()> fn);
    size_t random_key_index(size_t num_keys, WorkloadType type, size_t op_index);
    
    // Benchmark classifier support
    LevelClassifier level_classifier_;
    bool classifier_loaded_ = false;
    size_t current_num_keys_ = 0;
    
    // Statistics helpers
    double calculate_percentile(const std::vector<uint64_t>& latencies, double percentile);
};

// ─── System Comparator ─────────────────────────────────────
class SystemComparator {
public:
    void add_result(const WorkloadMetrics& metrics);
    void print_comparison_table();
    void generate_comparison_chart();
    
private:
    std::map<std::string, std::vector<WorkloadMetrics>> results_by_system;
};

} // namespace bench
} // namespace lsm
