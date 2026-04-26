# 🌲 Learned LSM-Tree Database

A high-performance **LSM-tree based key-value store** in C++ with **ML-enhanced Bloom filters**, inspired by the paper *"Learned LSM-trees: Two Approaches Using Learned Bloom Filters"* (Fidalgo & Ye, Harvard, 2025).

## Features

- **Complete LSM-Tree Engine** — MemTable (Skip List), WAL, SSTables, Leveled Compaction
- **Bloom Filters** — Integrated into SSTables with ~0.82% false positive rate
- **ML Classifier** — Gradient Boosted Trees predict which levels to skip during reads
- **Interactive CLI** — put/get/delete/scan/load/flush/compact/predict/train
- **71 Unit Tests** — Comprehensive test coverage with Google Test
- **Crash Recovery** — Write-Ahead Log with CRC32 checksums

## Quick Start

### 1. Prerequisites
```bash
# C++ build tools
brew install cmake    # macOS
# or: sudo apt install cmake g++   # Linux

# Python ML dependencies
pip3 install numpy scikit-learn mmh3
```

### 2. Build the Database
```bash
git clone <repo-url> LSM
cd LSM
mkdir build && cd build
cmake ..
cmake --build .
```

### 3. Run Tests (71 tests)
```bash
cd build
ctest --output-on-failure
```

### 4. Train the ML Classifier (do this FIRST before using predict)
```bash
cd LSM    # project root, NOT build/
python3 ml/train_classifier.py
```
This trains per-level GBT classifiers and saves models to `data/models/`.

### 5. Run the Database
```bash
cd build
./lsm_db ../data    # ../data points to LSM/data where models are stored
```

### 6. Try It Out
```
lsm> put name Alice
  OK (39 μs)
lsm> put age 25
  OK (7 μs)
lsm> get name
  Alice (3 μs)
lsm> delete age
  OK (49 μs)
lsm> load 5000
  Loaded 5000 entries in 45 ms (111111 ops/sec)
lsm> flush
  Flushed (4303 μs)
lsm> stats
  Levels: 2, SSTables: 1, Entries: 5001
lsm> predict key_000500
  ML Prediction: L0=SKIP L1=SKIP L2=CHECK (4 μs)
lsm> predict nonexistent
  ML Prediction: L0=SKIP L1=SKIP L2=SKIP → Key likely NOT in database (5 μs)
lsm> scan key_000000 key_000005
  key_000000 → value_0_xxx...
  key_000001 → value_1_xxx...
  ...
lsm> quit
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `put <key> <value>` | Insert or update a key-value pair |
| `get <key>` | Look up a key |
| `delete <key>` | Delete a key |
| `scan <start> <end>` | Range scan (inclusive) |
| `flush` | Force flush MemTable to disk |
| `compact` | Force compaction |
| `stats` | Show database statistics |
| `load <count>` | Bulk insert N key-value pairs |
| `predict <key>` | ML: predict which level contains the key |
| `train` | Train the ML classifier (runs Python) |
| `help` | Show all commands |
| `quit` | Exit |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client CLI / API                       │
├─────────────────────────────────────────────────────────┤
│                   LSM-Tree Engine                         │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────┐     │
│  │   WAL   │  │ MemTable │  │   Read Path Router   │     │
│  │(append) │→ │(SkipList)│  │ Traditional / ML     │     │
│  └─────────┘  └────┬─────┘  └──────────┬──────────┘     │
│                     │flush              │                 │
│              ┌──────▼──────┐            │                 │
│              │   Level 0   │ ◄──────────┤                 │
│              │ + Bloom/ML  │            │                 │
│              └──────┬──────┘            │                 │
│              ┌──────▼──────┐            │                 │
│              │   Level 1+  │ ◄──────────┘                 │
│              │ + Bloom/ML  │                              │
│              └─────────────┘                              │
├─────────────────────────────────────────────────────────┤
│              Compaction Engine (Background)               │
└─────────────────────────────────────────────────────────┘
```

## ML Classifier (Paper Approach 1)

### How It Works
1. **Python** trains per-level GBT classifiers using sklearn
2. Models are exported as JSON to `data/models/`
3. **C++** loads the JSON models and runs inference during GET
4. For each key, the classifier predicts which levels to CHECK or SKIP
5. Skipping levels avoids unnecessary Bloom filter checks → faster reads

### Training Results
```
Level 0 classifier: 93.3% accuracy
Level 1 classifier: 80.9% accuracy
Level 2 classifier: 81.2% accuracy
Non-existent keys:  100% correctly identified
```

## Project Structure

```
LSM/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/                    # Header files
│   ├── common/                 # Types, config, MurmurHash3
│   ├── memtable/               # Skip List + MemTable
│   ├── wal/                    # Write-Ahead Log
│   ├── sstable/                # SSTable format, writer, reader
│   ├── bloom/                  # Bloom filter
│   ├── tree/                   # LSM-tree + Level manager
│   └── ml/                     # ML feature engineering + classifier
├── src/                        # Implementation files
├── tests/                      # Google Test files (71 tests)
├── ml/                         # Python ML training scripts
├── docs/                       # Technical documentation
│   └── REPORT.md               # Detailed technical report
└── data/                       # Runtime data (gitignored)
    ├── wal/                    # WAL files
    ├── sstables/               # SSTable files
    └── models/                 # Trained ML models (JSON)
```

## Configuration (Monkey Paper)

| Parameter | Default | Description |
|-----------|---------|-------------|
| MemTable size | 1 MB | Flush threshold |
| Size ratio (T) | 10 | Level capacity multiplier |
| Bloom bits/key | 10 | ~0.82% false positive rate |
| Block size | 4 KB | SSTable data block size |

## Based On

- **Paper**: [Learned LSM-trees: Two Approaches Using Learned Bloom Filters](https://arxiv.org/abs/2508.00882) (Fidalgo & Ye, Harvard, 2025)
- **Monkey Paper**: Optimal Navigable Key-Value Store (Dayan & Idreos, 2018)
- **Inspired by**: LevelDB, RocksDB, Cassandra

## What's Remaining (for contributors)

- [ ] **Learned Bloom Filter** (Paper Approach 2) — Replace Bloom filters with ML models + backup filters for 70-80% memory savings
- [ ] **Benchmarking Suite** — Automated performance testing with 6 workload types (random, sequential, level-targeted)
- [ ] **Performance Analysis** — 3-way comparison: Traditional vs Classifier vs Learned BF
- [ ] **Docker Deployment** — Containerized deployment
- [ ] **Train on Real Data** — Use actual SSTable metadata for ML training instead of synthetic data

## License

MIT
