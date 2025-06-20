# Needle Benchmark Results

This document contains benchmark results comparing the built-in Go implementation
of the HNSW index with other popular libraries. Each benchmark uses synthetic
vectors generated with the same random seed to ensure reproducibility.

| Library    | Dataset        | Build Time | Avg Latency | P95 Latency | QPS  | Recall @10 |
|------------|----------------|-----------:|------------:|------------:|-----:|-----------:|
| needle     | synthetic (10k, dim=128) | TODO | TODO | TODO | TODO | TODO |
| hnswlib    | synthetic (10k, dim=128) | TODO | TODO | TODO | TODO | TODO |

`Build Time` measures the time to construct the index. `Avg Latency` and
`P95 Latency` measure query performance over 100 random queries. `QPS`
represents queries per second. `Recall @10` compares approximate results to
the exhaustive ground truth using squared Euclidean distance.

These numbers are placeholders until the benchmarks are executed in an
environment with the required dependencies.
