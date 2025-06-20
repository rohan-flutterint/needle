#!/usr/bin/env python3
"""Benchmark hnswlib on a synthetic dataset.

This script mirrors the Go benchmark in `needle_benchmark.go`. It generates
random vectors using the same seed, builds an hnswlib index and measures
performance and recall.
"""

import argparse
import numpy as np
import time

try:
    import hnswlib
except ImportError:
    hnswlib = None  # The environment may not have hnswlib installed.


# Generate reproducible synthetic data
def generate_data(n, dim):
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


def euclidean_squared(a, b):
    diff = a - b
    return np.sum(diff * diff)


def compute_ground_truth(base, queries, k):
    gt = []
    for q in queries:
        dists = np.sum((base - q) ** 2, axis=1)
        top = np.argsort(dists)[:k]
        gt.append(top)
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--efC", type=int, default=200)
    parser.add_argument("--efS", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if hnswlib is None:
        raise SystemExit("hnswlib not installed")

    base = generate_data(args.n, args.dim)
    queries = generate_data(args.queries, args.dim)
    gt = compute_ground_truth(base, queries, args.k)

    index = hnswlib.Index(space="l2", dim=args.dim)

    start = time.time()
    index.init_index(max_elements=args.n, ef_construction=args.efC, M=args.m)
    index.add_items(base, np.arange(args.n))
    index.set_ef(args.efS)
    build_time = time.time() - start

    latencies = []
    hits = 0
    start = time.time()
    for qi, q in enumerate(queries):
        t0 = time.time()
        labels, _ = index.knn_query(q, k=args.k)
        latencies.append(time.time() - t0)
        hits += len(set(labels[0]).intersection(set(gt[qi])))
    total = time.time() - start

    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    recall = hits / float(len(queries) * args.k)
    qps = len(queries) / total

    print(f"build_time_ms {build_time*1000:.2f}")
    print(f"avg_latency_ms {avg*1000:.2f}")
    print(f"p95_latency_ms {p95*1000:.2f}")
    print(f"qps {qps:.2f}")
    print(f"recall_at_{args.k} {recall:.4f}")


if __name__ == "__main__":
    main()
