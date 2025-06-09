# Needle Benchmarks

## Test Environment

- **OS**: macOS (darwin 24.5.0)
- **Architecture**: arm64
- **CPU**: Apple M2 Pro
- **Go Version**: 1.24.3

## Basic Operations

| Operation | Iterations | Time/Op | Allocs/Op | Bytes/Op |
|-----------|------------|---------|-----------|----------|
| Add       | 10,000     | 110.6µs | 1,068     | 375,302  |
| Search    | 109,537    | 12.6µs  | 79        | 24,686   |

## Batch Operations

| Operation | Iterations | Time/Op | Allocs/Op | Bytes/Op | Notes |
|-----------|------------|---------|-----------|----------|-------|
| AddBatch  | 10,000     | 282.4µs | 2,366     | 810,294  | 100 vectors/batch = 2.8µs/vector |

## Concurrent Operations

| Operation        | Iterations | Time/Op | Allocs/Op | Bytes/Op |
|------------------|------------|---------|-----------|----------|
| Concurrent Add   | 10,000     | 191.1µs | 1,783     | 653,444  |
| Concurrent Search| 39,724     | 28.4µs  | 356       | 164,986  |

## High Volume Performance

| Operation    | Iterations | Time/Op | Allocs/Op | Bytes/Op |
|--------------|------------|---------|-----------|----------|
| High Volume  | 10,000     | 109.9µs | 769       | 303,608  |
