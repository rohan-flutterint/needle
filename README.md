# Needle

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/needle)](https://goreportcard.com/report/github.com/TFMV/needle)
[![Go Reference](https://pkg.go.dev/badge/github.com/TFMV/needle.svg)](https://pkg.go.dev/github.com/TFMV/needle)

**High-Performance HNSW Vector Search in Go**

Needle is a high-performance implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search in Go. It uses Apache Arrow for efficient vector storage and provides a thread-safe API for concurrent operations.

## Features

- **Efficient Vector Storage**: Uses Apache Arrow for chunked vector storage with zero-copy operations
- **Thread-Safe**: Supports concurrent insertions and searches with mutex locks
- **Memory Efficient**: Zero-allocation search operations using object pools and optimized memory management
- **High Performance**: Optimized neighbor selection with partial sorting and efficient graph traversal
- **Batch Operations**: Efficient batch vector addition with minimal allocations
- **Configurable**: Adjustable parameters for search quality vs. speed trade-offs
- **Production Ready**: Comprehensive test suite and benchmarks

## Installation

```bash
go get github.com/TFMV/needle
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/apache/arrow-go/v18/arrow/memory"
    "github.com/TFMV/needle"
)

func main() {
    // Create a new HNSW index
    dim := 128          // vector dimension
    m := 16            // max connections per node
    efConstruction := 200 // construction search width
    efSearch := 100    // search width
    chunkSize := 1000  // vectors per chunk
    g := needle.NewGraph(dim, m, efConstruction, efSearch, chunkSize, memory.DefaultAllocator)

    // Add vectors
    vec1 := []float64{1.0, 2.0, 3.0} // your vector data
    g.Add(1, vec1)

    // Search for nearest neighbors
    query := []float64{1.1, 2.1, 3.1}
    results, err := g.Search(query, 10) // find 10 nearest neighbors
    if err != nil {
        panic(err)
    }
    fmt.Println("Nearest neighbors:", results)
}
```

## Parameters

- `dim`: Vector dimension
- `m`: Maximum number of connections per node (higher values = better recall, more memory)
- `efConstruction`: Search width during construction (higher values = better graph quality)
- `efSearch`: Search width during queries (higher values = better recall, slower)
- `chunkSize`: Number of vectors per Arrow chunk (affects memory usage)

## Performance Characteristics

- **Single-threaded Add**: ~110μs per vector
- **Single-threaded Search**: ~12μs per query
- **Batch Add**: ~2.8μs per vector (100 vectors/batch)
- **Concurrent Add**: ~191μs per vector
- **Concurrent Search**: ~28μs per query
- **High Volume Search**: ~116μs per query (with 100,000 vectors)

## Implementation Details

### Vector Storage

Vectors are stored in Apache Arrow chunks for efficient memory management and vectorized operations. The chunked storage provides:

- Zero-copy vector access
- Efficient memory usage with configurable chunk sizes
- Support for large datasets
- Optimized vector operations

### Graph Structure

The HNSW graph is implemented with:

- Hierarchical layers for fast approximate search
- Bidirectional connections for better search quality
- Thread-safe operations using mutex locks
- Object pools for zero-allocation search operations
- Optimized neighbor selection with partial sorting
- Efficient memory management for neighbor lists

### Search Algorithm

The search process is optimized for performance:

1. Starts at the top layer
2. Navigates down through layers using greedy search
3. Uses priority queues for efficient neighbor selection
4. Implements partial sorting for better performance
5. Leverages object pools to minimize allocations
6. Supports concurrent operations with thread safety

## Testing

The implementation includes:

- Unit tests for core functionality
- Edge case tests
- High-volume tests (10,000 vectors)
- Concurrent operation tests
- Benchmarks for various operations

Run tests with:

```bash
go test -v -bench=. -benchmem
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
