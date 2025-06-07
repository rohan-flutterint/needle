package hnsw

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/memory"
)

func TestNewGraph(t *testing.T) {
	g := NewGraph(2, 2, 4, 4, 100, memory.DefaultAllocator)
	if g == nil {
		t.Fatal("NewGraph returned nil")
	}
	if g.dim != 2 {
		t.Errorf("expected dim=2, got %d", g.dim)
	}
	if g.m != 2 {
		t.Errorf("expected m=2, got %d", g.m)
	}
	if g.efConstruction != 4 {
		t.Errorf("expected efConstruction=4, got %d", g.efConstruction)
	}
	if g.efSearch != 4 {
		t.Errorf("expected efSearch=4, got %d", g.efSearch)
	}
	if g.chunkSize != 100 {
		t.Errorf("expected chunkSize=100, got %d", g.chunkSize)
	}
}

func TestAddAndSearch(t *testing.T) {
	// For small test graphs, set m to be at least the number of points
	points := [][]float64{
		{0, 0},
		{1, 1},
		{2, 2},
	}
	m := len(points) // Ensure m is at least the number of points
	g := NewGraph(2, m, 4, 4, 100, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	for i, p := range points {
		if err := g.Add(i, p); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Search for nearest neighbors
	query := []float64{0.1, 0.1}
	results, err := g.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	// First result should be point 0 (closest to query)
	if results[0] != 0 {
		t.Errorf("expected first result to be 0, got %d", results[0])
	}
}

func TestChunking(t *testing.T) {
	// For small test graphs, set m to be at least the number of points
	numPoints := 5
	m := numPoints                                        // Ensure m is at least the number of points
	g := NewGraph(2, m, 4, 4, 2, memory.DefaultAllocator) // small chunk size
	g.levelFunc = func() int { return 0 }

	// Add points to force chunking
	for i := 0; i < numPoints; i++ {
		if err := g.Add(i, []float64{float64(i), float64(i)}); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Verify chunks were created
	if len(g.vectors) < 2 {
		t.Errorf("expected at least 2 chunks, got %d", len(g.vectors))
	}
}

func TestConcurrentAccess(t *testing.T) {
	// For small test graphs, set m to be at least the number of points
	numPoints := 10
	m := numPoints // Ensure m is at least the number of points
	g := NewGraph(2, m, 4, 4, 100, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	// Add initial points
	for i := 0; i < numPoints; i++ {
		if err := g.Add(i, []float64{float64(i), float64(i)}); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Concurrent searches
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := []float64{float64(i), float64(i)}
			results, err := g.Search(query, 3)
			if err != nil {
				t.Errorf("Search failed: %v", err)
				return
			}
			if len(results) != 3 {
				t.Errorf("expected 3 results, got %d", len(results))
			}
		}()
	}
	wg.Wait()
}

func BenchmarkAdd(b *testing.B) {
	// For benchmarks, use standard HNSW parameters
	g := NewGraph(128, 16, 200, 100, 1000, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	vec := make([]float64, 128)
	for i := range vec {
		vec[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec[0] = rand.Float64() // change vector slightly
		if err := g.Add(i, vec); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSearch(b *testing.B) {
	// For benchmarks, use standard HNSW parameters
	g := NewGraph(128, 16, 200, 100, 1000, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	// Add some points first
	vec := make([]float64, 128)
	for i := 0; i < 1000; i++ {
		for j := range vec {
			vec[j] = rand.Float64()
		}
		if err := g.Add(i, vec); err != nil {
			b.Fatal(err)
		}
	}

	// Generate a query vector
	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query[0] = rand.Float64() // change query slightly
		if _, err := g.Search(query, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConcurrentAdd(b *testing.B) {
	dim := 128
	m := 16
	efConstruction := 200
	efSearch := 100
	chunkSize := 1000
	alloc := memory.NewGoAllocator()

	g := NewGraph(dim, m, efConstruction, efSearch, chunkSize, alloc)

	// Pre-generate random vectors
	vectors := make([][]float64, b.N)
	for i := range vectors {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		vectors[i] = vec
	}

	for i := 0; b.Loop(); i++ {
		err := g.Add(i, vectors[i])
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConcurrentSearch(b *testing.B) {
	dim := 128
	m := 16
	efConstruction := 200
	efSearch := 100
	chunkSize := 1000
	alloc := memory.NewGoAllocator()

	g := NewGraph(dim, m, efConstruction, efSearch, chunkSize, alloc)

	// Add some vectors first
	numVectors := 10000
	for i := 0; i < numVectors; i++ {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		err := g.Add(i, vec)
		if err != nil {
			b.Fatal(err)
		}
	}

	// Generate random query vectors
	queries := make([][]float64, b.N)
	for i := range queries {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		queries[i] = vec
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			_, err := g.Search(queries[i], 10)
			if err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
}
