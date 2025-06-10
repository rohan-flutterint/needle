package needle

import (
	"math/rand"
	"sync"
	"sync/atomic"
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

func TestAddBatch(t *testing.T) {
	g := NewGraph(2, 5, 10, 10, 100, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	// Test batch add
	items := make([]struct {
		ID  int
		Vec []float64
	}, 10)

	for i := 0; i < 10; i++ {
		items[i] = struct {
			ID  int
			Vec []float64
		}{
			ID:  i,
			Vec: []float64{float64(i), float64(i)},
		}
	}

	err := g.AddBatch(items)
	if err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	if g.Len() != 10 {
		t.Errorf("expected 10 nodes, got %d", g.Len())
	}

	// Test search after batch add
	query := []float64{1.1, 1.1}
	results, err := g.Search(query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
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

func TestHighVolume(t *testing.T) {
	// Test parameters
	dim := 128
	numPoints := 10000
	m := 16
	efConstruction := 200
	efSearch := 100
	k := 10

	// Create graph
	g := NewGraph(dim, m, efConstruction, efSearch, 1000, memory.DefaultAllocator)

	// Generate random points
	points := make([][]float64, numPoints)
	for i := range points {
		points[i] = make([]float64, dim)
		for j := range points[i] {
			points[i][j] = rand.Float64()
		}
	}

	// Add points
	for i, p := range points {
		if err := g.Add(i, p); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Verify graph properties
	if g.Len() != numPoints {
		t.Errorf("expected %d points, got %d", numPoints, g.Len())
	}

	// Test search accuracy
	for i := 0; i < 100; i++ { // Run 100 random queries
		// Generate query point
		query := make([]float64, dim)
		for j := range query {
			query[j] = rand.Float64()
		}

		// Get approximate nearest neighbors
		results, err := g.Search(query, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Verify result count
		if len(results) != k {
			t.Errorf("expected %d results, got %d", k, len(results))
		}

		// Verify results are unique
		seen := make(map[int]bool)
		for _, r := range results {
			if seen[r] {
				t.Errorf("duplicate result found: %d", r)
			}
			seen[r] = true
		}

		// Verify approximate ordering (should be roughly sorted by distance)
		dists := make([]float64, len(results))
		for i, r := range results {
			idx := g.idToIdx[r]
			dists[i] = euclideanSquared(query, g.getVectorFast(idx))
		}
		for i := 1; i < len(dists); i++ {
			if dists[i] < dists[i-1]*0.5 { // Allow some approximation
				t.Errorf("results not approximately sorted: %v", dists)
				break
			}
		}
	}
}

func TestEdgeCases(t *testing.T) {
	g := NewGraph(2, 5, 10, 10, 100, memory.DefaultAllocator)

	// Test empty graph
	results, err := g.Search([]float64{0, 0}, 1)
	if err != nil {
		t.Fatalf("Search on empty graph failed: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results from empty graph, got %d", len(results))
	}

	// Test single point
	err = g.Add(1, []float64{1, 1})
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	results, err = g.Search([]float64{0, 0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}

	// Test requesting more neighbors than points
	results, err = g.Search([]float64{0, 0}, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result when requesting too many, got %d", len(results))
	}

	// Test dimension mismatch
	err = g.Add(2, []float64{1, 1, 1})
	if err == nil {
		t.Error("expected error for dimension mismatch, got nil")
	}
	_, err = g.Search([]float64{1}, 1)
	if err == nil {
		t.Error("expected error for dimension mismatch in search, got nil")
	}
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

func BenchmarkAddBatch(b *testing.B) {
	g := NewGraph(128, 16, 200, 100, 1000, memory.DefaultAllocator)
	g.levelFunc = func() int { return 0 }

	// Pre-generate batch items
	batchSize := 100
	batches := make([][]struct {
		ID  int
		Vec []float64
	}, (b.N+batchSize-1)/batchSize)

	for i := range batches {
		batch := make([]struct {
			ID  int
			Vec []float64
		}, batchSize)
		for j := range batch {
			vec := make([]float64, 128)
			for k := range vec {
				vec[k] = rand.Float64()
			}
			batch[j] = struct {
				ID  int
				Vec []float64
			}{
				ID:  i*batchSize + j,
				Vec: vec,
			}
		}
		batches[i] = batch
	}

	b.ResetTimer()
	batchIdx := 0
	for i := 0; i < b.N; i += batchSize {
		if batchIdx >= len(batches) {
			break
		}
		if err := g.AddBatch(batches[batchIdx]); err != nil {
			b.Fatal(err)
		}
		batchIdx++
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

	g := NewGraph(dim, m, efConstruction, efSearch, chunkSize, memory.DefaultAllocator)

	// Pre-generate random vectors
	vectors := make([][]float64, b.N)
	for i := range vectors {
		vectors[i] = make([]float64, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float64()
		}
	}

	// Use atomic counter for IDs
	var counter int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		// Each goroutine gets its own slice of vectors
		for pb.Next() {
			id := atomic.AddInt64(&counter, 1) - 1
			if int(id) >= len(vectors) {
				continue
			}
			err := g.Add(int(id), vectors[id])
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkConcurrentSearch(b *testing.B) {
	dim := 128
	m := 16
	efConstruction := 200
	efSearch := 100
	chunkSize := 1000
	numPoints := 10000

	g := NewGraph(dim, m, efConstruction, efSearch, chunkSize, memory.DefaultAllocator)

	// Add points first
	for i := 0; i < numPoints; i++ {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		if err := g.Add(i, vec); err != nil {
			b.Fatal(err)
		}
	}

	// Pre-generate queries
	queries := make([][]float64, b.N)
	for i := range queries {
		queries[i] = make([]float64, dim)
		for j := range queries[i] {
			queries[i][j] = rand.Float64()
		}
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i >= len(queries) {
				continue
			}
			_, err := g.Search(queries[i], 10)
			if err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
}

func BenchmarkHighVolume(b *testing.B) {
	// Benchmark parameters
	dim := 128
	numPoints := 100000
	m := 16
	efConstruction := 200
	efSearch := 100

	// Create graph
	g := NewGraph(dim, m, efConstruction, efSearch, 1000, memory.DefaultAllocator)

	// Generate random points
	points := make([][]float64, numPoints)
	for i := range points {
		points[i] = make([]float64, dim)
		for j := range points[i] {
			points[i][j] = rand.Float64()
		}
	}

	// Add points (not counted in benchmark)
	for i, p := range points {
		if err := g.Add(i, p); err != nil {
			b.Fatal(err)
		}
	}

	// Generate query points
	queries := make([][]float64, b.N)
	for i := range queries {
		queries[i] = make([]float64, dim)
		for j := range queries[i] {
			queries[i][j] = rand.Float64()
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := g.Search(queries[i], 10)
		if err != nil {
			b.Fatal(err)
		}
	}
}
