package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"time"

	"github.com/TFMV/needle"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// generateData generates n vectors of dimension dim using a fixed seed for reproducibility.
func generateData(n, dim int) [][]float64 {
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, n)
	for i := range data {
		vec := make([]float64, dim)
		for j := range vec {
			vec[j] = rng.Float64()
		}
		data[i] = vec
	}
	return data
}

// computeGroundTruth performs exhaustive search to find the top k neighbors for each query.
func computeGroundTruth(base, queries [][]float64, k int) [][]int {
	result := make([][]int, len(queries))
	for qi, q := range queries {
		dists := make([]struct {
			idx  int
			dist float64
		}, len(base))
		for i, v := range base {
			dists[i].idx = i
			dists[i].dist = euclideanSquared(q, v)
		}
		sort.Slice(dists, func(i, j int) bool { return dists[i].dist < dists[j].dist })
		top := make([]int, k)
		for j := 0; j < k; j++ {
			top[j] = dists[j].idx
		}
		result[qi] = top
	}
	return result
}

func euclideanSquared(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func main() {
	var (
		dim       = flag.Int("dim", 128, "vector dimension")
		n         = flag.Int("n", 10000, "dataset size")
		q         = flag.Int("queries", 100, "number of queries")
		m         = flag.Int("m", 16, "HNSW M parameter")
		efC       = flag.Int("efC", 200, "efConstruction")
		efS       = flag.Int("efS", 100, "efSearch")
		chunkSize = flag.Int("chunk", 1000, "chunk size")
		k         = flag.Int("k", 10, "neighbors per query")
	)
	flag.Parse()

	base := generateData(*n, *dim)
	queries := generateData(*q, *dim)

	gt := computeGroundTruth(base, queries, *k)

	// build index
	start := time.Now()
	g := needle.NewGraph(*dim, *m, *efC, *efS, *chunkSize, memory.DefaultAllocator)
	items := make([]struct {
		ID  int
		Vec []float64
	}, len(base))
	for i, vec := range base {
		items[i].ID = i
		items[i].Vec = vec
	}
	if err := g.AddBatch(items); err != nil {
		panic(err)
	}
	buildTime := time.Since(start)

	// get memory usage after build
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	usedMB := mem.Alloc / (1024 * 1024)

	// run queries
	latencies := make([]time.Duration, len(queries))
	start = time.Now()
	for i, vec := range queries {
		t0 := time.Now()
		res, err := g.Search(vec, *k)
		if err != nil {
			panic(err)
		}
		latencies[i] = time.Since(t0)

		// compute recall for this query
		hit := 0
		gtSet := make(map[int]struct{}, *k)
		for _, idx := range gt[i] {
			gtSet[idx] = struct{}{}
		}
		for _, id := range res {
			if _, ok := gtSet[id]; ok {
				hit++
			}
		}
		gt[i] = []int{hit} // reuse slice to store hits
	}
	totalQueryTime := time.Since(start)

	// compute recall
	hits := 0
	for i := range gt {
		hits += gt[i][0]
	}
	recall := float64(hits) / float64(len(queries)*(*k))

	// average and p95 latency
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	sum := time.Duration(0)
	for _, l := range latencies {
		sum += l
	}
	avgLat := sum / time.Duration(len(latencies))
	p95 := latencies[int(float64(len(latencies))*0.95)]
	qps := float64(len(latencies)) / totalQueryTime.Seconds()

	fmt.Printf("build_time_ms %.2f\n", float64(buildTime.Microseconds())/1000)
	fmt.Printf("avg_latency_ms %.2f\n", float64(avgLat.Microseconds())/1000)
	fmt.Printf("p95_latency_ms %.2f\n", float64(p95.Microseconds())/1000)
	fmt.Printf("qps %.2f\n", qps)
	fmt.Printf("memory_mb %d\n", usedMB)
	fmt.Printf("recall_at_%d %.4f\n", *k, recall)
}
