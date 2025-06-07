// Package hnsw provides a high-performance, low-allocation HNSW index using arrow-go/v18.
package hnsw

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// Node represents a point in the HNSW graph.
type Node struct {
	ID        int     // user-provided ID
	idx       int     // internal index in storage
	level     int     // maximum layer
	neighbors [][]int // neighbor indices per level
}

// NewNode creates a new node with properly initialized neighbor slices.
func NewNode(id, idx, level int) *Node {
	return &Node{
		ID:        id,
		idx:       idx,
		level:     level,
		neighbors: make([][]int, level+1), // Pre-allocate all levels
	}
}

// Graph is the main HNSW index structure.
type Graph struct {
	// HNSW parameters
	m              int
	efConstruction int
	efSearch       int

	// chunked arrow storage for vectors (flat float64 values)
	dim       int
	chunkSize int
	allocator memory.Allocator
	vectors   []*array.Float64      // stores flat float64 values: rows * dim
	builder   *array.Float64Builder // current chunk builder

	// graph structure
	maxLevel   int
	enterPoint *Node
	nodes      []*Node
	idToIdx    map[int]int
	levelFunc  func() int // function to determine node level

	// pools for zero-allocation search queues
	pqPool  sync.Pool // *minHeap
	resPool sync.Pool // *maxHeap

	mu sync.RWMutex
}

// NewGraph initializes an HNSW index.
func NewGraph(dim, m, efConstruction, efSearch, chunkSize int, alloc memory.Allocator) *Graph {
	maxEF := max(efSearch, efConstruction)
	g := &Graph{
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		dim:            dim,
		chunkSize:      chunkSize,
		allocator:      alloc,
		vectors:        make([]*array.Float64, 0),
		builder:        array.NewFloat64Builder(alloc),
		nodes:          make([]*Node, 0),
		idToIdx:        make(map[int]int),
		levelFunc:      randomLevel, // default to random level
	}
	// pools
	g.pqPool = sync.Pool{New: func() any {
		hs := make(minHeap, 0, maxEF)
		heap.Init(&hs)
		return &hs
	}}
	g.resPool = sync.Pool{New: func() any {
		hs := make(maxHeap, 0, maxEF)
		heap.Init(&hs)
		return &hs
	}}
	return g
}

// Add inserts a new point into the index.
func (g *Graph) Add(id int, vec []float64) error {
	if len(vec) != g.dim {
		return fmt.Errorf("vector dimension mismatch: got %d, want %d", len(vec), g.dim)
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	// append to builder & flush if needed
	g.builder.AppendValues(vec, nil)
	if g.builder.Len()/g.dim >= g.chunkSize {
		chunk := g.builder.NewArray().(*array.Float64)
		g.vectors = append(g.vectors, chunk)
		g.builder = array.NewFloat64Builder(g.allocator)
	}

	// prepare node
	pos := len(g.nodes)
	g.idToIdx[id] = pos
	n := NewNode(id, pos, g.levelFunc())
	if n.level > g.maxLevel {
		g.maxLevel = n.level
		g.enterPoint = n
	}

	// Add node to graph before any references
	g.nodes = append(g.nodes, n)

	// link into graph
	if g.enterPoint != nil && pos > 0 {
		cur := g.enterPoint
		// navigate down
		for lvl := g.maxLevel; lvl > n.level; lvl-- {
			cur = g.greedySearchLayer(vec, cur, lvl)
		}
		// connect layers
		for lvl := min(n.level, g.maxLevel); lvl >= 0; lvl-- {
			cand := g.searchLayer(vec, cur, lvl, g.efConstruction)
			nbrs := selectNeighbors(cand, g.m)

			// Add bidirectional connections
			for _, c := range nbrs {
				ni := c.idx
				if ni >= len(g.nodes) {
					continue
				}
				peer := g.nodes[ni]

				// Connect n -> peer
				n.neighbors = appendLevel(n.neighbors, lvl, ni)

				// Connect peer -> n
				peer.neighbors = appendLevel(peer.neighbors, lvl, n.idx)
			}

			// Update current node for next level
			cur = g.greedySearchLayer(vec, cur, lvl)
		}
	}
	return nil
}

// Search finds the k nearest neighbors to query.
func (g *Graph) Search(query []float64, k int) ([]int, error) {
	if len(query) != g.dim {
		return nil, fmt.Errorf("query dimension mismatch: got %d, want %d", len(query), g.dim)
	}
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.nodes) == 0 {
		return nil, nil
	}

	// flush builder
	if g.builder.Len() > 0 {
		chunk := g.builder.NewArray().(*array.Float64)
		g.vectors = append(g.vectors, chunk)
		g.builder = array.NewFloat64Builder(g.allocator)
	}

	// descent
	ep := g.enterPoint
	if ep == nil {
		// If no enter point, use the first node
		ep = g.nodes[0]
	}
	for lvl := g.maxLevel; lvl > 0; lvl-- {
		ep = g.greedySearchLayer(query, ep, lvl)
	}
	// ef search
	cands := g.searchLayer(query, ep, 0, g.efSearch)
	if len(cands) == 0 {
		// If no candidates found, return the closest node
		closest := g.nodes[0]
		minDist := euclidean(query, g.getVector(0))
		for i := 1; i < len(g.nodes); i++ {
			dist := euclidean(query, g.getVector(i))
			if dist < minDist {
				minDist = dist
				closest = g.nodes[i]
			}
		}
		return []int{closest.ID}, nil
	}
	top := selectNeighbors(cands, k)
	if len(top) == 0 {
		return nil, nil
	}
	out := make([]int, len(top))
	for i, c := range top {
		out[i] = g.nodes[c.idx].ID
	}
	return out, nil
}

// appendLevel ensures neighbors slice has length lvl+1, then appends idx.
func appendLevel(neigh [][]int, lvl, idx int) [][]int {
	if len(neigh) <= lvl {
		n := make([][]int, lvl+1)
		copy(n, neigh)
		neigh = n
	}
	if neigh[lvl] == nil {
		neigh[lvl] = make([]int, 0)
	}
	neigh[lvl] = append(neigh[lvl], idx)
	return neigh
}

// greedySearchLayer performs a greedy walk.
func (g *Graph) greedySearchLayer(vec []float64, entry *Node, lvl int) *Node {
	if entry == nil {
		return nil
	}

	cur := entry
	dMin := euclidean(vec, g.getVector(cur.idx))
	improved := true
	for improved {
		improved = false
		if len(cur.neighbors) <= lvl {
			return cur
		}
		for _, ni := range cur.neighbors[lvl] {
			if ni >= len(g.nodes) {
				continue
			}
			d := euclidean(vec, g.getVector(ni))
			if d < dMin {
				dMin = d
				cur = g.nodes[ni]
				improved = true
			}
		}
	}
	return cur
}

// searchLayer performs EF search using pooled heaps.
func (g *Graph) searchLayer(query []float64, entry *Node, lvl, ef int) []*candidate {
	if entry == nil {
		return nil
	}

	visited := make(map[int]struct{})

	pqPtr := g.pqPool.Get().(*minHeap)
	*pqPtr = (*pqPtr)[:0]
	heap.Init(pqPtr)
	resPtr := g.resPool.Get().(*maxHeap)
	*resPtr = (*resPtr)[:0]
	heap.Init(resPtr)

	d0 := euclidean(query, g.getVector(entry.idx))
	heap.Push(pqPtr, &candidate{idx: entry.idx, dist: d0})
	heap.Push(resPtr, &candidate{idx: entry.idx, dist: d0})
	visited[entry.idx] = struct{}{}

	for pqPtr.Len() > 0 {
		c := heap.Pop(pqPtr).(*candidate)
		if resPtr.Len() > 0 && c.dist > (*resPtr)[0].dist {
			break
		}

		if c.idx >= len(g.nodes) {
			continue
		}
		node := g.nodes[c.idx]
		if node == nil {
			continue
		}

		nbrs := node.neighbors
		if len(nbrs) <= lvl {
			continue
		}

		for _, ni := range nbrs[lvl] {
			if ni >= len(g.nodes) {
				continue
			}
			if _, ok := visited[ni]; ok {
				continue
			}
			visited[ni] = struct{}{}
			d := euclidean(query, g.getVector(ni))
			if resPtr.Len() < ef {
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
			} else if d < (*resPtr)[0].dist {
				heap.Pop(resPtr)
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
			}
		}
	}

	// Convert results to slice
	out := make([]*candidate, resPtr.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(resPtr).(*candidate)
	}
	// return pools
	g.pqPool.Put(pqPtr)
	g.resPool.Put(resPtr)
	return out
}

// selectNeighbors picks top m by distance.
func selectNeighbors(cands []*candidate, m int) []*candidate {
	if len(cands) == 0 {
		return nil
	}
	if len(cands) <= m {
		return cands
	}
	// Sort by distance
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	// Return top m
	return cands[:m]
}

// candidate for search.
type candidate struct {
	idx  int
	dist float64
}

// minHeap for PQ.
type minHeap []*candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// maxHeap for results.
type maxHeap []*candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// getVector retrieves internal vector by idx.
func (g *Graph) getVector(idx int) []float64 {
	off := idx * g.dim
	out := make([]float64, g.dim)
	o := off
	for _, chunk := range g.vectors {
		n := chunk.Len()
		if o < n {
			d := min(n-o, g.dim)
			for i := range d {
				out[i] = chunk.Value(o + i)
			}
			return out
		}
		o -= n
	}
	return out
}

// Len returns the number of indexed points.
func (g *Graph) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes)
}

func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

// randomLevel samples a layer.
func randomLevel() int {
	lvl := 0
	for rand.Float64() < 1.0/math.E {
		lvl++
	}
	return lvl
}

// euclidean distance.
func euclidean(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
