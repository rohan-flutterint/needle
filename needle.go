// Package needle provides a high performance HNSW index in pure Arrow and Go.

package needle

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

// VisitedList represents a reusable visited tracking structure inspired by hnswlib
type VisitedList struct {
	curV uint16
	mass []uint16
}

// NewVisitedList creates a new visited list with the given capacity
func NewVisitedList(capacity int) *VisitedList {
	return &VisitedList{
		curV: 1,
		mass: make([]uint16, capacity),
	}
}

// Reset prepares the visited list for reuse
func (vl *VisitedList) Reset() {
	vl.curV++
	if vl.curV == 0 {
		// Overflow case - reset the entire array
		for i := range vl.mass {
			vl.mass[i] = 0
		}
		vl.curV = 1
	}
}

// IsVisited checks if an index has been visited
func (vl *VisitedList) IsVisited(idx int) bool {
	if idx >= len(vl.mass) {
		// Extend the slice if needed
		oldLen := len(vl.mass)
		newMass := make([]uint16, idx+1)
		copy(newMass, vl.mass)
		vl.mass = newMass
		// Initialize new elements to 0
		for i := oldLen; i < len(newMass); i++ {
			vl.mass[i] = 0
		}
	}
	return vl.mass[idx] == vl.curV
}

// Visit marks an index as visited
func (vl *VisitedList) Visit(idx int) {
	if idx >= len(vl.mass) {
		// Extend the slice if needed
		oldLen := len(vl.mass)
		newMass := make([]uint16, idx+1)
		copy(newMass, vl.mass)
		vl.mass = newMass
		// Initialize new elements to 0
		for i := oldLen; i < len(newMass); i++ {
			vl.mass[i] = 0
		}
	}
	vl.mass[idx] = vl.curV
}

// VisitedListPool manages a pool of VisitedList instances
type VisitedListPool struct {
	pool    []*VisitedList
	mutex   sync.Mutex
	maxSize int
}

// NewVisitedListPool creates a new visited list pool
func NewVisitedListPool(maxSize int) *VisitedListPool {
	pool := &VisitedListPool{
		pool:    make([]*VisitedList, 0, 4),
		maxSize: maxSize,
	}
	// Pre-allocate a few lists
	for i := 0; i < 2; i++ {
		pool.pool = append(pool.pool, NewVisitedList(maxSize))
	}
	return pool
}

// Get retrieves a visited list from the pool
func (vlp *VisitedListPool) Get() *VisitedList {
	vlp.mutex.Lock()
	defer vlp.mutex.Unlock()

	if len(vlp.pool) > 0 {
		vl := vlp.pool[len(vlp.pool)-1]
		vlp.pool = vlp.pool[:len(vlp.pool)-1]
		vl.Reset()
		return vl
	}

	// Create new if pool is empty
	return NewVisitedList(vlp.maxSize)
}

// Return returns a visited list to the pool
func (vlp *VisitedListPool) Return(vl *VisitedList) {
	vlp.mutex.Lock()
	defer vlp.mutex.Unlock()

	if len(vlp.pool) < 8 { // Limit pool size
		vlp.pool = append(vlp.pool, vl)
	}
}

// Node represents a point in the HNSW graph.
type Node struct {
	ID        int     // user-provided ID
	idx       int     // internal index in storage
	level     int     // maximum layer
	neighbors [][]int // neighbor indices per level
}

// NewNode creates a new node with properly initialized neighbor slices.
func NewNode(id, idx, level int) *Node {
	neighbors := make([][]int, level+1)
	// Pre-allocate neighbor slices to reduce allocations
	for i := range neighbors {
		// Use 2*m capacity for better connectivity
		neighbors[i] = make([]int, 0, 32)
	}
	return &Node{
		ID:        id,
		idx:       idx,
		level:     level,
		neighbors: neighbors,
	}
}

// Graph is the main HNSW index structure.
type Graph struct {
	// HNSW parameters
	m              int
	efConstruction int
	efSearch       int
	ml             float64 // level generation parameter

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

	// pools for zero-allocation operations
	pqPool      sync.Pool        // *minHeap
	resPool     sync.Pool        // *maxHeap
	vecPool     sync.Pool        // []float64 for getVector
	candPool    sync.Pool        // []*candidate for reuse
	intPool     sync.Pool        // []int for neighbor lists
	visitedPool *VisitedListPool // visited list pool

	mu sync.RWMutex
}

// NewGraph initializes an HNSW index.
func NewGraph(dim, m, efConstruction, efSearch, chunkSize int, alloc memory.Allocator) *Graph {
	// Ensure efConstruction is at least M, following hnswlib best practices
	if efConstruction < m {
		efConstruction = m
	}
	maxEF := max(efSearch, efConstruction)
	g := &Graph{
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		ml:             1.0 / math.Log(2.0), // Better level distribution
		dim:            dim,
		chunkSize:      chunkSize,
		allocator:      alloc,
		vectors:        make([]*array.Float64, 0),
		builder:        array.NewFloat64Builder(alloc),
		nodes:          make([]*Node, 0),
		idToIdx:        make(map[int]int),
		levelFunc:      nil,                       // Will be set below
		visitedPool:    NewVisitedListPool(50000), // Pre-allocate for large graphs
	}

	// Set level function with proper ml
	g.levelFunc = func() int {
		return g.randomLevel()
	}

	// Initialize pools for better memory management
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
	g.vecPool = sync.Pool{New: func() any {
		return make([]float64, dim)
	}}
	g.candPool = sync.Pool{New: func() any {
		return make([]*candidate, 0, maxEF)
	}}
	g.intPool = sync.Pool{New: func() any {
		return make([]int, 0, m*4)
	}}

	return g
}

// AddBatch inserts multiple points into the index efficiently.
func (g *Graph) AddBatch(items []struct {
	ID  int
	Vec []float64
}) error {
	if len(items) == 0 {
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Validate all vectors first
	for _, item := range items {
		if len(item.Vec) != g.dim {
			return fmt.Errorf("vector dimension mismatch: got %d, want %d", len(item.Vec), g.dim)
		}
	}

	// Batch append vectors to builder
	for _, item := range items {
		g.builder.AppendValues(item.Vec, nil)
		if g.builder.Len()/g.dim >= g.chunkSize {
			chunk := g.builder.NewArray().(*array.Float64)
			g.vectors = append(g.vectors, chunk)
			g.builder = array.NewFloat64Builder(g.allocator)
		}
	}

	// Process nodes in batch
	nodes := make([]*Node, len(items))
	startPos := len(g.nodes)

	for i, item := range items {
		pos := startPos + i
		g.idToIdx[item.ID] = pos
		nodes[i] = NewNode(item.ID, pos, g.levelFunc())

		// Update max level if needed
		if nodes[i].level > g.maxLevel {
			g.maxLevel = nodes[i].level
			g.enterPoint = nodes[i]
		}
	}

	// Add all nodes to graph first
	g.nodes = append(g.nodes, nodes...)

	// Connect nodes to graph
	for i, item := range items {
		n := nodes[i]
		pos := startPos + i

		// Skip first node
		if pos == 0 {
			g.enterPoint = n
			g.maxLevel = n.level
			continue
		}

		// Find enter point if none exists
		if g.enterPoint == nil {
			g.enterPoint = g.nodes[0]
			g.maxLevel = g.enterPoint.level
		}

		// Connect to existing graph
		g.connectNodeToGraph(n, item.Vec)
	}

	return nil
}

// connectNodeToGraph connects a node to the existing graph structure with enhanced hnswlib-style algorithm.
func (g *Graph) connectNodeToGraph(n *Node, vec []float64) {
	cur := g.enterPoint
	curDist := euclideanSquaredFast(vec, g.getVectorFast(cur.idx))

	// Enhanced navigation down through levels with better tracking
	for lvl := g.maxLevel; lvl > n.level; lvl-- {
		changed := true
		for changed {
			changed = false
			if lvl >= len(cur.neighbors) {
				break
			}

			for _, ni := range cur.neighbors[lvl] {
				if ni >= len(g.nodes) {
					continue
				}
				nbrVec := g.getVectorFast(ni)
				d := euclideanSquaredFast(vec, nbrVec)
				if d < curDist {
					curDist = d
					cur = g.nodes[ni]
					changed = true
					break
				}
			}
		}
	}

	// Connect at each level from top to bottom with enhanced algorithm
	for lvl := min(n.level, g.maxLevel); lvl >= 0; lvl-- {
		// Use higher ef for better connectivity, especially at level 0
		ef := g.efConstruction
		if lvl == 0 {
			ef = max(g.efConstruction, g.m*2)
		}

		candidates := g.searchLayerFast(vec, cur, lvl, ef)
		if len(candidates) == 0 {
			candidates = []*candidate{{idx: cur.idx, dist: curDist}}
		}

		// Enhanced neighbor selection
		mMax := g.m
		if lvl == 0 {
			mMax = g.m * 2 // Allow more connections at level 0
		}
		g.mutuallyConnectNewElement(n, candidates, lvl, mMax)

		// Update current node for next level - use closest candidate
		if len(candidates) > 0 {
			cur = g.nodes[candidates[0].idx]
			curDist = candidates[0].dist
		}
	}
}

// mutuallyConnectNewElement performs bidirectional connections with enhanced pruning
func (g *Graph) mutuallyConnectNewElement(newNode *Node, candidates []*candidate, level, mMax int) []*candidate {
	selected := g.selectNeighborsHeuristic(candidates, g.m)

	// Connect new node to selected neighbors
	if level < len(newNode.neighbors) {
		for _, c := range selected {
			newNode.neighbors[level] = append(newNode.neighbors[level], c.idx)
		}
	}

	// Bidirectionally connect and prune existing neighbors
	for _, c := range selected {
		ni := c.idx
		if ni >= len(g.nodes) {
			continue
		}
		peer := g.nodes[ni]

		if level < len(peer.neighbors) {
			// Add new connection
			peer.neighbors[level] = append(peer.neighbors[level], newNode.idx)

			// Enhanced pruning when over capacity
			if len(peer.neighbors[level]) > mMax {
				g.pruneNeighborsHeuristic(peer, level)
			}
		}
	}

	return selected
}

// pruneNeighborsHeuristic efficiently prunes neighbors using improved hnswlib-style heuristic.
func (g *Graph) pruneNeighborsHeuristic(node *Node, lvl int) {
	mMax := g.m
	if lvl == 0 {
		mMax = g.m * 2
	}

	if lvl >= len(node.neighbors) || len(node.neighbors[lvl]) <= mMax {
		return
	}

	// Get pooled candidate slice
	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	// Calculate distances to all neighbors
	nodeVec := g.getVectorFast(node.idx)
	for _, nbrIdx := range node.neighbors[lvl] {
		if nbrIdx >= len(g.nodes) {
			continue
		}
		nbrVec := g.getVectorFast(nbrIdx)
		d := euclideanSquaredFast(nodeVec, nbrVec)
		candidates = append(candidates, &candidate{idx: nbrIdx, dist: d})
	}

	// Use adaptive selection based on level and candidate count
	var selected []*candidate
	if lvl == 0 && len(candidates) > mMax*3 {
		// Use advanced heuristic only for level 0 with many candidates
		selected = g.selectNeighborsAdvancedHeuristic(candidates, mMax)
	} else {
		// Use simpler selection for better performance and recall balance
		selected = g.selectNeighborsHeuristic(candidates, mMax)
	}

	// Get pooled int slice for new neighbors
	newNbrs := g.intPool.Get().([]int)[:0]
	defer g.intPool.Put(newNbrs)

	for _, s := range selected {
		newNbrs = append(newNbrs, s.idx)
	}

	// Replace neighbors
	node.neighbors[lvl] = node.neighbors[lvl][:len(newNbrs)]
	copy(node.neighbors[lvl], newNbrs)
}

// selectNeighborsAdvancedHeuristic implements the full hnswlib neighbor selection heuristic
func (g *Graph) selectNeighborsAdvancedHeuristic(candidates []*candidate, m int) []*candidate {
	if len(candidates) <= m {
		return candidates
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	selected := make([]*candidate, 0, m)

	// Always include the closest candidate
	selected = append(selected, candidates[0])

	// For each remaining candidate, check diversity constraint
	for i := 1; i < len(candidates) && len(selected) < m; i++ {
		cand := candidates[i]
		candVec := g.getVectorFast(cand.idx)

		shouldAdd := true
		// Check against all selected neighbors (full hnswlib heuristic)
		for _, sel := range selected {
			selVec := g.getVectorFast(sel.idx)
			distToSel := euclideanSquaredFast(candVec, selVec)

			// If candidate is closer to any selected neighbor than to query, skip
			if distToSel < cand.dist {
				shouldAdd = false
				break
			}
		}

		if shouldAdd {
			selected = append(selected, cand)
		}
	}

	// If we didn't select enough diverse neighbors, fill with closest remaining
	if len(selected) < m {
		for i := 1; i < len(candidates) && len(selected) < m; i++ {
			found := false
			for _, sel := range selected {
				if sel.idx == candidates[i].idx {
					found = true
					break
				}
			}
			if !found {
				selected = append(selected, candidates[i])
			}
		}
	}

	return selected
}

// Add inserts a single point into the index.
func (g *Graph) Add(id int, vec []float64) error {
	return g.AddBatch([]struct {
		ID  int
		Vec []float64
	}{{ID: id, Vec: vec}})
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

	if k > len(g.nodes) {
		k = len(g.nodes)
	}

	// Flush builder if needed
	if g.builder.Len() > 0 {
		chunk := g.builder.NewArray().(*array.Float64)
		g.vectors = append(g.vectors, chunk)
		g.builder = array.NewFloat64Builder(g.allocator)
	}

	// For small graphs, do optimized exhaustive search
	if len(g.nodes) <= g.m {
		return g.exhaustiveSearch(query, k), nil
	}

	// Fast HNSW search
	return g.hnswSearch(query, k), nil
}

// exhaustiveSearch performs optimized exhaustive search for small graphs.
func (g *Graph) exhaustiveSearch(query []float64, k int) []int {
	candidates := g.candPool.Get().([]*candidate)[:0]
	defer g.candPool.Put(candidates)

	for i := range g.nodes {
		vec := g.getVectorFast(i)
		d := euclideanSquaredFast(query, vec)
		candidates = append(candidates, &candidate{idx: i, dist: d})
	}

	top := selectNeighborsFast(candidates, k)
	out := make([]int, len(top))
	for i, c := range top {
		out[i] = g.nodes[c.idx].ID
	}
	return out
}

// hnswSearch performs optimized HNSW search.
func (g *Graph) hnswSearch(query []float64, k int) []int {
	ep := g.enterPoint
	if ep == nil {
		ep = g.nodes[0]
	}

	// Descent through layers
	for lvl := g.maxLevel; lvl > 0; lvl-- {
		next := g.greedySearchLayerFast(query, ep, lvl)
		if next != nil {
			ep = next
		}
	}

	// Search at level 0 with higher ef for better recall
	ef := max(g.efSearch, k*2)
	candidates := g.searchLayerFast(query, ep, 0, ef)
	if len(candidates) == 0 {
		return g.exhaustiveSearch(query, k)
	}

	top := selectNeighborsFast(candidates, k)
	out := make([]int, len(top))
	for i, c := range top {
		out[i] = g.nodes[c.idx].ID
	}
	return out
}

// greedySearchLayerFast performs optimized greedy search.
func (g *Graph) greedySearchLayerFast(vec []float64, entry *Node, lvl int) *Node {
	if entry == nil || lvl >= len(entry.neighbors) {
		return entry
	}

	cur := entry
	curVec := g.getVectorFast(cur.idx)
	dMin := euclideanSquared(vec, curVec)

	improved := true
	for improved {
		improved = false

		for _, ni := range cur.neighbors[lvl] {
			if ni >= len(g.nodes) {
				continue
			}

			nbrVec := g.getVectorFast(ni)
			d := euclideanSquaredFast(vec, nbrVec)
			if d < dMin {
				dMin = d
				cur = g.nodes[ni]
				improved = true
				break // Take first improvement for speed
			}
		}
	}
	return cur
}

// searchLayerFast performs optimized layer search with better termination.
func (g *Graph) searchLayerFast(query []float64, entry *Node, lvl, ef int) []*candidate {
	if entry == nil {
		return nil
	}

	visited := g.visitedPool.Get()
	defer g.visitedPool.Return(visited)

	pqPtr := g.pqPool.Get().(*minHeap)
	*pqPtr = (*pqPtr)[:0]
	resPtr := g.resPool.Get().(*maxHeap)
	*resPtr = (*resPtr)[:0]

	defer func() {
		g.pqPool.Put(pqPtr)
		g.resPool.Put(resPtr)
	}()

	entryVec := g.getVectorFast(entry.idx)
	d0 := euclideanSquaredFast(query, entryVec)
	heap.Push(pqPtr, &candidate{idx: entry.idx, dist: d0})
	heap.Push(resPtr, &candidate{idx: entry.idx, dist: d0})
	visited.Visit(entry.idx)

	lowerBound := d0

	for pqPtr.Len() > 0 {
		c := heap.Pop(pqPtr).(*candidate)

		// Better termination condition borrowed from hnswlib
		if resPtr.Len() >= ef && c.dist > lowerBound {
			break
		}

		node := g.nodes[c.idx]
		if lvl >= len(node.neighbors) {
			continue
		}

		for _, ni := range node.neighbors[lvl] {
			if ni >= len(g.nodes) {
				continue
			}
			if visited.IsVisited(ni) {
				continue
			}
			visited.Visit(ni)

			nbrVec := g.getVectorFast(ni)
			d := euclideanSquaredFast(query, nbrVec)

			// Improved candidate management
			if resPtr.Len() < ef {
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
				if d < lowerBound {
					lowerBound = d
				}
			} else if d < (*resPtr)[0].dist {
				heap.Pop(resPtr)
				heap.Push(resPtr, &candidate{idx: ni, dist: d})
				heap.Push(pqPtr, &candidate{idx: ni, dist: d})
				lowerBound = (*resPtr)[0].dist
			}
		}
	}

	out := make([]*candidate, resPtr.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(resPtr).(*candidate)
	}
	return out
}

// getVectorFast retrieves vector with optimized memory access patterns.
func (g *Graph) getVectorFast(idx int) []float64 {
	// Create result slice directly for better performance
	result := make([]float64, g.dim)

	off := idx * g.dim
	o := off

	// Optimized chunk traversal with early exit
	for chunkIdx, chunk := range g.vectors {
		n := chunk.Len()
		if o < n {
			d := min(n-o, g.dim)

			// Optimized bulk copy with loop unrolling for common dimensions
			i := 0
			for i < d-4 {
				result[i] = chunk.Value(o + i)
				result[i+1] = chunk.Value(o + i + 1)
				result[i+2] = chunk.Value(o + i + 2)
				result[i+3] = chunk.Value(o + i + 3)
				i += 4
			}
			for i < d {
				result[i] = chunk.Value(o + i)
				i++
			}

			// If we have more data to read, continue with next chunk
			if d < g.dim {
				remaining := g.dim - d
				o = 0
				for j := chunkIdx + 1; j < len(g.vectors) && remaining > 0; j++ {
					nextChunk := g.vectors[j]
					nextN := nextChunk.Len()
					nextD := min(nextN, remaining)

					for k := 0; k < nextD; k++ {
						result[d+k] = nextChunk.Value(k)
					}
					remaining -= nextD
					d += nextD
					if remaining == 0 {
						break
					}
				}
			}
			return result
		}
		o -= n
	}
	return result
}

// selectNeighborsFast optimized neighbor selection.
func selectNeighborsFast(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	// Use partial sort for better performance
	if m < len(cands)/4 {
		// For small m, use heap-based selection
		return selectNeighborsHeap(cands, m)
	}

	// For larger m, use full sort
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	return cands[:m]
}

// selectNeighborsHeap uses heap-based selection for small m.
func selectNeighborsHeap(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}

	// Build max heap of size m
	maxHeap := make(maxHeap, 0, m)

	for _, c := range cands {
		if maxHeap.Len() < m {
			heap.Push(&maxHeap, c)
		} else if c.dist < maxHeap[0].dist {
			heap.Pop(&maxHeap)
			heap.Push(&maxHeap, c)
		}
	}

	// Extract results
	result := make([]*candidate, maxHeap.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(&maxHeap).(*candidate)
	}

	return result
}

// euclideanSquared computes squared euclidean distance (faster, same ordering).
func euclideanSquared(a, b []float64) float64 {
	var sum float64
	// Unroll loop for better performance on common dimensions
	i := 0
	for i < len(a)-3 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
		i += 4
	}
	for i < len(a) {
		d := a[i] - b[i]
		sum += d * d
		i++
	}
	return sum
}

// euclideanSquaredFast computes squared euclidean distance with optimizations from hnswlib.
func euclideanSquaredFast(a, b []float64) float64 {
	var sum float64

	// Unroll loop by 8 for better performance (inspired by hnswlib SIMD approach)
	i := 0
	for i <= len(a)-8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
		i += 8
	}

	// Handle remaining elements
	for i < len(a) {
		d := a[i] - b[i]
		sum += d * d
		i++
	}
	return sum
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

// randomLevel samples a layer with proper ml parameter.
func (g *Graph) randomLevel() int {
	lvl := 0
	for rand.Float64() < 1.0/math.E && lvl < 16 { // Cap at 16 levels
		lvl++
	}
	return lvl
}

// selectNeighborsHeuristic implements the improved neighbor selection heuristic.
func (g *Graph) selectNeighborsHeuristic(candidates []*candidate, m int) []*candidate {
	if len(candidates) <= m {
		return candidates
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	// Optimized diversity heuristic: only apply to smaller candidate sets
	// and use a simpler diversity check for performance
	if len(candidates) <= m*2 {
		// Simple selection for small candidate sets
		return candidates[:m]
	}

	selected := make([]*candidate, 0, m)
	selected = append(selected, candidates[0]) // Always include closest

	// Use a more efficient diversity check - only check against closest selected
	for i := 1; i < len(candidates) && len(selected) < m; i++ {
		cand := candidates[i]
		shouldAdd := true

		// Only check against the closest selected neighbor for efficiency
		if len(selected) > 0 {
			closest := selected[0]
			candVec := g.getVectorFast(cand.idx)
			closestVec := g.getVectorFast(closest.idx)
			distToClosest := euclideanSquaredFast(candVec, closestVec)

			// If candidate is much closer to closest selected than to query, skip
			if distToClosest < cand.dist*0.9 {
				shouldAdd = false
			}
		}

		if shouldAdd {
			selected = append(selected, cand)
		}
	}

	return selected
}
