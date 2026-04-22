# 2D Image Path Finder

Find paths between two points through black pixels in a binary image.

## Setup

```
python3 -m venv .venv
pip install -r requirements.txt
```

## Usage

```
python pathfinder.py <image> <start_row,col> <end_row,col> [--connectivity 4|8] [--mode reachable|path] [-o output.png]

python pathfinder.py data/2D_paths/small-ring.png 0,0 4,4 -o data/output_ring.png
python pathfinder.py data/2D_paths/bars.png 0,0 10,10 -o data/output_bars.png
python pathfinder.py data/2D_paths/polygons.png 0,0 99,99 -o data/output_polygons.png

```

```
pytest pathfinder.py -v
```

## Approach

The image is an implicit graph: black pixels are nodes, adjacent black pixels are edges (4 or 8-connected).

**Phase 1 — Reachability**: BFS from start; returns `True` if end is reached. Iterative to avoid recursion limits on large images.

**Phase 2 — Path finding**: BFS with parent tracking. Backtracks from end to reconstruct the path. BFS guarantees the shortest path (by pixel count) since all edges have equal weight. Annotated output draws the path in red with green/blue endpoint markers.


## non-overlapping paths between 2 pairs of sources and sinks (s1, e1), and (s2, e2)
Given 2 pairs (S1,E1) and (S2,E2), find two paths that share no pixels.

*My initial intuition was to use greedy approach here, get a path for first pair, try finding a path for second pair that doesn't cross the path between first pair. If that doesn't work, swap the order of pairs and try again. But I quickly realized, if two pair's shortest paths overlap, then I might end up without solution and will have to try all paths in first pair and try to find a disjoint path from second pair of source and sink.*  

Did a quick digging and found out it is acutually a well studied problem caller k-disjoint paths. But Edmonds-karp algorithm works with capacity on edge and I have pixels as vertices. So needed to basterdize the algorithm slightly.

1. **Vertex split**: each black pixel becomes `in` → `out` (capacity 1), converting vertex-disjointness to edge-disjointness
2. **Neighbor edges**: `out` of one pixel → `in` of adjacent pixel (capacity 1)
3. **Super-source/sink**: S feeds both starts, both ends drain to T
4. **2× BFS on residual graph**: first BFS finds one path; second BFS finds another, using reverse edges to reroute the first path at conflict points if needed
5. **Flow = 2** → decompose into paths; **flow < 2** → no disjoint paths exist

```
python disjoint_paths.py data/2D_paths/bars.png 0,0  0,4 0,6 10,10 -o data/output_disjoint_bars.png
python disjoint_paths.py data/2D_paths/small-ring.png 0,0 4,0 0,4 4,4 -o data/output_disjoint_ring.png
python disjoint_paths.py data/2D_paths/polygons.png 0,0 99,90 0,10 99,99 -o data/output_disjoint_polygons.png

```
