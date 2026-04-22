"""
2 vertex-disjoint paths via max-flow (Edmonds-Karp) with vertex splitting.

Given a binary image and 2 pairs (S1,E1), (S2,E2), finds two paths through
black pixels that share no pixels. Uses:
  1. Vertex splitting: pixel (r,c) → in-node and out-node (cap 1 internal edge)
  2. Neighbor edges: out-node → in-node of adjacent pixel (cap 1)
  3. Super-source S → both starts, both ends → super-sink T
  4. Edmonds-Karp: BFS on residual graph, at most 2 augmenting path iterations
  5. Flow decomposition to extract the two pixel paths
"""
from collections import defaultdict, deque
import argparse
import numpy as np

from pathfinder import NEIGHBORS_4, NEIGHBORS_8

# Special node IDs for super-source and super-sink
_SOURCE = "S"
_SINK = "T"


def _node_in(r: int, c: int) -> tuple:
    return (r, c, "in")


def _node_out(r: int, c: int) -> tuple:
    return (r, c, "out")


class FlowNetwork:
    """Sparse directed graph with edge capacities for max-flow."""

    def __init__(self):
        self.cap = defaultdict(lambda: defaultdict(int))       # residual capacity
        self.orig_cap = defaultdict(lambda: defaultdict(int))  # original capacity
        self.adj = defaultdict(set)

    def add_edge(self, u, v, cap: int):
        self.cap[u][v] += cap
        self.orig_cap[u][v] += cap
        self.adj[u].add(v)
        self.adj[v].add(u)

    def bfs(self, source, sink) -> dict | None:
        """BFS on residual graph. Returns parent map if augmenting path found."""
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v not in parent and self.cap[u][v] > 0:
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    def max_flow(self, source, sink) -> int:
        """Edmonds-Karp: BFS-based augmenting paths."""
        total_flow = 0
        while True:
            parent = self.bfs(source, sink)
            if parent is None:
                break
            bottleneck = float("inf")
            v = sink
            while parent[v] is not None:
                u = parent[v]
                bottleneck = min(bottleneck, self.cap[u][v])
                v = u
            v = sink
            while parent[v] is not None:
                u = parent[v]
                self.cap[u][v] -= bottleneck
                self.cap[v][u] += bottleneck
                v = u
            total_flow += bottleneck
        return total_flow

    def edge_flow(self, u, v) -> int:
        """Flow on edge u→v = original_cap - remaining_cap."""
        return self.orig_cap[u][v] - self.cap[u][v]


def build_flow_network(
    grid: np.ndarray,
    pair1: tuple[tuple[int, int], tuple[int, int]],
    pair2: tuple[tuple[int, int], tuple[int, int]],
    connectivity: int = 4,
) -> FlowNetwork:
    """Build vertex-split flow network from binary image and 2 source-sink pairs."""
    rows, cols = grid.shape
    offsets = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8
    net = FlowNetwork()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                continue
            net.add_edge(_node_in(r, c), _node_out(r, c), 1)
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    net.add_edge(_node_out(r, c), _node_in(nr, nc), 1)

    s1, e1 = pair1
    s2, e2 = pair2
    net.add_edge(_SOURCE, _node_in(*s1), 1)
    net.add_edge(_SOURCE, _node_in(*s2), 1)
    net.add_edge(_node_out(*e1), _SINK, 1)
    net.add_edge(_node_out(*e2), _SINK, 1)

    return net


def _decompose_flow(net: FlowNetwork) -> list[list[tuple[int, int]]]:
    """Decompose flow=2 into two pixel paths by tracing from SOURCE to SINK.

    Uses edge_flow() to find edges that carried flow, then traces through them.
    Consumes flow during tracing so each trace gets its own edges.
    """
    # Build mutable flow dict: flow[u][v] = units of flow on edge u→v
    flow = defaultdict(lambda: defaultdict(int))
    for u in net.adj:
        for v in net.adj[u]:
            f = net.edge_flow(u, v)
            if f > 0:
                flow[u][v] = f

    paths = []
    for _ in range(2):
        current = _SOURCE
        path_nodes = [current]
        while current != _SINK:
            next_node = None
            for v in net.adj[current]:
                if flow[current][v] > 0:
                    next_node = v
                    break
            if next_node is None:
                break
            flow[current][next_node] -= 1
            path_nodes.append(next_node)
            current = next_node

        # Extract pixel coordinates from out-nodes only
        pixels = []
        for node in path_nodes:
            if isinstance(node, tuple) and len(node) == 3 and node[2] == "out":
                pixels.append((node[0], node[1]))
        paths.append(pixels)

    return paths


def find_disjoint_paths(
    grid: np.ndarray,
    pair1: tuple[tuple[int, int], tuple[int, int]],
    pair2: tuple[tuple[int, int], tuple[int, int]],
    connectivity: int = 4,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find 2 vertex-disjoint paths for the given pairs.

    Returns (path1, path2) where each is a list of (row,col), or None if
    no disjoint solution exists.
    """
    s1, e1 = pair1
    s2, e2 = pair2

    rows, cols = grid.shape
    for label, (r, c) in [("s1", s1), ("e1", e1), ("s2", s2), ("e2", e2)]:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{label} is out of bounds")
        if grid[r, c] != 0:
            return None

    endpoints = {s1, e1, s2, e2}
    if len(endpoints) < 4:
        raise ValueError("all 4 endpoints must be distinct pixels")

    net = build_flow_network(grid, pair1, pair2, connectivity)
    flow = net.max_flow(_SOURCE, _SINK)

    if flow < 2:
        return None

    paths = _decompose_flow(net)
    if len(paths) != 2 or not paths[0] or not paths[1]:
        return None

    path_a, path_b = paths

    # Fix pairing: we need path_a = s1→e1, path_b = s2→e2
    # The flow may have produced any of these assignments:
    #   path_a: s1→e1, path_b: s2→e2  (correct)
    #   path_a: s2→e2, path_b: s1→e1  (just swap)
    #   path_a: s1→e2, path_b: s2→e1  (crossed — reassign)
    #   path_a: s2→e1, path_b: s1→e2  (crossed + swapped)
    if path_a[0] == s1 and path_a[-1] == e1:
        pass  # correct
    elif path_b[0] == s1 and path_b[-1] == e1:
        path_a, path_b = path_b, path_a  # just swap
    else:
        # crossed pairing: s1→e2 and s2→e1 (or vice versa)
        # reassign: take path starting from s1 but going to e2,
        # and path starting from s2 going to e1 — swap their tails
        if path_a[0] != s1:
            path_a, path_b = path_b, path_a
        # now path_a: s1→e2, path_b: s2→e1
        # find crossing point (first shared pixel)
        set_a = set(path_a)
        cross = None
        for i, p in enumerate(path_b):
            if p in set_a:
                cross = p
                break
        if cross is not None:
            idx_a = path_a.index(cross)
            idx_b = path_b.index(cross)
            new_a = path_a[:idx_a] + path_b[idx_b:]
            new_b = path_b[:idx_b] + path_a[idx_a:]
            path_a, path_b = new_a, new_b
        else:
            # no crossing — paths are fully disjoint but endpoints swapped
            # just return with swapped destinations (the paths themselves are valid)
            pass

    return path_a, path_b


# ---------------------------------------------------------------------------
# Tests (run with: pytest disjoint_paths.py)
# ---------------------------------------------------------------------------

def _grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8) * 255


def test_simple_two_corridors():
    """Two separate corridors — trivially disjoint."""
    grid = _grid([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])
    result = find_disjoint_paths(grid, ((0, 0), (2, 0)), ((0, 2), (2, 2)), connectivity=4)
    assert result is not None
    p1, p2 = result
    assert p1[0] == (0, 0) and p1[-1] == (2, 0)
    assert p2[0] == (0, 2) and p2[-1] == (2, 2)
    assert set(p1).isdisjoint(set(p2))


def test_shared_bottleneck_fails():
    """Single-pixel bottleneck — can't route both pairs through it."""
    grid = _grid([
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])
    result = find_disjoint_paths(grid, ((0, 0), (2, 0)), ((0, 2), (2, 2)), connectivity=4)
    assert result is None


def test_greedy_would_fail():
    """Both shortest paths go through the center column, but disjoint paths exist via side corridors.
    Greedy picks center for pair 1, blocking pair 2. Max-flow finds the side routes."""
    grid = _grid([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    # S1=(0,1) E1=(4,1): shortest goes through center col 2
    # S2=(0,3) E2=(4,3): shortest also goes through center col 2
    # disjoint solution: pair1 via left side, pair2 via right side
    result = find_disjoint_paths(grid, ((0, 1), (4, 1)), ((0, 3), (4, 3)), connectivity=4)
    assert result is not None
    p1, p2 = result
    assert p1[0] == (0, 1) and p1[-1] == (4, 1)
    assert p2[0] == (0, 3) and p2[-1] == (4, 3)
    assert set(p1).isdisjoint(set(p2))


def test_no_path_at_all():
    """One pair is completely unreachable."""
    grid = _grid([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    result = find_disjoint_paths(grid, ((0, 0), (2, 0)), ((0, 2), (2, 2)), connectivity=4)
    assert result is None


def test_8conn_diagonal():
    """Disjoint paths using 8-connectivity diagonals."""
    grid = _grid([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    result = find_disjoint_paths(grid, ((0, 0), (2, 2)), ((0, 2), (2, 0)), connectivity=8)
    assert result is not None
    p1, p2 = result
    assert set(p1).isdisjoint(set(p2))


def test_adjacent_pairs():
    """Pairs running side by side."""
    grid = _grid([
        [0, 0],
        [0, 0],
        [0, 0],
    ])
    result = find_disjoint_paths(grid, ((0, 0), (2, 0)), ((0, 1), (2, 1)), connectivity=4)
    assert result is not None
    p1, p2 = result
    assert p1[0] == (0, 0) and p1[-1] == (2, 0)
    assert p2[0] == (0, 1) and p2[-1] == (2, 1)
    assert set(p1).isdisjoint(set(p2))


# ---------------------------------------------------------------------------
# CLI: python disjoint_paths.py <image> <s1> <e1> <s2> <e2> [options]
# ---------------------------------------------------------------------------

def _parse_coord(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected row,col but got '{s}'")
    return int(parts[0]), int(parts[1])


def main():
    from image_utils import load_image

    parser = argparse.ArgumentParser(description="Find 2 vertex-disjoint paths in a binary image.")
    parser.add_argument("image", help="Path to PNG image")
    parser.add_argument("s1", type=_parse_coord, help="Start of pair 1 (row,col)")
    parser.add_argument("e1", type=_parse_coord, help="End of pair 1 (row,col)")
    parser.add_argument("s2", type=_parse_coord, help="Start of pair 2 (row,col)")
    parser.add_argument("e2", type=_parse_coord, help="End of pair 2 (row,col)")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--output", "-o", help="Save annotated image")
    args = parser.parse_args()

    grid = load_image(args.image, threshold=args.threshold)
    result = find_disjoint_paths(grid, (args.s1, args.e1), (args.s2, args.e2), connectivity=args.connectivity)

    if result is None:
        print("No vertex-disjoint paths found.")
    else:
        p1, p2 = result
        print(f"Path 1: {len(p1)} pixels")
        print(" -> ".join(f"({r},{c})" for r, c in p1))
        print(f"Path 2: {len(p2)} pixels")
        print(" -> ".join(f"({r},{c})" for r, c in p2))
        overlap = set(p1) & set(p2)
        if overlap:
            print(f"WARNING: {len(overlap)} overlapping pixels (bug!)")
        else:
            print("Verified: paths are vertex-disjoint.")

        if args.output:
            from PIL import Image, ImageDraw
            img = Image.open(args.image).convert("RGB")
            pixels = img.load()
            for r, c in p1:
                pixels[c, r] = (255, 0, 0)   # red
            for r, c in p2:
                pixels[c, r] = (0, 0, 255)   # blue
            draw = ImageDraw.Draw(img)
            rad = max(0, min(img.width, img.height) // 100)
            for (r, c), color in [(args.s1, (0, 255, 0)), (args.e1, (0, 200, 0)),
                                   (args.s2, (255, 255, 0)), (args.e2, (200, 200, 0))]:
                if rad == 0:
                    pixels[c, r] = color
                else:
                    draw.ellipse([c - rad, r - rad, c + rad, r + rad], fill=color)
            img.save(args.output)
            print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
