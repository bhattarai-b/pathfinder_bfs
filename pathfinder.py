from collections import deque
import argparse
import numpy as np

# 4-connected neighbor offsets (up, down, left, right)
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 8-connected neighbor offsets (adds diagonals)
NEIGHBORS_8 = NEIGHBORS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _grid(rows: list[list[int]]) -> np.ndarray:
    """Test helper: 0=black(traversable), 1=white(wall) → scaled to 0/255."""
    return np.array(rows, dtype=np.uint8) * 255


class ImageGraph:
    """Treats a binary image as an implicit graph where black pixels (0) are traversable."""

    def __init__(self, grid: np.ndarray, connectivity: int = 4):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if grid.ndim != 2:
            raise ValueError("grid must be 2D")

        self.grid = grid
        self.rows, self.cols = grid.shape
        self.offsets = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8

    def _is_valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    # we can only traverse to black pixels 
    def _is_traversable(self, r: int, c: int) -> bool:
        return self.grid[r, c] == 0

    def _neighbors(self, r: int, c: int):
        for dr, dc in self.offsets:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and self._is_traversable(nr, nc):
                yield nr, nc

    def is_reachable(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """
        BFS reachability check from start to end through black pixels.
        
        Why BFS?: 
            - I had though about using DFS, but quickly realized if the image is too large or has long corridors, DFS might hit Python's recursion limit.
            - BFS uses a queue and is guaranteed to find the shortest path in an unweighted grid, which is a nice property for reachability checks.
        """
        
        sr, sc = start
        er, ec = end

        if not self._is_valid(sr, sc) or not self._is_valid(er, ec):
            raise ValueError("start or end is out of bounds")
        if not self._is_traversable(sr, sc) or not self._is_traversable(er, ec):
            return False

        if start == end:
            return True

        visited = np.zeros((self.rows, self.cols), dtype=bool)
        visited[sr, sc] = True
        queue = deque()
        queue.append((sr, sc))

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if nr == er and nc == ec:
                    return True
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        return False

    def find_path(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
        """BFS with parent tracking. Returns the shortest path as a list of (row, col), or None."""
        sr, sc = start
        er, ec = end

        if not self._is_valid(sr, sc) or not self._is_valid(er, ec):
            raise ValueError("start or end is out of bounds")
        if not self._is_traversable(sr, sc) or not self._is_traversable(er, ec):
            return None

        if start == end:
            return [start]

        # parent stores (pr, pc) for each visited cell; (-1,-1) = no parent (start)
        parent = np.full((self.rows, self.cols, 2), -1, dtype=np.int32)
        parent[sr, sc] = (sr, sc)  # mark start visited with self-reference
        queue = deque()
        queue.append((sr, sc))
        found = False

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if parent[nr, nc, 0] != -1:
                    continue  # already visited
                parent[nr, nc] = (r, c)
                if nr == er and nc == ec:
                    found = True
                    break
                queue.append((nr, nc))
            if found:
                break

        if not found:
            return None

        # backtrack from end to start
        path = []
        r, c = er, ec
        while (r, c) != (sr, sc):
            path.append((r, c))
            pr, pc = parent[r, c]
            r, c = int(pr), int(pc)
        path.append((sr, sc))
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Tests (run with: pytest pathfinder.py)
# ---------------------------------------------------------------------------

def test_basic_reachability():
    # horizontal corridor
    assert ImageGraph(_grid([[0, 0, 0, 0, 0]]), 4).is_reachable((0, 0), (0, 4)) is True
    # path around wall
    assert ImageGraph(_grid([[0, 1, 0], [0, 0, 0]]), 4).is_reachable((0, 0), (0, 2)) is True
    # wall blocks completely
    assert ImageGraph(_grid([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), 4).is_reachable((0, 0), (0, 2)) is False
    # start == end
    assert ImageGraph(_grid([[0]]), 4).is_reachable((0, 0), (0, 0)) is True


def test_white_start_or_end():
    assert ImageGraph(_grid([[1, 0], [0, 0]]), 4).is_reachable((0, 0), (1, 1)) is False
    assert ImageGraph(_grid([[0, 0], [0, 1]]), 4).is_reachable((0, 0), (1, 1)) is False


def test_4conn_vs_8conn():
    diag = _grid([[0, 1], [1, 0]])
    assert ImageGraph(diag, 4).is_reachable((0, 0), (1, 1)) is False
    assert ImageGraph(diag, 8).is_reachable((0, 0), (1, 1)) is True

    staircase = _grid([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert ImageGraph(staircase, 4).is_reachable((0, 0), (2, 2)) is False
    assert ImageGraph(staircase, 8).is_reachable((0, 0), (2, 2)) is True


def test_maze_and_extremes():
    # spiral
    spiral = _grid([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ])
    assert ImageGraph(spiral, 4).is_reachable((0, 0), (4, 4)) is True
    # all black
    assert ImageGraph(np.zeros((10, 10), dtype=np.uint8), 4).is_reachable((0, 0), (9, 9)) is True
    # all white except endpoints (disconnected)
    w = np.full((5, 5), 255, dtype=np.uint8); w[0, 0] = 0; w[4, 4] = 0
    assert ImageGraph(w, 4).is_reachable((0, 0), (4, 4)) is False


def test_find_path():
    # straight corridor returns correct path
    g = ImageGraph(_grid([[0, 0, 0, 0, 0]]), 4)
    path = g.find_path((0, 0), (0, 4))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (0, 4)
    assert len(path) == 5  # shortest = 5 pixels
    # every pixel is black and consecutive pixels are neighbors
    for i in range(len(path) - 1):
        r1, c1 = path[i]; r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) <= 2  # at most 1 step in each axis

    # path around wall
    g2 = ImageGraph(_grid([[0, 1, 0], [0, 0, 0]]), 4)
    path2 = g2.find_path((0, 0), (0, 2))
    assert path2 is not None and path2[0] == (0, 0) and path2[-1] == (0, 2)

    # no path → None
    g3 = ImageGraph(_grid([[0, 1, 0], [0, 1, 0]]), 4)
    assert g3.find_path((0, 0), (0, 2)) is None

    # start == end
    assert ImageGraph(_grid([[0]]), 4).find_path((0, 0), (0, 0)) == [(0, 0)]

    # white start/end → None
    assert ImageGraph(_grid([[1, 0]]), 4).find_path((0, 0), (0, 1)) is None

    # 8-conn diagonal
    g8 = ImageGraph(_grid([[0, 1], [1, 0]]), 8)
    p8 = g8.find_path((0, 0), (1, 1))
    assert p8 == [(0, 0), (1, 1)]


def test_errors():
    import pytest
    g = ImageGraph(_grid([[0, 0]]), 4)
    with pytest.raises(ValueError, match="out of bounds"):
        g.is_reachable((0, 0), (5, 5))
    with pytest.raises(ValueError, match="out of bounds"):
        g.find_path((0, 0), (5, 5))
    with pytest.raises(ValueError, match="connectivity"):
        ImageGraph(_grid([[0]]), 6)
    with pytest.raises(ValueError, match="2D"):
        ImageGraph(np.array([0, 0, 0], dtype=np.uint8))


# ---------------------------------------------------------------------------
# CLI (run with: python pathfinder.py <image> <start> <end>)
# ---------------------------------------------------------------------------

def _parse_coord(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected row,col but got '{s}'")
    return int(parts[0]), int(parts[1])


def main():
    from image_utils import load_image

    parser = argparse.ArgumentParser(description="Check pixel reachability in a binary image.")
    parser.add_argument("image", help="Path to PNG image")
    parser.add_argument("start", type=_parse_coord, help="Start pixel as row,col")
    parser.add_argument("end", type=_parse_coord, help="End pixel as row,col")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--mode", choices=["reachable", "path"], default="path")
    parser.add_argument("--output", "-o", help="Save annotated image to this path")
    args = parser.parse_args()

    grid = load_image(args.image, threshold=args.threshold)
    graph = ImageGraph(grid, connectivity=args.connectivity)

    if args.mode == "reachable":
        result = graph.is_reachable(args.start, args.end)
        print(f"Reachable: {result}")
    else:
        path = graph.find_path(args.start, args.end)
        if path is None:
            print("No path found.")
        else:
            print(f"Path found: {len(path)} pixels")
            print(" -> ".join(f"({r},{c})" for r, c in path))
            if args.output:
                from image_utils import save_annotated
                save_annotated(args.image, path, args.output, args.start, args.end)
                print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
