import threading
import time
import math
import csv
import tracemalloc
from queue import PriorityQueue
from typing import List, Tuple

class PacmanGame:
    def __init__(self, maze: List[List[str]], initial_pacman_pos: Tuple[int, int]):
        self.maze = maze
        self.pacman_pos = initial_pacman_pos
        self._lock = threading.Lock()

    def update_pacman_position(self, new_pos: Tuple[int, int]) -> bool:
        if not self.is_valid_position(new_pos):
            return False
        with self._lock:
            self.pacman_pos = new_pos
            return True

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        return (0 <= row < len(self.maze) and 0 <= col < len(self.maze[0]) and self.maze[row][col] != 'x')

    def get_pacman_position(self) -> Tuple[int, int]:
        with self._lock:
            return self.pacman_pos

class Ghost(threading.Thread):
    def __init__(self, start_pos: Tuple[int, int], ghost_type: str, maze: List[List[str]], game: PacmanGame):
        super().__init__()
        self.start_pos = start_pos
        self.ghost_type = ghost_type
        self.maze = maze
        self.game = game
        self.path = []
        self.visited = set()
        self.open_nodes = 0
        self.execution_time = 0
        self.memory_used = 0
        self.parent = {}

    def is_valid_move(self, x: int, y: int) -> bool:
        return (0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] != 'x')

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def reconstruct_path(self, end: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while end in self.parent:
            path.append(end)
            end = self.parent[end]
        path.append(self.start_pos)
        return list(reversed(path))

    def astar(self, start_pos: Tuple[int, int]) -> bool:
        target = self.game.get_pacman_position()
        open_set = PriorityQueue()
        open_set.put((0, start_pos))
        g_score = {start_pos: 0}
        self.parent[start_pos] = None
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while not open_set.empty():
            _, current = open_set.get()
            self.open_nodes += 1
            if current == target:
                self.path = self.reconstruct_path(current)
                return True

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid_move(*neighbor):
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, target)
                    open_set.put((f_score, neighbor))
                    self.parent[neighbor] = current

        return False

    def run(self):
        tracemalloc.start()
        start_time = time.time()
        self.astar(self.start_pos)
        self.execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak / 1024
        tracemalloc.stop()
        self.write_results()

    def write_results(self):
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nGhost {self.ghost_type} Results (A*):\n")
            f.write(f"  Path length: {len(self.path)}\n")
            f.write(f"  Path: {self.path}\n")
            f.write(f"  Open nodes: {self.open_nodes}\n")
            f.write(f"  Execution time: {self.execution_time:.4f} seconds\n")
            f.write(f"  Memory used: {self.memory_used:.2f} KB\n")
            f.write("-" * 50 + "\n")

def read_maze(filename: str) -> Tuple[List[List[str]], List[Tuple[Tuple[int, int], str]], Tuple[int, int]]:
    maze = []
    ghosts = []
    pacman_pos = None
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for i, row in enumerate(reader):
            maze_row = []
            for j, cell in enumerate(row):
                if cell in ['b', 'i', 'p', 'c']:
                    ghosts.append(((i, j), cell))
                    maze_row.append(' ')
                elif cell == 'P':
                    pacman_pos = (i, j)
                    maze_row.append(cell)
                else:
                    maze_row.append(cell)
            maze.append(maze_row)
    return maze, ghosts, pacman_pos

def main():
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("A* Pathfinding Results\n")
        f.write("=" * 50 + "\n")

    maze, ghosts, pacman_pos = read_maze('res/level/level.csv')
    game = PacmanGame(maze, pacman_pos)
    ghost_threads = []
    for ghost_pos, ghost_type in ghosts:
        ghost = Ghost(ghost_pos, ghost_type, maze, game)
        ghost_threads.append(ghost)
        ghost.start()
    for ghost in ghost_threads:
        ghost.join()

if __name__ == "__main__":
    main()
