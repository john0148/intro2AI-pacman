import threading
import time
import os
import psutil
from queue import Queue
import csv
from typing import List, Tuple, Set, Dict, Optional
import tracemalloc
from collections import deque

class PacmanGame:
    def __init__(self, maze: List[List[str]], initial_pacman_pos: Tuple[int, int]):
        self.maze = maze
        self.pacman_pos = initial_pacman_pos
        self._lock = threading.Lock()  # Lock để đồng bộ hóa việc truy cập vị trí pacman
        
    def update_pacman_position(self, new_pos: Tuple[int, int]) -> bool:
        """
        Cập nhật vị trí mới cho Pacman
        Args:
            new_pos: Tuple[int, int] - vị trí mới (row, col)
        Returns:
            bool - True nếu cập nhật thành công, False nếu vị trí không hợp lệ
        """
        if not self.is_valid_position(new_pos):
            return False
            
        with self._lock:
            self.pacman_pos = new_pos
            return True
            
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Kiểm tra xem vị trí có hợp lệ để Pacman di chuyển tới không
        """
        row, col = pos
        if 0 <= row < len(self.maze) and 0 <= col < len(self.maze[0]):
            return self.maze[row][col] != 'x'
        return False
        
    def get_pacman_position(self) -> Tuple[int, int]:
        """
        Lấy vị trí hiện tại của Pacman một cách thread-safe
        """
        with self._lock:
            return self.pacman_pos

class Ghost(threading.Thread):
    def __init__(self, start_pos: Tuple[int, int], ghost_type: str, maze: List[List[str]], game: PacmanGame):
        super().__init__()
        self.start_pos = start_pos
        self.ghost_type = ghost_type
        self.maze = maze
        self.game = game
        self.path: List[Tuple[int, int]] = []
        self.execution_time = 0
        self.memory_used = 0

    def is_valid_move(self, x: int, y: int, visited_this_run: Set[Tuple[int, int]]) -> bool:
        return (0 <= x < len(self.maze) and
                0 <= y < len(self.maze[0]) and
                self.maze[x][y] != 'x' and
                (x, y) not in visited_this_run)

    def dfs(self, start_pos: Tuple[int, int]) -> Tuple[bool, Set[Tuple[int, int]], int]:
        stack = [(start_pos, [start_pos])]
        visited_this_run = set()
        open_nodes_count = 0
        path_found = False
        self.path = []

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        target_pos = self.game.get_pacman_position()

        while stack:
            current_pos, current_path = stack.pop()
            open_nodes_count += 1

            if current_pos == target_pos:
                self.path = current_path
                path_found = True
                visited_this_run.add(current_pos)
                break

            if current_pos not in visited_this_run:
                visited_this_run.add(current_pos)

                for dx, dy in directions:
                    next_x, next_y = current_pos[0] + dx, current_pos[1] + dy
                    next_pos = (next_x, next_y)

                    if (0 <= next_x < len(self.maze) and
                        0 <= next_y < len(self.maze[0]) and
                        self.maze[next_x][next_y] != 'x'):
                        if next_pos not in visited_this_run:
                            stack.append((next_pos, current_path + [next_pos]))

        if not path_found:
            self.path = [start_pos]

        return path_found, visited_this_run, open_nodes_count

    def run(self):
        tracemalloc.start()
        start_time = time.time()
        
        _path_found, _visited, _open_count = self.dfs(self.start_pos)
        
        self.execution_time = time.time() - start_time
        
        current, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak / 1024  # Convert to KB
        tracemalloc.stop()

        self.write_results(_open_count)

    def write_results(self, open_nodes_count=None):
        nodes_to_write = open_nodes_count if open_nodes_count is not None else "N/A"
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nGhost {self.ghost_type} Results (DFS - Standalone Run):\n")
            f.write(f"  Path length: {len(self.path)}\n")
            f.write(f"  Path: {self.path}\n")
            f.write(f"  Open nodes in this run: {nodes_to_write}\n")
            f.write(f"  Execution time for this run: {self.execution_time:.4f} seconds\n")
            f.write(f"  Memory used for this run: {self.memory_used:.2f} KB\n")
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
                if cell in ['b', 'i', 'p', 'c']:  # Ghost positions
                    ghosts.append(((i, j), cell))
                    maze_row.append(' ')
                elif cell == 'P':  # Pacman position
                    pacman_pos = (i, j)
                    maze_row.append(cell)
                else:
                    maze_row.append(cell)
            maze.append(maze_row)
    
    return maze, ghosts, pacman_pos

def main():
    # Xóa nội dung file output.txt nếu tồn tại
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("DFS Pathfinding Results (Standalone)\n")
        f.write("=" * 50 + "\n")

    # Đọc mê cung từ file CSV
    maze, ghosts, pacman_pos = read_maze('res/level/level.csv')
    
    # Khởi tạo game với vị trí Pacman cố định
    game = PacmanGame(maze, pacman_pos)
    
    # Tạo và start các thread cho mỗi ghost
    ghost_threads = []
    for ghost_pos, ghost_type in ghosts:
        ghost = Ghost(ghost_pos, ghost_type, maze, game)
        ghost_threads.append(ghost)
        ghost.start()
    
    # Đợi tất cả các ghost hoàn thành
    for ghost in ghost_threads:
        ghost.join()

if __name__ == "__main__":
    main()
