import threading
import time
import os
import psutil
from queue import PriorityQueue
import csv
from typing import List, Tuple, Set, Dict, Optional
import tracemalloc

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
        self.parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

    def is_valid_move(self, x: int, y: int, visited_this_run: Set[Tuple[int, int]]) -> bool:
        """
        Kiểm tra xem một vị trí có hợp lệ để ghost di chuyển tới không
        """
        return (0 <= x < len(self.maze) and
                0 <= y < len(self.maze[0]) and
                self.maze[x][y] != 'x' and
                (x, y) not in visited_this_run)
    
    def reconstruct_path(self, target_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Tái tạo đường đi từ điểm đích trở về điểm bắt đầu
        """
        path = []
        current: Optional[Tuple[int, int]] = target_pos
        while current is not None and current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(self.start_pos)
        return list(reversed(path))

    def ucs(self, start_pos: Tuple[int, int]) -> Tuple[bool, Set[Tuple[int, int]], int]:
        """
        Thuật toán UCS (Uniform Cost Search) tìm đường đi từ ghost đến pacman
        Returns:
            Tuple gồm: (đã tìm thấy đường, tập các nút đã thăm, số nút đã mở)
        """
        target = self.game.get_pacman_position()
        open_set = PriorityQueue()
        # Khởi tạo với (chi phí, vị trí)
        open_set.put((0, start_pos))
        
        # Chi phí từ điểm bắt đầu tới các điểm
        cost_so_far = {start_pos: 0}
        self.parent = {start_pos: None}
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited_this_run = {start_pos}
        open_nodes_count = 0
        path_found = False
        self.path = []

        while not open_set.empty():
            # Lấy nút có chi phí thấp nhất
            current_cost, current = open_set.get()
            open_nodes_count += 1
            
            # Nếu tìm thấy đích
            if current == target:
                self.path = self.reconstruct_path(current)
                path_found = True
                break

            # Duyệt các nút lân cận
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_move(*neighbor, visited_this_run):
                    continue
                
                # Chi phí đồng nhất: từng bước có chi phí bằng 1
                new_cost = cost_so_far[current] + 1
                
                # Nếu nút chưa được thăm hoặc tìm thấy đường đi tốt hơn
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    open_set.put((new_cost, neighbor))
                    self.parent[neighbor] = current
                    visited_this_run.add(neighbor)

        if not path_found:
            self.path = [start_pos]

        return path_found, visited_this_run, open_nodes_count

    def run(self):
        tracemalloc.start()
        start_time = time.time()
        
        _path_found, _visited, _open_count = self.ucs(self.start_pos)
        
        self.execution_time = time.time() - start_time
        
        current, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak / 1024
        tracemalloc.stop()
        
        self.write_results(_open_count)

    def write_results(self, open_nodes_count=None):
        nodes_to_write = open_nodes_count if open_nodes_count is not None else "N/A"
        
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nGhost {self.ghost_type} Results (UCS - Standalone Run):\n")
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
        f.write("UCS Pathfinding Results (Standalone)\n")
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