import pygame
import sys
import csv
import os
from typing import List, Tuple, Dict, Set
import time
import tracemalloc

# Import algorithms from the provided files
from DFS import Ghost as DFSGhost
from BFS import Ghost as BFSGhost
from BFS import PacmanGame # Renamed import

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 10
BUTTON_HEIGHT = 30
INFO_HEIGHT = 100  # Giảm chiều cao của bảng thông tin
PADDING = 10
FONT_SIZE = 12
BUTTON_WIDTH = 70
BUTTON_MARGIN = 8
ANIMATION_SPEED = 5  # Frames per animation change
GHOST_MOVE_SPEED = 0.1  # Ghost movement speed (cells per frame)
GHOST_OFFSET = 3
LEVEL_BUTTON_WIDTH = 60  # Width for level buttons

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
GREEN = (0, 255, 0)

# Game elements colors
ELEMENT_COLORS = {
    " ": BLACK,      # Empty tile
    "x": BLUE,       # Wall
    ".": WHITE,      # PacGum
    "o": YELLOW,     # SuperPacGum
    "-": GRAY,       # GhostDoor
    "P": YELLOW,     # Pacman
    "b": RED,        # RedGhost
    "i": CYAN,       # BlueGhost
    "p": PINK,       # PinkGhost
    "c": ORANGE      # OrangeGhost
}

class PacmanGameUI:
    def __init__(self, maze_file: str):
        self.maze_file = maze_file
        self.maze = []
        self.initial_ghosts = []
        self.initial_pacman_pos = None
        self.original_maze = []
        self.rows = 0
        self.cols = 0
        
        self.active_algorithm = None
        self.selected_algorithm = None  # Currently selected algorithm (might not be active yet)
        self.selected_level = None  # Track the currently selected level button
        self.level_buttons = []  # Dynamic level buttons
        self.show_level_buttons = False  # Whether to show level buttons

        # Metrics for individual runs (used for info panel while playing)
        self.paths = {}
        self.open_nodes_last_run = {}
        self.execution_times_last_run = {}
        self.memory_used_last_run = {}

        # --- Cumulative Performance Metrics --- 
        self.cumulative_exec_time: Dict[str, float] = {}
        self.cumulative_expanded_nodes: Dict[str, int] = {}
        self.cumulative_peak_memory: Dict[str, float] = {}
        self.metrics_finalized = False
        self.game_start_time = time.perf_counter()
        self.game_end_time: Optional[float] = None
        # --- End Cumulative Metrics --- 

        self.running = False
        self.scroll_y = 0
        self.max_scroll = 0
        self.animation_frame = 0
        self.frame_counter = 0
        
        self.ghost_positions = {}
        self.ghost_order = ['b', 'p', 'i', 'c']
        self.ghost_algorithms = {}  # For storing which algorithm to use for each ghost

        self.load_images()
        
        # Add left padding for buttons area - make it wide enough for both algorithm and level buttons
        self.left_panel_width = 2 * BUTTON_WIDTH + 3 * PADDING + BUTTON_MARGIN
        
        # Initialize screen with default dimensions (will be updated after maze loading)
        self.screen_width = self.left_panel_width + 500  # Default width
        self.screen_height = 500 + BUTTON_HEIGHT + INFO_HEIGHT  # Default height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pacman Pathfinding Visualization")

        # Create info surface
        self.info_surface = pygame.Surface((self.screen_width - 2*PADDING, INFO_HEIGHT*2))
        self.info_rect = pygame.Rect(PADDING, 500 + BUTTON_HEIGHT + PADDING,
                                   self.screen_width - 2*PADDING, INFO_HEIGHT - 2*PADDING)

        self.font = pygame.font.SysFont('Arial', FONT_SIZE)

        # Create algorithm buttons on the left side
        self.create_algorithm_buttons()
        
        # Now load the maze and initialize game state
        self.maze, self.initial_ghosts, self.initial_pacman_pos = self.read_maze(self.maze_file)
        self.original_maze = [row[:] for row in self.maze]  # Deep copy
        
        # Update screen dimensions based on actual maze size
        self.screen_width = self.left_panel_width + self.cols * CELL_SIZE
        self.screen_height = self.rows * CELL_SIZE + INFO_HEIGHT  # Removed BUTTON_HEIGHT to make UI shorter
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Update info rectangle to be at the bottom left, below the maze
        self.info_rect = pygame.Rect(PADDING, self.rows * CELL_SIZE + PADDING,
                                   self.screen_width - 2*PADDING, INFO_HEIGHT - 2*PADDING)

        self.ghost_positions = {ghost_type: {"pos": list(pos), "path_index": 0, "moving": False}
                              for pos, ghost_type in self.initial_ghosts}
                              
        self.ghosts = self.initial_ghosts
        self.pacman_pos = self.initial_pacman_pos
        
        # Create a game state object using the imported PacmanGame
        self.game_state = PacmanGame(self.maze, self.initial_pacman_pos)

        self.game_over = False
        self.win = False
        self._initialize_cumulative_metrics() # Call helper to initialize

    def _initialize_cumulative_metrics(self):
        """Initialize or reset cumulative metrics for all potential ghosts."""
        self.cumulative_exec_time = {g_type: 0.0 for _, g_type in self.initial_ghosts}
        self.cumulative_expanded_nodes = {g_type: 0 for _, g_type in self.initial_ghosts}
        self.cumulative_peak_memory = {g_type: 0.0 for _, g_type in self.initial_ghosts}
        self.metrics_finalized = False
        self.game_start_time = time.perf_counter()
        self.game_end_time = None

    def load_images(self):
        """Load and scale all game images"""
        # Load Pacman animation frames
        self.pacman_frames = []
        for i in range(1, 5):
            img = pygame.image.load(f"image/pacman-{i}.png")
            img = pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
            self.pacman_frames.append(img)
        
        # Load ghost images
        self.ghost_images = {
            'b': pygame.transform.scale(pygame.image.load("image/RedGhost.png"), (CELL_SIZE, CELL_SIZE)),
            'i': pygame.transform.scale(pygame.image.load("image/BlueGhost.png"), (CELL_SIZE, CELL_SIZE)),
            'p': pygame.transform.scale(pygame.image.load("image/PinkGhost.png"), (CELL_SIZE, CELL_SIZE)),
            'c': pygame.transform.scale(pygame.image.load("image/OrangeGhost.png"), (CELL_SIZE, CELL_SIZE))
        }

    def read_maze(self, filename: str) -> Tuple[List[List[str]], List[Tuple[Tuple[int, int], str]], Tuple[int, int]]:
        maze = []
        ghosts = []
        pacman_pos = None
        
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                for i, row in enumerate(reader):
                    maze_row = []
                    for j, cell in enumerate(row):
                        if cell in ['b', 'i', 'p', 'c']:  # Ghost positions
                            ghosts.append(((i, j), cell))
                            maze_row.append(' ')  # Empty space where ghost is
                        elif cell == 'P':  # Pacman position
                            pacman_pos = (i, j)
                            maze_row.append(cell)
                        else:
                            maze_row.append(cell)
                    maze.append(maze_row)
                
                # Make sure we have a valid maze with at least one row
                if not maze:
                    raise ValueError("Empty maze file")
                
                # Verify all rows have the same length
                expected_cols = len(maze[0])
                for i, row in enumerate(maze):
                    if len(row) != expected_cols:
                        print(f"Warning: Row {i} has inconsistent length. Padding with empty spaces.")
                        # Pad row if needed
                        while len(row) < expected_cols:
                            row.append(' ')
                        # Trim row if too long
                        if len(row) > expected_cols:
                            maze[i] = row[:expected_cols]
                
        except Exception as e:
            print(f"Error reading maze file: {e}")
            # Create a small default maze if file can't be read
            maze = [["x", "x", "x", "x", "x"], 
                    ["x", "P", " ", " ", "x"], 
                    ["x", " ", "x", " ", "x"], 
                    ["x", "b", " ", " ", "x"], 
                    ["x", "x", "x", "x", "x"]]
            ghosts = [((3, 1), "b")]
            pacman_pos = (1, 1)
            
        # Update the rows and cols properties
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows > 0 else 0
            
        return maze, ghosts, pacman_pos
    
    def create_algorithm_buttons(self):
        """Create algorithm buttons on the left side of the screen"""
        # Position calculations for the left side - add padding to create separation from maze
        button_x = PADDING
        
        # Create algorithm buttons
        self.buttons = [
            {"text": "DFS", "x": button_x, "y": PADDING, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT, "enabled": True, "type": "algorithm"},
            {"text": "BFS", "x": button_x, "y": PADDING + BUTTON_HEIGHT + BUTTON_MARGIN, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT, "enabled": True, "type": "algorithm"},
            {"text": "UCS", "x": button_x, "y": PADDING + 2*(BUTTON_HEIGHT + BUTTON_MARGIN), 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT, "enabled": False, "type": "algorithm"},
            {"text": "Astar", "x": button_x, "y": PADDING + 3*(BUTTON_HEIGHT + BUTTON_MARGIN), 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT, "enabled": False, "type": "algorithm"},
            {"text": "Parallel", "x": button_x, "y": PADDING + 4*(BUTTON_HEIGHT + BUTTON_MARGIN), 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT, "enabled": True, "type": "algorithm"}
        ]
        
        # Initialize empty level buttons (will be populated when algorithm is selected)
        self.level_buttons = []
    
    def create_level_buttons(self, algorithm):
        """Create level buttons based on selected algorithm"""
        if not algorithm:
            self.level_buttons = []
            return
        
        # Position calculations - level buttons will be to the right of algorithm buttons
        button_x = PADDING + BUTTON_WIDTH + BUTTON_MARGIN
        
        self.level_buttons = []
        for i in range(5):
            self.level_buttons.append({
                "text": f"level_{i+1}", 
                "x": button_x, 
                "y": PADDING + i*(BUTTON_HEIGHT + BUTTON_MARGIN), 
                "width": LEVEL_BUTTON_WIDTH, 
                "height": BUTTON_HEIGHT, 
                "enabled": True,
                "type": "level",
                "algorithm": algorithm
            })

    def load_level(self, algorithm, level):
        """Load a level for the specified algorithm"""
        if algorithm == "Parallel":
            # Set different algorithms for different ghosts
            self.ghost_algorithms = {
                'b': "BFS",  # Red Ghost -> BFS
                'i': "BFS",  # Blue Ghost -> BFS
                'p': "DFS",  # Pink Ghost -> DFS
                'c': "DFS"   # Orange Ghost -> DFS
            }
            level_file = f"res/level/Parallel/{level}.csv"
        else:
            # Set the same algorithm for all ghosts
            self.ghost_algorithms = {ghost_type: algorithm for ghost_type in self.ghost_order}
            level_file = f"res/level/{algorithm}/{level}.csv"
        
        # Check if file exists
        if not os.path.exists(level_file):
            print(f"Warning: Level file {level_file} not found.")
            return False
        
        # Load the maze file
        self.maze_file = level_file
        self.reset_game()  # Reset game state with new maze
        self.active_algorithm = algorithm
        
        # Run algorithm if not Parallel
        if algorithm != "Parallel":
            self.run_algorithm(algorithm)
        else:
            self.run_parallel_algorithms()
        
        return True
    
    def run_parallel_algorithms(self):
        """Run different algorithms for different ghosts in parallel"""
        self.active_algorithm = "Parallel"
        
        # Clear only the last run metrics
        self.paths = {}
        self.open_nodes_last_run = {}
        self.execution_times_last_run = {}
        self.memory_used_last_run = {}
        
        # Run algorithm for each ghost
        for ghost_type in list(self.ghost_positions.keys()):
            algorithm = self.ghost_algorithms.get(ghost_type)
            if not algorithm:
                continue
                
            ghost_info = self.ghost_positions[ghost_type]
            current_pos_float = ghost_info["pos"]
            current_pos_int = (int(round(current_pos_float[0])), int(round(current_pos_float[1])))

            if not self.game_state.is_valid_position(current_pos_int):
                print(f"Warning: Ghost {ghost_type} at invalid position {current_pos_int}. Skipping recalculation.")
                continue

            if current_pos_int == self.pacman_pos:
                self.paths[ghost_type] = [current_pos_int]
                self.open_nodes_last_run[ghost_type] = 0
                self.execution_times_last_run[ghost_type] = 0
                self.memory_used_last_run[ghost_type] = 0
                ghost_info["moving"] = False
                ghost_info["path_index"] = 0
                continue

            # --- Metrics Measurement Start --- 
            tracemalloc.start()
            start_time = time.perf_counter()
            
            ghost = None
            path_found = False
            visited_nodes_this_run = set()
            open_nodes_this_run = 0

            try:
                if algorithm == "DFS":
                    ghost = DFSGhost(current_pos_int, ghost_type, self.maze, self.game_state)
                    path_found, visited_nodes_this_run, open_nodes_this_run = ghost.dfs(current_pos_int)
                else:  # BFS
                    ghost = BFSGhost(current_pos_int, ghost_type, self.maze, self.game_state)
                    path_found, visited_nodes_this_run, open_nodes_this_run = ghost.bfs(current_pos_int)
            finally:
                exec_time_this_run = time.perf_counter() - start_time
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                peak_mem_kb = peak_mem / 1024
                tracemalloc.stop()
            # --- Metrics Measurement End --- 

            # --- Store metrics and update ghost movement --- 
            ghost_path = ghost.path if ghost and ghost.path else [current_pos_int]
            self.paths[ghost_type] = ghost_path
            self.open_nodes_last_run[ghost_type] = open_nodes_this_run
            self.execution_times_last_run[ghost_type] = exec_time_this_run
            self.memory_used_last_run[ghost_type] = peak_mem_kb

            # Update cumulative metrics
            if not self.metrics_finalized:
                self.cumulative_exec_time[ghost_type] = self.cumulative_exec_time.get(ghost_type, 0) + exec_time_this_run
                self.cumulative_expanded_nodes[ghost_type] = self.cumulative_expanded_nodes.get(ghost_type, 0) + open_nodes_this_run
                self.cumulative_peak_memory[ghost_type] = max(self.cumulative_peak_memory.get(ghost_type, 0), peak_mem_kb)

            # Start ghost movement
            ghost_info["moving"] = True
            ghost_info["path_index"] = 0
            if ghost_path:
                ghost_info["pos"] = list(ghost_path[0])
            else:
                ghost_info["pos"] = list(current_pos_int)
                
    def run_algorithm(self, algorithm: str):
        """Run the selected pathfinding algorithm"""
        if algorithm not in ["DFS", "BFS", "Parallel"]:
            print(f"Algorithm {algorithm} not implemented yet")
            return
            
        if algorithm == "Parallel":
            self.run_parallel_algorithms()
            return
            
        self.active_algorithm = algorithm
        
        # Clear only the last run metrics
        self.paths = {}
        self.open_nodes_last_run = {}
        self.execution_times_last_run = {}
        self.memory_used_last_run = {}
        
        # Iterate over a copy of keys
        for ghost_type in list(self.ghost_positions.keys()): 
            ghost_info = self.ghost_positions[ghost_type]
            current_pos_float = ghost_info["pos"]
            current_pos_int = (int(round(current_pos_float[0])), int(round(current_pos_float[1])))

            if not self.game_state.is_valid_position(current_pos_int):
                 print(f"Warning: Ghost {ghost_type} at invalid position {current_pos_int}. Skipping recalculation.")
                 continue

            if current_pos_int == self.pacman_pos:
                 self.paths[ghost_type] = [current_pos_int]
                 self.open_nodes_last_run[ghost_type] = 0
                 self.execution_times_last_run[ghost_type] = 0
                 self.memory_used_last_run[ghost_type] = 0
                 ghost_info["moving"] = False
                 ghost_info["path_index"] = 0
                 continue

            # --- Metrics Measurement Start --- 
            tracemalloc.start()
            start_time = time.perf_counter()
            
            ghost = None
            path_found = False
            visited_nodes_this_run = set()
            open_nodes_this_run = 0

            try:
                if algorithm == "DFS":
                    ghost = DFSGhost(current_pos_int, ghost_type, self.maze, self.game_state)
                    path_found, visited_nodes_this_run, open_nodes_this_run = ghost.dfs(current_pos_int)
                else:  # BFS
                    ghost = BFSGhost(current_pos_int, ghost_type, self.maze, self.game_state)
                    path_found, visited_nodes_this_run, open_nodes_this_run = ghost.bfs(current_pos_int)
            finally:
                exec_time_this_run = time.perf_counter() - start_time
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                peak_mem_kb = peak_mem / 1024
                tracemalloc.stop()
            # --- Metrics Measurement End --- 

            # --- Store Last Run Metrics (for info panel) --- 
            ghost_path = ghost.path if ghost and ghost.path else [current_pos_int]
            self.paths[ghost_type] = ghost_path
            self.open_nodes_last_run[ghost_type] = open_nodes_this_run
            self.execution_times_last_run[ghost_type] = exec_time_this_run
            self.memory_used_last_run[ghost_type] = peak_mem_kb
            # --- End Store Last Run Metrics --- 

            # --- Update Cumulative Metrics (if game not over) --- 
            if not self.metrics_finalized:
                self.cumulative_exec_time[ghost_type] += exec_time_this_run
                self.cumulative_expanded_nodes[ghost_type] += open_nodes_this_run
                self.cumulative_peak_memory[ghost_type] = max(self.cumulative_peak_memory[ghost_type], peak_mem_kb)
            # --- End Update Cumulative Metrics --- 

            # Start or continue ghost movement
            ghost_info["moving"] = True
            ghost_info["path_index"] = 0
            if ghost_path:
                 ghost_info["pos"] = list(ghost_path[0])
            else:
                 ghost_info["pos"] = list(current_pos_int)

    def update_ghost_positions(self):
        """Update ghost positions based on their paths"""
        for ghost_type, ghost_info in self.ghost_positions.items():
            if ghost_info["moving"] and ghost_type in self.paths:
                path = self.paths[ghost_type]
                current_index = ghost_info["path_index"]
                
                if current_index < len(path) - 1:
                    # Get current and next positions
                    current_pos = path[current_index]
                    next_pos = path[current_index + 1]
                    
                    # Calculate movement
                    dx = (next_pos[1] - current_pos[1]) * GHOST_MOVE_SPEED
                    dy = (next_pos[0] - current_pos[0]) * GHOST_MOVE_SPEED
                    
                    # Update position
                    ghost_info["pos"][0] += dy
                    ghost_info["pos"][1] += dx
                    
                    # Check if reached next position
                    distance_to_next = abs(ghost_info["pos"][0] - next_pos[0]) + abs(ghost_info["pos"][1] - next_pos[1])
                    if distance_to_next < GHOST_MOVE_SPEED:
                        ghost_info["pos"] = [next_pos[0], next_pos[1]]
                        ghost_info["path_index"] += 1
                        
                        # Check if this is the last position (Pacman's position)
                        if ghost_info["path_index"] == len(path) - 1:
                            ghost_info["moving"] = False  # Stop moving when reaching Pacman
                else:
                    # Reset to start of path when reached end
                    if len(path) > 0:
                        ghost_info["pos"] = [path[0][0], path[0][1]]
                        ghost_info["path_index"] = 0

    def move_pacman(self, new_pos: Tuple[int, int]):
        """Move Pacman to a new position if valid"""
        row, col = new_pos
        if 0 <= row < self.rows and 0 <= col < self.cols and self.maze[row][col] != 'x':
            # Check if Pacman is eating a dot or super dot
            is_dot = self.maze[row][col] == '.' or self.maze[row][col] == 'o'
            self.maze[self.pacman_pos[0]][self.pacman_pos[1]] = ' '
            self.maze[row][col] = 'P'
            self.pacman_pos = (row, col)
            
            # Update game state for algorithms
            self.game_state.update_pacman_position(new_pos)
            
            # Check for win/game over AFTER moving and updating state
            if is_dot and self.check_win_condition():
                self.win = True
                self.finalize_metrics() # Finalize metrics on win
                print("Win condition met!")
            elif self.check_collision():
                 self.game_over = True
                 self.finalize_metrics() # Finalize metrics on game over
                 print("Collision detected after Pacman move!")
            elif self.active_algorithm and not self.win:
                 # Re-run the algorithm if the game is still active
                 if self.active_algorithm == "Parallel":
                     self.run_parallel_algorithms()
                 else:
                     self.run_algorithm(self.active_algorithm)
            
            return True
        return False
    
    def draw(self):
        """Draw the game state"""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Update animation frame
        self.frame_counter = (self.frame_counter + 1) % ANIMATION_SPEED
        if self.frame_counter == 0:
            self.animation_frame = (self.animation_frame + 1) % len(self.pacman_frames)
        
        # Update ghost positions
        self.update_ghost_positions()
        
        # Draw maze - offset by left panel width to accommodate buttons
        maze_offset_x = self.left_panel_width
        
        # Ensure we have valid maze dimensions
        rows = len(self.maze)
        if rows > 0:
            cols = len(self.maze[0])
            
            # Draw each cell of the maze
            for i in range(rows):
                for j in range(cols):
                    cell = self.maze[i][j]
                    if cell == 'P':  # Draw Pacman
                        # Draw Pacman with current animation frame
                        self.screen.blit(self.pacman_frames[self.animation_frame], 
                                       (maze_offset_x + j * CELL_SIZE, i * CELL_SIZE))
                    else:
                        color = ELEMENT_COLORS.get(cell, BLACK)
                        # Draw cell
                        pygame.draw.rect(self.screen, color, 
                                        (maze_offset_x + j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    
                    # Draw cell border
                    pygame.draw.rect(self.screen, DARK_GRAY, 
                                    (maze_offset_x + j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # Group ghosts by position
        ghost_groups = {}
        for ghost_type, ghost_info in self.ghost_positions.items():
            pos_key = (int(ghost_info["pos"][0]), int(ghost_info["pos"][1]))
            if pos_key not in ghost_groups:
                ghost_groups[pos_key] = []
            ghost_groups[pos_key].append(ghost_type)
        
        # Draw ghosts with offset when overlapping
        for pos_key, ghosts_at_pos in ghost_groups.items():
            # Sort ghosts by predefined order
            ghosts_at_pos.sort(key=lambda x: self.ghost_order.index(x))
            
            # Calculate offsets based on number of ghosts
            num_ghosts = len(ghosts_at_pos)
            for i, ghost_type in enumerate(ghosts_at_pos):
                ghost_img = self.ghost_images.get(ghost_type)
                if ghost_img:
                    ghost_info = self.ghost_positions[ghost_type]
                    base_x = maze_offset_x + ghost_info["pos"][1] * CELL_SIZE
                    base_y = ghost_info["pos"][0] * CELL_SIZE
                    
                    # Apply offset based on position in group
                    if num_ghosts > 1:
                        offset_x = (i - (num_ghosts - 1)/2) * GHOST_OFFSET
                        offset_y = -i * GHOST_OFFSET  # Move up slightly for each ghost
                    else:
                        offset_x = 0
                        offset_y = 0
                    
                    self.screen.blit(ghost_img, 
                                   (base_x + offset_x, base_y + offset_y))
        
        # Draw algorithm buttons
        for button in self.buttons:
            # Button background - default color
            color = LIGHT_GRAY if button["enabled"] else DARK_GRAY
            
            # Only highlight the currently selected algorithm
            if button["enabled"] and button["text"] == self.selected_algorithm:
                color = GREEN
                
            pygame.draw.rect(self.screen, color, 
                            (button["x"], button["y"], button["width"], button["height"]))
            
            # Button text
            text = self.font.render(button["text"], True, BLACK if button["enabled"] else GRAY)
            text_rect = text.get_rect(center=(button["x"] + button["width"]//2, 
                                             button["y"] + button["height"]//2))
            self.screen.blit(text, text_rect)
        
        # Draw level buttons if algorithm is selected
        if self.show_level_buttons and self.level_buttons:
            for button in self.level_buttons:
                # Only highlight the level button if it's selected and belongs to current algorithm
                if self.selected_level == button["text"] and self.selected_algorithm == button["algorithm"]:
                    color = GREEN
                else:
                    color = LIGHT_GRAY
                
                pygame.draw.rect(self.screen, color, 
                                (button["x"], button["y"], button["width"], button["height"]))
                
                text = self.font.render(button["text"], True, BLACK)
                text_rect = text.get_rect(center=(button["x"] + button["width"]//2, 
                                                 button["y"] + button["height"]//2))
                self.screen.blit(text, text_rect)
        
        # --- Draw Info Panel --- 
        self.info_surface.fill(BLACK)
        info_y = PADDING

        # --- Info Panel Content Logic --- 
        if self.active_algorithm and not self.game_over and not self.win:
            algorithm_text = self.font.render(f"Algorithm: {self.active_algorithm} (Live)", True, WHITE)
            self.info_surface.blit(algorithm_text, (0, info_y))
            info_y += FONT_SIZE + 5

            # Draw metrics for each ghost (Last Run)
            for ghost_type in self.ghost_order:
                if ghost_type in self.paths:
                    ghost_name = { 'b': 'Red', 'i': 'Blue', 'p': 'Pink', 'c': 'Orange'}.get(ghost_type, ghost_type)
                    color = ELEMENT_COLORS.get(ghost_type, WHITE)
                    
                    # For parallel mode, show which algorithm each ghost is using
                    if self.active_algorithm == "Parallel" and ghost_type in self.ghost_algorithms:
                        ghost_algo = self.ghost_algorithms[ghost_type]
                        ghost_text = self.font.render(f"{ghost_name} ({ghost_algo}):", True, color)
                    else:
                        ghost_text = self.font.render(f"{ghost_name}:", True, color)
                        
                    self.info_surface.blit(ghost_text, (0, info_y))
                    info_y += FONT_SIZE + 5

                    path_len = len(self.paths.get(ghost_type, []))
                    expanded_nodes_lr = self.open_nodes_last_run.get(ghost_type, 0)
                    exec_time = self.execution_times_last_run.get(ghost_type, 0)
                    mem_used = self.memory_used_last_run.get(ghost_type, 0)

                    path_text = self.font.render(
                        f"  Last Run -> Len: {path_len} | Expanded: {expanded_nodes_lr} | "
                        f"Time: {exec_time:.4f}s | Mem: {mem_used:.2f} KB",
                        True, WHITE)
                    self.info_surface.blit(path_text, (0, info_y))
                    info_y += FONT_SIZE + 10
        elif not self.active_algorithm and not self.game_over and not self.win:
            # Display instructions in a more compact format at the bottom left
            instructions = [
                "Algorithms: Left buttons | Levels: Right buttons",
                "Controls: Arrow keys | Scroll: Ctrl+Up/Down | Reset: R"
            ]
            for instruction in instructions:
                text = self.font.render(instruction, True, WHITE)
                self.info_surface.blit(text, (0, info_y))
                info_y += FONT_SIZE + 5
        elif self.game_over or self.win:
             status_text = "Game Over!" if self.game_over else "You Win!"
             status_render = self.font.render(status_text, True, RED if self.game_over else GREEN)
             self.info_surface.blit(status_render, (0, info_y))
             info_y += FONT_SIZE + 5

        # --- Blit Info Panel and Scroll Bar --- 
        self.max_scroll = max(0, info_y - self.info_rect.height)
        self.screen.blit(self.info_surface, self.info_rect,
                        (0, self.scroll_y, self.info_rect.width, self.info_rect.height))
        if self.max_scroll > 0:
            scroll_bar_height = max(20, self.info_rect.height * self.info_rect.height / info_y)
            scroll_bar_y_ratio = self.scroll_y / self.max_scroll if self.max_scroll > 0 else 0
            scroll_bar_pos_y = self.info_rect.top + (self.info_rect.height - scroll_bar_height) * scroll_bar_y_ratio
            pygame.draw.rect(self.screen, GRAY,
                           (self.screen_width - PADDING - 5, 
                            scroll_bar_pos_y,
                            5, scroll_bar_height))

        # --- Draw Game Over / Win Messages AND FINAL METRICS --- 
        if self.game_over or self.win:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message_font = pygame.font.SysFont('Arial', 36, bold=True)
            message_text = "GAME OVER" if self.game_over else "YOU WIN!"
            message_color = RED if self.game_over else GREEN
            center_x = self.screen_width // 2
            base_y = self.screen_height // 2

            text_surface = message_font.render(message_text, True, message_color)
            text_rect = text_surface.get_rect(center=(center_x, base_y - 50))
            self.screen.blit(text_surface, text_rect)

            restart_font = pygame.font.SysFont('Arial', 18)
            restart_text = restart_font.render("Press R to Restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(center_x, base_y + 10))
            self.screen.blit(restart_text, restart_rect)

            # --- Draw Final Metrics (only on Game Over) --- 
            if self.game_over and self.metrics_finalized:
                metrics_y = base_y + 40
                metrics_font = pygame.font.SysFont('Arial', 14)
                line_height = metrics_font.get_linesize() + 2

                if self.game_end_time is not None:
                     total_game_duration = self.game_end_time - self.game_start_time
                     duration_text = f"Total Game Duration: {total_game_duration:.3f} seconds"
                     duration_render = metrics_font.render(duration_text, True, WHITE)
                     duration_rect = duration_render.get_rect(center=(center_x, metrics_y))
                     self.screen.blit(duration_render, duration_rect)
                     metrics_y += line_height * 1.5

                title_text = f"Final Metrics ({self.active_algorithm}):"                 
                title_render = metrics_font.render(title_text, True, WHITE)
                title_rect = title_render.get_rect(center=(center_x, metrics_y))
                self.screen.blit(title_render, title_rect)
                metrics_y += line_height

                for ghost_type in self.ghost_order:
                    if ghost_type in self.cumulative_exec_time:
                        ghost_name = { 'b': 'Red', 'i': 'Blue', 'p': 'Pink', 'c': 'Orange'}.get(ghost_type, ghost_type)
                        color = ELEMENT_COLORS.get(ghost_type, WHITE)
                        
                        total_time = self.cumulative_exec_time.get(ghost_type, 0.0)
                        expanded_nodes = self.cumulative_expanded_nodes.get(ghost_type, 0)
                        peak_mem = self.cumulative_peak_memory.get(ghost_type, 0.0)

                        metrics_line = f"{ghost_name}: Time={total_time:.3f}s | Expanded Nodes={expanded_nodes} | Mem={peak_mem:.1f}KB"
                        metrics_render = metrics_font.render(metrics_line, True, color)
                        metrics_rect = metrics_render.get_rect(center=(center_x, metrics_y))
                        self.screen.blit(metrics_render, metrics_rect)
                        metrics_y += line_height

        # --- Update Display --- 
        pygame.display.flip()
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click events"""
        x, y = pos
        
        # Check if any algorithm button was clicked
        for button in self.buttons:
            if (button["x"] <= x <= button["x"] + button["width"] and
                button["y"] <= y <= button["y"] + button["height"] and
                button["enabled"]):
                # Clear previous selections when changing algorithms
                if self.selected_algorithm != button["text"]:
                    # Reset selected level and clear active algorithm 
                    self.selected_level = None
                    self.active_algorithm = None
                
                # Select algorithm and show level buttons
                self.selected_algorithm = button["text"]
                self.create_level_buttons(button["text"])
                self.show_level_buttons = True
                return
        
        # Check if any level button was clicked
        if self.show_level_buttons:
            for button in self.level_buttons:
                if (button["x"] <= x <= button["x"] + button["width"] and
                    button["y"] <= y <= button["y"] + button["height"]):
                    # Set the selected level
                    self.selected_level = button["text"]
                    # Load the level for the selected algorithm
                    self.load_level(button["algorithm"], button["text"])
                    return

    def handle_key(self, key):
        """Handle keyboard events"""
        # Allow reset key even if game over/won
        if key == pygame.K_r:
            print("Restarting game via R key...")
            
            # Preserve the selected algorithm
            selected_algo = self.selected_algorithm
            
            # Reset the game
            self.reset_game()
            
            # Restore the algorithm selection UI state
            if selected_algo:
                self.selected_algorithm = selected_algo
                self.create_level_buttons(selected_algo)
                self.show_level_buttons = True
                
            return

        # Only handle movement/scroll if game is active
        if not self.game_over and not self.win:
            row, col = self.pacman_pos

            if key == pygame.K_UP:
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Scroll info panel up
                    self.scroll_y = max(0, self.scroll_y - 20)
                else:
                    self.move_pacman((row - 1, col))
            elif key == pygame.K_DOWN:
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Scroll info panel down
                    self.scroll_y = min(self.max_scroll, self.scroll_y + 20)
                else:
                    self.move_pacman((row + 1, col))
            elif key == pygame.K_LEFT:
                self.move_pacman((row, col - 1))
            elif key == pygame.K_RIGHT:
                self.move_pacman((row, col + 1))
            # Add escape key to cancel algorithm selection
            elif key == pygame.K_ESCAPE:
                self.selected_algorithm = None
                self.show_level_buttons = False
                self.level_buttons = []

    def run(self):
        """Main game loop"""
        self.running = True
        clock = pygame.time.Clock()
        
        while self.running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Handle clicks and key presses regardless of game state,
                # the handlers themselves will check if the game is over/won.
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            # --- Game Logic Update ---
            # Only update game state if the game is active
            if not self.game_over and not self.win:
                # Update animation frame (can happen even if game over/won for visual feedback)
                self.frame_counter = (self.frame_counter + 1) % ANIMATION_SPEED
                if self.frame_counter == 0:
                    self.animation_frame = (self.animation_frame + 1) % len(self.pacman_frames)

                # Update ghost positions (only if game active and algorithm running)
                if self.active_algorithm:
                     self.update_ghost_positions()
                     # Check for collision after ghost movement
                     if self.check_collision():
                         self.game_over = True
                         self.finalize_metrics() # Finalize metrics
                         print("Collision detected after ghost move!")

                # Note: Win condition is checked within move_pacman

            # --- Drawing ---
            self.draw() # Draw the current state (including game over/win messages)

            clock.tick(60) # Limit frame rate

        pygame.quit()
        sys.exit()

    def reset_game(self):
        """Resets the game to its initial state."""
        print("Resetting game...")
        # Reread the maze and initial positions
        self.maze, self.initial_ghosts, self.initial_pacman_pos = self.read_maze(self.maze_file)
        self.original_maze = [row[:] for row in self.maze] # Update original maze copy too
        self.ghosts = self.initial_ghosts
        self.pacman_pos = self.initial_pacman_pos

        # Reset ghost positions and state
        self.ghost_positions = {ghost_type: {"pos": list(pos), "path_index": 0, "moving": False}
                              for pos, ghost_type in self.ghosts}

        # Reset game state object for algorithms
        self.game_state = PacmanGame(self.maze, self.pacman_pos)

        # Reset last run metrics
        self.paths = {}
        self.open_nodes_last_run = {}
        self.execution_times_last_run = {}
        self.memory_used_last_run = {}
        
        # Reset cumulative metrics and flags
        self._initialize_cumulative_metrics()
        
        # Reset game state flags
        self.game_over = False
        self.win = False
        self.scroll_y = 0 # Reset scroll

        # Reset animation
        self.animation_frame = 0
        self.frame_counter = 0

        # Keep selected_algorithm but clear active_algorithm
        # This allows the user to still see level buttons for the selected algorithm
        self.active_algorithm = None 

        print("Game reset complete.")

    def check_collision(self) -> bool:
        """Checks if any ghost has collided with Pacman."""
        pacman_coords = self.pacman_pos
        for ghost_type, ghost_info in self.ghost_positions.items():
            ghost_coords = (int(round(ghost_info["pos"][0])), int(round(ghost_info["pos"][1])))
            if ghost_coords == pacman_coords:
                return True
        return False

    def check_win_condition(self) -> bool:
        """Checks if all dots and super dots have been eaten."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r][c] == '.' or self.maze[r][c] == 'o':
                    return False # Found a dot, not won yet
        return True # No dots found, win!

    def finalize_metrics(self):
         """Mark metrics as final and record end time."""
         if not self.metrics_finalized:
             self.metrics_finalized = True
             self.game_end_time = time.perf_counter()
             print("Metrics finalized.")

# Main function
def main():
    # Ensure level directories exist
    level_dirs = ["BFS", "DFS", "UCS", "Astar", "Parallel"]
    for algo_dir in level_dirs:
        dir_path = f"res/level/{algo_dir}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            # Create a sample level in each directory
            for i in range(1, 6):
                level_file = f"{dir_path}/level_{i}.csv"
                if not os.path.exists(level_file):
                    with open(level_file, 'w') as f:
                        if algo_dir == "Parallel":
                            # Create a larger maze with multiple ghosts for parallel mode
                            f.write("x;x;x;x;x;x;x\n")
                            f.write("x;P; ; ; ; ;x\n")
                            f.write("x; ;x; ; ;b;x\n")
                            f.write("x; ; ; ;x; ;x\n")
                            f.write("x;c; ;x; ; ;x\n")
                            f.write("x; ; ; ; ;p;x\n")
                            f.write("x;i; ; ; ; ;x\n")
                            f.write("x;x;x;x;x;x;x")
                        else:
                            # Create a simple maze for other algorithms
                            f.write("x;x;x;x;x\n")
                            f.write("x;P; ; ;x\n")
                            f.write("x; ;x; ;x\n")
                            f.write("x;b; ; ;x\n")
                            f.write("x;x;x;x;x")
    
    # Check if level.csv exists, if not use BFS level 1
    maze_file = "res/level/level.csv"
    if not os.path.exists(maze_file):
        print(f"Warning: {maze_file} not found. Using a sample maze.")
        maze_file = "res/level/BFS/level_1.csv"
    
    # Create and run the game
    game = PacmanGameUI(maze_file)
    game.run()

if __name__ == "__main__":
    main()