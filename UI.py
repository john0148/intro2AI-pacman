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
from DFS import PacmanGame as DFSPacmanGame  # We'll use this just for type compatibility

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 15
BUTTON_HEIGHT = 35
INFO_HEIGHT = 200
PADDING = 10
FONT_SIZE = 16
BUTTON_WIDTH = 90
BUTTON_MARGIN = 10
ANIMATION_SPEED = 5  # Frames per animation change
GHOST_MOVE_SPEED = 0.1  # Ghost movement speed (cells per frame)
GHOST_OFFSET = 4  # Pixel offset for overlapping ghosts

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
        self.maze, self.ghosts, self.pacman_pos = self.read_maze(maze_file)
        self.original_maze = [row[:] for row in self.maze]  # Deep copy
        self.rows = len(self.maze)
        self.cols = len(self.maze[0]) if self.rows > 0 else 0
        self.active_algorithm = None
        self.paths = {}
        self.open_nodes = {}
        self.execution_times = {}
        self.memory_used = {}
        self.running = False
        self.scroll_y = 0  # Add scroll position
        self.max_scroll = 0  # Add maximum scroll value
        self.animation_frame = 0
        self.frame_counter = 0
        
        # Ghost animation variables
        self.ghost_positions = {ghost_type: {"pos": list(pos), "path_index": 0, "moving": False} 
                              for pos, ghost_type in self.ghosts}
        
        # Ghost drawing order (to maintain consistent layering)
        self.ghost_order = ['b', 'p', 'i', 'c']  # Red, Pink, Blue, Orange
        
        # Load images
        self.load_images()
        
        # Create a game state object for the algorithms
        self.game_state = DFSPacmanGame(self.maze, self.pacman_pos)
        
        # Initialize screen
        self.screen_width = self.cols * CELL_SIZE
        self.screen_height = self.rows * CELL_SIZE + BUTTON_HEIGHT + INFO_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pacman Pathfinding Visualization")
        
        # Create info surface
        self.info_surface = pygame.Surface((self.screen_width - 2*PADDING, INFO_HEIGHT*2))
        self.info_rect = pygame.Rect(PADDING, self.rows * CELL_SIZE + BUTTON_HEIGHT + PADDING, 
                                   self.screen_width - 2*PADDING, INFO_HEIGHT - 2*PADDING)
        
        # Initialize font
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
        
        # Create buttons
        self.buttons = [
            {"text": "DFS", "x": PADDING, "y": self.rows * CELL_SIZE + PADDING, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT - 2*PADDING, "enabled": True},
            {"text": "BFS", "x": PADDING + BUTTON_WIDTH + BUTTON_MARGIN, "y": self.rows * CELL_SIZE + PADDING, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT - 2*PADDING, "enabled": True},
            {"text": "UCS", "x": PADDING + 2*(BUTTON_WIDTH + BUTTON_MARGIN), "y": self.rows * CELL_SIZE + PADDING, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT - 2*PADDING, "enabled": False},
            {"text": "A*", "x": PADDING + 3*(BUTTON_WIDTH + BUTTON_MARGIN), "y": self.rows * CELL_SIZE + PADDING, 
             "width": BUTTON_WIDTH, "height": BUTTON_HEIGHT - 2*PADDING, "enabled": False}
        ]
    
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
            
        return maze, ghosts, pacman_pos
    
    def run_algorithm(self, algorithm: str):
        """Run the selected pathfinding algorithm"""
        if algorithm not in ["DFS", "BFS"]:
            print(f"Algorithm {algorithm} not implemented yet")
            return
            
        self.active_algorithm = algorithm
        
        # Clear previous results
        self.paths = {}
        self.open_nodes = {}
        self.execution_times = {}
        self.memory_used = {}
        
        # Reset ghost positions to their current positions and mark them as not moving
        for ghost_type in self.ghost_positions:
            if not self.ghost_positions[ghost_type]["moving"]:  # Only reset if ghost is not already moving
                pos = next((pos for pos, g_type in self.ghosts if g_type == ghost_type), None)
                if pos:
                    self.ghost_positions[ghost_type] = {
                        "pos": list(pos),
                        "path_index": 0,
                        "moving": False
                    }
        
        # Run the algorithm for each ghost
        for ghost_pos, ghost_type in self.ghosts:
            # Skip if ghost is already at Pacman's position
            if self.ghost_positions[ghost_type]["moving"] == False and \
               int(self.ghost_positions[ghost_type]["pos"][0]) == self.pacman_pos[0] and \
               int(self.ghost_positions[ghost_type]["pos"][1]) == self.pacman_pos[1]:
                continue
                
            # Start memory tracking
            tracemalloc.start()
            start_time = time.time()
            
            # Create the appropriate ghost object based on the algorithm
            if algorithm == "DFS":
                ghost = DFSGhost(ghost_pos, ghost_type, self.maze, self.game_state)
                # Run DFS algorithm
                ghost.dfs(ghost_pos)
            else:  # BFS
                ghost = BFSGhost(ghost_pos, ghost_type, self.maze, self.game_state)
                # Run BFS algorithm
                ghost.bfs(ghost_pos)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_used = peak / 1024  # Convert to KB
            tracemalloc.stop()
            
            # Store results
            self.paths[ghost_type] = ghost.path
            self.open_nodes[ghost_type] = ghost.open_nodes
            self.execution_times[ghost_type] = execution_time
            self.memory_used[ghost_type] = memory_used
            
            # Start ghost movement only if not already at Pacman's position
            if ghost_type in self.ghost_positions and not self.ghost_positions[ghost_type]["moving"]:
                current_pos = self.ghost_positions[ghost_type]["pos"]
                if int(current_pos[0]) != self.pacman_pos[0] or int(current_pos[1]) != self.pacman_pos[1]:
                    self.ghost_positions[ghost_type]["moving"] = True
                    self.ghost_positions[ghost_type]["path_index"] = 0

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
            # Update maze
            self.maze[self.pacman_pos[0]][self.pacman_pos[1]] = ' '
            self.maze[row][col] = 'P'
            self.pacman_pos = (row, col)
            
            # Update game state for algorithms
            self.game_state.update_pacman_position(new_pos)
            
            # Re-run the active algorithm if any
            if self.active_algorithm:
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
        
        # Draw maze
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.maze[i][j]
                if cell == 'P':  # Draw Pacman
                    # Draw Pacman with current animation frame
                    self.screen.blit(self.pacman_frames[self.animation_frame], 
                                   (j * CELL_SIZE, i * CELL_SIZE))
                else:
                    color = ELEMENT_COLORS.get(cell, BLACK)
                    # Draw cell
                    pygame.draw.rect(self.screen, color, 
                                    (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
                # Draw cell border
                pygame.draw.rect(self.screen, DARK_GRAY, 
                                (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
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
                    base_x = ghost_info["pos"][1] * CELL_SIZE
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
        
        # Draw buttons
        for button in self.buttons:
            # Button background
            color = LIGHT_GRAY if button["enabled"] else DARK_GRAY
            if button["enabled"] and self.active_algorithm == button["text"]:
                color = GREEN
                
            pygame.draw.rect(self.screen, color, 
                            (button["x"], button["y"], button["width"], button["height"]))
            
            # Button text
            text = self.font.render(button["text"], True, BLACK if button["enabled"] else GRAY)
            text_rect = text.get_rect(center=(button["x"] + button["width"]//2, 
                                             button["y"] + button["height"]//2))
            self.screen.blit(text, text_rect)
        
        # Draw info panel
        # Clear info surface
        self.info_surface.fill(BLACK)
        
        # Draw info panel content
        info_y = PADDING
        
        # Draw algorithm info
        if self.active_algorithm:
            algorithm_text = self.font.render(f"Algorithm: {self.active_algorithm}", True, WHITE)
            self.info_surface.blit(algorithm_text, (0, info_y))
            info_y += FONT_SIZE + 5
            
            # Draw metrics for each ghost
            for ghost_type in self.ghost_order:  # Use consistent order
                if ghost_type in self.paths:
                    ghost_name = {
                        'b': 'Red Ghost',
                        'i': 'Blue Ghost',
                        'p': 'Pink Ghost',
                        'c': 'Orange Ghost'
                    }.get(ghost_type, ghost_type)
                    
                    color = ELEMENT_COLORS.get(ghost_type, WHITE)
                    ghost_text = self.font.render(f"{ghost_name}:", True, color)
                    self.info_surface.blit(ghost_text, (0, info_y))
                    info_y += FONT_SIZE + 5
                    
                    path_text = self.font.render(
                        f"  Path length: {len(self.paths.get(ghost_type, []))} | "
                        f"Open nodes: {self.open_nodes.get(ghost_type, 0)} | "
                        f"Time: {self.execution_times.get(ghost_type, 0):.4f}s | "
                        f"Memory: {self.memory_used.get(ghost_type, 0):.2f} KB", 
                        True, WHITE)
                    self.info_surface.blit(path_text, (0, info_y))
                    info_y += FONT_SIZE + 10
        else:
            instructions = [
                "Click on a search algorithm button to visualize pathfinding",
                "Use arrow keys to move Pacman",
                "Hold Ctrl + Up/Down arrows to scroll info panel",
                "The algorithm will re-run each time Pacman moves"
            ]
            
            for instruction in instructions:
                text = self.font.render(instruction, True, WHITE)
                self.info_surface.blit(text, (0, info_y))
                info_y += FONT_SIZE + 5
        
        # Update max scroll value
        self.max_scroll = max(0, info_y - INFO_HEIGHT + 2*PADDING)
        
        # Draw the visible portion of the info surface
        self.screen.blit(self.info_surface, self.info_rect, 
                        (0, self.scroll_y, self.info_rect.width, self.info_rect.height))
        
        # Draw scroll bar if needed
        if self.max_scroll > 0:
            scroll_bar_height = max(20, (INFO_HEIGHT - 2*PADDING) * (INFO_HEIGHT - 2*PADDING) / (info_y + 2*PADDING))
            scroll_bar_pos = (INFO_HEIGHT - 2*PADDING - scroll_bar_height) * self.scroll_y / self.max_scroll
            pygame.draw.rect(self.screen, GRAY, 
                           (self.screen_width - PADDING - 5, 
                            self.rows * CELL_SIZE + BUTTON_HEIGHT + PADDING + scroll_bar_pos,
                            5, scroll_bar_height))
        
        # Update display
        pygame.display.flip()
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click events"""
        x, y = pos
        
        # Check if any button was clicked
        for button in self.buttons:
            if (button["x"] <= x <= button["x"] + button["width"] and
                button["y"] <= y <= button["y"] + button["height"] and
                button["enabled"]):
                self.run_algorithm(button["text"])
                return
    
    def handle_key(self, key):
        """Handle keyboard events"""
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
    
    def run(self):
        """Main game loop"""
        self.running = True
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            
            self.draw()
            clock.tick(30)
        
        pygame.quit()
        sys.exit()

# Main function
def main():
    # Check if level.csv exists, if not create a sample one
    maze_file = "res/level/level.csv"
    if not os.path.exists(maze_file):
        print(f"Warning: {maze_file} not found. Using a sample maze.")
        maze_file = "res/level/sample_level.csv"
        with open(maze_file, 'w') as f:
            f.write("x;x;x;x;x\nx;P; ; ;x\nx; ;x; ;x\nx;b; ; ;x\nx;x;x;x;x")
    
    # Create and run the game
    game = PacmanGameUI(maze_file)
    game.run()

if __name__ == "__main__":
    main()