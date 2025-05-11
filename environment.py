import numpy as np
import random

class PacManEnvironment:
    
    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, width=20, height=20, num_ghosts=4, max_steps=1000):
        self.width = width
        self.height = height
        self.num_ghosts = num_ghosts
        self.max_steps = max_steps
        
        # Game state
        self.pacman_pos = None
        self.ghost_positions = []
        self.food_positions = []
        self.walls = []
        self.steps = 0
        self.score = 0
        
        self.reset()
    
    def reset(self):
        self.steps = 0
        self.score = 0
        
        self.walls = []
        for x in range(self.width):
            self.walls.append((x, 0))
            self.walls.append((x, self.height - 1))
        for y in range(self.height):
            self.walls.append((0, y))
            self.walls.append((self.width - 1, y))
        
        # random internal walls
        num_internal_walls = int(0.1 * self.width * self.height)
        for _ in range(num_internal_walls):
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)
            if (x, y) not in self.walls:
                self.walls.append((x, y))
        
        # Place the Pac-Man
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if (x, y) not in self.walls:
                self.pacman_pos = (x, y)
                break
        
        # Place the ghosts
        self.ghost_positions = []
        for _ in range(self.num_ghosts):
            while True:
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) not in self.walls and (x, y) != self.pacman_pos and (x, y) not in self.ghost_positions:
                    self.ghost_positions.append((x, y))
                    break
        
        
        self.food_positions = []
        food_count = int(0.2 * self.width * self.height)
        for _ in range(food_count):
            while True:
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if ((x, y) not in self.walls and 
                    (x, y) != self.pacman_pos and 
                    (x, y) not in self.ghost_positions and
                    (x, y) not in self.food_positions):
                    self.food_positions.append((x, y))
                    break
        
        return self._get_state()
    
    def step(self, action):
        self.steps += 1
        
        x, y = self.pacman_pos
        if action == self.UP and (x, y - 1) not in self.walls:
            self.pacman_pos = (x, y - 1)
        elif action == self.RIGHT and (x + 1, y) not in self.walls:
            self.pacman_pos = (x + 1, y)
        elif action == self.DOWN and (x, y + 1) not in self.walls:
            self.pacman_pos = (x, y + 1)
        elif action == self.LEFT and (x - 1, y) not in self.walls:
            self.pacman_pos = (x - 1, y)
        
        reward = -0.1 
        if self.pacman_pos in self.food_positions:
            self.food_positions.remove(self.pacman_pos)
            reward += 1.0
            self.score += 1
        
        new_ghost_positions = []
        for ghost_pos in self.ghost_positions:
            gx, gy = ghost_pos
            possible_moves = []
            
            if (gx, gy - 1) not in self.walls:  
                possible_moves.append((gx, gy - 1))
            if (gx + 1, gy) not in self.walls: 
                possible_moves.append((gx + 1, gy))
            if (gx, gy + 1) not in self.walls:  
                possible_moves.append((gx, gy + 1))
            if (gx - 1, gy) not in self.walls: 
                possible_moves.append((gx - 1, gy))
            
            if possible_moves:
                new_ghost_pos = random.choice(possible_moves)
                new_ghost_positions.append(new_ghost_pos)
            else:
                new_ghost_positions.append(ghost_pos)
        
        self.ghost_positions = new_ghost_positions
        
        done = False
        
        if self.pacman_pos in self.ghost_positions:
            reward -= 10.0
            done = True
        
        if not self.food_positions:
            reward += 10.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done, {"score": self.score}
    
    def _get_state(self):
       
        state = np.zeros((5, self.height, self.width), dtype=np.float32)
        
        for wall_x, wall_y in self.walls:
            state[0, wall_y, wall_x] = 1
        
        px, py = self.pacman_pos
        state[1, py, px] = 1
        
    
        for ghost_x, ghost_y in self.ghost_positions:
            state[2, ghost_y, ghost_x] = 1
        
    
        for food_x, food_y in self.food_positions:
            state[3, food_y, food_x] = 1
        
        for y in range(self.height):
            for x in range(self.width):
                if (state[0, y, x] == 0 and  
                    state[1, y, x] == 0 and  
                    state[2, y, x] == 0 and  
                    state[3, y, x] == 0):   
                    state[4, y, x] = 1
        
        return state