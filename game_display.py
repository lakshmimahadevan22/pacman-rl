import pygame
import os
import numpy as np

class PacManDisplay:
    
    def __init__(self, width, height, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        
    
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pac-Man RL")
        
        # Load assets
        self.load_assets()
    
    def load_assets(self):
        self.images = {}
        
        self.colors = {
            "wall": (0, 0, 255),      
            "pacman": (255, 255, 0),  
            "ghost": (255, 0, 0),     
            "food": (255, 255, 255), 
            "empty": (0, 0, 0)        
        }
        
        
        asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        os.makedirs(asset_dir, exist_ok=True)
        
        for item in ["pacman", "ghost", "wall", "food", "empty"]:
            image_path = os.path.join(asset_dir, f"{item}.png")
            try:
                image = pygame.image.load(image_path)
                self.images[item] = pygame.transform.scale(image, (self.cell_size, self.cell_size))
            except pygame.error:
                surface = pygame.Surface((self.cell_size, self.cell_size))
                surface.fill(self.colors[item])
                self.images[item] = surface
    
    def render(self, state):
        self.screen.fill((0, 0, 0)) 
        
        wall_layer = state[0]
        pacman_layer = state[1]
        ghost_layer = state[2]
        food_layer = state[3]
        empty_layer = state[4]
        
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                if wall_layer[y, x] == 1:
                    self.screen.blit(self.images["wall"], rect)
                elif pacman_layer[y, x] == 1:
                    self.screen.blit(self.images["pacman"], rect)
                elif ghost_layer[y, x] == 1:
                    self.screen.blit(self.images["ghost"], rect)
                elif food_layer[y, x] == 1:
                    self.screen.blit(self.images["food"], rect)
                elif empty_layer[y, x] == 1:
                    self.screen.blit(self.images["empty"], rect)
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()