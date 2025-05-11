import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

def create_simple_assets():
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    os.makedirs(asset_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(1, 1))
    circle = plt.Circle((0.5, 0.5), 0.4, color='yellow')
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(asset_dir, 'pacman.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(1, 1))
    ghost_body = plt.Rectangle((0.2, 0.2), 0.6, 0.6, color='red')
    ax.add_patch(ghost_body)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(asset_dir, 'ghost.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(1, 1))
    wall = plt.Rectangle((0, 0), 1, 1, color='blue')
    ax.add_patch(wall)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(asset_dir, 'wall.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(1, 1))
    circle = plt.Circle((0.5, 0.5), 0.1, color='white')
    ax.add_patch(circle)
    ax.set_facecolor('black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(asset_dir, 'food.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(1, 1))
    empty = plt.Rectangle((0, 0), 1, 1, color='black')
    ax.add_patch(empty)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.savefig(os.path.join(asset_dir, 'empty.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_training_progress(episode_rewards, avg_window=100, filename="training_progress.png"):

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.5, label='Episode Reward')
    
   
    if len(episode_rewards) >= avg_window:
        moving_avg = np.convolve(episode_rewards, np.ones(avg_window)/avg_window, mode='valid')
        plt.plot(range(avg_window-1, len(episode_rewards)), moving_avg, label=f'{avg_window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def set_random_seeds(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False