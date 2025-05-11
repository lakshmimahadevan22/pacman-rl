import os
import numpy as np
import torch
import pygame
import time
import argparse
from collections import deque

from environment import PacManEnvironment
from agent import DQNAgent
from game_display import PacManDisplay
from utils import create_simple_assets, plot_training_progress, set_random_seeds

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pac-Man RL Agent')
    parser.add_argument('--width', type=int, default=15, help='Width of the grid')
    parser.add_argument('--height', type=int, default=15, help='Height of the grid')
    parser.add_argument('--num_ghosts', type=int, default=3, help='Number of ghosts')
    parser.add_argument('--cell_size', type=int, default=30, help='Size of each cell in pixels')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--memory_size', type=int, default=10000, help='Size of replay memory')
    parser.add_argument('--target_update', type=int, default=10, help='Update target network every n episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.1, help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Exploration decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--render', action='store_true', help='Render the game')
    parser.add_argument('--render_delay', type=float, default=0.05, help='Delay between frames (seconds)')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--load_model', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def train(args):
    set_random_seeds(args.seed)
    create_simple_assets()
    
    env = PacManEnvironment(
        width=args.width, 
        height=args.height, 
        num_ghosts=args.num_ghosts,
        max_steps=args.max_steps
    )
    
    state_shape = (5, args.height, args.width) 
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=4, 
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate
    )
    
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    
    display = None
    if args.render:
        display = PacManDisplay(args.width, args.height, args.cell_size)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    episode_rewards = []
    best_avg_reward = float('-inf')
    reward_window = deque(maxlen=100)
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < args.max_steps:
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            agent.replay()
            
            if args.render and display:
                display.render(state)
                time.sleep(args.render_delay)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if display:
                            display.close()
                        return
        
        if episode % args.target_update == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        reward_window.append(episode_reward)
        
        avg_reward = np.mean(reward_window) if reward_window else episode_reward
        
        if avg_reward > best_avg_reward and len(reward_window) == reward_window.maxlen:
            best_avg_reward = avg_reward
            agent.save(os.path.join(args.save_dir, 'best_model.pth'))
        
        if episode % 50 == 0:
            agent.save(os.path.join(args.save_dir, f'model_episode_{episode}.pth'))
            plot_training_progress(episode_rewards, filename=os.path.join(args.save_dir, 'training_progress.png'))
        
        print(f"Episode {episode+1}/{args.episodes}, Score: {info['score']}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    agent.save(os.path.join(args.save_dir, 'final_model.pth'))
    
    plot_training_progress(episode_rewards, filename=os.path.join(args.save_dir, 'training_progress.png'))
    
    if display:
        display.close()

def evaluate(args):
    set_random_seeds(args.seed)
    
    env = PacManEnvironment(
        width=args.width, 
        height=args.height, 
        num_ghosts=args.num_ghosts,
        max_steps=args.max_steps
    )
    
    display = PacManDisplay(args.width, args.height, args.cell_size)
    
    state_shape = (5, args.height, args.width)
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=4,
        epsilon=0.0 
    )
    
    model_path = args.load_model or os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Please train a model first.")
        return
    
    episode_scores = []
    episode_rewards = []
    
    for episode in range(args.eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < args.max_steps:
            
            action = agent.act(state, training=False)
            
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
        
            display.render(state)
            time.sleep(args.render_delay)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    display.close()
                    return
        
        episode_scores.append(info['score'])
        episode_rewards.append(episode_reward)
        
        print(f"Evaluation Episode {episode+1}/{args.eval_episodes}, Score: {info['score']}, Reward: {episode_reward:.2f}")
    
    print("\nEvaluation Summary:")
    print(f"Average Score: {np.mean(episode_scores):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    
    
    display.close()

def main():
    args = parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)

if __name__ == "__main__":
    main()