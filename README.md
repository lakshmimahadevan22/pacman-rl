# Pac-Man RL: Deep Q-Learning Agent

This project implements a reinforcement learning agent that learns to play Pac-Man through Deep Q-Learning. The agent navigates a grid-based maze, attempts to eat all food pellets while avoiding ghosts. 

## Features

- **Grid-based Environment**: Customizable maze with walls, food pellets, and ghosts  
- **Deep Q-Network (DQN)**: Neural network architecture with convolutional layers for spatial reasoning  
- **Experience Replay**: Stores and reuses past experiences to stabilize learning  
- **Target Network**: Separate network for generating target Q-values to reduce oscillation  
- **Exploration Strategy**: Epsilon-greedy approach with decaying randomness  
- **Visualization**: Real-time game rendering with Pygame  
- **Training Progress Tracking**: Performance metrics plotting using Matplotlib  
- **Model Checkpointing**: Automatic saving of best-performing models  

## Project Structure

- `environment.py`: Implements the Pac-Man game environment with state management  
- `agent.py`: Contains the DQN agent with learning and decision-making capabilities  
- `model.py`: Defines the neural network architecture for the Q-function approximation  
- `game_display.py`:  Visualization of the game using Pygame  
- `utils.py`: Utility functions for creating assets, plotting, and setting random seeds  
- `main.py`: Main script for training and evaluating the agent  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lakshmimahadevan22/pacman-rl.git
cd pacman-rl
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install torch numpy pygame matplotlib
```


### Training

To train the agent with default parameters:

```bash
python main.py
```

With custom parameters:

```bash
python main.py --width 15 --height 15 --num_ghosts 3 --episodes 500 --render
```
 

### Evaluation

To evaluate a trained agent:

```bash
python main.py --eval --load_model models/best_model.pth --eval_episodes 10
```

## How it Works

### Environment

The environment is a grid-based representation of the Pac-Man game:

- Pac-Man tries to eat all food pellets while avoiding ghosts  
- Walls restrict movement  
- Ghosts move randomly  
- The episode ends when Pac-Man collects all food, collides with a ghost, or reaches maximum steps  

### State Representation

The game state is represented as a 5-channel grid:

1. Walls  
2. Pac-Man position  
3. Ghost positions  
4. Food positions  
5. Empty spaces  

### Action Space

The agent can choose from 4 actions:

- `0`: Move UP  
- `1`: Move RIGHT  
- `2`: Move DOWN  
- `3`: Move LEFT  

### Reward Structure

- `+1.0` for eating a food pellet  
- `-0.1` for each step (encourages efficient paths)  
- `-10.0` for colliding with a ghost  
- `+10.0` for collecting all food  

## Training Process

The agent learns through the following process:

1. **Exploration vs. Exploitation**: Uses epsilon-greedy strategy, starting with mostly random actions and gradually shifting to learned policy  
2. **Experience Collection**: Stores `(state, action, reward, next_state, done)` tuples in replay memory  
3. **Batch Learning**: Samples random batches from experience replay for training  
4. **Q-Value Updates**: Updates Q-values using the Bellman equation.
5. **Target Network**: Periodically updates target network to stabilize training  

**The training progress is tracked by**:

- Episode rewards  
- Food pellets collected  
- Moving average of rewards  
- Exploration rate (`epsilon`)  

## Results

The agent learns to:

- Avoid walls and ghosts  
- Seek food pellets efficiently  
- Develop strategic navigation patterns  

Training metrics are saved to the `models` directory, with plots showing the learning progress over time.

