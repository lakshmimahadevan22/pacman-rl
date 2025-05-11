import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model import DQN

class DQNAgent:
    def __init__(self, state_shape, num_actions, 
                 memory_size=10000, batch_size=64, 
                 gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.1, epsilon_decay=0.995,
                 learning_rate=0.0001):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.q_network = DQN(state_shape, num_actions).to(self.device)
        self.target_network = DQN(state_shape, num_actions).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states_np = np.array([experience[0] for experience in minibatch])
        actions_np = np.array([[experience[1]] for experience in minibatch])
        rewards_np = np.array([[experience[2]] for experience in minibatch])
        next_states_np = np.array([experience[3] for experience in minibatch])
        dones_np = np.array([[experience[4]] for experience in minibatch])
        
        # Converting numpy arrays to tensors
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).to(self.device)
        
        # Computing Q values for current states
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Computing Q values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        
        # Computing target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
    
    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_network()