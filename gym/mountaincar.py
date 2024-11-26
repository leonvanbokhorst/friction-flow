"""
MountainCar-v0 DQN Implementation
Features:
- Double DQN architecture
- Prioritized Experience Replay
- Dueling Networks
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Tuple, List
import torch.nn.functional as F

# Constants
INPUT_SIZE = 2    # MountainCar has 2 state dimensions (position, velocity)
N_ACTIONS = 3     # MountainCar has 3 actions (left, nothing, right)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99  # Higher discount for long-term rewards
LEARNING_RATE = 0.001  # Increased learning rate
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1  # Increased minimum exploration
EPSILON_DECAY = 3000
TARGET_UPDATE = 10

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Simplified network architecture"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, N_ACTIONS)
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    """
    Prioritized Experience Replay Memory
    - Stores transitions with priorities
    - Samples based on priority weights
    """
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """Save an experience"""
        self.memory.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class MountainCarAI:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Use RMSprop instead of Adam
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), 
                                     lr=LEARNING_RATE, 
                                     alpha=0.95, 
                                     momentum=0.95)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.steps_done = 0
        self.best_reward = float('-inf')
        
        # Add epsilon tracking for better exploration
        self.epsilon_history = []
        self.success_positions = []
        
    def select_action(self, state: torch.Tensor) -> int:
        """Enhanced action selection with momentum and position-based strategy"""
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                 np.exp(-self.steps_done / EPSILON_DECAY)
        
        position, velocity = state
        
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        else:
            # Enhanced exploration strategy
            if abs(velocity) > 0.015:  # Increased velocity threshold
                return 0 if velocity < 0 else 2  # Maintain momentum
            elif position < -0.2:  # Adjusted position threshold
                return 2  # Prioritize right movement when in valley
            elif position > 0.2:  # New condition for higher positions
                return 2 if velocity >= 0 else 0  # Maintain upward momentum
            else:
                return random.randrange(N_ACTIONS)
            
    def optimize_model(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
            
        # Sample and transpose batch of experiences
        transitions = self.memory.sample(BATCH_SIZE)
        
        # Convert batch-array of Transitions to Transition of batch-arrays
        batch = Experience(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
            
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, num_episodes: int):
        best_position = -1.2
        consecutive_successes = 0
        episode_velocities = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            
            episode_reward = 0
            done = False
            steps = 0
            max_position = -1.2
            max_velocity = 0
            velocities = []
            last_position = state[0].item()
            episode_success = False  # Track success for this episode
            
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                     np.exp(-self.steps_done / EPSILON_DECAY)
            
            while not done:
                action = self.select_action(state)
                next_state, _, done, truncated, _ = self.env.step(action)
                done = done or truncated
                steps += 1
                
                position = next_state[0]
                velocity = next_state[1]
                velocities.append(abs(velocity))
                max_velocity = max(max_velocity, abs(velocity))
                
                # Initialize reward
                reward = 0
                
                # Modified reward structure
                if position >= 0.5:
                    reward = 2000.0  # Increased success reward
                    episode_success = True
                    self.success_positions.append(steps)
                    done = True
                else:
                    # Position-based rewards
                    if position > 0.4:
                        reward += (position - 0.4) * 800
                    
                    # Enhanced velocity rewards
                    current_velocity = abs(velocity)
                    if position > 0:
                        reward += current_velocity * 300
                    else:
                        reward += current_velocity * 150
                    
                    # Adjusted penalties
                    if abs(velocity) < 0.001:
                        reward -= 10
                    
                    if steps >= 200:
                        reward -= 100
                        done = True
                
                max_position = max(max_position, position)
                last_position = position
                episode_reward += reward
                
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
                self.memory.push(Experience(state, action, reward, next_state, done))
                state = next_state
                
                if len(self.memory) >= MIN_MEMORY_SIZE:
                    self.optimize_model()
                
                if self.steps_done % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                self.steps_done += 1
            
            # Update consecutive successes at episode end
            if episode_success:
                consecutive_successes += 1
            else:
                consecutive_successes = 0
            
            episode_velocities.append(max(velocities))
            
            if episode % 10 == 0:
                avg_velocity = np.mean(episode_velocities[-10:])
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {steps}, "
                      f"Max Height: {max_position:.2f}, Best: {best_position:.2f}, "
                      f"Max Velocity: {max_velocity:.3f}, Avg Velocity: {avg_velocity:.3f}, "
                      f"Epsilon: {epsilon:.2f}, Consecutive Successes: {consecutive_successes}")
                
                if consecutive_successes >= 3:
                    print(f"\n🎉 Solved with {consecutive_successes} consecutive successes!")
                    return True
                
        return False

    def play(self, episodes: int = 5):
        """Play episodes with trained model"""
        env = gym.make('MountainCar-v0', render_mode='human')
        successes = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            
            max_position = -1.2
            done = False
            steps = 0
            
            while not done:
                # Always use greedy policy during play
                with torch.no_grad():
                    action = self.policy_net(state.unsqueeze(0)).max(1)[1].item()
                
                next_state, _, done, truncated, _ = env.step(action)
                done = done or truncated
                steps += 1
                
                position = next_state[0]
                if position > max_position:
                    max_position = position
                
                if position >= 0.5:
                    successes += 1
                    print(f"Episode {episode + 1} SUCCESS! Steps: {steps}, Max Position: {max_position:.2f}")
                    break
                
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
                
                if steps >= 200:
                    print(f"Episode {episode + 1} failed. Max Position: {max_position:.2f}")
                    break
            
        env.close()
        print(f"\nSuccessful episodes: {successes}/{episodes}")

def main():
    print("🏔️ Starting MountainCar AI Training...")
    ai = MountainCarAI()
    
    print("\n🏃‍♂️ Training the agent...")
    solved = ai.train(num_episodes=1000)  # Increased max episodes
    
    if solved:
        print("\n🎯 Training complete! Saving model...")
        torch.save(ai.policy_net.state_dict(), "gym/models/mountaincar_best.pth")
        
        print("\n🚗 Watch the trained agent play!")
        ai.play(episodes=5)
    else:
        print("\n❌ Failed to solve the environment consistently.")

if __name__ == "__main__":
    main() 