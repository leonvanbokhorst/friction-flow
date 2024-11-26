"""
Acrobot-v1 DQN Implementation
Features:
- Double DQN architecture
- Experience Replay
- Continuous state space handling
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List
import torch.nn.functional as F
import math

# Constants
INPUT_SIZE = 6      # Acrobot has 6 state dimensions
N_ACTIONS = 3       # Acrobot has 3 actions (negative, neutral, positive torque)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.0003
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1  # Higher minimum exploration
EPSILON_DECAY = 2000
TARGET_UPDATE = 10
SUCCESS_HEIGHT = 1.9
NEAR_SUCCESS_HEIGHT = 1.7  # New threshold for "almost there"
STABILITY_THRESHOLD = 1.5  # Reduced threshold for more precise control
POSITION_MEMORY = 10  # Remember last N positions for stability
CURRICULUM_PHASES = [
    (-1.0, 100),   # Phase 1: Just get above horizontal
    (0.0, 200),    # Phase 2: Get upward
    (1.0, 300),    # Phase 3: Get high
    (1.9, 400)     # Phase 4: Full height
]

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        
        # Define network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.memory.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class AcrobotAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('Acrobot-v1')
        self.n_actions = self.env.action_space.n
        
        # Get number of state observations
        state_size = self.env.observation_space.shape[0]
        
        # Create networks
        self.policy_net = DQN(state_size, self.n_actions).to(self.device)
        self.target_net = DQN(state_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                       math.exp(-self.steps_done / EPSILON_DECAY)
        
        # Get current height
        cos_theta1, sin_theta1 = state[0].item(), state[1].item()
        cos_theta2, sin_theta2 = state[2].item(), state[3].item()
        tip_height = -cos_theta1 - cos_theta2
        
        # More exploration when stuck
        if self.stuck_counter > 50:
            eps_threshold = max(eps_threshold, 0.5)
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state = state.unsqueeze(0)
                return self.policy_net(state).argmax().view(1, 1)
        else:
            # Smart random actions
            if tip_height < -1.0:  # When low, prefer upward actions
                action = random.choice([1, 2])  # Prefer stronger actions
            elif tip_height < 0.0:  # Mid-low
                action = random.choice([0, 1, 2])  # All actions
            else:  # When high
                action = random.choice([0, 2])  # More controlled actions
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self, experiences):
        if len(experiences) < MIN_MEMORY_SIZE:
            return
            
        state_batch = torch.stack([experience.state for experience in experiences])
        action_batch = torch.tensor([experience.action for experience in experiences], device=self.device)
        reward_batch = torch.tensor([experience.reward for experience in experiences], device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack([experience.next_state for experience in experiences])
        done_batch = torch.tensor([experience.done for experience in experiences], device=self.device, dtype=torch.float32)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1]
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1))
            target_q_values = reward_batch + GAMMA * next_q_values.squeeze(1) * (1 - done_batch)
            
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, num_episodes: int):
        self.stuck_counter = 0
        prev_max_height = float('-inf')
        current_phase = 0
        episode_rewards = []  # Track all episode rewards
        consecutive_solves = 0  # Track consecutive successes
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            
            episode_reward = 0
            max_height = float('-inf')
            steps = 0
            done = False
            episode_success = False  # Track if episode reached success height
            
            # Get current target height from curriculum
            target_height, _ = CURRICULUM_PHASES[min(current_phase, len(CURRICULUM_PHASES)-1)]
            
            while not done:
                action = self.select_action(state)
                next_state, _, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                steps += 1
                
                # Calculate height
                cos_theta1, sin_theta1 = next_state[0], next_state[1]
                cos_theta2, sin_theta2 = next_state[2], next_state[3]
                tip_height = -cos_theta1 - cos_theta2
                max_height = max(max_height, tip_height)
                
                # Curriculum-based reward
                reward = 0
                
                # Base reward for height improvement
                if tip_height > target_height:
                    reward += 10.0
                    if current_phase < len(CURRICULUM_PHASES) - 1:
                        current_phase += 1
                
                # Progressive rewards
                reward += (tip_height - target_height) * 5.0
                
                # Bonus for exceeding previous max
                if tip_height > prev_max_height:
                    reward += 5.0
                    self.stuck_counter = 0
                else:
                    self.stuck_counter += 1
                
                # Success reward
                if tip_height > SUCCESS_HEIGHT:
                    episode_success = True
                    reward += 100.0
                
                # Small time penalty
                reward -= 0.1
                
                # Store experience
                self.memory.push(Experience(state, action, reward, next_state, done))
                
                if len(self.memory) >= MIN_MEMORY_SIZE:
                    experiences = self.memory.sample(BATCH_SIZE)
                    self.optimize_model(experiences)
                    
                    if steps % TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
            
            # Update tracking variables
            episode_rewards.append(episode_reward)
            if episode_success:
                consecutive_solves += 1
            else:
                consecutive_solves = 0
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                         np.exp(-self.steps_done / EPSILON_DECAY)
                print(f"Episode {episode}, Steps: {steps}, Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, Max Height: {max_height:.2f}, "
                      f"Epsilon: {epsilon:.2f}, Consecutive Solves: {consecutive_solves}")
                
                if consecutive_solves >= 5:  # Need 5 consecutive successful episodes
                    print(f"\n🎉 Solved with {consecutive_solves} consecutive successes!")
                    return True
            
            prev_max_height = max_height
        
        return False

    def play(self, episodes: int = 5):
        """Play episodes with trained model"""
        self.policy_net.eval()
        env = gym.make('Acrobot-v1', render_mode='human')
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            
            total_reward = 0
            done = False
            steps = 0
            max_height = float('-inf')
            successes = 0  # Track number of times we reach success height
            
            while not done:
                with torch.no_grad():
                    state = state.unsqueeze(0)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()
                
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                steps += 1
                
                # Calculate height
                cos_theta1, sin_theta1 = next_state[0], next_state[1]
                cos_theta2, sin_theta2 = next_state[2], next_state[3]
                tip_height = -cos_theta1 - cos_theta2
                max_height = max(max_height, tip_height)
                
                # Count successes without ending episode
                if tip_height > SUCCESS_HEIGHT:
                    successes += 1
                
                total_reward += reward
                
                # Only end if we hit step limit or truly fail
                if steps >= 200 or truncated:
                    done = True
                
                if done:
                    status = f"SUCCESS! ({successes} times)" if successes > 0 else \
                             "ALMOST!" if max_height > NEAR_SUCCESS_HEIGHT else "FAILED!"
                    print(f"Episode {episode + 1} {status} Steps: {steps}, "
                          f"Max Height: {max_height:.2f}, Total Reward: {total_reward:.2f}")
                    break
                
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        
        self.policy_net.train()  # Set back to training mode

def main():
    print("\n🤸 Starting Acrobot AI Training...\n")
    
    ai = AcrobotAI()
    
    print("🏃‍♂️ Training the agent...")
    solved = ai.train(num_episodes=1000)
    
    if solved:
        print("\n🎯 Training complete! Saving model...")
        torch.save(ai.policy_net.state_dict(), "gym/models/acrobot_best.pth")
        
        print("\n🤸 Watch the trained agent play!")
        ai.play()
    else:
        print("\n❌ Failed to solve the environment consistently.")

if __name__ == "__main__":
    main() 