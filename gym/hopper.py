"""
Hopper-v4 DQN Implementation
Features:
- Double DQN architecture with continuous action space
- Experience Replay with prioritization
- Advanced reward shaping for stable hopping
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple
import torch.nn.functional as F
import math
import pickle
from tqdm import tqdm

# Constants
STATE_DIM = 11      # Hopper state dimensions
ACTION_DIM = 3      # Hopper action dimensions
ACTION_HIGH = 1.0   # Action space limits

# Hyperparameters (adjusted for continuous control)
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.0003
MEMORY_SIZE = 100000
MIN_MEMORY_SIZE = 5000
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 2000
TARGET_UPDATE = 10
TAU = 0.005  # Soft update parameter

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * ACTION_HIGH

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling
        
    def push(self, experience: Experience, error: float = None):
        priority = max(self.priorities, default=1.0) if error is None else abs(error)
        self.memory.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], torch.Tensor, np.ndarray]:
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        weights = (len(self.memory) * probs[indices]) ** -self.beta
        weights /= weights.max()
        
        return [self.memory[idx] for idx in indices], torch.FloatTensor(weights), indices
        
    def __len__(self):
        return len(self.memory)

class HopperAI:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.env = gym.make('Hopper-v5')
        
        # Initialize networks
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=5
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.current_avg_reward = 0.0
        self.best_weights = None
        self.best_reward = float('-inf')
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.last_recovery_episode = 0
        self.recovery_cooldown = 10
        
    def select_action(self, state: torch.Tensor, best_avg_reward: float) -> torch.Tensor:
        # Ensure state is a tensor on the correct device
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)

        with torch.no_grad():
            action = self.actor(state)
            
        # Adjust exploration based on recovery state
        base_epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                      math.exp(-self.steps_done / EPSILON_DECAY)
        
        # Reduce noise if in recovery mode
        if self.recovery_attempts > 0:
            base_epsilon *= 0.5
            
        performance_ratio = max(0.1, min(1.0, self.current_avg_reward / best_avg_reward))
        noise_scale = base_epsilon * (1.0 - performance_ratio)
        
        noise = torch.randn_like(action) * noise_scale
        return torch.clamp(action + noise, -ACTION_HIGH, ACTION_HIGH)
    
    def optimize_model(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
            
        experiences, weights, indices = self.memory.sample(BATCH_SIZE)
        weights = weights.to(self.device)
        
        state_batch = torch.stack([e.state for e in experiences]).to(self.device)
        action_batch = torch.stack([e.action for e in experiences]).to(self.device)
        reward_batch = torch.tensor([e.reward for e in experiences], 
                                  device=self.device, dtype=torch.float32)
        next_state_batch = torch.stack([e.next_state for e in experiences]).to(self.device)
        done_batch = torch.tensor([e.done for e in experiences], 
                                device=self.device, dtype=torch.float32)
        
        # Gradient clipping and loss scaling
        critic_loss_scale = min(1.0, max(0.1, self.current_avg_reward / self.best_reward))
        actor_loss_scale = critic_loss_scale  # Scale both losses similarly
        
        # Update critic
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_actions)
            target_Q = reward_batch.unsqueeze(1) + GAMMA * target_Q * (1 - done_batch).unsqueeze(1)
        
        current_Q = self.critic(state_batch, action_batch)
        critic_loss = (weights.unsqueeze(1) * F.mse_loss(current_Q, target_Q, reduction='none')).mean()
        (critic_loss * critic_loss_scale).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)  # Reduced from 1.0
        self.critic_optimizer.step()
        
        # Update actor with reduced frequency when unstable
        if self.steps_done % 2 == 0:  # Update actor every other step
            self.actor_optimizer.zero_grad()
            actor_actions = self.actor(state_batch)
            actor_loss = -(weights.unsqueeze(1) * self.critic(state_batch, actor_actions)).mean()
            (actor_loss * actor_loss_scale).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
        
        # Soft update targets
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        # Update priorities
        with torch.no_grad():
            td_error = abs(target_Q - current_Q).detach()
            for idx, error in zip(indices, td_error):
                self.memory.priorities[idx] = error.item()

    def save_best_weights(self, episode_reward: float):
        """Save weights if we achieve better performance"""
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_weights = {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }
    
    def restore_best_weights(self):
        """Restore best weights if performance drops significantly"""
        if self.best_weights is not None:
            self.actor.load_state_dict(self.best_weights['actor'])
            self.critic.load_state_dict(self.best_weights['critic'])
            
    def train(self, num_episodes: int, previous_best: float = float('-inf')) -> bool:
        best_avg_reward = previous_best
        episode_rewards = []
        recovery_threshold = 0.3  # Threshold for recovery
        warmup_episodes = 10  # Allow some episodes to establish baseline
        
        pbar = tqdm(range(num_episodes), desc="Training Episodes")
        for episode in pbar:
            state, _ = self.env.reset()
            # Convert state to tensor here as well
            state = torch.FloatTensor(state).to(self.device)
            
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, best_avg_reward)
                next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                
                # Convert next_state to tensor before storing
                next_state = torch.FloatTensor(next_state).to(self.device)
                
                # Store experience
                self.memory.push(Experience(state, action, reward, next_state, done))
                if episode >= warmup_episodes:  # Only optimize after warmup
                    self.optimize_model()
                
                state = next_state
                episode_reward += reward
                self.steps_done += 1
            
            episode_rewards.append(episode_reward)
            
            # Update current average with smaller window
            window_size = min(20, len(episode_rewards))
            self.current_avg_reward = np.mean(episode_rewards[-window_size:])
            
            # Save best weights if performance improves
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_weights = {
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'reward': episode_reward
                }
            
            # Check for recovery after warmup period
            if episode > warmup_episodes and len(episode_rewards) > window_size:
                recent_avg = np.mean(episode_rewards[-window_size:])
                should_recover = (
                    recent_avg < (self.best_reward * recovery_threshold) and
                    episode - self.last_recovery_episode > self.recovery_cooldown and
                    self.recovery_attempts < self.max_recovery_attempts
                )
                
                if should_recover:
                    print(f"\n⚠️ Performance drop detected! Attempting recovery (attempt {self.recovery_attempts + 1}/{self.max_recovery_attempts})")
                    self.restore_best_weights()
                    # Reduce exploration reset
                    self.steps_done = max(0, self.steps_done - EPSILON_DECAY // 8)  # Less aggressive reset
                    # Add minimal noise to weights
                    self._add_noise_to_weights(scale=0.01)  # Reduced noise
                    self.recovery_attempts += 1
                    self.last_recovery_episode = episode
                    
                    # Clear recent memory to avoid learning from poor experiences
                    self.memory = ReplayMemory(MEMORY_SIZE)
            
            # Reset recovery counter if performing well
            if episode_reward > (self.best_reward * 0.7):  # More lenient threshold
                self.recovery_attempts = max(0, self.recovery_attempts - 1)
            
            # Update progress bar
            if episode_reward > best_avg_reward:
                best_avg_reward = episode_reward
                pbar.set_description(f"Reward: {episode_reward:.1f} | Avg: {self.current_avg_reward:.1f} | Best: {best_avg_reward:.1f} 📊")
            else:
                pbar.set_description(f"Reward: {episode_reward:.1f} | Avg: {self.current_avg_reward:.1f} | Best: {best_avg_reward:.1f}")
            
            if self.current_avg_reward > 2500:
                print(f"\n🎉 Environment solved in {episode} episodes!")
                return True
        
        return False

    def load_model(self, checkpoint: dict) -> None:
        """Load pretrained model weights from checkpoint."""
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

    def play(self, episodes: int = 5):
        """Demonstrate trained agent"""
        self.actor.eval()
        # Add try-finally to ensure proper cleanup
        try:
            env = gym.make('Hopper-v5', render_mode='human')
            for episode in range(episodes):
                state, _ = env.reset()
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                total_reward = 0
                
                while True:
                    with torch.no_grad():
                        action = self.actor(state)
                    next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                    done = terminated or truncated
                    total_reward += reward
                    
                    env.render()  # Ensure each frame is rendered
                    
                    if done:
                        print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f}")
                        break
                        
                    state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
            
        finally:
            env.close()
        self.actor.train()

    def _add_noise_to_weights(self, scale: float = 0.02):
        """Add small noise to weights to escape local optima"""
        with torch.no_grad():
            for param in self.actor.parameters():
                noise = torch.randn_like(param.data) * scale * param.data.abs()
                param.data.add_(noise)
            for param in self.critic.parameters():
                noise = torch.randn_like(param.data) * scale * param.data.abs()
                param.data.add_(noise)

def main():
    print("\n🦘 Starting Hopper AI Training...\n")
    
    ai = HopperAI()
    global_best_reward = float('-inf')  # Track best reward across all phases
    
    # Training phases
    for phase in range(10):
        print(f"\n📈 Training Phase {phase + 1}")
        
        try:
            checkpoint = torch.load("gym/models/hopper_best.pth", weights_only=True)
            ai.load_model(checkpoint)
            global_best_reward = float(checkpoint.get('best_reward', float('-inf')))
            print(f"Loaded best model (previous best reward: {global_best_reward:.1f})")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No saved model found: {e}")
            print("Starting with untrained agent")
        
        # Pass global best reward to train
        success = ai.train(num_episodes=100, previous_best=global_best_reward)
        
        if success:
            print("🎯 Reached solution threshold! Moving to demonstration.")
            break
    
    # Load and play best model
    print("\n🦘 Loading best model for demonstration...")
    best_checkpoint = torch.load("gym/models/hopper_best.pth")
    ai.load_model(best_checkpoint)
    print("🦘 Watching the best hopper performance...")
    ai.play(episodes=20)  # Show multiple episodes of best performance

if __name__ == "__main__":
    main() 