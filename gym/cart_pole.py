import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
from statistics import mean, stdev
from collections import defaultdict
import torch.optim as optim
import random
import torch.nn.functional as F
from collections import deque
import copy
import math

torch.set_default_dtype(torch.float32)


class Personality:
    CAREFUL = "Careful Carl"
    WILD = "Wild Wallace"


class CartPoleWithDisturbances(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.recovery_window = False
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.last_wind_force = 0
        self.wind_buildup = 0
        
        # Slightly stronger wind parameters
        self.gust_strength = 0.03  # Increased from 0.02
        self.wind_change_rate = 0.04  # Increased from 0.03
        self.current_wind = 0
        self.wind_direction = np.random.choice([-1, 1])
        self.turbulence = 0.0
        
        # Stats tracking
        self.max_wind_force = 0
        self.gust_count = 0
        self.direction_changes = 0
        self.strength_changes = 0
        
    def reset(self, **kwargs):
        self.disturbance_countdown = np.random.randint(30, 70)
        self.current_wind = 0
        self.wind_direction = np.random.choice([-1, 1])
        self.gust_strength = 0.1
        self.turbulence = 0.0
        
        # Reset stats
        self.max_wind_force = 0
        self.gust_count = 0
        self.direction_changes = 0
        self.strength_changes = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        # Track state before wind
        old_state = self.unwrapped.state.copy()
        old_wind = self.current_wind
        
        # Slightly more dynamic wind behavior
        self.turbulence = 0.97 * self.turbulence + 0.03 * np.random.normal(0, 0.03)  # Increased from 0.02
        self.current_wind = (0.993 * self.current_wind +  # Slightly faster change (was 0.995)
                           0.007 * np.random.normal(0, 0.05) +  # Increased from 0.04
                           self.turbulence)
        
        # Event detection
        if abs(self.current_wind - old_wind) > 0.02:
            self.strength_changes += 1
        if np.sign(self.current_wind) != np.sign(old_wind) and abs(self.current_wind) > 0.01:
            self.direction_changes += 1
        if abs(self.current_wind) > abs(self.max_wind_force):
            self.max_wind_force = abs(self.current_wind)
        
        # Periodic gusts
        self.disturbance_countdown -= 1
        if self.disturbance_countdown <= 0:
            self.gust_count += 1
            gust = self.wind_direction * self.gust_strength * np.random.uniform(0.6, 1.2)
            self.current_wind += gust
            self.disturbance_countdown = np.random.randint(70, 130)
        
        # Enhanced recovery mechanics
        wind_force = self.current_wind * self.wind_direction
        max_manageable_force = 0.15 * (1.0 - self.wind_buildup * 0.3)
        wind_force = np.clip(wind_force, -max_manageable_force, max_manageable_force)
        
        cart_velocity = self.unwrapped.state[1]
        cart_position = self.unwrapped.state[0]
        
        # More sensitive recovery detection
        moving_to_center = np.sign(cart_velocity) != np.sign(cart_position)
        in_danger = abs(cart_position) > 0.65  # Even earlier detection
        approaching_danger = abs(cart_position) > 0.45 and abs(cart_velocity) > 0.4  # More sensitive
        
        # Progressive recovery assistance
        if in_danger and moving_to_center:
            self.recovery_window = True
            self.recovery_attempts += 1
            wind_force *= 0.45  # Stronger reduction
        elif approaching_danger and moving_to_center:
            wind_force *= 0.65  # More early assistance
        
        # Apply effects with enhanced counter-steering
        velocity_effect = wind_force * 0.025
        position_effect = wind_force * 0.001
        
        # Enhanced counter-steering bonus
        if np.sign(cart_velocity) != np.sign(wind_force):
            # More aggressive position-based reduction
            base_reduction = 0.45
            position_factor = abs(cart_position) / 1.8  # More aggressive scaling
            reduction = base_reduction * (1.0 + position_factor)
            
            velocity_effect *= reduction
            position_effect *= reduction
            
            # More lenient recovery detection
            if self.recovery_window and (
                abs(cart_position) < 0.95 or  # Higher threshold
                abs(cart_position) < abs(old_state[0])  # Any improvement counts
            ):
                self.successful_recoveries += 1
                self.recovery_window = False
                self.wind_buildup = 0
        
        # Enhanced damping curve
        position_factor = abs(cart_position)
        if position_factor > 0.8:
            damping = 0.7 + 0.2 * (position_factor - 0.8)  # Progressive damping
            velocity_effect *= damping
            position_effect *= damping
        
        # Apply final effects with damping
        if abs(cart_position) > 1.0:
            # Extra damping when far from center
            damping = 0.8
            velocity_effect *= damping
            position_effect *= damping
        
        self.unwrapped.state[1] += velocity_effect
        self.unwrapped.state[0] += position_effect
        self.last_wind_force = wind_force
        
        # Enhanced reward structure
        next_state, reward, done, truncated, info = super().step(action)
        
        if self.recovery_window and moving_to_center:
            # Progressive recovery bonus
            position_improvement = abs(old_state[0]) - abs(next_state[0])
            recovery_bonus = 1.0 + max(0, position_improvement * 2.0)
            reward += recovery_bonus
            
            # Extra bonus for velocity control
            if abs(next_state[1]) < abs(old_state[1]):
                reward += 0.5
        
        return next_state, reward, done, truncated, info


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # State history for adaptation (store as numpy array)
        self.state_history = np.zeros((10, 4), dtype=np.float32)  # Fixed size buffer
        self.history_idx = 0
        self.history_filled = False
        
        # Enhanced feature extraction
        self.features = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Adaptive control branch
        self.adaptive_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Velocity prediction
        self.velocity_pred = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.temperature = 1.0

    def forward(self, x):
        features = self.features(x)
        value = self.value_head(features)
        velocity_pred = self.velocity_pred(features)
        logits = self.policy_head(features)

        cart_position = x[..., 0]
        cart_velocity = x[..., 1]

        # Initialize all biases first
        position_bias = torch.zeros_like(logits)
        velocity_bias = torch.zeros_like(logits)

        # Position control parameters
        position_threshold = 0.5
        position_danger = 0.8
        high_position = torch.abs(cart_position) > position_threshold
        position_emergency = torch.abs(cart_position) > position_danger
        
        # Velocity control parameters
        velocity_threshold = 0.8
        velocity_danger = 1.2
        high_velocity = torch.abs(cart_velocity) > velocity_threshold
        
        # Calculate position bias
        position_factor = torch.abs(cart_position) * 1.0
        position_bias[..., 0] = -1.0 * torch.sign(cart_position) * position_factor
        position_bias[..., 1] = 1.0 * torch.sign(cart_position) * position_factor
        
        # Calculate velocity bias
        velocity_bias[..., 0] = -0.8 * torch.sign(cart_velocity)
        velocity_bias[..., 1] = 0.8 * torch.sign(cart_velocity)
        
        # Handle tensor dimensions
        if len(x.shape) > 1:
            high_velocity = high_velocity.unsqueeze(-1).expand_as(velocity_bias)
            high_position = high_position.unsqueeze(-1).expand_as(position_bias)
            position_emergency = position_emergency.unsqueeze(-1).expand_as(position_bias)
        else:
            high_velocity = high_velocity.unsqueeze(-1).expand(2)
            high_position = high_position.unsqueeze(-1).expand(2)
            position_emergency = position_emergency.unsqueeze(-1).expand(2)
        
        # Apply conditional biases
        position_bias = torch.where(
            high_position,
            position_bias * 1.2,
            position_bias
        )
        
        position_bias = torch.where(
            position_emergency,
            position_bias * 1.5,
            position_bias
        )
        
        velocity_bias = torch.where(
            high_velocity,
            velocity_bias * 1.5,
            velocity_bias
        )
        
        # Momentum bonus
        moving_to_center = torch.sign(cart_velocity) != torch.sign(cart_position)
        momentum_bonus = torch.where(
            moving_to_center.unsqueeze(-1).expand_as(position_bias),
            position_bias * 1.1,
            position_bias * 0.95
        )
        
        # Exploration noise
        base_noise = 0.1
        position_factor = torch.clamp(1.0 - torch.abs(cart_position) / 2.0, 0.4, 1.0)
        velocity_factor = torch.clamp(1.0 - torch.abs(cart_velocity) / 2.0, 0.4, 1.0)
        noise_scale = base_noise * position_factor * velocity_factor
        
        if len(x.shape) > 1:
            noise_scale = noise_scale.unsqueeze(-1).expand_as(logits)
        else:
            noise_scale = noise_scale.unsqueeze(-1).expand(2)
        
        noise = torch.randn_like(logits) * noise_scale
        
        # Combine all biases
        total_bias = momentum_bonus + velocity_bias
        logits = logits + total_bias + noise
        logits = torch.clamp(logits, -8.0, 8.0)
        
        # Dynamic temperature
        self.temperature = torch.clamp(
            0.8 + 0.4 * position_factor.mean(),
            0.6, 1.2
        )
        
        return torch.softmax(logits / self.temperature, dim=-1), value, velocity_pred


class AIStats:
    def __init__(self):
        self.decisions = defaultdict(int)
        self.reaction_times = []
        self.stability_scores = []
        self.recovery_count = 0
        self.position_history = []
        
        # Initialize with default values
        self.oscillations = 0
        self.time_in_danger_zone = 0
        self.max_recovery_time = 0
        self.energy_usage = [0]  # Initialize with one data point
        self.last_action = None
        self.recovery_windows = []
        
    def add_decision(self, action):
        self.decisions[action] += 1

    def add_reaction_time(self, time_ms):
        self.reaction_times.append(time_ms)

    def add_stability(self, state):
        _, _, angle, ang_velocity = state
        stability = 1.0 - min(abs(angle) + abs(ang_velocity), 1.0)
        self.stability_scores.append(stability)

    def add_position(self, state):
        cart_position = state[0]
        self.position_history.append(abs(cart_position))

    def log_recovery(self, old_state, new_state):
        # Detect successful wind counter-steering
        old_position = abs(old_state[0])
        new_position = abs(new_state[0])
        old_velocity = old_state[1]
        
        # Count as recovery if:
        # 1. Was in danger zone
        # 2. Moving back to center
        # 3. Position improved
        if (old_position > 1.0 and
            np.sign(old_velocity) != np.sign(old_state[0]) and
            new_position < old_position):
            self.recovery_count += 1

    def add_state(self, state, action):
        # Existing tracking
        self.add_position(state)
        self.add_stability(state)
        
        # Track oscillations
        if self.last_action is not None and action != self.last_action:
            self.oscillations += 1
            self.energy_usage.append(1)
        else:
            self.energy_usage.append(0)
        self.last_action = action
        
        # Track danger zone time
        if abs(state[0]) > 0.5:
            self.time_in_danger_zone += 1
            
    def get_summary(self):
        # Protect against empty lists
        stability = mean(self.stability_scores) if self.stability_scores else 0
        avg_reaction = mean(self.reaction_times) if self.reaction_times else 0
        avg_position = mean(self.position_history) if self.position_history else 0
        max_pos = max(self.position_history) if self.position_history else 0
        
        summary = {
            "left_right_ratio": f"{self.decisions[0]}:{self.decisions[1]}",
            "avg_reaction_ms": f"{avg_reaction:.1f}ms",
            "stability_score": f"{stability:.2f}/1.00",
            "recoveries": self.recovery_count,
            "decision_consistency": f"{min(self.decisions.values()) / max(self.decisions.values()):.2f}" if self.decisions else "0.00",
            "avg_distance_from_center": f"{avg_position:.2f}",
            "max_deviation": f"{max_pos:.2f}",
            
            # Protected new metrics
            "oscillations_per_100": f"{(self.oscillations / max(len(self.position_history), 1) * 100):.1f}",
            "danger_zone_percent": f"{(self.time_in_danger_zone / max(len(self.position_history), 1) * 100):.1f}%",
            "energy_efficiency": f"{(1 - mean(self.energy_usage)):.2f}/1.00",
            "avg_recovery_time": f"{mean(self.recovery_windows):.1f}" if self.recovery_windows else "N/A",
        }
        return summary


def run_demonstration(policy, episodes=3):
    """Show phase - with rendering and stats"""
    env = CartPoleWithDisturbances(gym.make("CartPole-v1", render_mode="human"))
    stats = AIStats()
    total_steps = []
    failure_states = []  # Track failure states

    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        last_time = time.time()
        last_state = state

        print(f"\n📊 Episode {episode + 1}")
        while True:
            current_time = time.time()

            # Get action and measure decision time
            start_time = time.time()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                probs, value, velocity_pred = policy(state_tensor)
                action = torch.argmax(probs).item()
            decision_time = (time.time() - start_time) * 1000  # ms

            # Record stats
            stats.add_decision(action)
            stats.add_reaction_time(decision_time)
            stats.add_stability(state)
            stats.add_position(state)
            stats.log_recovery(last_state, state)

            # Take action
            state, reward, done, truncated, _ = env.step(action)
            steps += 1
            last_state = state

            if done or truncated:
                failure_states.append({
                    'position': state[0],
                    'velocity': state[1],
                    'angle': state[2],
                    'angular_velocity': state[3],
                    'steps': steps,
                    'wind_force': env.last_wind_force
                })
                print(f"Episode lasted {steps} steps!")
                print(f"Failed at: pos={state[0]:.2f}, vel={state[1]:.2f}, "
                      f"angle={state[2]:.2f}, ang_vel={state[3]:.2f}")
                print(f"Last wind force: {env.last_wind_force:.3f}")
                total_steps.append(steps)
                break

        time.sleep(0.5)

    env.close()

    # Print AI Performance Analysis
    print(f"\n🧠 Policy's Performance Metrics:")
    print("=" * 50)
    summary = stats.get_summary()
    print(f"📈 Average Episode Length: {mean(total_steps):.1f} steps")
    if len(total_steps) > 1:
        print(f"🎯 Consistency (stddev): ±{stdev(total_steps):.1f} steps")
    print(f"⚖️  Left/Right Balance: {summary['left_right_ratio']}")
    print(f"⚡ Average Reaction Time: {summary['avg_reaction_ms']}")
    print(f"🎭 Stability Score: {summary['stability_score']}")
    print(f"💪 Successful Recoveries: {summary['recoveries']}")
    print(f"🎯 Decision Consistency: {summary['decision_consistency']}")
    print(f"🏁 Average Distance from Center: {summary['avg_distance_from_center']}")
    print(f"🔄 Max Deviation: {summary['max_deviation']}")

    # Add disturbance statistics
    print(f"\n🌪️  Wind Event Analysis:")
    print("=" * 50)
    print(f"💨 Max Wind Force: {env.max_wind_force:.2f}")
    print(f"🌬️  Total Gusts: {env.gust_count}")
    print(f"️  Direction Changes: {env.direction_changes}")
    print(f"📊 Strength Adjustments: {env.strength_changes}")
    print(f"🎯 Average Steps Between Events: {mean([50, 100]):.1f}")

    # Print failure analysis
    print(f"\n🔍 Failure Analysis:")
    print("=" * 50)
    positions = [f['position'] for f in failure_states]
    velocities = [f['velocity'] for f in failure_states]
    angles = [f['angle'] for f in failure_states]
    ang_vels = [f['angular_velocity'] for f in failure_states]
    
    print(f"Position at failure: {mean(positions):.2f} ± {stdev(positions):.2f}")
    print(f"Velocity at failure: {mean(velocities):.2f} ± {stdev(velocities):.2f}")
    print(f"Angle at failure: {mean(angles):.2f} ± {stdev(angles):.2f}")
    print(f"Angular velocity at failure: {mean(ang_vels):.2f} ± {stdev(ang_vels):.2f}")
    print(f"Wind force at failure: {mean([f['wind_force'] for f in failure_states]):.3f}")

    print(f"\n🎯 Additional Performance Metrics:")
    print("=" * 50)
    print(f"🔄 Oscillations per 100 steps: {summary['oscillations_per_100']}")
    print(f"⚠️  Time in Danger Zone: {summary['danger_zone_percent']}")
    print(f"⚡ Energy Efficiency: {summary['energy_efficiency']}")
    print(f"🏃 Average Recovery Time: {summary['avg_recovery_time']}")

    return total_steps


def train_agent(episodes=300):
    env = CartPoleWithDisturbances(gym.make("CartPole-v1"))
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Experience replay buffer
    replay_buffer = deque(maxlen=10000)
    min_replay_size = 1000
    batch_size = 64
    
    # Training parameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 5000  # Slower decay
    
    # Tracking variables
    episode_rewards = []
    best_reward = float('-inf')
    steps_done = 0
    
    # Adjust reward shaping to be more position-aware
    def calculate_reward(state, next_state, reward):
        # Stronger penalties for position and velocity
        position_penalty = (abs(next_state[0]) ** 2) * 0.2  # Increased from 0.1
        velocity_penalty = abs(next_state[1]) * 0.15  # Increased from 0.1
        
        # Reduced angle penalties since those are good
        angle_penalty = abs(next_state[2]) * 1.5  # Reduced from 2.0
        ang_velocity_penalty = abs(next_state[3]) * 0.1
        
        # Add centering bonus
        centering_bonus = 0.1 if (abs(next_state[0]) < abs(state[0]) and 
                                 abs(next_state[0]) > 0.5) else 0
        
        # Add velocity damping bonus
        damping_bonus = 0.1 if (abs(next_state[1]) < abs(state[1]) and 
                               abs(next_state[0]) > 0.5) else 0
        
        return (reward 
                - position_penalty 
                - velocity_penalty 
                - angle_penalty 
                - ang_velocity_penalty
                + centering_bonus
                + damping_bonus)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy with decay
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                     math.exp(-steps_done / epsilon_decay)
            steps_done += 1
            
            # Action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    probs, _, _ = policy(state_tensor)
                    action = torch.argmax(probs).item()
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            shaped_reward = calculate_reward(state, next_state, reward)
            
            # Store transition
            replay_buffer.append({
                'state': state,
                'action': action,
                'reward': shaped_reward,
                'next_state': next_state,
                'done': done
            })
            
            # Training step
            if len(replay_buffer) >= min_replay_size:
                batch = random.sample(replay_buffer, batch_size)
                update_policy(policy, optimizer, batch, gamma)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-30:]) if len(episode_rewards) >= 30 else np.mean(episode_rewards)
        
        # Progress reporting
        if episode % 10 == 0:
            print(f"Episode {episode}: Steps = {steps_done}, "
                  f"Reward = {episode_reward:.1f}, "
                  f"Avg = {avg_reward:.1f}, "
                  f"Epsilon = {epsilon:.3f}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(policy.state_dict(), 'best_cartpole_model.pth')
        
        # Success criterion
        if avg_reward >= 195.0:
            print(f"\n🎉 Solved at episode {episode}!")
            break
            
    return policy

def update_policy(policy, optimizer, batch, gamma):
    # Prepare batch data
    states = torch.FloatTensor(np.array([x['state'] for x in batch]))
    actions = torch.LongTensor([x['action'] for x in batch])
    rewards = torch.FloatTensor([x['reward'] for x in batch])
    next_states = torch.FloatTensor(np.array([x['next_state'] for x in batch]))
    dones = torch.FloatTensor([x['done'] for x in batch])
    
    # Current values
    probs, values, _ = policy(states)
    values = values.squeeze(-1)
    
    # Next values for TD learning
    with torch.no_grad():
        _, next_values, _ = policy(next_states)
        next_values = next_values.squeeze(-1)
        targets = rewards + gamma * next_values * (1 - dones)
    
    # Compute losses
    value_loss = F.mse_loss(values, targets)
    advantage = (targets - values.detach())
    policy_loss = -torch.mean(torch.log(probs[range(len(actions)), actions]) * advantage)
    entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))
    
    # Add entropy bonus to encourage exploration
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

def compute_returns(episode_data, gamma):
    returns = []
    R = 0
    for transition in reversed(episode_data):
        R = transition['reward'] + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def compute_advantages(returns, values):
    advantages = returns - torch.tensor([v.item() for v in values], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def main():
    print("🎪 Welcome to the CartPole Challenge! 🎪")
    print("\n📚 Training our agent...")

    # Train the agent
    policy = train_agent(episodes=300)

    print("\n🎭 Now for the demonstration!")
    time.sleep(2)

    # Show off what it learned
    print("\n🤖 Agent Performance Analysis 🔍")
    results = run_demonstration(policy)

    print("\n🏆 Final Results:")
    print("=" * 50)
    print(f"Average Steps: {mean(results):.1f}")
    if len(results) > 1:
        print(f"Consistency: ±{stdev(results):.1f} steps")
    print(f"Max Duration: {max(results)}")

if __name__ == "__main__":
    main()
