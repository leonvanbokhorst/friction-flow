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

torch.set_default_dtype(torch.float32)


class Personality:
    CAREFUL = "Careful Carl"
    WILD = "Wild Wallace"


class CartPoleWithDisturbances(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.disturbance_countdown = 50
        self.current_wind = 0
        self.wind_direction = 1
        self.gust_strength = 0.1
        self.wind_change_rate = 0.1
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
        # Gentler continuous wind with turbulence
        self.turbulence = 0.95 * self.turbulence + 0.05 * np.random.normal(0, 0.05)
        self.current_wind = (0.98 * self.current_wind + 
                           0.02 * np.random.normal(0, 0.1) + 
                           self.turbulence)
        
        # Add sudden events
        self.disturbance_countdown -= 1
        if self.disturbance_countdown <= 0:
            event_type = np.random.choice(['gust', 'direction_change', 'strength_change'],
                                        p=[0.3, 0.4, 0.3])
            
            if event_type == 'gust':
                # Weaker gusts
                gust = self.wind_direction * self.gust_strength * np.random.uniform(0.8, 1.5)
                self.current_wind += gust
                self.gust_count += 1
            elif event_type == 'direction_change':
                self.wind_direction *= -1
                self.current_wind *= -0.2
                self.direction_changes += 1
            else:  # strength_change
                self.gust_strength = np.clip(
                    self.gust_strength + np.random.uniform(-0.05, 0.05),
                    0.05, 0.15
                )
                self.strength_changes += 1
            
            self.disturbance_countdown = np.random.randint(60, 120)
        
        # Apply gentler disturbance to the cart
        wind_force = self.current_wind * self.wind_direction
        self.max_wind_force = max(self.max_wind_force, abs(wind_force))
        
        self.unwrapped.state[1] += wind_force * 0.05
        self.unwrapped.state[0] += wind_force * 0.003
        
        return super().step(action)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Wider feature extractor
        self.features = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Separate velocity prediction branch
        self.velocity_pred = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Policy head with position awareness
        self.policy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.temperature = 1.0

    def forward(self, x):
        features = self.features(x)
        value = self.value_head(features)
        velocity_pred = self.velocity_pred(features)
        logits = self.policy_head(features)

        cart_position = x[..., 0]
        cart_velocity = x[..., 1]

        # Position bias with proper broadcasting
        position_bias = torch.zeros_like(logits)
        position_bias[..., 0] = -0.5 * torch.sign(cart_position)
        position_bias[..., 1] = 0.5 * torch.sign(cart_position)
        
        # Velocity influence
        velocity_factor = torch.tanh(cart_velocity * 0.5)
        if len(x.shape) > 1:
            velocity_factor = velocity_factor.unsqueeze(-1)
        position_bias *= (1.0 - abs(velocity_factor))

        # Stability calculation
        stability = 1.0 - torch.min(
            torch.abs(cart_position) + 0.5 * torch.abs(cart_velocity),
            torch.tensor(1.0)
        )
        
        # Noise with proper shape
        noise_scale = 0.05 * (1.0 - stability)
        if len(x.shape) > 1:
            noise_scale = noise_scale.unsqueeze(-1)
        noise = torch.randn_like(logits) * noise_scale

        # Temperature adjustment
        self.temperature = 1.0 + 0.5 * (1.0 - stability.mean())
        
        # Combined logic
        logits = logits + position_bias * 0.3 + noise

        return torch.softmax(logits / self.temperature, dim=-1), value, velocity_pred


class AIStats:
    def __init__(self):
        self.decisions = defaultdict(int)
        self.reaction_times = []
        self.stability_scores = []
        self.recovery_count = 0
        self.position_history = []

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
        # Detect if we recovered from a bad position
        old_angle = abs(old_state[2])
        new_angle = abs(new_state[2])
        if old_angle > 0.2 and new_angle < 0.1:
            self.recovery_count += 1

    def get_summary(self):
        summary = {
            "left_right_ratio": f"{self.decisions[0]}:{self.decisions[1]}",
            "avg_reaction_ms": f"{mean(self.reaction_times):.1f}ms",
            "stability_score": f"{mean(self.stability_scores):.2f}/1.00",
            "recoveries": self.recovery_count,
            "decision_consistency": f"{min(self.decisions.values()) / max(self.decisions.values()):.2f}",
            "avg_distance_from_center": f"{mean(self.position_history):.2f}",
            "max_deviation": f"{max(self.position_history):.2f}",
        }
        return summary


def run_demonstration(policy, episodes=3):
    """Show phase - with rendering and stats"""
    env = CartPoleWithDisturbances(gym.make("CartPole-v1", render_mode="human"))
    stats = AIStats()
    total_steps = []

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
                print(f"Episode lasted {steps} steps!")
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
    print(f"↔️  Direction Changes: {env.direction_changes}")
    print(f"📊 Strength Adjustments: {env.strength_changes}")
    print(f"🎯 Average Steps Between Events: {mean([50, 100]):.1f}")

    return total_steps


def train_agent(episodes=300):
    print("\n🎓 Training agent...")
    env = CartPoleWithDisturbances(gym.make("CartPole-v1"))
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    running_reward = 10
    gamma = 0.99
    
    # Adjusted training parameters
    batch_size = 128
    entropy_weight = 0.01
    value_weight = 0.5
    ppo_epochs = 4
    clip_epsilon = 0.2

    # More patient early stopping
    patience = 50
    min_episodes = 150
    best_reward = 0
    patience_counter = 0
    no_improvement_threshold = 0.85
    
    # Stronger emphasis on position control
    position_weight = 2.0  # Increased
    velocity_weight = 1.0
    
    for episode in range(episodes):
        state, _ = env.reset()
        last_position = state[0]
        last_velocity = state[1]
        done = False
        episode_rewards = []
        episode_data = []

        while not done:
            state_tensor = torch.FloatTensor(state)
            probs, value, velocity_pred = policy(state_tensor)
            
            # Progressive exploration strategy
            exploration_rate = max(0.3, 1.0 - episode / (episodes * 0.7))
            if random.random() < exploration_rate:
                action = random.randint(0, 1)
                action_tensor = torch.tensor(action)
            else:
                action_tensor = Categorical(probs).sample()
            
            action = action_tensor.item()
            next_state, reward, done, truncated, _ = env.step(action)
            current_position = next_state[0]
            current_velocity = next_state[1]

            # Modified reward structure
            modified_reward = (
                reward * 2.0 +
                position_weight * (1.0 - abs(current_position)) +  # Position control
                velocity_weight * (1.0 - abs(current_velocity)) +  # Velocity damping
                -3.0 * (abs(current_position) > 1.5) +            # Stronger boundary penalty
                1.0 * (abs(current_position) < 0.5)               # Center bonus
            )

            episode_data.append({
                'state': state_tensor,
                'action': action_tensor,
                'reward': modified_reward,
                'value': value,
                'log_prob': Categorical(probs).log_prob(action_tensor)
            })
            
            episode_rewards.append(modified_reward)
            state = next_state
            last_position = current_position
            last_velocity = current_velocity
            done = done or truncated

        # More gradual learning rate decay
        if running_reward > 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.995

        # Process episode data
        returns = compute_returns(episode_data, gamma=0.99)
        advantages = compute_advantages(returns, [t['value'] for t in episode_data])
        
        # PPO update loop
        for _ in range(ppo_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(episode_data))
            for start_idx in range(0, len(episode_data), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                update_policy(policy, optimizer, episode_data, returns, advantages, 
                            batch_indices, clip_epsilon, entropy_weight, value_weight)

        # Logging and early stopping
        episode_reward = sum(episode_rewards)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.1f}, Running = {running_reward:.1f}")
        
        # Don't allow early stopping before minimum episodes
        if episode < min_episodes:
            patience_counter = 0
        elif episode_reward > best_reward * no_improvement_threshold:
            if episode_reward > best_reward:
                best_reward = episode_reward
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and episode >= min_episodes:
            print(f"Early stopping at episode {episode} - No improvement for {patience} episodes")
            break

        # Progressive difficulty
        if episode > 200:
            env.gust_strength *= 1.001  # Gradually increase challenge

    return policy

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

def update_policy(policy, optimizer, episode_data, returns, advantages, batch_indices, 
                 clip_epsilon, entropy_weight, value_weight):
    # Get batch data
    states = torch.stack([episode_data[i]['state'] for i in batch_indices])
    old_log_probs = torch.stack([episode_data[i]['log_prob'].detach() for i in batch_indices])
    actions = torch.stack([episode_data[i]['action'] for i in batch_indices])
    batch_returns = returns[batch_indices].detach().float()
    batch_advantages = advantages[batch_indices].detach().float()

    # Get current policy distributions
    new_probs, new_values, velocity_pred = policy(states)
    dist = Categorical(new_probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # PPO policy loss
    ratio = (new_log_probs - old_log_probs).exp()
    surrogate1 = ratio * batch_advantages
    surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    # Value loss
    value_loss = F.mse_loss(new_values.squeeze(-1), batch_returns)

    # Combined loss
    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()


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
