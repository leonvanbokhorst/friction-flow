import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from time import sleep


class Personality:
    CAREFUL = "Careful Carl"
    WILD = "Wild Wallace"
    BALANCED = "Balanced Bob"


class PersonalityNetwork(nn.Module):
    def __init__(self, personality):
        super().__init__()
        self.personality = personality

        # Base network for all personalities
        self.network = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))

        # Personality traits
        if personality == Personality.CAREFUL:
            self.temperature = 0.7
        elif personality == Personality.WILD:
            self.temperature = 1.3
        else:
            self.temperature = 1.0

    def forward(self, x):
        return torch.softmax(self.network(x) / self.temperature, dim=-1)


def run_personality(personality, episodes=3):
    print(f"\n🤖 Watching {personality} in action!")
    env = gym.make("CartPole-v1", render_mode="human")
    policy = PersonalityNetwork(personality)

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}")
        while True:
            with torch.no_grad():  # No training, just watching
                state_tensor = torch.FloatTensor(state)
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()  # Take best action

                state, reward, done, truncated, _ = env.step(action)
                steps += 1
                episode_reward += reward

                # Add some personality commentary
                if steps % 50 == 0:
                    if personality == Personality.CAREFUL:
                        print("🎯 Staying cautious...")
                    elif personality == Personality.WILD:
                        print("🎪 Living on the edge!")
                    else:
                        print("😎 Keeping it smooth...")

                if done or truncated:
                    print(f"Lasted for {steps} steps!")
                    break

        sleep(1)  # Brief pause between episodes

    env.close()
    return steps


def personality_contest():
    print("🎪 Welcome to the CartPole Personality Contest! 🎪")
    print("Each personality will show off their style...")
    sleep(2)

    for personality in [Personality.CAREFUL, Personality.WILD, Personality.BALANCED]:
        run_personality(personality)
        print(f"\n{personality} has finished their performance!")
        sleep(1)  # Give a moment between personalities

    print("\n🎭 The show is complete! Who was your favorite?")


if __name__ == "__main__":
    personality_contest()
