import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ollama import Client
import asyncio
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SocialAgent(nn.Module):
    def __init__(
        self,
        initial_bias,
        susceptibility,
        stubbornness,
        is_influencer=False,
        confirmation_bias_strength=0.5,
    ):
        """
        Initialize a SocialAgent with psychological and social attributes.

        Args:
            initial_bias (float): The initial bias level of the agent.
            susceptibility (float): How susceptible the agent is to influence.
            stubbornness (float): How resistant the agent is to change.
            is_influencer (bool): Whether the agent is an influencer.
            confirmation_bias_strength (float): Strength of confirmation bias.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float))
        self.susceptibility = susceptibility
        self.stubbornness = stubbornness
        self.is_influencer = is_influencer
        self.confirmation_bias_strength = confirmation_bias_strength
        self.interaction_history = []
        self.memory_length = 10
        self.emotional_state = nn.Parameter(torch.rand(1))

        # Updated input size to 5 to match all our inputs
        self.belief_network = nn.Sequential(
            nn.Linear(5, 12),  # Changed from 4 to 5 inputs
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Tanh(),
        )

        # Initialize weights
        for layer in self.belief_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Add trauma memory and emotional contagion parameters
        self.trauma_memories = []  # Store traumatic events
        self.trauma_decay_rate = 0.95  # How quickly trauma fades
        self.emotional_contagion_rate = 0.3
        self.is_bridge_builder = False  # New flag for bridge builders
        self.group_memberships = []  # Track multiple group memberships

        # Add bias influence tracking
        self.bias_influence_history = []
        self.received_influences = []
        self.influence_strength = 0.0  # Track how much this agent influences others

        # Add projection layer for attention
        self.projection = nn.Linear(5, 12)  # Project from 5 features to 12 dimensions

        # Add attention mechanism for selective influence
        self.attention = nn.MultiheadAttention(embed_dim=12, num_heads=4)

        # Add uncertainty quantification
        self.uncertainty = nn.Parameter(torch.rand(1))

    def calculate_confirmation_bias(self, social_influence):
        """
        Calculate the confirmation bias effect on social influence acceptance.

        Args:
            social_influence (float): The influence from social interactions.

        Returns:
            float: The acceptance rate after applying confirmation bias.
        """
        bias_difference = abs(self.bias.item() - social_influence)
        acceptance_rate = 1.0 - (bias_difference * self.confirmation_bias_strength)
        return max(0.1, min(1.0, acceptance_rate))  # Clamp between 0.1 and 1.0

    def update_emotional_state(
        self, social_influence, neighboring_emotions=None, crisis_intensity=0
    ):
        """
        Update the emotional state of the agent based on various factors.

        Args:
            social_influence (float): Influence from social interactions.
            neighboring_emotions (torch.Tensor, optional): Emotions of neighbors.
            crisis_intensity (float, optional): Intensity of a crisis event.
        """
        # Update emotion based on social influence, neighbors, and crisis
        delta = abs(self.bias.item() - social_influence)

        # Factor in emotional contagion from neighbors
        emotional_influence = 0
        if neighboring_emotions is not None:
            emotional_influence = (
                torch.mean(neighboring_emotions) - self.emotional_state
            ) * self.emotional_contagion_rate

        # Factor in crisis impact
        crisis_impact = crisis_intensity * (1 - self.stubbornness)

        self.emotional_state.data = torch.clamp(
            self.emotional_state
            - 0.1 * delta
            + emotional_influence
            + crisis_impact
            + 0.05 * torch.rand(1),
            0,
            1,
        )

        # Store trauma if crisis is significant
        if crisis_intensity > 0.5:
            self.trauma_memories.append((crisis_intensity, 0))  # (intensity, age)

    def forward(self, social_influence, new_information, external_influence):
        """
        Forward pass to update agent's bias based on inputs.

        Args:
            social_influence (float): Influence from social interactions.
            new_information (float): New information affecting the agent.
            external_influence (float): External influence on the agent.

        Returns:
            torch.Tensor: The decision output from the belief network.
        """
        self.update_emotional_state(social_influence)
        # Modify susceptibility based on emotional state
        effective_susceptibility = self.susceptibility * (
            1 + 0.5 * self.emotional_state.item()
        )

        # Calculate confirmation bias effect
        confirmation_factor = self.calculate_confirmation_bias(social_influence)

        # Add memory effect
        self.interaction_history.append(social_influence)
        if len(self.interaction_history) > self.memory_length:
            self.interaction_history.pop(0)

        weighted_history = sum(
            [x * (0.9**i) for i, x in enumerate(reversed(self.interaction_history))]
        ) / len(self.interaction_history)

        # Prepare inputs including confirmation bias
        inputs = torch.tensor(
            [
                self.bias.item(),
                social_influence * confirmation_factor,  # Apply confirmation bias
                new_information,
                external_influence,
                weighted_history,  # Add historical influence
            ],
            dtype=torch.float,
        )

        # Project inputs to higher dimension for attention
        current_state = inputs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 5]
        current_state = self.projection(current_state)  # Shape: [1, 1, 12]

        # Apply attention mechanism
        attention_output = self.attention(
            query=current_state, key=current_state, value=current_state
        )[0]

        # Get decision from belief network
        decision = self.belief_network(inputs)

        # Influencers are more resistant to change
        if not self.is_influencer:
            with torch.no_grad():
                bias_update = (
                    (decision.item() - self.bias.item())
                    * effective_susceptibility
                    * (1 - self.stubbornness)
                )
                self.bias.add_(bias_update)

        return decision

    def calculate_bias_impact(self, other_bias):
        """
        Calculate the impact of this agent's bias on another agent.

        Args:
            other_bias (float): The bias of another agent.

        Returns:
            float: The calculated impact.
        """
        bias_diff = abs(self.bias.item() - other_bias)
        impact = (1 - bias_diff) * self.influence_strength
        return impact

    def record_influence(self, influence_value, source_bias):
        """
        Record the influence received and its source.

        Args:
            influence_value (float): The value of the influence received.
            source_bias (float): The bias of the source of influence.
        """
        self.received_influences.append(
            {
                "value": influence_value,
                "source_bias": source_bias,
                "time": len(self.bias_influence_history),
            }
        )

    def analyze_individual_bias(self):
        """
        Analyze the bias patterns of individual agents.

        Returns:
            dict: A dictionary containing bias-related metrics.
        """
        return {
            'bias_values': [float(agent.bias.item()) for agent in self.agents],
            'susceptibility': [float(agent.susceptibility) for agent in self.agents],
            'confirmation_bias': [float(agent.confirmation_bias_strength) for agent in self.agents],
            'emotional_state': [float(agent.emotional_state.item()) for agent in self.agents]
        }


class SocialNetwork:
    def __init__(
        self,
        num_agents=20,
        num_influencers=2,
        num_echo_chambers=2,
        num_bridge_builders=2,
    ):
        """
        Initialize a SocialNetwork with agents and their connections.

        Args:
            num_agents (int): Total number of agents in the network.
            num_influencers (int): Number of influencer agents.
            num_echo_chambers (int): Number of echo chambers.
            num_bridge_builders (int): Number of bridge builders.
        """
        self.num_echo_chambers = num_echo_chambers
        self.num_bridge_builders = num_bridge_builders
        self.num_base_agents = num_agents  # Store original number of agents
        self.total_agents = num_agents + num_bridge_builders
        self.agents = []
        self.external_influences = torch.zeros(self.total_agents)
        self.group_identities = torch.zeros(self.total_agents, num_echo_chambers)

        # Initialize connection matrix with final size
        self.connections = torch.zeros((self.total_agents, self.total_agents))

        # Create regular agents and influencers
        agents_per_chamber = (num_agents - num_influencers) // num_echo_chambers

        # Create regular agents in echo chambers
        for chamber in range(num_echo_chambers):
            chamber_bias = 1.0 if chamber == 0 else -1.0
            for _ in range(agents_per_chamber):
                initial_bias = np.random.normal(chamber_bias, 0.2)
                agent = SocialAgent(
                    initial_bias=initial_bias,
                    susceptibility=np.random.uniform(0.1, 0.9),
                    stubbornness=np.random.uniform(0.1, 0.9),
                    confirmation_bias_strength=np.random.uniform(0.3, 0.7),
                )
                self.agents.append(agent)

        # Create influencers
        for i in range(num_influencers):
            influencer_bias = 1.0 if i == 0 else -1.0
            influencer = SocialAgent(
                initial_bias=influencer_bias,
                susceptibility=0.1,
                stubbornness=0.9,
                is_influencer=True,
                confirmation_bias_strength=0.8,
            )
            self.agents.append(influencer)

        # Create initial echo chamber connections
        self.create_echo_chamber_connections(
            num_agents, agents_per_chamber, num_influencers
        )

        # Initialize crisis parameters
        self.current_crisis = None
        self.crisis_history = []

        # Add bridge builders
        self.add_bridge_builders(num_bridge_builders)

        # Add bias tracking metrics
        self.bias_spread_history = []
        self.chamber_polarization = []
        self.influence_network = nx.DiGraph()

        # Add metrics tracking
        self.metrics = {
            "entropy": [],
            "clustering_coefficient": [],
            "opinion_diversity": [],
            "chamber_metrics": {}  # Add this for group dynamics tracking
        }

        # Add bias tracking
        self.bias_history = []
        self.crisis_start = None
        self.crisis_end = None

    def add_bridge_builder_connections(self, builder_idx, chambers):
        """
        Add connections for bridge builders to selected chambers.

        Args:
            builder_idx (int): Index of the bridge builder agent.
            chambers (list): List of chamber indices to connect.
        """
        agents_per_chamber = (self.num_base_agents - 2) // self.num_echo_chambers

        # For each chamber this bridge builder connects to
        for chamber in chambers:
            chamber_start = chamber * agents_per_chamber
            chamber_end = chamber_start + agents_per_chamber

            # Connect to agents in the chamber
            for i in range(chamber_start, chamber_end):
                # Add bidirectional connections
                connection_strength = np.random.uniform(0.3, 0.7)
                self.connections[builder_idx, i] = connection_strength
                self.connections[i, builder_idx] = connection_strength

        # Renormalize connections
        row_sums = self.connections.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.connections = self.connections / row_sums

    def create_echo_chamber_connections(
        self, num_agents, agents_per_chamber, num_influencers
    ):
        """
        Create connections within and between echo chambers.

        Args:
            num_agents (int): Total number of agents.
            agents_per_chamber (int): Number of agents per chamber.
            num_influencers (int): Number of influencer agents.
        """
        # Only create connections for the base agents initially
        for chamber in range(self.num_echo_chambers):
            start_idx = chamber * agents_per_chamber
            end_idx = start_idx + agents_per_chamber

            # Connect agents within chambers
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        self.connections[i, j] = np.random.uniform(0.5, 1.0)

        # Add weak connections between chambers
        for i in range(num_agents - num_influencers):
            for j in range(num_agents - num_influencers):
                if i // agents_per_chamber != j // agents_per_chamber:
                    if np.random.random() < 0.1:
                        self.connections[i, j] = np.random.uniform(0.0, 0.2)

        # Connect influencers
        influencer_indices = range(num_agents - num_influencers, num_agents)
        for inf_idx in influencer_indices:
            for i in range(num_agents):
                if i != inf_idx:
                    chamber_influence = (
                        1.0 if inf_idx % 2 == i // agents_per_chamber else 0.5
                    )
                    self.connections[i, inf_idx] = (
                        np.random.uniform(0.3, 0.8) * chamber_influence
                    )

        # Normalize connections
        row_sums = self.connections.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        self.connections = self.connections / row_sums

    def trigger_crisis(self, intensity, duration, target_chamber=None):
        """
        Initiate a crisis event affecting the network.

        Args:
            intensity (float): Intensity of the crisis.
            duration (int): Duration of the crisis in time steps.
            target_chamber (int, optional): Specific chamber affected.
        """
        self.current_crisis = {
            "intensity": intensity,
            "duration": duration,
            "target_chamber": target_chamber,
            "remaining": duration,
        }
        self.crisis_history.append(self.current_crisis)

    def simulate_step(self, time_step):
        """
        Perform a simulation step, updating agent states.

        Args:
            time_step (int): The current time step in the simulation.

        Returns:
            torch.Tensor: Updated biases of all agents.
        """
        current_biases = torch.tensor([agent.bias.item() for agent in self.agents])
        current_emotions = torch.tensor(
            [agent.emotional_state.item() for agent in self.agents]
        )

        # Handle crisis events
        crisis_intensity = 0
        if self.current_crisis:
            crisis_intensity = self.current_crisis["intensity"]
            self.current_crisis["remaining"] -= 1
            if self.current_crisis["remaining"] <= 0:
                self.current_crisis = None

        # Update agent states
        new_biases = []
        for i, agent in enumerate(self.agents):
            # Calculate neighboring emotions
            neighboring_emotions = current_emotions[self.connections[i] > 0]

            # Calculate social influence
            social_influence = (self.connections[i] * current_biases).sum()

            # Update agent with crisis and emotional contagion
            decision = agent(
                social_influence,
                new_information=0.0,
                external_influence=self.external_influences[i],
            )
            agent.update_emotional_state(
                social_influence,
                neighboring_emotions=neighboring_emotions,
                crisis_intensity=crisis_intensity,
            )

            new_biases.append(agent.bias.item())

        return torch.tensor(new_biases)

    def run_simulation(self, steps=100):
        """
        Run the simulation for a specified number of steps.

        Args:
            steps (int): Number of simulation steps to run.

        Returns:
            torch.Tensor: History of biases over time.
        """
        bias_history = []
        self.bias_history = []  # Store for crisis analysis
        
        for step in range(steps):
            if self.current_crisis and step == self.current_crisis['remaining']:
                self.crisis_start = step
            if self.current_crisis and step == self.current_crisis['remaining'] + self.current_crisis['duration']:
                self.crisis_end = step
                
            new_biases = self.simulate_step(step)
            bias_history.append(new_biases.clone())
            self.bias_history.append(new_biases.tolist())
            
        return torch.stack(bias_history)

    def visualize_network(self):
        """
        Visualize the social network structure using networkx.
        """
        G = nx.DiGraph()

        # Add nodes
        for i in range(len(self.agents)):
            G.add_node(
                i,
                bias=self.agents[i].bias.item(),
                influencer=self.agents[i].is_influencer,
            )

        # Add edges (connections)
        for i in range(len(self.agents)):
            for j in range(len(self.agents)):
                weight = self.connections[i, j].item()
                if weight > 0.01:  # Only add significant connections
                    G.add_edge(i, j, weight=weight)

        # Create plot with specific axes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate network layout
        pos = nx.spring_layout(G)

        # Draw nodes
        node_colors = [G.nodes[n]["bias"] for n in G.nodes()]
        influencer_sizes = [
            1000 if G.nodes[n]["influencer"] else 300 for n in G.nodes()
        ]

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=influencer_sizes,
            cmap=plt.cm.RdYlBu,
            ax=ax,
        )

        # Draw edges with varying thickness based on weight
        edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, ax=ax)

        plt.title(
            "Social Network Structure\nNode color = bias, Size = influence, Edge thickness = connection strength"
        )

        # Add colorbar with proper normalization
        norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
        plt.colorbar(sm, ax=ax)

        plt.show()

    def plot_simulation(self, bias_history):
        """
        Plot the evolution of agent biases over time.

        Args:
            bias_history (torch.Tensor): History of biases over time.
        """
        plt.figure(figsize=(15, 8))

        # Calculate agents per chamber
        regular_agents = len(self.agents) - 2  # Subtract influencers
        agents_per_chamber = regular_agents // self.num_echo_chambers

        # Plot regular agents
        for chamber in range(self.num_echo_chambers):
            start_idx = chamber * agents_per_chamber
            end_idx = start_idx + agents_per_chamber

            for i in range(start_idx, end_idx):
                plt.plot(
                    bias_history[:, i].numpy(),
                    alpha=0.4,
                    color="red" if chamber == 0 else "blue",
                    label=f"Chamber {chamber + 1}" if i == start_idx else "",
                )

        # Plot influencers with distinct style
        for i in range(-2, 0):  # Last two agents are influencers
            plt.plot(
                bias_history[:, i].numpy(),
                linewidth=3,
                linestyle="--",
                label=f"Influencer {abs(i)}",
            )

        plt.xlabel("Time Steps")
        plt.ylabel("Bias Level")
        plt.title(
            "Evolution of Agent Biases Over Time\nSolid lines = regular agents, Dashed lines = influencers"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def add_bridge_builders(self, num_bridge_builders):
        """
        Add bridge builder agents to the network.

        Args:
            num_bridge_builders (int): Number of bridge builders to add.
        """
        for i in range(num_bridge_builders):
            # Select random chambers to bridge
            chambers = np.random.choice(self.num_echo_chambers, 2, replace=False)

            # Create bridge builder agent
            agent = SocialAgent(
                initial_bias=0.0,  # Start neutral
                susceptibility=np.random.uniform(0.4, 0.8),
                stubbornness=np.random.uniform(0.2, 0.5),
                confirmation_bias_strength=0.3,  # More open to different views
            )
            agent.is_bridge_builder = True
            agent.group_memberships = chambers.tolist()
            self.agents.append(agent)

            # Add connections for this bridge builder
            self.add_bridge_builder_connections(len(self.agents) - 1, chambers)

    def analyze_individual_bias(self):
        """
        Analyze individual agent bias patterns.

        Returns:
            dict: A dictionary containing bias-related metrics.
        """
        return {
            'bias_values': [float(agent.bias.item()) for agent in self.agents],
            'susceptibility': [float(agent.susceptibility) for agent in self.agents],
            'confirmation_bias': [float(agent.confirmation_bias_strength) for agent in self.agents],
            'emotional_state': [float(agent.emotional_state.item()) for agent in self.agents]
        }

    def analyze_bias_evolution(self, bias_history):
        """
        Analyze how biases change over time.

        Args:
            bias_history (torch.Tensor): History of biases over time.

        Returns:
            dict: A dictionary containing temporal metrics.
        """
        return {
            'convergence_rate': float(calculate_convergence(bias_history)),
            'stability_metrics': self.analyze_network_stability(),
            'opinion_shifts': self.track_opinion_shifts(bias_history),
            'critical_points': self.identify_critical_points(bias_history)
        }

    def track_opinion_shifts(self, bias_history):
        """
        Track major changes in opinions over time.

        Args:
            bias_history (torch.Tensor): History of biases over time.

        Returns:
            list: A list of significant opinion shifts.
        """
        shifts = []
        for t in range(1, len(bias_history)):
            shift = torch.mean(torch.abs(bias_history[t] - bias_history[t-1]))
            if shift > 0.1:  # Significant shift threshold
                shifts.append({'time': t, 'magnitude': float(shift)})
        return shifts

    def identify_critical_points(self, bias_history):
        """
        Identify points where significant changes occurred.

        Args:
            bias_history (torch.Tensor): History of biases over time.

        Returns:
            list: A list of critical points with variance peaks.
        """
        critical_points = []
        variance_history = [float(torch.var(biases)) for biases in bias_history]
        
        for t in range(1, len(variance_history)-1):
            if (variance_history[t] > variance_history[t-1] and 
                variance_history[t] > variance_history[t+1]):
                critical_points.append({
                    'time': t,
                    'variance_peak': variance_history[t]
                })
        return critical_points

    def analyze_crisis_response(self):
        """
        Analyze network response to crisis events.

        Returns:
            dict: A dictionary containing crisis response metrics.
        """
        if not hasattr(self, 'bias_history') or self.crisis_start is None:
            return {
                'pre_crisis_variance': None,
                'crisis_variance': None,
                'post_crisis_variance': None,
                'recovery_time': None
            }
        
        pre_crisis = self.bias_history[:self.crisis_start]
        crisis_period = self.bias_history[self.crisis_start:self.crisis_end]
        post_crisis = self.bias_history[self.crisis_end:]
        
        return {
            'pre_crisis_variance': float(torch.var(torch.tensor(pre_crisis))) if len(pre_crisis) > 0 else None,
            'crisis_variance': float(torch.var(torch.tensor(crisis_period))) if len(crisis_period) > 0 else None,
            'post_crisis_variance': float(torch.var(torch.tensor(post_crisis))) if len(post_crisis) > 0 else None,
            'recovery_time': self.calculate_recovery_time()
        }

    def calculate_recovery_time(self):
        """
        Calculate how long it takes for the network to stabilize after a crisis.

        Returns:
            int: The recovery time in time steps.
        """
        if not hasattr(self, 'bias_history') or self.crisis_end is None:
            return None
            
        post_crisis = self.bias_history[self.crisis_end:]
        if not post_crisis:
            return None
            
        baseline_variance = torch.var(torch.tensor(self.bias_history[:self.crisis_start]))
        for t, biases in enumerate(post_crisis):
            if torch.var(torch.tensor(biases)) <= baseline_variance * 1.1:  # Within 10% of baseline
                return t
        return len(post_crisis)  # If never recovered

    def analyze_network_stability(self):
        """
        Analyze network stability over time.

        Returns:
            dict: A dictionary containing stability metrics.
        """
        stability_metrics = {
            'variance_trend': [],
            'convergence_speed': [],
            'bridge_load': {}  # Track how much each bridge builder is utilized
        }
        
        # Track bridge builder utilization
        for idx, agent in enumerate(self.agents):
            if agent.is_bridge_builder:
                connections = self.connections[idx]
                stability_metrics['bridge_load'][idx] = {
                    'active_connections': int((connections > 0.1).sum().item()),
                    'total_influence': float(connections.sum().item())
                }
        
        return stability_metrics

    def visualize_bias_spread(self):
        """
        Visualize the spread of bias across the network.
        """
        plt.figure(figsize=(10, 6))
        biases = [agent.bias.item() for agent in self.agents]
        plt.hist(biases, bins=20, alpha=0.7)
        plt.title("Distribution of Biases Across Network")
        plt.xlabel("Bias Value")
        plt.ylabel("Number of Agents")
        plt.grid(True, alpha=0.3)
        plt.show()

    def analyze_group_dynamics(self):
        """
        Analyze bias patterns at the echo chamber level.

        Returns:
            dict: A dictionary containing chamber-level metrics.
        """
        chamber_metrics = {}
        agents_per_chamber = (self.num_base_agents - 2) // self.num_echo_chambers
        
        for chamber in range(self.num_echo_chambers):
            start_idx = chamber * agents_per_chamber
            end_idx = start_idx + agents_per_chamber
            chamber_agents = self.agents[start_idx:end_idx]
            
            chamber_metrics[f"chamber_{chamber}"] = {
                'mean_bias': float(np.mean([agent.bias.item() for agent in chamber_agents])),
                'bias_variance': float(np.var([agent.bias.item() for agent in chamber_agents])),
                'echo_chamber_strength': float(self.calculate_chamber_isolation(start_idx, end_idx)),
                'external_influence_resistance': float(self.measure_external_resistance(chamber_agents))
            }
        
        # Store metrics for tracking over time
        self.metrics["chamber_metrics"] = chamber_metrics
        return chamber_metrics

    def calculate_chamber_isolation(self, start_idx, end_idx):
        """
        Calculate how isolated an echo chamber is.

        Args:
            start_idx (int): Start index of the chamber.
            end_idx (int): End index of the chamber.

        Returns:
            float: The isolation metric of the chamber.
        """
        chamber_connections = self.connections[start_idx:end_idx, :]
        internal_connections = chamber_connections[:, start_idx:end_idx].sum()
        total_connections = chamber_connections.sum()
        return float(internal_connections / total_connections if total_connections > 0 else 0)

    def measure_external_resistance(self, chamber_agents):
        """
        Measure how resistant the chamber is to external influence.

        Args:
            chamber_agents (list): List of agents in the chamber.

        Returns:
            float: The resistance metric of the chamber.
        """
        return float(np.mean([agent.stubbornness for agent in chamber_agents]))

    def measure_polarization(self):
        """
        Measure network polarization using multiple metrics.

        Returns:
            dict: A dictionary containing polarization metrics.
        """
        # Get current biases
        biases = torch.tensor([agent.bias.item() for agent in self.agents])
        
        # Calculate overall variance
        variance = float(torch.var(biases))
        
        # Calculate chamber polarization
        chamber_polarization = {}
        agents_per_chamber = (self.num_base_agents - 2) // self.num_echo_chambers
        
        for chamber in range(self.num_echo_chambers):
            start_idx = chamber * agents_per_chamber
            end_idx = start_idx + agents_per_chamber
            chamber_biases = biases[start_idx:end_idx]
            
            chamber_polarization[f"chamber_{chamber}"] = {
                'mean': float(torch.mean(chamber_biases)),
                'variance': float(torch.var(chamber_biases)),
                'extremity': float(torch.max(torch.abs(chamber_biases)))
            }
        
        # Calculate distance between chambers
        chamber_means = [stats['mean'] for stats in chamber_polarization.values()]
        chamber_distances = []
        for i in range(len(chamber_means)):
            for j in range(i + 1, len(chamber_means)):
                chamber_distances.append(abs(chamber_means[i] - chamber_means[j]))
        
        return {
            'overall_variance': variance,
            'chamber_polarization': chamber_polarization,
            'max_chamber_distance': max(chamber_distances) if chamber_distances else 0,
            'avg_chamber_distance': sum(chamber_distances) / len(chamber_distances) if chamber_distances else 0,
            'extremity_index': float(torch.max(torch.abs(biases)))
        }

    def analyze_bridge_effectiveness(self):
        """
        Analyze the effectiveness of bridge builders in connecting echo chambers.

        Returns:
            dict: A dictionary containing bridge effectiveness metrics.
        """
        bridge_metrics = {}
        
        # Identify bridge builders
        bridge_builders = [(i, agent) for i, agent in enumerate(self.agents) 
                          if hasattr(agent, 'is_bridge_builder') and agent.is_bridge_builder]
        
        for idx, bridge_agent in bridge_builders:
            # Get connection strengths
            connections = self.connections[idx]
            
            # Calculate metrics for this bridge builder
            bridge_metrics[f"bridge_{idx}"] = {
                'connection_count': int((connections > 0.1).sum().item()),
                'total_influence': float(connections.sum().item()),
                'chambers_connected': bridge_agent.group_memberships,
                'avg_connection_strength': float(connections[connections > 0.1].mean().item()),
                'bias_difference': float(abs(bridge_agent.bias.item())),  # Difference from neutral (0)
                'emotional_state': float(bridge_agent.emotional_state.item())
            }
        
        # Calculate overall effectiveness metrics
        if bridge_builders:
            overall_metrics = {
                'total_bridges': len(bridge_builders),
                'avg_connections_per_bridge': float(np.mean([m['connection_count'] 
                    for m in bridge_metrics.values()])),
                'avg_influence': float(np.mean([m['total_influence'] 
                    for m in bridge_metrics.values()])),
                'bridge_bias_variance': float(np.var([m['bias_difference'] 
                    for m in bridge_metrics.values()]))
            }
        else:
            overall_metrics = {
                'total_bridges': 0,
                'avg_connections_per_bridge': 0.0,
                'avg_influence': 0.0,
                'bridge_bias_variance': 0.0
            }
            
        return {
            'individual_bridges': bridge_metrics,
            'overall_metrics': overall_metrics
        }

    def test_network_resilience(self):
        """
        Test the network's resilience to crises and external influences.

        Returns:
            dict: A dictionary containing resilience metrics.
        """
        resilience_metrics = {
            'pre_crisis_variance': None,
            'crisis_variance': None,
            'post_crisis_variance': None,
            'recovery_time': None
        }

        if not hasattr(self, 'bias_history') or self.crisis_start is None:
            return resilience_metrics
        
        pre_crisis = self.bias_history[:self.crisis_start]
        crisis_period = self.bias_history[self.crisis_start:self.crisis_end]
        post_crisis = self.bias_history[self.crisis_end:]
        
        resilience_metrics['pre_crisis_variance'] = float(torch.var(torch.tensor(pre_crisis))) if len(pre_crisis) > 0 else None
        resilience_metrics['crisis_variance'] = float(torch.var(torch.tensor(crisis_period))) if len(crisis_period) > 0 else None
        resilience_metrics['post_crisis_variance'] = float(torch.var(torch.tensor(post_crisis))) if len(post_crisis) > 0 else None
        resilience_metrics['recovery_time'] = self.calculate_recovery_time()
        
        return resilience_metrics


def calculate_convergence(bias_history: torch.Tensor) -> float:
    """
    Calculate how quickly the network converges.

    Args:
        bias_history (torch.Tensor): History of biases over time.

    Returns:
        float: The convergence rate of the network.
    """
    final_variance = torch.var(bias_history[-1])
    initial_variance = torch.var(bias_history[0])
    return 1 - (final_variance / initial_variance)


def run_comparative_analysis(num_trials: int = 5) -> Dict[str, List[float]]:
    """
    Run multiple trials comparing crisis vs no-crisis scenarios.

    Args:
        num_trials (int): Number of trials to run.

    Returns:
        dict: A dictionary containing variance and convergence metrics.
    """
    results = {
        "crisis_variance": [],
        "normal_variance": [],
        "crisis_convergence": [],
        "normal_convergence": [],
    }

    for i in range(num_trials):
        print(f"Running trial {i+1}/{num_trials}")

        # With crisis
        network = SocialNetwork(
            num_agents=20, num_influencers=2, num_echo_chambers=2, num_bridge_builders=2
        )
        network.trigger_crisis(intensity=0.8, duration=10, target_chamber=0)
        bias_history_crisis = network.run_simulation(steps=100)

        # Without crisis
        network = SocialNetwork(
            num_agents=20, num_influencers=2, num_echo_chambers=2, num_bridge_builders=2
        )
        bias_history_normal = network.run_simulation(steps=100)

        # Calculate metrics
        results["crisis_variance"].append(torch.var(bias_history_crisis[-1]).item())
        results["normal_variance"].append(torch.var(bias_history_normal[-1]).item())
        results["crisis_convergence"].append(
            calculate_convergence(bias_history_crisis).item()
        )
        results["normal_convergence"].append(
            calculate_convergence(bias_history_normal).item()
        )

    # Plot results with updated parameter names
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.boxplot(
        [results["crisis_variance"], results["normal_variance"]],
        tick_labels=["With Crisis", "Without Crisis"],
    )
    plt.title("Final Opinion Variance")
    plt.ylabel("Variance")

    plt.subplot(1, 2, 2)
    plt.boxplot(
        [results["crisis_convergence"], results["normal_convergence"]],
        tick_labels=["With Crisis", "Without Crisis"],
    )
    plt.title("Convergence Rate")
    plt.ylabel("Convergence Score")

    plt.tight_layout()
    plt.show()

    return results


def analyze_bridge_impact(bridge_counts=[0, 2, 4, 6], num_trials=3):
    """
    Analyze the impact of different numbers of bridge builders.

    Args:
        bridge_counts (list): List of bridge builder counts to test.
        num_trials (int): Number of trials per bridge count.

    Returns:
        dict: A dictionary containing convergence and variance metrics.
    """
    results = {"bridge_count": [], "convergence": [], "final_variance": []}

    for bridges in bridge_counts:
        print(f"Testing with {bridges} bridge builders...")
        for trial in range(num_trials):
            network = SocialNetwork(
                num_agents=20,
                num_influencers=2,
                num_echo_chambers=2,
                num_bridge_builders=bridges,
            )
            bias_history = network.run_simulation(steps=100)

            results["bridge_count"].append(bridges)
            results["convergence"].append(calculate_convergence(bias_history).item())
            results["final_variance"].append(torch.var(bias_history[-1]).item())

    # Plot results
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    bridge_counts = sorted(set(results["bridge_count"]))
    convergence_by_bridges = [
        [
            v
            for i, v in enumerate(results["convergence"])
            if results["bridge_count"][i] == bc
        ]
        for bc in bridge_counts
    ]
    plt.boxplot(convergence_by_bridges, labels=bridge_counts)
    plt.title("Convergence Rate by Number of Bridge Builders")
    plt.xlabel("Number of Bridge Builders")
    plt.ylabel("Convergence Score")

    plt.subplot(1, 2, 2)
    variance_by_bridges = [
        [
            v
            for i, v in enumerate(results["final_variance"])
            if results["bridge_count"][i] == bc
        ]
        for bc in bridge_counts
    ]
    plt.boxplot(variance_by_bridges, labels=bridge_counts)
    plt.title("Final Opinion Variance by Number of Bridge Builders")
    plt.xlabel("Number of Bridge Builders")
    plt.ylabel("Variance")

    plt.tight_layout()
    plt.show()

    return results


class SimulationAnalyzer:
    def __init__(self, model_name: str = "hermes3:latest"):
        """
        Initialize the SimulationAnalyzer with a specified LLM model.

        Args:
            model_name (str): The name of the LLM model to use.
        """
        self.client = Client(host="http://localhost:11434")
        self.model = model_name

    async def analyze_results(self, simulation_data: Dict[str, Any]) -> str:
        """
        Analyze simulation results using an LLM with enhanced storytelling.

        Args:
            simulation_data (dict): The data from the simulation to analyze.

        Returns:
            str: A narrative analysis of the simulation results.
        """
        try:
            # Log the simulation data
            logging.info("Simulation Data: %s", simulation_data)

            # Format the simulation data into a narrative-focused prompt
            prompt = f"""You are a social network researcher analyzing a fascinating simulation of bias dynamics in social networks. 
            Tell a compelling story about what happened in this simulation, using the following data:

            Individual Dynamics:
            - Bias Distribution: {simulation_data['individual_metrics']['bias_values']}
            - Emotional States: {simulation_data['individual_metrics']['emotional_state']}
            - Susceptibility Patterns: {simulation_data['individual_metrics']['susceptibility']}

            Group Behavior:
            - Echo Chamber Metrics: {simulation_data['group_metrics']}
            - Polarization Index: {simulation_data['network_metrics']['overall_variance']}
            - Bridge Builder Impact: {simulation_data['bridge_builder_metrics']}

            Temporal Evolution:
            - Convergence Rate: {simulation_data['temporal_metrics']['convergence_rate']}
            - Critical Points: {simulation_data['temporal_metrics']['critical_points']}
            - Opinion Shifts: {simulation_data['temporal_metrics']['opinion_shifts']}

            Crisis Impact:
            - Pre/Post Crisis Changes: {simulation_data['crisis_metrics']}
            - Network Resilience: {simulation_data['resilience_metrics']}

            Please weave these findings into an engaging narrative that explains:
            1. How did individual biases evolve and influence each other?
            2. What role did echo chambers play in the network?
            3. How effective were bridge builders in reducing polarization?
            4. What happened during crisis events?
            5. What lessons can we learn about managing bias in social networks?

            Frame this as a story about how ideas and beliefs spread through our social networks, 
            using concrete examples from the data to illustrate key points.
            """

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.7}  # Add creativity while maintaining accuracy
            )
            return response["response"]
        except Exception as e:
            logging.error("Error analyzing results: %s", str(e))
            return f"Error analyzing results: {str(e)}"

def run_comprehensive_simulation() -> Tuple[Dict[str, Any], str]:
    """
    Run a comprehensive simulation with all analysis metrics.

    Returns:
        tuple: A tuple containing the simulation data and narrative.
    """
    network = SocialNetwork(
        num_agents=12,
        num_influencers=2,
        num_echo_chambers=2,
        num_bridge_builders=1
    )

    # Collect initial metrics
    initial_metrics = {
        'individual_metrics': network.analyze_individual_bias(),
        'group_metrics': network.analyze_group_dynamics(),
        'network_metrics': network.measure_polarization()
    }

    # Run simulation with crisis event
    print("Running simulation...")
    network.trigger_crisis(intensity=0.8, duration=10, target_chamber=0)
    bias_history = network.run_simulation(steps=100)

    # Collect comprehensive metrics
    simulation_data = {
        'individual_metrics': network.analyze_individual_bias(),
        'group_metrics': network.analyze_group_dynamics(),
        'network_metrics': network.measure_polarization(),
        'temporal_metrics': network.analyze_bias_evolution(bias_history),
        'bridge_builder_metrics': network.analyze_bridge_effectiveness(),
        'crisis_metrics': network.analyze_crisis_response(),
        'resilience_metrics': network.test_network_resilience()
    }

    # Get LLM analysis
    analyzer = SimulationAnalyzer()
    narrative = asyncio.run(analyzer.analyze_results(simulation_data))

    # Print results
    print("\nSimulation Analysis Narrative:")
    print("=" * 80)
    print(narrative)
    print("=" * 80)

    # Visualize key findings
    network.visualize_network()
    network.visualize_bias_spread()
    network.plot_simulation(bias_history)

    return simulation_data, narrative

if __name__ == "__main__":
    simulation_data, narrative = run_comprehensive_simulation()
