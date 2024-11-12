import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class SocialAgent(nn.Module):
    def __init__(
        self,
        initial_bias,
        susceptibility,
        stubbornness,
        is_influencer=False,
        confirmation_bias_strength=0.5,
    ):
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

    def calculate_confirmation_bias(self, social_influence):
        """Calculate how much the agent accepts/rejects information based on current bias"""
        bias_difference = abs(self.bias.item() - social_influence)
        acceptance_rate = 1.0 - (bias_difference * self.confirmation_bias_strength)
        return max(0.1, min(1.0, acceptance_rate))  # Clamp between 0.1 and 1.0

    def update_emotional_state(self, social_influence, neighboring_emotions=None, crisis_intensity=0):
        # Update emotion based on social influence, neighbors, and crisis
        delta = abs(self.bias.item() - social_influence)
        
        # Factor in emotional contagion from neighbors
        emotional_influence = 0
        if neighboring_emotions is not None:
            emotional_influence = (torch.mean(neighboring_emotions) - self.emotional_state) * self.emotional_contagion_rate
        
        # Factor in crisis impact
        crisis_impact = crisis_intensity * (1 - self.stubbornness)
        
        self.emotional_state.data = torch.clamp(
            self.emotional_state - 0.1 * delta + emotional_influence + crisis_impact + 0.05 * torch.rand(1),
            0, 1
        )
        
        # Store trauma if crisis is significant
        if crisis_intensity > 0.5:
            self.trauma_memories.append((crisis_intensity, 0))  # (intensity, age)

    def forward(self, social_influence, new_information, external_influence):
        self.update_emotional_state(social_influence)
        # Modify susceptibility based on emotional state
        effective_susceptibility = self.susceptibility * (1 + 0.5 * self.emotional_state.item())

        # Calculate confirmation bias effect
        confirmation_factor = self.calculate_confirmation_bias(social_influence)

        # Add memory effect
        self.interaction_history.append(social_influence)
        if len(self.interaction_history) > self.memory_length:
            self.interaction_history.pop(0)
            
        weighted_history = sum(
            [x * (0.9 ** i) for i, x in enumerate(reversed(self.interaction_history))]
        ) / len(self.interaction_history)
        
        # Prepare inputs including confirmation bias
        inputs = torch.tensor(
            [
                self.bias.item(),
                social_influence * confirmation_factor,  # Apply confirmation bias
                new_information,
                external_influence,
                weighted_history  # Add historical influence
            ],
            dtype=torch.float,
        )

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


class SocialNetwork:
    def __init__(self, num_agents=20, num_influencers=2, num_echo_chambers=2, num_bridge_builders=2):
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
        self.create_echo_chamber_connections(num_agents, agents_per_chamber, num_influencers)
        
        # Initialize crisis parameters
        self.current_crisis = None
        self.crisis_history = []
        
        # Add bridge builders
        self.add_bridge_builders(num_bridge_builders)

    def add_bridge_builder_connections(self, builder_idx, chambers):
        """Add connections for bridge builders to their chosen chambers"""
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

    def create_echo_chamber_connections(self, num_agents, agents_per_chamber, num_influencers):
        """Create connection matrix with echo chambers and influencers"""
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
                    chamber_influence = 1.0 if inf_idx % 2 == i // agents_per_chamber else 0.5
                    self.connections[i, inf_idx] = np.random.uniform(0.3, 0.8) * chamber_influence

        # Normalize connections
        row_sums = self.connections.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        self.connections = self.connections / row_sums

    def trigger_crisis(self, intensity, duration, target_chamber=None):
        """Initiate a crisis event"""
        self.current_crisis = {
            'intensity': intensity,
            'duration': duration,
            'target_chamber': target_chamber,
            'remaining': duration
        }
        self.crisis_history.append(self.current_crisis)

    def simulate_step(self, time_step):
        """Enhanced simulation step with crisis and emotional contagion"""
        current_biases = torch.tensor([agent.bias.item() for agent in self.agents])
        current_emotions = torch.tensor([agent.emotional_state.item() for agent in self.agents])
        
        # Handle crisis events
        crisis_intensity = 0
        if self.current_crisis:
            crisis_intensity = self.current_crisis['intensity']
            self.current_crisis['remaining'] -= 1
            if self.current_crisis['remaining'] <= 0:
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
                external_influence=self.external_influences[i]
            )
            agent.update_emotional_state(
                social_influence,
                neighboring_emotions=neighboring_emotions,
                crisis_intensity=crisis_intensity
            )
            
            new_biases.append(agent.bias.item())

        return torch.tensor(new_biases)

    def run_simulation(self, steps=100):
        """Run simulation with visualization of network structure"""
        bias_history = []

        for step in range(steps):
            new_biases = self.simulate_step(step)
            bias_history.append(new_biases.clone())

        return torch.stack(bias_history)

    def visualize_network(self):
        """Create a network visualization using networkx"""
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
        influencer_sizes = [1000 if G.nodes[n]["influencer"] else 300 for n in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=influencer_sizes,
            cmap=plt.cm.RdYlBu,
            ax=ax
        )

        # Draw edges with varying thickness based on weight
        edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, ax=ax)

        plt.title("Social Network Structure\nNode color = bias, Size = influence, Edge thickness = connection strength")
        
        # Add colorbar with proper normalization
        norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
        plt.colorbar(sm, ax=ax)
        
        plt.show()

    def plot_simulation(self, bias_history):
        """Enhanced plotting with echo chamber and influencer highlighting"""
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
        """Add agents with connections to multiple chambers"""
        for i in range(num_bridge_builders):
            # Select random chambers to bridge
            chambers = np.random.choice(self.num_echo_chambers, 2, replace=False)
            
            # Create bridge builder agent
            agent = SocialAgent(
                initial_bias=0.0,  # Start neutral
                susceptibility=np.random.uniform(0.4, 0.8),
                stubbornness=np.random.uniform(0.2, 0.5),
                confirmation_bias_strength=0.3  # More open to different views
            )
            agent.is_bridge_builder = True
            agent.group_memberships = chambers.tolist()
            self.agents.append(agent)
            
            # Add connections for this bridge builder
            self.add_bridge_builder_connections(len(self.agents) - 1, chambers)


# Example usage
if __name__ == "__main__":
    # Create network with 20 regular agents, 2 influencers, and 2 echo chambers
    network = SocialNetwork(num_agents=20, num_influencers=2, num_echo_chambers=2, num_bridge_builders=2)

    # Visualize initial network structure
    network.visualize_network()

    # Trigger a crisis event
    network.trigger_crisis(intensity=0.8, duration=10, target_chamber=0)

    # Run and plot simulation
    bias_history = network.run_simulation(steps=100)
    network.plot_simulation(bias_history)
