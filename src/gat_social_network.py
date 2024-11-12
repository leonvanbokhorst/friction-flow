from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Add logger configuration at the top of the file
logger = logging.getLogger(__name__)


@dataclass
class AgentNode:
    """Represents an agent in the relationship graph.

    Attributes:
        agent_id: Unique identifier for the agent
        attributes: Dictionary of agent characteristics
        state: Current emotional/behavioral state
    """

    agent_id: str
    attributes: Dict[str, float]
    state: Dict[str, float]


class RelationshipGraph:
    """Manages and analyzes agent relationships using PyTorch Geometric."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the relationship graph with GAT architecture.

        Graph Attention Network (GAT) Explanation:
        - GAT is a neural network architecture designed for graph-structured data
        - It uses attention mechanisms to weigh the importance of different connections
        - Key components:
          1. Multi-head attention: Multiple parallel attention mechanisms (num_heads)
          2. Node features: Encoded from agent attributes and states
          3. Edge features: Encoded from relationship attributes
          4. Attention weights: Learned dynamically based on node and edge features

        Args:
            hidden_dim: Dimension of the hidden representations (default: 64)
            num_heads: Number of parallel attention mechanisms (default: 4)
            device: Computing device to use (default: CUDA if available, else CPU)
        """
        logger.info(
            f"Initializing RelationshipGraph (hidden_dim={hidden_dim}, num_heads={num_heads}, device={device})"
        )
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Initialize empty graph
        self.nx_graph = nx.DiGraph()

        # Calculate input dimensions based on agent attributes and state
        # 2 attributes (openness, extraversion) + 2 state variables (happiness, energy)
        input_dim = (
            4  # Update this if you change the number of attributes/state variables
        )

        # Neural network layers for relationship processing
        self.node_encoder = nn.Linear(input_dim, hidden_dim).to(device)
        self.edge_encoder = nn.Linear(3, hidden_dim).to(
            device
        )  # trust, influence, familiarity

        # GAT layer: Each head learns different aspects of relationships
        # Output dim per head is hidden_dim // num_heads to maintain constant total dimensions
        self.gat_layer = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,  # Concatenate outputs from different heads
        ).to(device)

    def add_agent(self, agent: AgentNode) -> None:
        """Adds a new agent to the relationship graph."""
        logger.debug(
            f"Adding agent {agent.agent_id} to graph with attributes={agent.attributes}, state={agent.state}"
        )
        self.nx_graph.add_node(
            agent.agent_id, attributes=agent.attributes, state=agent.state
        )

    def update_relationship(
        self,
        source_id: str,
        target_id: str,
        trust: float,
        influence: float,
        familiarity: float,
    ) -> None:
        """Updates or creates a relationship between two agents."""
        logger.debug(
            f"Updating relationship: {source_id} -> {target_id} "
            f"(trust={trust:.2f}, influence={influence:.2f}, familiarity={familiarity:.2f})"
        )
        if not self.nx_graph.has_edge(source_id, target_id):
            self.nx_graph.add_edge(
                source_id,
                target_id,
                trust=trust,
                influence=influence,
                familiarity=familiarity,
            )

    def compute_social_dynamics(self) -> torch.Tensor:
        """Computes relationship embeddings using Graph Attention Network (GAT).

        GAT Process:
        1. Node features are encoded from agent attributes and states
        2. Edge features are encoded from relationship attributes
        3. Multiple attention heads process the graph in parallel:
           - Each head computes attention weights for node pairs
           - Weights determine how much each neighbor influences the node
           - Different heads can capture different relationship patterns
        4. Final embeddings combine information from all attention heads

        Returns:
            torch.Tensor: Node embeddings capturing social dynamics
        """
        logger.info("Computing social dynamics using GAT")
        node_features = self._get_node_features()
        edge_index, edge_features = self._get_edge_features()

        # Convert to PyG Data object for GAT processing
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(
            self.device
        )

        # Apply GAT to learn relationship patterns
        node_embeddings = self.gat_layer(data.x, data.edge_index)
        logger.debug(f"Generated node embeddings shape: {node_embeddings.shape}")
        return node_embeddings

    def analyze_communities(self) -> List[List[str]]:
        """Detects communities using the Louvain method.

        Louvain Method Explanation:
        1. Optimization Algorithm:
           - Maximizes modularity (measure of network division quality)
           - Modularity high when communities have dense internal connections
             but sparse connections between communities

        2. Two-Phase Process:
           Phase 1 - Local Optimization:
           - Iteratively moves nodes between communities
           - Each move must increase overall modularity
           - Continues until no moves improve modularity

           Phase 2 - Network Aggregation:
           - Creates super-nodes from found communities
           - Builds new network from these super-nodes
           - Repeats Phase 1 on this new network

        3. Advantages:
           - Automatically determines number of communities
           - Handles different scales of communities
           - Computationally efficient for large networks

        Returns:
            List[List[str]]: List of communities, where each community is a list of agent IDs
        """
        logger.info("Analyzing communities using Louvain method")

        # Convert to undirected graph for community detection
        # Louvain method works on undirected graphs as community membership
        # is inherently symmetric
        undirected_graph = self.nx_graph.to_undirected()

        # Detect communities using Louvain algorithm
        communities = list(nx.community.louvain_communities(undirected_graph))

        logger.debug(f"Detected {len(communities)} communities: {communities}")
        return communities

    def get_influence_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Finds all influence paths between two agents."""
        logger.debug(f"Finding influence paths from {source_id} to {target_id}")
        paths = list(nx.all_simple_paths(self.nx_graph, source_id, target_id, cutoff=3))
        logger.debug(f"Found {len(paths)} influence paths: {paths}")
        return paths

    def _get_node_features(self) -> torch.Tensor:
        """Extracts and encodes node features."""
        features = []
        for node in self.nx_graph.nodes(data=True):
            node_data = torch.tensor(
                [*node[1]["attributes"].values(), *node[1]["state"].values()],
                dtype=torch.float32,
                device=self.device,
            )
            features.append(node_data)

        features_tensor = torch.stack(features)
        return self.node_encoder(features_tensor)

    def _get_edge_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts and encodes edge features."""
        edge_index = []
        features = []

        for source, target, data in self.nx_graph.edges(data=True):
            source_idx = list(self.nx_graph.nodes()).index(source)
            target_idx = list(self.nx_graph.nodes()).index(target)
            edge_index.append([source_idx, target_idx])

            edge_data = torch.tensor(
                [data["trust"], data["influence"], data["familiarity"]],
                dtype=torch.float32,
                device=self.device,
            )
            features.append(edge_data)

        if not edge_index:  # Handle empty graph case
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0, 3), device=self.device),
            )

        edge_index_tensor = torch.tensor(edge_index, device=self.device).t()
        features_tensor = torch.stack(features)
        return edge_index_tensor, self.edge_encoder(features_tensor)

    def visualize(self, title: str = "Relationship Graph") -> None:
        """Visualizes the relationship graph with edge weights and node attributes."""
        # Create figure with a specific size and layout for colorbars
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create layout for graph
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)

        # Draw nodes
        node_sizes = []
        node_colors = []
        for node, data in self.nx_graph.nodes(data=True):
            node_sizes.append(1000 * data["attributes"]["extraversion"])
            node_colors.append(data["attributes"]["openness"])

        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.7,
            ax=ax,
        )

        # Draw edges
        edges = []
        edge_colors = []
        edge_widths = []
        for u, v, data in self.nx_graph.edges(data=True):
            edges.append((u, v))
            edge_colors.append(data["influence"])
            edge_widths.append(data["trust"] * 2)

        nx.draw_networkx_edges(
            self.nx_graph,
            pos,
            edgelist=edges,
            edge_color=edge_colors,
            edge_cmap=plt.cm.coolwarm,
            width=edge_widths,
            edge_vmin=0,
            edge_vmax=1,
            arrows=True,
            arrowsize=20,
            ax=ax,
        )

        # Add labels
        nx.draw_networkx_labels(self.nx_graph, pos, ax=ax)

        # Add colorbars
        sm1 = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm1.set_array([])
        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
        sm2.set_array([])

        # Adjust layout to make room for colorbars
        plt.subplots_adjust(right=0.85)
        cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.3])
        cbar_ax2 = fig.add_axes([0.87, 0.15, 0.02, 0.3])

        fig.colorbar(sm1, cax=cbar_ax1, label="Node Openness Level")
        fig.colorbar(sm2, cax=cbar_ax2, label="Edge Influence Level")

        ax.set_title(title)
        ax.axis("off")
        plt.show()


def demo_relationship_analysis():
    """Demonstrates social network analysis with a classroom social dynamics example.

    This demo creates a simulated classroom environment with 5 students, each having distinct
    personalities and relationship dynamics. It showcases:
    1. How different personality types interact
    2. Formation of social groups and communities
    3. Influence pathways between students
    4. Complex relationship networks and their analysis
    """
    # Configure logging to show timestamps and message levels
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("=== Starting Classroom Social Network Analysis ===")

    # Initialize our social network with a 64-dimensional hidden layer
    # This dimensionality allows for rich representation of social dynamics
    graph = RelationshipGraph(hidden_dim=64)

    # Create our cast of students, each with unique personality traits
    students = {
        # Emma: The natural leader
        "Emma": AgentNode(
            agent_id="Emma",
            attributes={
                "openness": 0.9,  # Very open to new experiences
                "extraversion": 0.8,  # Highly outgoing
            },
            state={
                "happiness": 0.8,  # Generally very positive
                "energy": 0.9,  # High energy levels
            },
        ),
        # James: The thoughtful introvert
        "James": AgentNode(
            agent_id="James",
            attributes={
                "openness": 0.4,  # More traditional in approach
                "extraversion": 0.3,  # Prefers smaller groups
            },
            state={
                "happiness": 0.6,  # Contentedly reserved
                "energy": 0.5,  # Moderate energy levels
            },
        ),
        # Sofia: The social bridge
        "Sofia": AgentNode(
            agent_id="Sofia",
            attributes={
                "openness": 0.7,  # Adaptable and accepting
                "extraversion": 0.6,  # Balanced social energy
            },
            state={
                "happiness": 0.7,  # Generally positive
                "energy": 0.8,  # Good energy levels
            },
        ),
        # Lucas: The social butterfly
        "Lucas": AgentNode(
            agent_id="Lucas",
            attributes={
                "openness": 0.6,  # Moderately open
                "extraversion": 0.7,  # Quite social
            },
            state={"happiness": 0.9, "energy": 0.7},  # Very happy  # Good energy
        ),
        # Mia: The selective socializer
        "Mia": AgentNode(
            agent_id="Mia",
            attributes={
                "openness": 0.8,  # Very open-minded
                "extraversion": 0.4,  # Selective in social interactions
            },
            state={"happiness": 0.7, "energy": 0.6},  # Content  # Moderate energy
        ),
    }

    # Add each student to our social network graph
    logger.info("\n🎓 Creating classroom social network...")
    for student in students.values():
        graph.add_agent(student)

    # Define the web of relationships between students
    # Each relationship has three aspects:
    # - trust: How much they trust each other (0-1)
    # - influence: How much they affect each other's decisions (0-1)
    # - familiarity: How well they know each other (0-1)
    relationships = [
        # Emma's relationships - she's central to the social network
        ("Emma", "Sofia", 0.9, 0.8, 0.9),  # Best friends with strong mutual trust
        ("Emma", "Lucas", 0.8, 0.7, 0.8),  # Good friends who often interact
        ("Emma", "James", 0.6, 0.7, 0.5),  # Mentoring relationship with quiet James
        # Sofia's relationships - she connects different social circles
        ("Sofia", "Emma", 0.9, 0.7, 0.9),  # Reciprocal strong friendship
        ("Sofia", "Mia", 0.8, 0.6, 0.7),  # Bridge to the more introverted students
        ("Sofia", "Lucas", 0.7, 0.5, 0.8),  # Friendly but not extremely close
        # James's relationships - fewer but deeper connections
        ("James", "Mia", 0.8, 0.4, 0.6),  # Connected through shared introversion
        ("James", "Emma", 0.7, 0.3, 0.5),  # Looks up to Emma's leadership
        # Lucas's relationships - broad but not always deep
        ("Lucas", "Emma", 0.8, 0.6, 0.8),  # Strong connection with the group leader
        ("Lucas", "Sofia", 0.7, 0.5, 0.8),  # Regular social interaction
        ("Lucas", "Mia", 0.6, 0.4, 0.5),  # Casual friendship
        # Mia's relationships - selective but strong
        ("Mia", "Sofia", 0.8, 0.5, 0.7),  # Comfortable friendship
        ("Mia", "James", 0.9, 0.6, 0.7),  # Strong bond between similar personalities
    ]

    # Create all the relationships in our graph
    logger.info("\n🤝 Establishing student relationships...")
    for source, target, trust, influence, familiarity in relationships:
        graph.update_relationship(source, target, trust, influence, familiarity)

    # Analyze the social network using various metrics
    logger.info("\n📊 Analyzing classroom social dynamics...")
    # Generate embeddings that capture the network structure
    embeddings = graph.compute_social_dynamics()

    # Detect natural groupings of students
    logger.info("\n👥 Detecting social groups...")
    communities = graph.analyze_communities()

    # Find how influence flows from Emma to Mia
    logger.info("\n🔍 Analyzing social influence paths...")
    influence_paths = graph.get_influence_paths("Emma", "Mia")

    # Output key insights about the social network
    logger.info("\n📈 Social Network Insights:")
    logger.info(f"• Number of students: {len(students)}")
    logger.info(f"• Number of relationships: {len(relationships)}")
    logger.info(f"• Social groups detected: {len(communities)}")
    logger.info(f"• Influence paths from Emma to Mia: {len(influence_paths)}")

    logger.info("\n📊 Visualizing the social network...")
    graph.visualize("Classroom Social Network")

    logger.info("\n✨ Analysis complete! ✨")

    # Return the computed metrics for potential further analysis
    return embeddings, communities, influence_paths


if __name__ == "__main__":
    demo_relationship_analysis()
