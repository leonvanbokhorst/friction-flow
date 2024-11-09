import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import seaborn as sns

class InputVisualizer:
    """Visualizes simulation inputs"""
    
    def __init__(self, inputs: 'SimulationInputs'):
        self.inputs = inputs
        
    def create_network_graph(self, output_path: Path) -> None:
        """Create and save a network visualization of agents and relationships"""
        G = nx.Graph()
        
        # Add nodes (agents)
        for agent in self.inputs.agents:
            G.add_node(
                agent.id,
                role=agent.role.value,
                status=agent.initial_status
            )
        
        # Add edges (relationships)
        for rel in self.inputs.relationships:
            G.add_edge(
                rel.agent_a,
                rel.agent_b,
                weight=rel.emotional_bond
            )
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [
            'red' if G.nodes[n]['role'] == 'leader'
            else 'orange' if G.nodes[n]['role'] == 'influencer'
            else 'green' if G.nodes[n]['role'] == 'mediator'
            else 'blue'
            for n in G.nodes()
        ]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=[G.nodes[n]['status'] * 1000 for n in G.nodes()],
            alpha=0.7
        )
        
        # Draw edges with weight-based thickness
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Agent Relationship Network")
        plt.savefig(output_path)
        plt.close() 