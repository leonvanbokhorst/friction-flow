import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from friction_flow.inputs.agent_inputs import SimulationInputs

class InputVisualizer:
    """Visualizes simulation inputs"""
    
    def __init__(self, inputs: SimulationInputs):
        self.inputs = inputs
        self.plt = plt
        self.sns = sns
        
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
        
        # Draw nodes with colors based on role
        node_colors = [
            'red' if G.nodes[n]['role'] == 'leader'
            else 'orange' if G.nodes[n]['role'] == 'influencer'
            else 'green' if G.nodes[n]['role'] == 'mediator'
            else 'blue' if G.nodes[n]['role'] == 'follower'
            else 'gray'
            for n in G.nodes()
        ]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=[G.nodes[n]['status'] * 1000 for n in G.nodes()],
            alpha=0.7
        )
        
        # Draw edges with weight-based thickness
        nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
            alpha=0.5
        )
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Agent Relationship Network')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    def create_personality_distribution(self, output_path: Path) -> None:
        """Visualize personality trait distributions"""
        traits_data = {
            'openness': [],
            'conscientiousness': [],
            'extraversion': [],
            'agreeableness': [],
            'neuroticism': [],
            'dominance': [],
            'social_influence': []
        }
        
        for agent in self.inputs.agents:
            for trait, value in agent.personality.model_dump().items():
                traits_data[trait].append(value)
                
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(traits_data))
        plt.title('Personality Trait Distributions')
        plt.ylabel('Trait Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def create_role_distribution(self, output_path: Path) -> None:
        """Visualize distribution of social roles"""
        role_counts = {}
        for agent in self.inputs.agents:
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
            
        plt.figure(figsize=(8, 6))
        plt.pie(
            role_counts.values(),
            labels=role_counts.keys(),
            autopct='%1.1f%%',
            colors=sns.color_palette("husl", len(role_counts))
        )
        plt.title('Distribution of Social Roles')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()