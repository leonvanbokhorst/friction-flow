from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
from .agent_inputs import SimulationInputs
from ..utils.input_visualizer import InputVisualizer
import logging

logger = logging.getLogger(__name__)

class InputManager:
    """Manages loading, validation, and visualization of simulation inputs"""
    
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.visualizer: Optional[InputVisualizer] = None
        self.inputs: Optional[SimulationInputs] = None
        
    def load_inputs(self) -> SimulationInputs:
        """Load and validate simulation inputs"""
        with open(self.input_path) as f:
            input_dict = yaml.safe_load(f)
        self.inputs = SimulationInputs.parse_raw(json.dumps(input_dict))
        return self.inputs
        
    def generate_visualizations(self, output_dir: Path) -> None:
        """Generate all input visualizations"""
        if not self.inputs:
            raise ValueError("Inputs must be loaded before generating visualizations")
            
        self.visualizer = InputVisualizer(self.inputs)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate network visualization
        self.visualizer.create_network_graph(output_dir / "network.png")
        
        # Generate personality distribution
        self.visualizer.create_personality_distribution(output_dir / "personalities.png")
        
        # Generate role distribution
        self.visualizer.create_role_distribution(output_dir / "roles.png") 
        
    def verify_inputs(self) -> bool:
        """Verify input consistency and completeness"""
        if not self.inputs:
            return False
            
        try:
            # Check for isolated agents
            connected_agents = set()
            for rel in self.inputs.relationships:
                connected_agents.add(rel.agent_a)
                connected_agents.add(rel.agent_b)
                
            isolated_agents = set(
                agent.id for agent in self.inputs.agents
            ) - connected_agents
            
            if isolated_agents:
                logger.warning(f"Found isolated agents: {isolated_agents}")
                
            # Verify role distribution
            role_counts = {}
            for agent in self.inputs.agents:
                role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
                
            if role_counts.get(SocialRole.LEADER, 0) != 1:
                raise ValueError("Simulation requires exactly one leader")
                
            return True
            
        except Exception as e:
            logger.error(f"Input verification failed: {str(e)}")
            return False