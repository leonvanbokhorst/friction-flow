from typing import Dict, Any
import asyncio
from datetime import datetime
import logging
import random
from pathlib import Path

from ..config.simulation_config import SimulationConfig
from ..utils.event_logger import EventLogger
from ..utils.results_manager import ResultsManager
from ..inputs.agent_inputs import SimulationInputs

logger = logging.getLogger(__name__)

class SimulationCoordinator:
    """Coordinates all aspects of the social simulation"""
    
    def __init__(self, config: SimulationConfig, inputs: SimulationInputs, output_dir: Path):
        self.config = config
        self.inputs = inputs
        self.results_manager = ResultsManager(output_dir)
        self.simulation_time = 0
        self._observers = []
        
        # Initialize agent states from inputs
        self.agent_states = {
            agent.id: {
                "personality": agent.personality.dict(),
                "status": agent.initial_status,
                "relationships": {},
                "emotional_state": {"valence": 0.5, "arousal": 0.5}
            }
            for agent in inputs.agents
        }
        
        # Initialize relationships
        for rel in inputs.relationships:
            self.agent_states[rel.agent_a]["relationships"][rel.agent_b] = {
                "trust": rel.trust,
                "familiarity": rel.familiarity,
                "emotional_bond": rel.emotional_bond
            }
            self.agent_states[rel.agent_b]["relationships"][rel.agent_a] = {
                "trust": rel.trust,
                "familiarity": rel.familiarity,
                "emotional_bond": rel.emotional_bond
            }
        
    async def initialize_simulation(self) -> None:
        """Initialize simulation components"""
        logger.info("Initializing simulation with %d agents", self.config.num_agents)
        # For now, just log the initialization
        self._notify_observers({
            "type": "simulation_initialized",
            "timestamp": datetime.now().isoformat(),
            "num_agents": self.config.num_agents
        })
        
    async def run_simulation(self, duration: int) -> None:
        """Run the simulation for specified duration"""
        logger.info("Starting simulation for %d steps", duration)
        
        for step in range(duration):
            self.simulation_time = step
            await self._simulation_step()
            
            # Notify observers of step completion
            self._notify_observers({
                "type": "step_completed",
                "step": step,
                "timestamp": datetime.now().isoformat()
            })
            
            if step % 100 == 0:
                logger.info("Completed step %d", step)
                
        # Save results at the end
        self.results_manager.save_results()
        logger.info("Simulation completed")
    
    async def _simulation_step(self) -> None:
        """Execute one step of the simulation"""
        # For demonstration, log some dummy metrics
        self.results_manager.log_metric(
            "emotional_intensity",
            random.random()  # Replace with actual metric
        )
        self.results_manager.log_metric(
            "group_cohesion",
            random.random()  # Replace with actual metric
        )
        self.results_manager.log_metric(
            "interaction_count",
            random.randint(0, 10)  # Replace with actual metric
        )
        
        # Log the step event
        self.results_manager.log_event({
            "type": "step_completed",
            "step": self.simulation_time,
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.01)
        
    def attach_observer(self, observer) -> None:
        """Attach an observer to the simulation"""
        self._observers.append(observer)
        
    def _notify_observers(self, event: Dict[str, Any]) -> None:
        """Notify all observers of an event"""
        for observer in self._observers:
            observer(event)