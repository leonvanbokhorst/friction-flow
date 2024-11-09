from typing import Dict, Any
import asyncio
from datetime import datetime
import logging

from ..config.simulation_config import SimulationConfig
from ..utils.event_logger import EventLogger

logger = logging.getLogger(__name__)

class SimulationCoordinator:
    """Coordinates all aspects of the social simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.event_logger = EventLogger()
        self.simulation_time = 0
        self._observers = []
        
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
                
        logger.info("Simulation completed")
    
    async def _simulation_step(self) -> None:
        """Execute one step of the simulation"""
        # For now, just simulate some delay
        await asyncio.sleep(0.01)
        
    def attach_observer(self, observer) -> None:
        """Attach an observer to the simulation"""
        self._observers.append(observer)
        
    def _notify_observers(self, event: Dict[str, Any]) -> None:
        """Notify all observers of an event"""
        for observer in self._observers:
            observer(event)