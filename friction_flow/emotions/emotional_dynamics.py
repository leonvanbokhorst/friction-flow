from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from .emotional_state import EmotionalState
from ..agents.enhanced_social_agent import EnhancedSocialAgent
from ..core.base_classes import Observable

class GroupEmotionalDynamics(Observable):
    """Manages emotional dynamics within a group of agents"""

    def __init__(self, agents: List[EnhancedSocialAgent]):
        super().__init__()
        self.agents = agents
        self.emotional_network = self._initialize_emotional_network()
        self.subgroups: List[EmotionalSubgroup] = []
        
    def _initialize_emotional_network(self) -> Dict[str, Dict[str, float]]:
        """Initialize the emotional relationship network between agents"""
        network = {}
        for agent in self.agents:
            network[agent.id] = {}
            for other in self.agents:
                if other.id != agent.id:
                    network[agent.id][other.id] = 0.5  # Initial neutral connection
        return network

    async def process_emotional_interaction(
        self, 
        source: EnhancedSocialAgent, 
        emotion: EmotionalState.Emotion,
        intensity: float
    ) -> None:
        """Process emotional influence from one agent to others"""
        for target in self.agents:
            if target.id != source.id:
                influence = self._calculate_emotional_influence(source, target)
                await target.emotional_state.update_emotion(emotion, intensity * influence)
                self._notify_observers({
                    'type': 'emotional_interaction',
                    'source': source.id,
                    'target': target.id,
                    'emotion': emotion,
                    'intensity': intensity * influence
                })

    def _calculate_emotional_influence(
        self, 
        source: EnhancedSocialAgent, 
        target: EnhancedSocialAgent
    ) -> float:
        """Calculate emotional influence between two agents"""
        base_influence = self.emotional_network[source.id][target.id]
        status_factor = source.status.compute_effective_status("emotional")
        personality_factor = (
            0.4 * source.personality.extraversion +
            0.3 * source.personality.dominance +
            0.3 * (1 - target.personality.stability)
        )
        return min(1.0, base_influence * status_factor * personality_factor)

    def update_emotional_network(self, agent1_id: str, agent2_id: str, value: float) -> None:
        """Update emotional connection strength between two agents"""
        self.emotional_network[agent1_id][agent2_id] = max(0.0, min(1.0, value))
        self.emotional_network[agent2_id][agent1_id] = max(0.0, min(1.0, value)) 