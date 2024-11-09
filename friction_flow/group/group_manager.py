from typing import List, Dict, Set
import numpy as np
from dataclasses import dataclass

from ..agents.enhanced_social_agent import EnhancedSocialAgent
from ..core.base_classes import Observable
from ..emotions.emotional_state import EmotionalState

@dataclass
class Group:
    """Represents a social group within the simulation"""
    id: str
    members: Set[EnhancedSocialAgent]
    cohesion: float = 0.0
    emotional_alignment: float = 0.0

class GroupManager(Observable):
    """Manages group formation and dynamics"""
    
    def __init__(self, agents: List[EnhancedSocialAgent]):
        super().__init__()
        self.agents = agents
        self.groups: Dict[str, Group] = {}
        self.similarity_threshold = 0.6
        
    async def update_groups(self) -> None:
        """Update group compositions based on current agent states"""
        similarity_matrix = self._calculate_similarity_matrix()
        new_groups = self._identify_groups(similarity_matrix)
        
        # Update existing groups or create new ones
        updated_groups = {}
        for group_id, members in new_groups.items():
            if group_id in self.groups:
                self.groups[group_id].members = members
                self.groups[group_id].cohesion = self._calculate_group_cohesion(members)
                self.groups[group_id].emotional_alignment = (
                    self._calculate_emotional_alignment(members)
                )
            else:
                updated_groups[group_id] = Group(
                    id=group_id,
                    members=members,
                    cohesion=self._calculate_group_cohesion(members),
                    emotional_alignment=self._calculate_emotional_alignment(members)
                )
                
        self.groups = updated_groups
        self._notify_group_changes()
        
    def _calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate similarity matrix between all agents"""
        n_agents = len(self.agents)
        similarity_matrix = np.zeros((n_agents, n_agents))
        
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i != j:
                    similarity_matrix[i, j] = self._calculate_agent_similarity(
                        agent1, agent2
                    )
                    
        return similarity_matrix
        
    def _calculate_agent_similarity(
        self,
        agent1: EnhancedSocialAgent,
        agent2: EnhancedSocialAgent
    ) -> float:
        """Calculate similarity between two agents"""
        emotional_similarity = self._calculate_emotional_similarity(
            agent1.emotional_state,
            agent2.emotional_state
        )
        personality_similarity = self._calculate_personality_similarity(
            agent1.personality,
            agent2.personality
        )
        return 0.6 * emotional_similarity + 0.4 * personality_similarity 