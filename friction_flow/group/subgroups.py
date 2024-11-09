from dataclasses import dataclass
from typing import List, Set
import numpy as np

from ..agents.enhanced_social_agent import EnhancedSocialAgent
from ..emotions.emotional_state import EmotionalState

@dataclass
class EmotionalSubgroup:
    """Represents an emotional subgroup within the larger group"""
    
    members: Set[EnhancedSocialAgent]
    primary_emotion: EmotionalState.Emotion
    
    def calculate_cohesion(self) -> float:
        """Calculate the emotional cohesion of the subgroup"""
        if not self.members:
            return 0.0
            
        emotional_variance = np.var([
            member.emotional_state.current_emotions[self.primary_emotion]
            for member in self.members
        ])
        return 1 / (1 + emotional_variance) 