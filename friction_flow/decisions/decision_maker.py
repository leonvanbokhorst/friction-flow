from typing import Dict, Any, List, Optional
import numpy as np

from ..core.base_classes import Observable
from ..emotions.emotional_state import EmotionalState

class DecisionMaker(Observable):
    """Handles decision-making logic for agents"""
    
    def __init__(self):
        super().__init__()
        self.decision_history: List[Dict[str, Any]] = []
        
    async def make_decision(
        self,
        options: List[Dict[str, Any]],
        emotional_state: EmotionalState,
        personality_factors: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a decision based on emotional state and personality"""
        
        weighted_scores = []
        
        for option in options:
            emotional_score = self._calculate_emotional_alignment(
                option, emotional_state
            )
            personality_score = self._calculate_personality_fit(
                option, personality_factors
            )
            context_score = self._evaluate_context_fit(
                option, context
            )
            
            total_score = (
                0.4 * emotional_score +
                0.3 * personality_score +
                0.3 * context_score
            )
            weighted_scores.append((option, total_score))
        
        decision = max(weighted_scores, key=lambda x: x[1])[0]
        
        self._record_decision(decision, emotional_state, context)
        return decision
    
    def _calculate_emotional_alignment(
        self,
        option: Dict[str, Any],
        emotional_state: EmotionalState
    ) -> float:
        """Calculate how well an option aligns with current emotional state"""
        dominant_emotion, intensity = emotional_state.get_dominant_emotion()
        
        # Map emotions to decision-making tendencies
        emotion_weights = {
            EmotionalState.Emotion.HAPPY: {'risk_tolerance': 0.7, 'cooperation': 0.8},
            EmotionalState.Emotion.ANGRY: {'risk_tolerance': 0.8, 'cooperation': 0.3},
            EmotionalState.Emotion.ANXIOUS: {'risk_tolerance': 0.2, 'cooperation': 0.5},
            EmotionalState.Emotion.CONFIDENT: {'risk_tolerance': 0.9, 'cooperation': 0.6},
            EmotionalState.Emotion.DEFENSIVE: {'risk_tolerance': 0.3, 'cooperation': 0.4},
            EmotionalState.Emotion.NEUTRAL: {'risk_tolerance': 0.5, 'cooperation': 0.5}
        }
        
        weights = emotion_weights[dominant_emotion]
        option_risk = option.get('risk_level', 0.5)
        option_cooperation = option.get('cooperation_required', 0.5)
        
        return 1 - (
            abs(weights['risk_tolerance'] - option_risk) +
            abs(weights['cooperation'] - option_cooperation)
        ) / 2 