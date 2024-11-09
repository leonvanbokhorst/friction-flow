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
        # Extract option characteristics
        valence_alignment = 1 - abs(
            option.get('expected_valence', 0.5) - emotional_state.valence
        )
        arousal_alignment = 1 - abs(
            option.get('expected_arousal', 0.5) - emotional_state.arousal
        )
        
        return (valence_alignment + arousal_alignment) / 2
        
    def _calculate_personality_fit(
        self,
        option: Dict[str, Any],
        personality_factors: Dict[str, float]
    ) -> float:
        """Calculate how well an option fits with personality"""
        trait_weights = {
            'openness': option.get('novelty', 0.5),
            'conscientiousness': option.get('structure', 0.5),
            'extraversion': option.get('social_engagement', 0.5),
            'agreeableness': option.get('cooperation', 0.5),
            'neuroticism': option.get('risk', 0.5)
        }
        
        weighted_sum = sum(
            personality_factors.get(trait, 0.5) * weight
            for trait, weight in trait_weights.items()
        )
        return weighted_sum / len(trait_weights)
        
    def _evaluate_context_fit(
        self,
        option: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate how well an option fits the current context"""
        context_factors = {
            'time_pressure': context.get('time_pressure', 0.5),
            'social_support': context.get('social_support', 0.5),
            'risk_level': context.get('risk_level', 0.5)
        }
        
        option_requirements = {
            'time_pressure': option.get('urgency', 0.5),
            'social_support': option.get('support_needed', 0.5),
            'risk_level': option.get('risk_tolerance', 0.5)
        }
        
        fit_scores = [
            1 - abs(context_factors[factor] - option_requirements[factor])
            for factor in context_factors.keys()
        ]
        
        return sum(fit_scores) / len(fit_scores)