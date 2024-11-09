from enum import Enum
from typing import Dict, Tuple
import numpy as np

class EmotionalState:
    """Manages an agent's emotional state"""
    
    class Emotion(Enum):
        HAPPY = "happy"
        ANGRY = "angry"
        ANXIOUS = "anxious"
        CONFIDENT = "confident"
        DEFENSIVE = "defensive"
        NEUTRAL = "neutral"
        
    def __init__(self):
        self.current_emotions: Dict[Emotion, float] = {
            emotion: 0.0 for emotion in self.Emotion
        }
        self.current_emotions[self.Emotion.NEUTRAL] = 0.5
        
    async def update_emotion(self, emotion: Emotion, intensity: float) -> None:
        """Update intensity of an emotion"""
        self.current_emotions[emotion] = max(0.0, min(1.0, intensity))
        self._normalize_emotions()
        
    def get_dominant_emotion(self) -> Tuple[Emotion, float]:
        """Get the strongest current emotion"""
        dominant = max(self.current_emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
        
    def _normalize_emotions(self) -> None:
        """Ensure emotion intensities sum to 1.0"""
        total = sum(self.current_emotions.values())
        if total > 0:
            for emotion in self.current_emotions:
                self.current_emotions[emotion] /= total 