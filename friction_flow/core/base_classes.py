from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Observable:
    """Base class for objects that can be observed"""
    
    def __init__(self):
        self._observers: List = []
        
    def attach(self, observer) -> None:
        """Attach an observer to this object"""
        if observer not in self._observers:
            self._observers.append(observer)
            
    def detach(self, observer) -> None:
        """Detach an observer from this object"""
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
            
    def _notify_observers(self, event: Dict[str, Any]) -> None:
        """Notify all observers of an event"""
        for observer in self._observers:
            observer.update(event)

class Agent(ABC):
    """Base class for all agents"""
    
    @abstractmethod
    async def process_emotion(self, emotion: str, intensity: float) -> None:
        """Process an emotional input"""
        pass
        
    @abstractmethod
    async def make_decision(self, context: Dict[str, Any]) -> Any:
        """Make a decision based on given context"""
        pass 