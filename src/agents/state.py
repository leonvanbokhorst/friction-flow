from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseState(ABC):
    @abstractmethod
    def update_state(self, update: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def record_state_transition(self, transition: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def recognize_patterns(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def analyze_trends(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def project_future_state(self, time_horizon: int) -> Dict[str, Any]:
        pass
