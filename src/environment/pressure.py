from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePressure(ABC):
    @abstractmethod
    def apply_pressure(self, agent_id: str, pressure_type: str, intensity: float) -> None:
        pass

    @abstractmethod
    def assess_pressure(self, agent_id: str) -> Dict[str, float]:
        pass

    @abstractmethod
    def suggest_coping_strategy(self, agent_id: str, pressure_type: str) -> str:
        pass

    @abstractmethod
    def evaluate_risk(self, agent_id: str) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_environmental_pressure(self) -> Dict[str, Any]:
        pass
