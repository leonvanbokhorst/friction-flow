from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePressure(ABC):
    @abstractmethod
    def apply_pressure(self, target: str, pressure_type: str, intensity: float) -> None:
        pass

    @abstractmethod
    def evaluate_pressure(self, target: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def suggest_coping_strategy(self, target: str, pressure_type: str) -> str:
        pass

    @abstractmethod
    def evaluate_risk(self, target: str) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_environmental_pressure(self) -> Dict[str, Any]:
        pass
