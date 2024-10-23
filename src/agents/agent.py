from abc import ABC, abstractmethod
from typing import Dict, Any

<<<<<<< HEAD
class BaseSimulator(ABC):
=======
class BaseAgent(ABC):
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def update(self, context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def make_decision(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def communicate(self, message: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def handle_pressure(self, pressure: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def manage_relationships(self, action: str, target_agent: str, parameters: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def learn(self, information: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_agent_state(self) -> Dict[str, Any]:
        pass
