from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseContext(ABC):
    @abstractmethod
    def update_context(self, domain: str, update: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_current_context(self) -> Dict[str, Any]:
        pass

    @abstractmethod
<<<<<<< HEAD
    def interpret_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
=======
    def interpret_context(self, agent_id: str) -> Dict[str, Any]:
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def assess_impact(self, change: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
<<<<<<< HEAD
    def adapt_response(self, initial_response: str, context: Dict[str, Any]) -> Dict[str, Any]:
=======
    def adapt_response(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass
