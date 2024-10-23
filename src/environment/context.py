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
    def interpret_context(self, agent_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def assess_impact(self, change: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def adapt_response(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
