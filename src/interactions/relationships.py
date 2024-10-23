from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRelationships(ABC):
    @abstractmethod
    def update_relationship(self, agent1: str, agent2: str, update: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def form_alliance(self, agent1: str, agent2: str) -> None:
        pass

    @abstractmethod
    def handle_conflict(self, agent1: str, agent2: str, conflict_type: str) -> None:
        pass

    @abstractmethod
    def get_relationship_status(self, agent1: str, agent2: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_agent_network(self, agent: str) -> Dict[str, Dict[str, Any]]:
        pass
