from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class BaseMemory(ABC):
    @abstractmethod
    def form_memory(self, memory_type: str, content: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def recall(self, memory_type: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def apply_decay(self, memory_type: str = None) -> None:
        pass

    @abstractmethod
    def integrate_memories(self, memory_ids: List[str]) -> str:
        pass

    @abstractmethod
    def get_memory_types(self) -> List[str]:
        pass

    @abstractmethod
    def clear_memory(self, memory_type: str = None) -> None:
        pass
