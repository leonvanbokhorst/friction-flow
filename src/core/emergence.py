from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEmergenceHandler(ABC):
    @abstractmethod
    def detect_patterns(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def handle_emergence(self, pattern: Dict[str, Any]) -> None:
        pass
