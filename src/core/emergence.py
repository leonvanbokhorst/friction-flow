from abc import ABC, abstractmethod

class BaseEmergenceHandler(ABC):
    @abstractmethod
    def detect_patterns(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emergent patterns in the system state."""
        pass

    @abstractmethod
    def handle_emergence(self, pattern: Dict[str, Any]) -> None:
        """Handle an emergent pattern."""
        pass
