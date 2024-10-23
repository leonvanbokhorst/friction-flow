from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Dict, Any, List
=======
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04

class BaseEmergenceHandler(ABC):
    @abstractmethod
    def detect_patterns(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
<<<<<<< HEAD
=======
        """Detect emergent patterns in the system state."""
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def handle_emergence(self, pattern: Dict[str, Any]) -> None:
<<<<<<< HEAD
=======
        """Handle an emergent pattern."""
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass
