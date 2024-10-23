from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseIdentity(ABC):
    @abstractmethod
    def get_personal_history(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_professional_journey(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_core_motivations(self) -> List[str]:
        pass

    @abstractmethod
    def get_value_system(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_psychological_profile(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_behavioral_patterns(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_identity(self, updates: Dict[str, Any]) -> None:
        pass
