from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEnvironmentComponent(ABC):
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass
