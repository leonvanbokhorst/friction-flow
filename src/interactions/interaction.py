from abc import ABC, abstractmethod
from typing import Any

class BaseInteraction(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        pass
