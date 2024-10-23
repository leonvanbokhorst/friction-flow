from abc import ABC, abstractmethod

class BaseInteraction(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the interaction."""
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """Validate the interaction parameters."""
        pass
