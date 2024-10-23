from abc import ABC, abstractmethod

class BaseEnvironmentComponent(ABC):
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the component state."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the component."""
        pass
