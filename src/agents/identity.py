from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseIdentity(ABC):
    @abstractmethod
    def get_personal_history(self) -> Dict[str, Any]:
<<<<<<< HEAD
=======
        """
        Get the agent's personal history.

        Returns:
            Dict[str, Any]: A dictionary containing key events and experiences in the agent's personal life.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_professional_journey(self) -> Dict[str, Any]:
<<<<<<< HEAD
=======
        """
        Get the agent's professional journey.

        Returns:
            Dict[str, Any]: A dictionary detailing the agent's career path, achievements, and milestones.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_core_motivations(self) -> List[str]:
<<<<<<< HEAD
=======
        """
        Get the agent's core motivations.

        Returns:
            List[str]: A list of the agent's primary motivations and drivers.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_value_system(self) -> Dict[str, float]:
<<<<<<< HEAD
=======
        """
        Get the agent's value system.

        Returns:
            Dict[str, float]: A dictionary mapping values to their importance (0.0 to 1.0) for the agent.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_psychological_profile(self) -> Dict[str, Any]:
<<<<<<< HEAD
=======
        """
        Get the agent's psychological profile.

        Returns:
            Dict[str, Any]: A dictionary containing key psychological traits and characteristics.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_behavioral_patterns(self) -> Dict[str, Any]:
<<<<<<< HEAD
=======
        """
        Get the agent's behavioral patterns.

        Returns:
            Dict[str, Any]: A dictionary describing typical behaviors in various situations.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def update_identity(self, updates: Dict[str, Any]) -> None:
<<<<<<< HEAD
=======
        """
        Update aspects of the agent's identity.

        Args:
            updates (Dict[str, Any]): A dictionary containing the updates to apply to the identity.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass
