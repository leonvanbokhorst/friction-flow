from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class BaseMemory(ABC):
    @abstractmethod
    def form_memory(self, memory_type: str, content: Dict[str, Any]) -> str:
<<<<<<< HEAD
=======
        """
        Form a new memory of the specified type.

        Args:
            memory_type (str): The type of memory to form (episodic, semantic, emotional, procedural, social).
            content (Dict[str, Any]): The content of the memory.

        Returns:
            str: A unique identifier for the formed memory.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def recall(self, memory_type: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
<<<<<<< HEAD
=======
        """
        Recall a memory based on the given type and query.

        Args:
            memory_type (str): The type of memory to recall.
            query (Dict[str, Any]): The query to search for in the memory.

        Returns:
            Optional[Dict[str, Any]]: The recalled memory, or None if not found.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
<<<<<<< HEAD
=======
        """
        Update an existing memory.

        Args:
            memory_id (str): The unique identifier of the memory to update.
            updates (Dict[str, Any]): The updates to apply to the memory.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def apply_decay(self, memory_type: str = None) -> None:
<<<<<<< HEAD
=======
        """
        Apply decay to memories of the specified type, or all memories if no type is specified.

        Args:
            memory_type (str, optional): The type of memories to decay. If None, decay all types.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def integrate_memories(self, memory_ids: List[str]) -> str:
<<<<<<< HEAD
=======
        """
        Integrate multiple memories into a new, consolidated memory.

        Args:
            memory_ids (List[str]): List of memory IDs to integrate.

        Returns:
            str: The ID of the newly created, integrated memory.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def get_memory_types(self) -> List[str]:
<<<<<<< HEAD
=======
        """
        Get a list of all available memory types.

        Returns:
            List[str]: A list of memory type names.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass

    @abstractmethod
    def clear_memory(self, memory_type: str = None) -> None:
<<<<<<< HEAD
=======
        """
        Clear memories of the specified type, or all memories if no type is specified.

        Args:
            memory_type (str, optional): The type of memories to clear. If None, clear all types.
        """
>>>>>>> 81d85402ba0615a47504a03307dd0c8b5ae0ac04
        pass
