from abc import ABC, abstractmethod
from typing import Optional, Dict

class BaseResourceManagement(ABC):
    @abstractmethod
    def allocate_resource(self, agent_id: str, resource_type: str, amount: float) -> bool:
        """
        Allocate a resource to an agent.

        Args:
            agent_id (str): The ID of the agent receiving the resource.
            resource_type (str): The type of resource being allocated.
            amount (float): The amount of the resource to allocate.

        Returns:
            bool: True if the allocation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def consume_resource(self, agent_id: str, resource_type: str, amount: float) -> bool:
        """
        Consume a resource from an agent's holdings.

        Args:
            agent_id (str): The ID of the agent consuming the resource.
            resource_type (str): The type of resource being consumed.
            amount (float): The amount of the resource to consume.

        Returns:
            bool: True if the consumption was successful, False if insufficient resources.
        """
        pass

    @abstractmethod
    def trade_resource(self, sender: str, receiver: str, resource_type: str, amount: float) -> bool:
        """
        Trade a resource between two agents.

        Args:
            sender (str): The ID of the agent sending the resource.
            receiver (str): The ID of the agent receiving the resource.
            resource_type (str): The type of resource being traded.
            amount (float): The amount of the resource to trade.

        Returns:
            bool: True if the trade was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_resource_status(self, agent_id: str) -> Dict[str, float]:
        """
        Get the current resource status for an agent.

        Args:
            agent_id (str): The ID of the agent to check.

        Returns:
            Dict[str, float]: A dictionary containing the agent's current resource holdings.
        """
        pass
