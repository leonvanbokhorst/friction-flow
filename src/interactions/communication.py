from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseCommunication(ABC):
    @abstractmethod
    def send_message(self, sender: str, receiver: str, content: Dict[str, Any], channel: str) -> None:
        """
        Send a message from one agent to another.

        Args:
            sender (str): The ID of the sending agent.
            receiver (str): The ID of the receiving agent.
            content (Dict[str, Any]): The content of the message.
            channel (str): The communication channel (e.g., 'formal', 'informal', 'non-verbal', 'coded').
        """
        pass

    @abstractmethod
    def receive_message(self, receiver: str) -> Dict[str, Any]:
        """
        Receive a message for a specific agent.

        Args:
            receiver (str): The ID of the receiving agent.

        Returns:
            Dict[str, Any]: The received message.
        """
        pass

    @abstractmethod
    def interpret_context(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret the context of a received message.

        Args:
            message (Dict[str, Any]): The message to interpret.

        Returns:
            Dict[str, Any]: The interpreted context.
        """
        pass

    @abstractmethod
    def recognize_intent(self, message: Dict[str, Any]) -> str:
        """
        Recognize the intent of a message.

        Args:
            message (Dict[str, Any]): The message to analyze.

        Returns:
            str: The recognized intent.
        """
        pass

    @abstractmethod
    def generate_response(self, context: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Generate a response based on the context and recognized intent.

        Args:
            context (Dict[str, Any]): The interpreted context.
            intent (str): The recognized intent.

        Returns:
            Dict[str, Any]: The generated response.
        """
        pass

    @abstractmethod
    def handle_group_communication(self, sender: str, receivers: List[str], content: Dict[str, Any]) -> None:
        """
        Handle communication within a group of agents.

        Args:
            sender (str): The ID of the sending agent.
            receivers (List[str]): The IDs of the receiving agents.
            content (Dict[str, Any]): The content of the group message.
        """
        pass

    @abstractmethod
    def encode_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode a message for secure or specialized communication.

        Args:
            message (Dict[str, Any]): The original message.

        Returns:
            Dict[str, Any]: The encoded message.
        """
        pass

    @abstractmethod
    def decode_message(self, encoded_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode an encoded message.

        Args:
            encoded_message (Dict[str, Any]): The encoded message.

        Returns:
            Dict[str, Any]: The decoded message.
        """
        pass
