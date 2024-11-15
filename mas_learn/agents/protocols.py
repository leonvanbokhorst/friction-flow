"""
Communication protocols for multi-agent system interactions.
Defines message formats, validation, and routing logic.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class AgentMessage:
    """Message format for agent communication"""
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None

class CommunicationProtocol:
    """Handles agent communication routing and validation"""
    
    @staticmethod
    def validate_message(message: AgentMessage) -> bool:
        """Validate message format and content"""
        required_fields = ['sender', 'recipient', 'content', 'message_type']
        return all(hasattr(message, field) for field in required_fields)
    
    @staticmethod
    async def route_message(message: AgentMessage, agents: Dict[str, 'BaseAgent']):
        """Route message to appropriate agent"""
        if message.recipient in agents:
            return await agents[message.recipient].communicate(message.content)
        raise ValueError(f"Unknown recipient: {message.recipient}") 