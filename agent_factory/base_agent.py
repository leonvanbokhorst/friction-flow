from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from ollama import Client
import redis
from redis.exceptions import ConnectionError


class BaseAgent(ABC):
    def __init__(self, agent_id: str, personality: Dict[str, Any]):
        self.agent_id = agent_id
        self.personality = personality
        self.ollama_client = Client()
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, decode_responses=True
            )
            self.redis_client.ping()  # Test the connection
        except ConnectionError:
            print("Could not connect to Redis. Please ensure Redis server is running.")
            # Handle the error appropriately (e.g., fallback to alternative storage, exit gracefully)
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for the agent"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(f"agent_{self.agent_id}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_dir / f"{self.agent_id}.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama"""
        response = self.ollama_client.embeddings(
            model="nomic-embed-text", prompt=text  
        )
        return response["embedding"]

    def store_memory(self, memory: str):
        embedding = self.get_embedding(memory)
        memory_key = f"{self.agent_id}:memory:{datetime.now().isoformat()}"
        self.redis_client.hset(
            memory_key,
            mapping={"text": memory, "embedding": np.array(embedding).tobytes()},
        )

    def send_message(self, recipient_id: str, message: str):
        embedding = self.get_embedding(message)
        message_key = f"messages:{datetime.now().isoformat()}"
        self.redis_client.hset(
            message_key,
            mapping={
                "sender": self.agent_id,
                "recipient": recipient_id,
                "message": message,
                "embedding": np.array(embedding).tobytes(),
            },
        )

    def find_similar_memories(self, query: str, limit: int = 5) -> List[str]:
        """Find similar memories using semantic search"""
        query_embedding = self.get_embedding(query)

        # Get all memories for this agent
        memories = []
        for key in self.redis_client.scan_iter(f"{self.agent_id}:memory:*"):
            memory_data = self.redis_client.hgetall(key)
            embedding = np.frombuffer(memory_data["embedding"])
            similarity = np.dot(query_embedding, embedding)
            memories.append((similarity, memory_data["text"]))

        # Sort by similarity and return top matches
        memories.sort(reverse=True)
        return [m[1] for m in memories[:limit]]
