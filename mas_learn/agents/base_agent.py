from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from mas_learn.utils.logger import mas_logger
from mas_learn.memory.vector_memory import VectorMemory


class BaseAgent:
    def __init__(self, name: str, role: str, capabilities: List[str], orchestrator=None):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.orchestrator = orchestrator
        self.memory = VectorMemory()
        self.llm = Ollama(model="qwen2.5-coder:32b")  # qwen2.5-coder:14b

        # Set up agent-specific logger
        self.logger = mas_logger.setup_agent_logger(name)

        # Log initialization
        mas_logger.agent_activity(name, "initialized", {
            "role": role,
            "capabilities": capabilities
        })

    def _initialize_memory(self):
        """Initialize vector storage for agent memory"""
        embeddings = HuggingFaceEmbeddings()
        return Chroma(
            collection_name=f"{self.name}_memory",
            embedding_function=embeddings,
            persist_directory=f"./data/memory/{self.name}",
        )

    def _initialize_llm(self):
        """Initialize Ollama LLM with qwen2.5-coder model"""
        from langchain.llms import Ollama

        return Ollama(model="qwen2.5-coder:32b")

    async def log_activity(self, action: str, details: dict = None):
        """Log agent activity"""
        mas_logger.agent_activity(self.name, action, details)

    async def communicate(
        self, message: str, target_agent: Optional["BaseAgent"] = None
    ):
        """Communicate with other agents or respond to messages"""
        await self.log_activity("communicate", {
            "target": target_agent.name if target_agent else None,
            "message": message
        })
        
        if target_agent:
            return await self._send_message(message, target_agent)
        return await self._process_message(message)

    async def learn(self, information: str) -> None:
        """Store new information in agent's memory."""
        await self.log_activity("learn", {"information": information[:100] + "..."})
        await self.memory.add_memory(information)
        mas_logger.debug(f"Agent {self.name} learned: {information}", "BaseAgent")

    async def recall(self, query: str) -> List[str]:
        """Retrieve relevant information from agent's memory."""
        return await self.memory.similarity_search(query)

    async def report_to_user(self, cycle_info: Dict) -> str:
        """Report progress and ask for permission to continue"""
        message = f"[{self.name}] {cycle_info.get('action', 'Progress')} - {cycle_info.get('message', '')}"
        mas_logger.console(message)

        if cycle_info.get("requires_approval", True):
            return input("Continue? (yes/no): ")
        return "yes"

    async def _send_message(self, message: str, target_agent: "BaseAgent") -> str:
        """
        Send message to another agent

        Args:
            message: Message to send
            target_agent: Recipient agent

        Returns:
            str: Response from target agent
        """
        # Store communication in memory
        await self.learn(f"Sent message to {target_agent.name}: {message}")

        # Process message through target agent
        response = await target_agent._process_message(message)

        # Store response in memory
        await self.learn(f"Received response from {target_agent.name}: {response}")

        return response

    async def _process_message(self, message: str) -> str:
        """
        Process incoming messages using the LLM

        Args:
            message: Input message to process

        Returns:
            str: Processed response
        """
        prompt = f"""
        Agent: {self.name}
        Role: {self.role}
        Capabilities: {self.capabilities}
        
        Message to process:
        {message}
        
        Generate an appropriate response considering the agent's role and capabilities.
        """

        response = await self.llm.agenerate([prompt])
        return response.generations[0].text
