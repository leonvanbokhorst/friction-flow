from typing import Dict, Any
from base_agent import BaseAgent


class SpecializedAgent(BaseAgent):
    def __init__(self, agent_id: str, personality: Dict[str, Any]):
        super().__init__(agent_id, personality)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""You are {self.personality['name']}, a {self.personality['role']}.
        Traits: {', '.join(self.personality['traits'])}
        Goals: {', '.join(self.personality['goals'])}
        """

    def process_task(self, task: str) -> str:
        # Log task receipt
        self.logger.info(f"Processing task: {task}")

        # Generate response using Hermes
        response = self.ollama_client.chat(
            model="hermes3:latest",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
            ],
        )

        # Store interaction in memory
        self.store_memory(f"Task: {task}\nResponse: {response['message']['content']}")

        # Log completion
        self.logger.info(f"Completed task: {task}")

        return response["message"]["content"]
