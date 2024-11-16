from typing import Dict, Any, List
from base_agent import BaseAgent


class SpecializedAgent(BaseAgent):
    def __init__(self, agent_id: str, personality: Dict[str, Any]):
        super().__init__(agent_id, personality)
        self.system_prompt = self._build_system_prompt()
        self.expertise_areas = personality.get('expertise_areas', [])

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

    def collaborate_on_task(self, task: str, other_agents: List['SpecializedAgent']) -> str:
        try:
            # First, analyze the task from this agent's perspective
            initial_thoughts = self.process_task(f"Given my role as {self.personality['role']}, "
                                             f"what are my initial thoughts on: {task}")
            
            # Share insights with other agents
            for agent in other_agents:
                try:
                    self.send_message(agent.agent_id, initial_thoughts)
                except Exception as e:
                    self.logger.error(f"Failed to send message to {agent.agent_id}: {str(e)}")
            
            # Get relevant memories that might help
            try:
                relevant_memories = self.find_similar_memories(task)
            except Exception as e:
                self.logger.error(f"Failed to retrieve memories: {str(e)}")
                relevant_memories = []
            
            # Generate final response considering collaboration
            collaborative_prompt = f"""Task: {task}
            My Role: {self.personality['role']}
            My Initial Analysis: {initial_thoughts}
            Relevant Past Experiences: {relevant_memories}
            Other Team Members: {[f"{a.personality['name']} ({a.personality['role']})" for a in other_agents]}
            
            Provide a response that:
            1. Addresses the task from my expertise perspective
            2. Acknowledges how my input fits with the team
            3. Suggests points for collaboration with specific team members
            """
            
            return self.process_task(collaborative_prompt)
        except Exception as e:
            self.logger.error(f"Error in collaborate_on_task: {str(e)}")
            return f"Error in collaboration: {str(e)}"
