from pathlib import Path
from typing import List, Dict, Any
import json
import yaml
from base_agent import BaseAgent
from specialized_agent import SpecializedAgent
from langchain_ollama import OllamaLLM
from tqdm import tqdm

MODEL = "hermes3:latest"


class AgentFactory:
    def __init__(self):
        self.agents_dir = Path("agent_factory/agents")
        self.agents_dir.mkdir(exist_ok=True)
        self.agents: Dict[str, BaseAgent] = {}
        self.load_existing_agents()

    def load_existing_agents(self):
        for agent_file in self.agents_dir.glob("*.yaml"):
            with open(agent_file, "r") as f:
                agent_config = yaml.safe_load(f)
                self.create_agent(agent_config)

    def auto_generate_agents(self, num_agents: int = 50) -> List[BaseAgent]:
        """Automatically generates a diverse set of useful agents using Ollama."""
        llm = OllamaLLM(model=MODEL)

        system_prompt = """Generate a unique agent configuration in the following exact JSON format:
        {
            "agent_id": "agent_[number]",
            "name": "[professional name]",
            "role": "[specific professional role]",
            "traits": ["trait1", "trait2", "trait3"],
            "goals": ["goal1", "goal2"]
        }
        
        Important constraints:
        1. Each name MUST be completely unique from all other agents
        2. Roles must be diverse across different professional domains (tech, healthcare, business, creative, etc.)
        3. Traits should be unique combinations - avoid repeating common traits like "analytical" too often
        4. Goals should be specific to the role and not generic
        5. No two agents should have the same combination of role and goals
        
        Make the agent realistic for a professional environment. Return only valid JSON, no other text."""

        used_names = set()
        used_roles = set()
        agents_configs = []
        max_attempts = num_agents * 3  # Allow for some retries
        attempts = 0
        
        with tqdm(total=num_agents, desc="Generating agents", unit="agent") as pbar:
            while len(agents_configs) < num_agents and attempts < max_attempts:
                try:
                    agent_num = len(agents_configs) + 1
                    response = llm.invoke(
                        system_prompt + 
                        f"\n\nGenerate agent #{agent_num} following the exact format above." +
                        f"\nAvoid these names: {list(used_names)}" +
                        f"\nAvoid these roles: {list(used_roles)}"
                    )
                    config = json.loads(response)
                    
                    # Ensure agent_id is correct
                    config["agent_id"] = f"agent_{agent_num}"
                    
                    # Skip if name or role already exists
                    if config["name"] in used_names or config["role"] in used_roles:
                        attempts += 1
                        continue
                    
                    used_names.add(config["name"])
                    used_roles.add(config["role"])
                    agents_configs.append(config)
                    pbar.update(1)
                except Exception as e:
                    print(f"Failed to generate agent #{agent_num}: {str(e)}")
                    attempts += 1
                    continue

        if not agents_configs:
            raise ValueError("Failed to generate any valid agents")

        # Create agents from configurations
        created_agents = []
        for config in tqdm(agents_configs, desc="Creating agents", unit="agent"):
            try:
                agent = self.create_agent(config)
                created_agents.append(agent)
            except ValueError as e:
                print(f"Failed to create agent {config.get('agent_id', 'unknown')}: {str(e)}")
                continue

        if not created_agents:
            raise ValueError("Failed to create any agents")

        return created_agents

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validates agent configuration for required fields and uniqueness."""
        required_fields = ["agent_id", "name", "role", "traits", "goals"]
        if not all(field in config for field in required_fields):
            raise ValueError(f"Missing required fields. Required: {required_fields}")

        if config["agent_id"] in self.agents:
            raise ValueError(f"Agent ID {config['agent_id']} already exists")

        return True

    def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """Creates a new agent after validation."""
        if len(self.agents) >= 50:
            raise ValueError("Maximum number of agents (50) reached")

        self._validate_config(config)

        agent_id = config["agent_id"]

        # Save agent configuration
        config_path = self.agents_dir / f"{agent_id}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create agent instance
        agent = self._instantiate_agent(config)
        self.agents[agent_id] = agent
        return agent

    def _instantiate_agent(self, config: Dict[str, Any]) -> BaseAgent:
        personality = {
            "name": config["name"],
            "role": config["role"],
            "traits": config["traits"],
            "goals": config["goals"],
        }

        return SpecializedAgent(config["agent_id"], personality)


if __name__ == "__main__":
    factory = AgentFactory()

    agents = factory.auto_generate_agents(num_agents=10)

    # Use the agent
    response = agents[0].process_task("Analyze this dataset and provide insights")
    print(response)
