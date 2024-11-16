from pathlib import Path
from typing import List, Dict, Any
import json
import yaml
from base_agent import BaseAgent
from specialized_agent import SpecializedAgent
from langchain_ollama import OllamaLLM
from tqdm import tqdm
import logging
import asyncio

MODEL = "hermes3:latest"
EMBEDDING_MODEL = "nomic-embed-text"

# Add logging configuration at the top
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentFactory:
    def __init__(self):
        self.agents_dir = Path("agent_factory/agents")
        self.agents_dir.mkdir(exist_ok=True)
        self.agents: Dict[str, BaseAgent] = {}
        logger.info("Initializing AgentFactory")
        self.load_existing_agents()

    def load_existing_agents(self):
        logger.info("Loading existing agents from disk")
        agent_count = 0
        for agent_file in self.agents_dir.glob("*.yaml"):
            try:
                with open(agent_file, "r") as f:
                    agent_config = yaml.safe_load(f)
                    self.create_agent(agent_config)
                    agent_count += 1
            except Exception as e:
                logger.error(f"Failed to load agent from {agent_file}: {str(e)}")
        logger.info(f"Loaded {agent_count} existing agents")

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
                        system_prompt
                        + f"\n\nGenerate agent #{agent_num} following the exact format above."
                        + f"\nAvoid these names: {list(used_names)}"
                        + f"\nAvoid these roles: {list(used_roles)}"
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
                print(
                    f"Failed to create agent {config.get('agent_id', 'unknown')}: {str(e)}"
                )
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

    def generate_task_specific_agents(
        self, question: str, max_agents: int = 5
    ) -> List[BaseAgent]:
        """Generates a team of specialized agents based on the question/task."""

        llm = OllamaLLM(model=MODEL)
        analysis_prompt = """Given this question/task, determine the necessary expert roles needed to solve it effectively.
        
        Question: %s
        
        Provide your response in this exact JSON format, with no additional text or formatting:
        {
            "roles": [
                {
                    "role": "specific professional role",
                    "justification": "why this role is needed",
                    "key_responsibilities": ["responsibility1", "responsibility2"]
                }
            ]
        }

        Constraints:
        - Suggest %d or fewer roles
        - Each role must be directly relevant to the question
        - Roles should be complementary, not redundant""" % (
            question,
            max_agents,
        )

        try:
            # Make multiple attempts to get valid JSON
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    response = llm.invoke(analysis_prompt)
                    # Clean the response to ensure it's valid JSON
                    cleaned_response = response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response.split("```json")[1]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response.rsplit("```", 1)[0]
                    cleaned_response = cleaned_response.strip()

                    role_analysis = json.loads(cleaned_response)

                    # Validate expected structure
                    if (
                        not isinstance(role_analysis, dict)
                        or "roles" not in role_analysis
                    ):
                        continue

                    # Generate agents for each identified role
                    task_agents = []
                    for role_info in role_analysis["roles"]:
                        agent_config = self._generate_agent_config(role_info)
                        agent = self.create_agent(agent_config)
                        task_agents.append(agent)

                    return task_agents

                except json.JSONDecodeError:
                    if attempt == max_attempts - 1:
                        raise
                    continue

        except Exception as e:
            raise ValueError(f"Failed to generate task-specific agents: {str(e)}")

    def _generate_agent_config(self, role_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a specific agent configuration based on role requirements."""

        llm = OllamaLLM(model=MODEL)
        config_prompt = f"""Generate an agent configuration for this role:
        Role: {role_info['role']}
        Responsibilities: {role_info['key_responsibilities']}
        
        Return ONLY valid JSON in this exact format, with no additional text or markdown:
        {{
            "agent_id": "unique_id",
            "name": "professional name",
            "role": "{role_info['role']}",
            "traits": ["trait1", "trait2", "trait3"],
            "goals": ["goal1", "goal2"],
            "expertise_areas": ["area1", "area2"]
        }}
        """

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(config_prompt)
                # Clean the response to handle potential markdown or extra text
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response.split("```json")[1]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response.rsplit("```", 1)[0]
                cleaned_response = cleaned_response.strip()

                config = json.loads(cleaned_response)
                config["agent_id"] = f"agent_{len(self.agents) + 1}"
                return config

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1} failed to parse JSON: {str(e)}")
                if attempt == max_attempts - 1:
                    raise ValueError(
                        f"Failed to generate valid JSON after {max_attempts} attempts"
                    )
                continue

    def _create_paper_writing_agent(self) -> BaseAgent:
        """Creates a specialized agent for writing technical papers."""
        config = {
            "agent_id": f"paper_writer_{len(self.agents) + 1}",
            "name": "Dr. Technical Writer",
            "role": "Technical Paper Author",
            "traits": ["analytical", "organized", "detail-oriented"],
            "goals": ["synthesize information", "create coherent narrative"],
            "expertise_areas": ["technical writing", "research synthesis"],
        }
        return self.create_agent(config)

    def solve_problem_collaboratively(
        self, question: str, max_agents: int = 20, max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Coordinates multiple agents to solve a complex problem with iterative refinement."""
        logger.info(f"Starting collaborative problem solving for question: {question}")

        # Generate appropriate agents for the task
        task_agents = self.generate_task_specific_agents(question, max_agents)
        logger.info(f"Generated {len(task_agents)} task-specific agents")

        # Initial responses
        insights = {agent.personality["role"]: [] for agent in task_agents}
        convergence_count = 0
        iteration = 0

        while iteration < max_iterations and convergence_count < 2:
            new_insights_count = 0

            for agent in task_agents:
                current_role = agent.personality["role"]

                # Build context from previous insights
                previous_insights = "\n".join(
                    [
                        f"{role}: {', '.join(role_insights)}"
                        for role, role_insights in insights.items()
                        if role_insights
                    ]
                )

                # Generate prompt for new insight
                if iteration == 0:
                    prompt = f"Given the question: {question}\nProvide one key insight from your expertise."
                else:
                    prompt = f"""
                    Question: {question}
                    Previous insights:\n{previous_insights}
                    
                    Based on your expertise as {current_role}:
                    1. Provide ONE new insight that hasn't been mentioned
                    2. Suggest which team member should expand on your insight and why
                    """

                try:
                    response = self._get_safe_response(agent, prompt, task_agents)

                    # Parse response and add new insight
                    if response and "Error:" not in response:
                        insights[current_role].append(response)
                        new_insights_count += 1

                        # Allow suggested agent to respond immediately
                        if (
                            iteration > 0
                            and "should expand on this" in response.lower()
                        ):
                            # Extract suggested role and get corresponding agent
                            # (Implementation details omitted for brevity)
                            pass

                except Exception as e:
                    logger.error(
                        f"Failed to get response from {current_role}: {str(e)}"
                    )

            # Check for convergence
            if new_insights_count == 0:
                convergence_count += 1
            else:
                convergence_count = 0

            iteration += 1

        # Generate final paper
        paper_prompt = f"""
        Based on all insights collected:
        {json.dumps(insights, indent=2)}
        
        Generate a comprehensive research paper that:
        1. Synthesizes all insights
        2. Provides clear methodology
        3. Includes technical implementation details
        4. Discusses potential challenges and solutions
        5. Concludes with concrete recommendations
        
        Format as a proper academic paper with sections.
        """

        paper_agent = self._create_paper_writing_agent()
        final_paper = self._get_safe_response(paper_agent, paper_prompt, task_agents)

        return {
            "insights": insights,
            "paper": final_paper,
            "iterations": iteration,
            "converged": convergence_count >= 2,
        }

    def _get_safe_response(
        self, agent: BaseAgent, question: str, other_agents: List[BaseAgent]
    ) -> str:
        """Safely get response from an agent with proper encoding handling."""
        try:
            # Get raw response
            raw_response = agent.collaborate_on_task(question, other_agents)

            # Handle different response types
            if isinstance(raw_response, bytes):
                # First try to decode with utf-8 and ignore errors
                try:
                    return raw_response.decode("utf-8", errors="ignore")
                except Exception:
                    # Fallback to latin-1 which can handle any byte sequence
                    return raw_response.decode("latin-1", errors="replace")
            elif isinstance(raw_response, str):
                return raw_response
            else:
                return str(raw_response)

        except Exception as e:
            logger.error(f"Error in _get_safe_response: {str(e)}")
            return f"Error processing response: {str(e)}"


if __name__ == "__main__":
    factory = AgentFactory()
    question = "In what ways can we build a model that scores near 99 percent accuracy on the MNIST dataset?"

    results = factory.solve_problem_collaboratively(
        question, max_agents=5, max_iterations=5
    )

    # Print insights by role
    print("\n=== Collected Insights ===")
    for role, role_insights in results["insights"].items():
        print(f"\n{role}:")
        for insight in role_insights:
            print(f"- {insight}")

    print("\n=== Final Research Paper ===")
    print(results["paper"])
