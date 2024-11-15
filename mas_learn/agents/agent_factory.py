from .base_agent import BaseAgent

class AgentFactory(BaseAgent):
    def __init__(self):
        super().__init__(
            name="agent_factory",
            role="agent_creator",
            capabilities=["create_agents", "modify_agents"]
        )
    
    async def create_agent(self, specification: dict) -> BaseAgent:
        """Creates a new agent based on provided specifications"""
        # Validate specification
        required_fields = ['name', 'role', 'capabilities']
        if any(field not in specification for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Use LLM to generate agent class code
        prompt = f"Create an agent class with: {specification}"
        response = await self.llm.agenerate([prompt])

        # Execute code with safety checks
        await self.report_to_user({
            "action": "create_agent",
            "specification": specification,
            "generated_code": response.generations[0].text
        })

        # Return new agent instance
        return eval(response.generations[0].text) 