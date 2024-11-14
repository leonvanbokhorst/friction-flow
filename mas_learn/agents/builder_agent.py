from typing import Dict, List, Type, Any
from typing import Dict, List, Optional, Type
from .base_agent import BaseAgent
from .coder_agent import CoderAgent
import inspect
import ast
from pathlib import Path
import importlib
import re

class BuilderAgent(BaseAgent):
    def __init__(self, name: str, orchestrator=None):
        super().__init__(
            name=name,
            role="Agent Builder",
            capabilities=[
                "create_agent",
                "modify_agent",
                "analyze_agent",
                "teach_agent",
                "enhance_capabilities"
            ],
            orchestrator=orchestrator
        )
        self.agent_templates = {}
        self.created_agents = []
        # Create a dedicated CoderAgent for code analysis
        self.code_analyzer = CoderAgent("code_analyzer", orchestrator=orchestrator)
        self.created_agents.append(self.code_analyzer)

    async def create_agent(self, specification: Dict[str, Any]) -> BaseAgent:
        """
        Create a new agent based on specifications
        
        Args:
            specification: Dict containing agent specifications including:
                - name: Agent name
                - role: Agent role
                - capabilities: List of desired capabilities
                - custom_attributes: Additional attributes
                
        Returns:
            BaseAgent: Newly created agent instance
        """
        # First, analyze the specification and retrieve relevant patterns
        relevant_patterns = await self.recall(str(specification))
        
        # Generate agent class code
        agent_code = await self._generate_agent_code(specification, relevant_patterns)
        
        # Analyze the generated code for safety and quality
        analysis = await self._analyze_agent_code(agent_code)
        
        # Get user approval
        approval = await self.report_to_user({
            'type': 'agent_creation',
            'specification': specification,
            'code_analysis': analysis,
            'agent_code': agent_code
        })
        
        if approval.lower() != 'yes':
            raise ValueError("Agent creation cancelled by user")
            
        # Create and instantiate the agent
        agent_class = self._create_agent_class(agent_code)
        new_agent = agent_class(
            name=specification['name'],
            role=specification['role'],
            capabilities=specification['capabilities']
        )
        
        # Store creation in memory
        await self.learn(f"Created agent: {specification['name']} with role: {specification['role']}")
        self.created_agents.append(new_agent)
        
        return new_agent

    async def teach_agent(self, target_agent: BaseAgent, knowledge: str) -> bool:
        """
        Teach new capabilities or knowledge to an existing agent
        
        Args:
            target_agent: The agent to teach
            knowledge: The knowledge or capability to transfer
            
        Returns:
            bool: Success status
        """
        # Prepare teaching prompt
        teaching_context = await self.recall(knowledge)
        
        teaching_prompt = f"""
        Agent: {target_agent.name}
        Current Role: {target_agent.role}
        Current Capabilities: {target_agent.capabilities}
        
        New Knowledge to Integrate:
        {knowledge}
        
        Relevant Context:
        {teaching_context}
        
        Please process and integrate this knowledge.
        """
        
        # Communicate with target agent
        response = await self.communicate(teaching_prompt, target_agent)
        
        # Verify learning
        verification = await target_agent.recall(knowledge)
        success = len(verification) > 0
        
        await self.learn(f"Teaching session with {target_agent.name}: {'Success' if success else 'Failed'}")
        
        return success

    async def enhance_capabilities(self, target_agent: BaseAgent, new_capability: str) -> bool:
        """
        Add new capabilities to an existing agent
        
        Args:
            target_agent: Agent to enhance
            new_capability: Capability to add
            
        Returns:
            bool: Success status
        """
        # Generate capability implementation
        capability_code = await self._generate_capability_code(new_capability)
        
        # Analyze for safety
        analysis = await self._analyze_agent_code(capability_code)
        
        # Get approval
        approval = await self.report_to_user({
            'type': 'capability_enhancement',
            'target_agent': target_agent.name,
            'new_capability': new_capability,
            'analysis': analysis
        })
        
        if approval.lower() != 'yes':
            return False
            
        # Add capability
        success = await self._add_capability(target_agent, new_capability, capability_code)
        
        if success:
            target_agent.capabilities.append(new_capability)
            await self.learn(f"Enhanced {target_agent.name} with capability: {new_capability}")
            
        return success

    async def _generate_agent_code(self, spec: Dict[str, Any], patterns: List[str] = None) -> str:
        """Generate the Python code for a new agent class.
        
        Args:
            spec: Dictionary containing agent specifications
            patterns: Optional list of design patterns to incorporate
        
        Returns:
            str: Generated Python code for the agent class
        """
        class_name = spec["name"].title().replace("_", "") + "Agent"
        
        code = f'''
class {class_name}(BaseAgent):
    """A specialized agent for {spec['role']}.
    
    Capabilities:
    - {chr(10)+'    - '.join(spec['capabilities'])}
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role = "{spec['role']}"
        self.capabilities = {spec['capabilities']}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implement specific processing logic here
        return {{"status": "success", "result": input_data}}
'''
        return code

    async def _analyze_agent_code(self, code: str) -> Dict:
        """Analyze generated agent code for safety and quality"""
        return await self.code_analyzer.analyze_code(code)

    def _create_agent_class(self, agent_code: str) -> Type[BaseAgent]:
        namespace = {}
        # Add required imports to the namespace
        from typing import Dict, Any
        namespace['BaseAgent'] = BaseAgent
        namespace['Dict'] = Dict
        namespace['Any'] = Any
        
        exec(agent_code, namespace)
        
        # Get the newly created agent class from namespace
        agent_classes = [v for v in namespace.values() if isinstance(v, type) and issubclass(v, BaseAgent)]
        if not agent_classes:
            raise ValueError("No valid agent class found in generated code")
        return agent_classes[0]

    async def _add_capability(self, agent: BaseAgent, capability: str, implementation: str) -> bool:
        """Add new capability to existing agent"""
        try:
            exec(implementation, agent.__dict__)
            return True
        except Exception as e:
            await self.learn(f"Failed to add capability {capability}: {str(e)}")
            return False 