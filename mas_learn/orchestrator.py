from typing import Dict, List, Optional
from mas_learn.agents import (
    BuilderAgent,
    ResearcherAgent,
    CoderAgent,
    MLAgent,
    CodeExecutorAgent,
)
from mas_learn.utils.logger import mas_logger
from mas_learn.storage.results_store import ResultsStore, ExecutionResult
from datetime import datetime
import asyncio


class MultiAgentOrchestrator:
    def __init__(self):
        mas_logger.debug("Initializing Multi-Agent System", "Orchestrator")
        self.agents = {
            "builder": BuilderAgent("builder", orchestrator=self),
            "researcher": ResearcherAgent("researcher", orchestrator=self),
            "coder": CoderAgent("coder", orchestrator=self),
            "ml_engineer": MLAgent("ml_engineer", orchestrator=self),
            "executor": CodeExecutorAgent(orchestrator=self),
        }

        # Create system activity log
        self.system_logger = mas_logger.setup_agent_logger("system")
        mas_logger.console("Multi-Agent System initialized successfully")

        # Initialize results store
        self.results_store = ResultsStore()

    async def log_system_activity(self, action: str, details: dict = None):
        """Log system-wide activity"""
        mas_logger.agent_activity("system", action, details)

    async def execute_research_cycle(self, topic: str) -> dict:
        # Reference existing implementation structure
        reference_impl = {
            "startLine": 36,
            "endLine": 125
        }
        
        try:
            # Initialize research cycle
            cycle_id = await self.agents["researcher"].start_research_cycle(
                topic=topic,
                objectives=self._get_research_objectives(topic)
            )
            
            # Parallel execution of research and analysis
            results = await asyncio.gather(
                self.agents["researcher"].web_search(topic),
                self.agents["ml_engineer"].analyze_architecture_options({"objective": topic}),
                self.agents["coder"].prepare_implementation_plan({"objective": topic})
            )
            
            # Synthesize findings
            synthesis = await self.agents["researcher"].synthesize_findings(results[0])
            
            # Get architecture design
            architecture = await self.agents["ml_engineer"].design_architecture({
                "research_findings": synthesis,
                "technical_requirements": results[1]
            })
            
            # Generate implementation
            implementation = await self.agents["coder"].implement_solution({
                "architecture": architecture,
                "implementation_plan": results[2]
            })
            
            return {
                "status": "success",
                "cycle_id": cycle_id,
                "synthesis": synthesis,
                "architecture": architecture,
                "implementation": implementation
            }
            
        except Exception as e:
            await self.log_system_activity("research_cycle_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}

    async def store_execution_result(
        self,
        code: str,
        output: str,
        status: str,
        validation_results: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store execution result and return result ID"""
        result = ExecutionResult(
            code=code,
            output=output,
            timestamp=datetime.now(),
            status=status,
            validation_results=validation_results,
            metadata=metadata,
        )
        return await self.results_store.store_execution(result)
