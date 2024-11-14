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
        await self.log_system_activity("start_research_cycle", {"topic": topic})

        # Start research
        cycle_id = await self.agents["researcher"].start_research_cycle(
            topic=topic,
            objectives=[
                "Generate novel approach",
                "Implement solution",
                "Evaluate results",
            ],
        )

        # Log progress at each major step
        await self.log_system_activity(
            "research_cycle_progress", {"cycle_id": cycle_id, "phase": "ideation"}
        )

        # Generate ideas
        mas_logger.console("Generating ideas...")
        ideas = await self.agents["researcher"].generate_ideas(topic)

        # Get user feedback on ideas
        mas_logger.console("\nGenerated ideas:")
        for idx, idea in enumerate(ideas, 1):
            mas_logger.console(f"\n{idx}. {idea['title']}")

        approval = await self.agents["researcher"].report_to_user(
            {"phase": "ideation", "ideas": ideas}
        )

        if approval.lower() != "yes":
            mas_logger.console("Research cycle cancelled by user")
            return {"status": "cancelled", "ideas": []}

        try:
            # Format the first idea as a proper specification for the coder
            selected_idea = ideas[0]
            implementation_spec = {
                "title": selected_idea["title"],
                "architecture": selected_idea["architecture"],
                "implementation": {
                    "libraries": selected_idea.get("components", []),
                    "core_classes": selected_idea.get("implementation", {}).get(
                        "core_classes", []
                    ),
                    "data_pipeline": selected_idea.get("implementation", {}).get(
                        "data_pipeline", []
                    ),
                },
                "evaluation": selected_idea.get("evaluation", []),
                "description": selected_idea["description"],
            }

            code = await self.agents["coder"].write_code(implementation_spec)

            if not code:
                await self.log_system_activity(
                    "implementation_failed",
                    {"reason": "No code generated", "phase": "code_generation"},
                )
                return {
                    "status": "failed",
                    "ideas": ideas,
                    "error": "Code generation failed",
                }

            await self.store_execution_result(
                code=code,
                output="Code generated successfully",
                status="success",
                validation_results={"generation": "successful"},
                metadata={"topic": topic, "selected_idea": selected_idea["title"]},
            )

            return {
                "status": "success",
                "ideas": ideas,
                "implementation": {
                    "code": code,
                    "selected_idea": selected_idea["title"],
                },
            }

        except Exception as e:
            await self.log_system_activity(
                "implementation_error", {"error": str(e), "phase": "implementation"}
            )
            mas_logger.error(f"Implementation failed: {str(e)}", "Orchestrator")
            return {"status": "error", "ideas": ideas, "error": str(e)}

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
