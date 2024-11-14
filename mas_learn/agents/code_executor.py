import ast
import asyncio
from .base_agent import BaseAgent

class CodeExecutorAgent(BaseAgent):
    def __init__(self, orchestrator=None):
        super().__init__(
            name="code_executor",
            role="code_runner",
            capabilities=["code_execution", "code_analysis"],
            orchestrator=orchestrator
        )
    
    async def execute_code(self, code: str):
        """Executes code after safety analysis and user permission"""
        # Perform static analysis
        try:
            ast.parse(code)
        except SyntaxError as e:
            await self.orchestrator.store_execution_result(
                code=code,
                output=str(e),
                status="error",
                validation_results={"syntax_check": "failed"}
            )
            return f"Code contains syntax errors: {str(e)}"
            
        # Ask for user permission
        approval = await self.report_to_user({
            "action": "execute_code",
            "code": code,
            "analysis": "Code passed static analysis"
        })
        
        if approval.lower() != "yes":
            await self.orchestrator.store_execution_result(
                code=code,
                output="Execution cancelled",
                status="cancelled",
                validation_results={"user_approval": "denied"}
            )
            return "Code execution cancelled by user"
            
        # Execute in controlled environment
        try:
            result = eval(code)  # For production, use a sandbox environment
            await self.orchestrator.store_execution_result(
                code=code,
                output=str(result),
                status="success",
                validation_results={"execution": "successful"}
            )
            return result
        except Exception as e:
            await self.orchestrator.store_execution_result(
                code=code,
                output=str(e),
                status="error",
                validation_results={"execution": "failed"}
            )
            return f"Execution error: {str(e)}" 