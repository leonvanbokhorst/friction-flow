import ast
import asyncio
from .base_agent import BaseAgent
from typing import Optional, Dict

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
        try:
            # Get coder agent for code cleaning and recovery
            coder_agent = self.orchestrator.agents.get("coder")
            if coder_agent:
                code = await coder_agent._clean_code_block(code)
            
            # Perform static analysis
            try:
                ast.parse(code)
            except SyntaxError as e:
                # Attempt recovery through CoderAgent
                if coder_agent:
                    recovery_result = await coder_agent.recover_from_error(
                        e, code, {"stage": "syntax_validation"}
                    )
                    if recovery_result["status"] == "success":
                        code = await coder_agent._clean_code_block(recovery_result["fixed_code"])
                        print("\nRecovered from syntax errors:")
                        print("Changes made:", recovery_result["changes"])
                    else:
                        return f"Code contains syntax errors: {str(e)}"
                else:
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
                result = await self._safe_execute(code)
                await self.orchestrator.store_execution_result(
                    code=code,
                    output=str(result),
                    status="success",
                    validation_results={"execution": "successful"}
                )
                return result
            except Exception as e:
                # Attempt runtime error recovery
                coder_agent = self.orchestrator.agents.get("coder")
                if coder_agent:
                    recovery_result = await coder_agent.recover_from_error(
                        e, code, {"stage": "runtime_execution"}
                    )
                    if recovery_result["status"] == "success":
                        print("\nRecovered from runtime error:")
                        print("Changes made:", recovery_result["changes"])
                        return await self.execute_code(recovery_result["fixed_code"])
                    
                await self.orchestrator.store_execution_result(
                    code=code,
                    output=str(e),
                    status="error",
                    validation_results={"execution": "failed"}
                )
                return f"Execution error: {str(e)}"
            
        except Exception as e:
            return f"Execution failed: {str(e)}"
    
    async def _safe_execute(self, code: str, args: Optional[Dict] = None) -> str:
        """
        Execute code in a sandboxed environment with ML-specific allowances
        
        Args:
            code: Code to execute
            args: Optional arguments for execution
            
        Returns:
            str: Execution result
        """
        # Create a restricted globals dictionary with ML essentials
        safe_globals = {
            '__builtins__': {
                # Class building
                '__build_class__': __build_class__,
                # Basic operations
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'set': set,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'tuple': tuple,
                'min': min,
                'max': max,
                'sum': sum,
                'enumerate': enumerate,
                'zip': zip,
                # ML essentials
                '__import__': __import__,  # Required for importing ML libraries
                'isinstance': isinstance,
                'getattr': getattr,
                'hasattr': hasattr,
                'ValueError': ValueError,
                'TypeError': TypeError,
            }
        }
        
        # Pre-import common ML libraries
        try:
            safe_globals['np'] = __import__('numpy')
            safe_globals['pd'] = __import__('pandas')
            safe_globals['torch'] = __import__('torch')
        except ImportError:
            pass  # Continue if some libraries aren't available
        
        # Add provided arguments to globals
        if args:
            safe_globals.update(args)
        
        try:
            # Execute code in restricted environment
            exec(code, safe_globals, {})
            return str(safe_globals.get('result', 'Code executed successfully'))
        except Exception as e:
            raise RuntimeError(f"Safe execution failed: {str(e)}")