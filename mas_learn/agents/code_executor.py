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
    
    async def execute_code(self, code: str, globals: dict = None):
        """Execute code with custom globals dictionary"""
        try:
            # Use provided globals or create default ones
            global_vars = globals or {"__name__": "__main__"}
            
            # Add safe builtins
            if '__builtins__' not in global_vars:
                global_vars['__builtins__'] = {
                    'print': print,
                    'len': len,
                    'range': range,
                    'list': list,
                    'dict': dict,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'tuple': tuple,
                    'min': min,
                    'max': max,
                    'sum': sum,
                }
            
            # Execute the code with the specified globals
            exec(code, global_vars)
            return "Code executed successfully"
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