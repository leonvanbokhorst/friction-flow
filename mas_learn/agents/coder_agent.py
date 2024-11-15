from typing import Dict, List, Optional, Tuple, Any
from .base_agent import BaseAgent
import ast
import inspect
from pathlib import Path
import json
from mas_learn.utils.code_analyzer import CodeAnalyzer
import difflib

class CoderAgent(BaseAgent):
    def __init__(self, name: str, orchestrator=None):
        super().__init__(
            name=name,
            role="Code Developer",
            capabilities=[
                "write_code", 
                "enhance_code", 
                "document_code",
                "analyze_code",
                "execute_code"
            ],
            orchestrator=orchestrator
        )
        self.code_history = []
        self.code_analyzer = CodeAnalyzer()
        
    async def write_code(self, specification: str) -> str:
        """Write code based on specification"""
        # Validate specification completeness
        required_fields = ['architecture', 'components', 'evaluation']
        if isinstance(specification, str):
            try:
                spec_dict = json.loads(specification)
            except:
                raise ValueError("Specification must be a valid JSON string or dictionary")
        else:
            spec_dict = specification
        
        missing_fields = [field for field in required_fields if field not in spec_dict]
        if missing_fields:
            raise ValueError(f"Incomplete specification. Missing: {missing_fields}")
        
        await self.log_activity("write_code_start", {
            "specification": str(spec_dict)[:200] + "..." if len(str(spec_dict)) > 200 else str(spec_dict)
        })
        
        try:
            # Log the code generation attempt
            await self.log_activity("generating_code", {
                "status": "in_progress"
            })
            
            code = await self._generate_code(specification)
            
            # Log successful code generation
            await self.log_activity("code_generated", {
                "code_length": len(code),
                "status": "success"
            })
            
            return code
            
        except Exception as e:
            # Log the error
            await self.log_activity("code_generation_failed", {
                "error": str(e),
                "status": "failed"
            })
            return ""

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Perform static analysis on code to identify potential improvements
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dict containing analysis results
        """
        return self.code_analyzer.analyze_code(code)

    async def execute_code(self, code: str, args: Dict = None) -> Tuple[bool, str]:
        """
        Safely execute code with user permission
        
        Args:
            code: Code to execute
            args: Optional arguments for code execution
            
        Returns:
            Tuple of (success, result/error message)
        """
        # First, analyze the code for safety
        analysis = await self.analyze_code(code)
        
        # Prepare execution report
        execution_report = f"""
        Code Execution Request:
        
        Analysis Results:
        {analysis}
        
        Code to execute:
        {code}
        
        Arguments: {args or 'None'}
        
        Do you approve this code execution? (yes/no)
        """
        
        # Get user permission
        user_approval = await self.report_to_user({
            'type': 'code_execution',
            'report': execution_report
        })
        
        if user_approval.lower() != 'yes':
            return False, "Code execution cancelled by user"
            
        try:
            # Execute in restricted environment
            result = await self._safe_execute(code, args)
            return True, result
        except Exception as e:
            return False, f"Execution error: {str(e)}"

    async def prepare_implementation_plan(self, research_task: Dict) -> Dict:
        """
        Prepare a detailed implementation plan based on research task
        
        Args:
            research_task: Dictionary containing task specifications
            
        Returns:
            Dict: Implementation plan with code structure and components
        """
        try:
            # Create implementation plan using LLM
            prompt = f"""
            Create a detailed implementation plan for:
            {json.dumps(research_task, indent=2)}
            
            Include:
            1. Code structure and organization
            2. Required dependencies
            3. Core classes and functions
            4. Data processing pipeline
            5. Testing strategy
            
            Format as a structured JSON response.
            """
            
            response = await self.llm.ainvoke(prompt)
            
            # Parse and validate response
            try:
                plan = json.loads(response)
            except json.JSONDecodeError:
                # If response isn't valid JSON, create structured format
                plan = {
                    "code_structure": response.split("\n"),
                    "dependencies": [],
                    "core_components": [],
                    "data_pipeline": [],
                    "testing": []
                }
                
            await self.log_activity("implementation_plan_created", {
                "task": research_task["objective"],
                "plan_size": len(str(plan))
            })
            
            # Store implementation plan artifact
            await self.store_artifact(
                "implementation_plan",
                plan,
                {"task": research_task["objective"]}
            )
            
            return plan
            
        except Exception as e:
            await self.log_activity("implementation_plan_failed", {"error": str(e)})
            raise

    async def implement_solution(self, inputs: Dict) -> Dict:
        """Implement solution with error recovery"""
        try:
            # Extract inputs
            architecture = inputs.get("architecture", {})
            implementation_plan = inputs.get("implementation_plan", {})
            
            # Generate implementation code
            code = await self.write_code({
                "architecture": architecture,
                "components": implementation_plan.get("core_components", []),
                "evaluation": implementation_plan.get("testing", [])
            })
            
            # Clean the code before validation
            code = await self._clean_code_block(code)
            
            # Validate the generated code
            validation = await self.analyze_code(code)
            
            if validation.get("errors", []):
                # Attempt recovery if validation fails
                recovery_result = await self.recover_from_error(
                    ValueError("Code validation failed"),
                    code,
                    {"validation": validation, "inputs": inputs}
                )
                
                if recovery_result["status"] == "success":
                    code = await self._clean_code_block(recovery_result["fixed_code"])
                else:
                    raise ValueError("Failed to generate valid code")
            
            # Store implementation artifact
            await self.store_artifact(
                "solution_implementation",
                {
                    "code": code,
                    "architecture": architecture,
                    "plan": implementation_plan
                },
                {"status": "completed"}
            )
            
            return {
                "code": code,
                "status": "success",
                "architecture": architecture["design_summary"]
            }
            
        except Exception as e:
            # Attempt recovery for any implementation error
            recovery_result = await self.recover_from_error(
                e, code if 'code' in locals() else "", inputs
            )
            
            if recovery_result["status"] == "success":
                clean_code = await self._clean_code_block(recovery_result["fixed_code"])
                return {
                    "code": clean_code,
                    "status": "recovered",
                    "architecture": architecture["design_summary"],
                    "recovery_changes": recovery_result["changes"]
                }
                
            await self.log_activity("implementation_failed", {
                "error": str(e),
                "recovery_attempted": True
            })
            raise

    async def _safe_execute(self, code: str, args: Optional[Dict] = None) -> str:
        """
        Execute code in a sandboxed environment
        
        Args:
            code: Code to execute
            args: Optional arguments for execution
            
        Returns:
            str: Execution result
        """
        # Create a restricted globals dictionary
        safe_globals = {
            '__builtins__': {
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
            }
        }
        
        # Add provided arguments to globals
        if args:
            safe_globals.update(args)
        
        try:
            # Execute code in restricted environment
            exec(code, safe_globals, {})
            return str(safe_globals.get('result', 'Code executed successfully'))
        except Exception as e:
            raise RuntimeError(f"Safe execution failed: {str(e)}")

    async def _generate_code(self, specification: str) -> str:
        """Generate code based on specification using LLM"""
        prompt = f"""
        Generate Python code for the following specification:
        {specification}
        
        Requirements:
        1. Use modern Python features and best practices
        2. Include proper error handling
        3. Add comprehensive docstrings
        4. Include type hints
        5. Follow PEP 8 style guidelines
        
        Return only the implementation code without explanations.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.strip()
        except Exception as e:
            await self.log_activity("code_generation_error", {"error": str(e)})
            raise 

    async def recover_from_error(self, error: Exception, code: str, context: Dict) -> Dict:
        """
        Attempt to recover from coding errors by analyzing and fixing the problematic code
        
        Args:
            error: The exception that occurred
            code: The code that caused the error
            context: Additional context about the code generation attempt
            
        Returns:
            Dict containing:
                - fixed_code: Corrected code if recovery successful
                - status: Recovery status
                - changes: List of changes made
        """
        try:
            # Log recovery attempt
            await self.log_activity("error_recovery_start", {
                "error": str(error),
                "code_length": len(code)
            })
            
            # Analyze the error and code
            analysis = await self.analyze_code(code)
            
            # Create error recovery prompt
            recovery_prompt = f"""
            Fix the following code that generated an error:
            
            Error: {str(error)}
            
            Code:
            {code}
            
            Analysis Results:
            {analysis}
            
            Context:
            {json.dumps(context, indent=2)}
            
            Provide fixed code that resolves the error.
            Return only the corrected code without explanations.
            """
            
            # Generate fixed code
            fixed_code = await self.llm.ainvoke(recovery_prompt)
            
            # Validate fixed code
            validation_result = await self.analyze_code(fixed_code)
            
            if validation_result.get("errors", []):
                raise ValueError("Recovery attempt produced invalid code")
                
            return {
                "fixed_code": fixed_code,
                "status": "success",
                "changes": self._diff_changes(code, fixed_code)
            }
            
        except Exception as recovery_error:
            await self.log_activity("error_recovery_failed", {
                "original_error": str(error),
                "recovery_error": str(recovery_error)
            })
            return {
                "fixed_code": "",
                "status": "failed",
                "changes": []
            }

    async def _clean_code_block(self, code: str) -> str:
        """Remove markdown code block tags and line numbers from code
        
        Args:
            code: Raw code string that may contain markdown tags
            
        Returns:
            str: Clean code ready for execution
        """
        # Remove markdown code block tags if present
        code = code.strip()
        if code.startswith("```") and code.endswith("```"):
            # Extract language if specified
            first_line = code.split("\n")[0]
            if ":" in first_line:  # Handle ```python:filename.py
                first_line = first_line.split(":")[0]
            
            # Remove first and last lines containing ``` tags
            code_lines = code.split("\n")[1:-1]
            code = "\n".join(code_lines)
        
        return code.strip()

    def _diff_changes(self, original_code: str, updated_code: str) -> List[str]:
        """
        Compare original and updated code to generate a list of changes made.
        
        Args:
            original_code (str): The original code before recovery
            updated_code (str): The updated code after recovery
            
        Returns:
            List[str]: List of human-readable change descriptions
        """
        changes = []
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            updated_code.splitlines(keepends=True),
            fromfile='original',
            tofile='recovered',
            n=0  # Only show changed lines
        )
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                changes.append(f"Added: {line[1:].strip()}")
            elif line.startswith('-') and not line.startswith('---'):
                changes.append(f"Removed: {line[1:].strip()}")
                
        return changes if changes else ["No significant changes detected"]

    async def error_recovery(self, code: str, error_msg: str) -> Dict:
        """
        Attempt to recover from code errors by analyzing and fixing the issue.
        
        Args:
            code (str): The problematic code
            error_msg (str): The error message received
            
        Returns:
            Dict: Recovery results including status and changes made
        """
        try:
            original_code = code
            # Attempt to fix the code using LLM
            fixed_code = await self._fix_code(code, error_msg)
            
            if fixed_code:
                changes = self._diff_changes(original_code, fixed_code)
                return {
                    "status": "recovered",
                    "code": fixed_code,
                    "recovery_changes": changes
                }
            
            return {
                "status": "failed",
                "error": "Could not generate valid fix"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _fix_code(self, code: str, error_msg: str) -> Optional[str]:
        """
        Use LLM to attempt to fix the code based on the error message.
        
        Args:
            code (str): The problematic code
            error_msg (str): The error message received
            
        Returns:
            Optional[str]: Fixed code if successful, None otherwise
        """
        prompt = f"""
        Fix the following Python code that generated this error:
        ERROR: {error_msg}
        
        CODE:
        {code}
        
        Return only the fixed code without explanations.
        """
        
        try:
            response = await self.llm.agenerate([prompt])
            return response.generations[0].text.strip()
        except Exception:
            return None