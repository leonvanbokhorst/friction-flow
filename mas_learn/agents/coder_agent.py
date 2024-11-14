from typing import Dict, List, Optional, Tuple
from .base_agent import BaseAgent
import ast
import inspect
from pathlib import Path
import json

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

    async def analyze_code(self, code: str) -> Dict[str, any]:
        """
        Perform static analysis on code to identify potential improvements
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            tree = ast.parse(code)
            analysis = {
                'complexity': self._analyze_complexity(tree),
                'documentation': self._analyze_documentation(tree),
                'type_hints': self._analyze_type_hints(tree),
                'potential_issues': self._analyze_potential_issues(tree)
            }
            return analysis
        except SyntaxError as e:
            return {'error': f'Syntax error in code: {str(e)}'}

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
        
        Arguments: {args if args else 'None'}
        
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
            result = self._safe_execute(code, args)
            return True, result
        except Exception as e:
            return False, f"Execution error: {str(e)}"

    def _analyze_complexity(self, tree: ast.AST) -> Dict:
        """
        Analyze code complexity metrics including cyclomatic complexity,
        nesting depth, and cognitive complexity.
        
        Args:
            tree: Abstract Syntax Tree of the code
            
        Returns:
            Dict containing complexity metrics
        """
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.max_depth = 0
                self.current_depth = 0
                self.cognitive_complexity = 0
                
            def visit_If(self, node):
                self.complexity += 1
                self.current_depth += 1
                self.cognitive_complexity += self.current_depth
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_While(self, node):
                self.complexity += 1
                self.current_depth += 1
                self.cognitive_complexity += self.current_depth
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_For(self, node):
                self.complexity += 1
                self.current_depth += 1
                self.cognitive_complexity += self.current_depth
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit_Try(self, node):
                self.complexity += len(node.handlers)
                self.current_depth += 1
                self.cognitive_complexity += self.current_depth
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return {
            'cyclomatic_complexity': visitor.complexity,
            'max_nesting_depth': visitor.max_depth,
            'cognitive_complexity': visitor.cognitive_complexity
        }

    def _analyze_documentation(self, tree: ast.AST) -> Dict:
        """
        Analyze documentation coverage including docstrings and comments.
        
        Args:
            tree: Abstract Syntax Tree of the code
            
        Returns:
            Dict containing documentation metrics
        """
        class DocVisitor(ast.NodeVisitor):
            def __init__(self):
                self.total_functions = 0
                self.documented_functions = 0
                self.total_classes = 0
                self.documented_classes = 0
                
            def visit_FunctionDef(self, node):
                self.total_functions += 1
                if ast.get_docstring(node):
                    self.documented_functions += 1
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                self.total_classes += 1
                if ast.get_docstring(node):
                    self.documented_classes += 1
                self.generic_visit(node)
        
        visitor = DocVisitor()
        visitor.visit(tree)
        
        return {
            'function_coverage': {
                'total': visitor.total_functions,
                'documented': visitor.documented_functions,
                'percentage': (visitor.documented_functions / visitor.total_functions * 100) if visitor.total_functions > 0 else 100
            },
            'class_coverage': {
                'total': visitor.total_classes,
                'documented': visitor.documented_classes,
                'percentage': (visitor.documented_classes / visitor.total_classes * 100) if visitor.total_classes > 0 else 100
            }
        }

    def _analyze_type_hints(self, tree: ast.AST) -> Dict:
        """
        Analyze type hint coverage in functions and class methods.
        
        Args:
            tree: Abstract Syntax Tree of the code
            
        Returns:
            Dict containing type hint metrics
        """
        class TypeHintVisitor(ast.NodeVisitor):
            def __init__(self):
                self.total_functions = 0
                self.typed_functions = 0
                self.total_arguments = 0
                self.typed_arguments = 0
                self.return_hints = 0
                
            def visit_FunctionDef(self, node):
                self.total_functions += 1
                
                # Check return type hint
                if node.returns:
                    self.return_hints += 1
                
                # Check argument type hints
                for arg in node.args.args:
                    self.total_arguments += 1
                    if arg.annotation:
                        self.typed_arguments += 1
                        
                if self.total_arguments > 0 and self.typed_arguments == self.total_arguments and node.returns:
                    self.typed_functions += 1
                    
                self.generic_visit(node)
        
        visitor = TypeHintVisitor()
        visitor.visit(tree)
        
        return {
            'function_type_coverage': {
                'total': visitor.total_functions,
                'fully_typed': visitor.typed_functions,
                'percentage': (visitor.typed_functions / visitor.total_functions * 100) if visitor.total_functions > 0 else 100
            },
            'argument_type_coverage': {
                'total': visitor.total_arguments,
                'typed': visitor.typed_arguments,
                'percentage': (visitor.typed_arguments / visitor.total_arguments * 100) if visitor.total_arguments > 0 else 100
            },
            'return_type_coverage': {
                'total': visitor.total_functions,
                'typed': visitor.return_hints,
                'percentage': (visitor.return_hints / visitor.total_functions * 100) if visitor.total_functions > 0 else 100
            }
        }

    def _analyze_potential_issues(self, tree: ast.AST) -> List[str]:
        """
        Identify potential code issues and anti-patterns.
        
        Args:
            tree: Abstract Syntax Tree of the code
            
        Returns:
            List of identified issues with descriptions
        """
        class IssueVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.loop_depth = 0
                self.try_depth = 0
                
            def visit_Try(self, node):
                self.try_depth += 1
                if self.try_depth > 3:
                    self.issues.append("Excessive nested try blocks detected")
                if not node.handlers:
                    self.issues.append("Empty except clause detected")
                for handler in node.handlers:
                    if handler.type is None:
                        self.issues.append("Bare except clause detected - consider catching specific exceptions")
                self.generic_visit(node)
                self.try_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                if self.loop_depth > 4:
                    self.issues.append("Deep loop nesting detected - consider refactoring")
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_Compare(self, node):
                if isinstance(node.ops[0], (ast.Is, ast.IsNot)) and isinstance(node.comparators[0], ast.Constant):
                    if node.comparators[0].value is None:
                        self.issues.append("Use 'is None' or 'is not None' for None comparisons")
                self.generic_visit(node)
        
        visitor = IssueVisitor()
        visitor.visit(tree)
        
        return visitor.issues

    def _safe_execute(self, code: str, args: Optional[Dict] = None) -> str:
        """Execute code in a sandboxed environment"""
        # Implementation for safe code execution
        pass 

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