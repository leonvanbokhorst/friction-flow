from typing import Dict, List, Optional
import ast
import re
from dataclasses import dataclass
from enum import Enum

class SecurityRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class SecurityIssue:
    risk_level: SecurityRisk
    description: str
    line_number: int
    suggestion: str

class CodeAnalyzer:
    DANGEROUS_BUILTINS = {
        'eval': SecurityRisk.HIGH,
        'exec': SecurityRisk.HIGH,
        'globals': SecurityRisk.MEDIUM,
        'locals': SecurityRisk.MEDIUM,
        'open': SecurityRisk.MEDIUM,
    }

    DANGEROUS_MODULES = {
        'os': SecurityRisk.MEDIUM,
        'subprocess': SecurityRisk.HIGH,
        'pickle': SecurityRisk.HIGH,
        'marshal': SecurityRisk.HIGH,
        'shelve': SecurityRisk.MEDIUM
    }

    DANGEROUS_METHODS = {
        'system': SecurityRisk.HIGH,
        'popen': SecurityRisk.HIGH,
        'eval': SecurityRisk.HIGH,
        'exec': SecurityRisk.HIGH,
        'loads': SecurityRisk.MEDIUM
    }

    SECURITY_SUGGESTIONS = {
        'os': 'Consider using pathlib for file operations instead',
        'subprocess': 'Use subprocess.run() with input validation',
        'pickle': 'Use a secure serialization format like JSON',
        'marshal': 'Use a secure serialization format like JSON',
        'shelve': 'Consider using a proper database solution'
    }

    RISK_LEVELS = {
        'system': SecurityRisk.HIGH,
        'popen': SecurityRisk.HIGH,
        'eval': SecurityRisk.HIGH,
        'exec': SecurityRisk.HIGH,
        'loads': SecurityRisk.MEDIUM
    }

    @staticmethod
    def analyze_code(code: str) -> Dict:
        """Centralized code analysis functionality"""
        try:
            tree = ast.parse(code)
            return {
                'syntax_valid': True,
                'complexity': CodeAnalyzer._analyze_complexity(tree),
                'safety': CodeAnalyzer._check_safety(tree),
                'quality': CodeAnalyzer._assess_quality(tree),
                'documentation': CodeAnalyzer._analyze_documentation(tree),
                'type_hints': CodeAnalyzer._analyze_type_hints(tree)
            }
        except SyntaxError as e:
            return {
                'syntax_valid': False,
                'error': str(e)
            }
    
    @staticmethod
    def _analyze_complexity(tree: ast.AST) -> Dict:
        """Analyze code complexity metrics"""
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
    
    @staticmethod
    def _check_safety(tree: ast.AST) -> Dict:
        """Check code for security issues"""
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues: List[SecurityIssue] = []
                self.imported_modules = set()
                self.dangerous_calls = []
                
            def visit_Import(self, node):
                for name in node.names:
                    self.imported_modules.add(name.name)
                    if name.name in CodeAnalyzer.DANGEROUS_MODULES:
                        self.issues.append(SecurityIssue(
                            risk_level=CodeAnalyzer.DANGEROUS_MODULES[name.name],
                            description=f"Potentially dangerous module '{name.name}' imported",
                            line_number=node.lineno,
                            suggestion=CodeAnalyzer.SECURITY_SUGGESTIONS[name.name]
                        ))
                self.generic_visit(node)
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr in CodeAnalyzer.DANGEROUS_METHODS:
                    self.dangerous_calls.append({
                        "method": node.func.attr,
                        "line": node.lineno,
                        "risk": CodeAnalyzer.RISK_LEVELS[node.func.attr]
                    })

                self.generic_visit(node)

        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        return {
            'security_issues': [vars(issue) for issue in visitor.issues],
            'risk_level': max((issue.risk_level for issue in visitor.issues), 
                            default=SecurityRisk.LOW).value
        }
    
    @staticmethod
    def _assess_quality(tree: ast.AST) -> Dict:
        """Assess code quality metrics"""
        class QualityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.function_count = 0
                self.class_count = 0
                self.line_lengths = []
                self.variable_names = []
                
            def visit_FunctionDef(self, node):
                self.function_count += 1
                if len(node.args.args) > 5:
                    self.issues.append(f"Function '{node.name}' has too many parameters")
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                self.class_count += 1
                self.generic_visit(node)
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.variable_names.append(node.id)
                self.generic_visit(node)

        visitor = QualityVisitor()
        visitor.visit(tree)
        
        # Analyze variable naming
        naming_issues = [name for name in visitor.variable_names 
                        if not re.match(r'^[a-z][a-z0-9_]*$', name)]
        
        return {
            'issues': visitor.issues,
            'metrics': {
                'function_count': visitor.function_count,
                'class_count': visitor.class_count,
                'naming_issues': naming_issues
            }
        }

    @staticmethod
    def _analyze_documentation(tree: ast.AST) -> Dict:
        """Analyze documentation coverage"""
        # Reusing existing implementation from CoderAgent
        reference_implementation = """
        See CoderAgent._analyze_documentation implementation at:
        """
        # Reference to existing implementation:

    @staticmethod
    def _analyze_type_hints(tree: ast.AST) -> Dict:
        """Analyze type hint coverage and correctness"""
        class TypeHintVisitor(ast.NodeVisitor):
            def __init__(self):
                self.functions_with_hints = 0
                self.total_functions = 0
                self.variables_with_hints = 0
                self.total_variables = 0
                self.type_hint_issues = []
                
            def visit_FunctionDef(self, node):
                self.total_functions += 1
                # Check return type annotation
                if node.returns:
                    self.functions_with_hints += 1
                # Check argument type annotations
                if any(arg.annotation for arg in node.args.args):
                    self.functions_with_hints += 1
                self.generic_visit(node)
                
            def visit_AnnAssign(self, node):
                self.total_variables += 1
                if node.annotation:
                    self.variables_with_hints += 1
                self.generic_visit(node)
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.total_variables += 1
                self.generic_visit(node)

        visitor = TypeHintVisitor()
        visitor.visit(tree)
        
        # Calculate coverage percentages
        function_coverage = (visitor.functions_with_hints / visitor.total_functions * 100 
                           if visitor.total_functions > 0 else 0)
        variable_coverage = (visitor.variables_with_hints / visitor.total_variables * 100 
                           if visitor.total_variables > 0 else 0)
        
        return {
            'function_type_coverage': round(function_coverage, 2),
            'variable_type_coverage': round(variable_coverage, 2),
            'total_functions': visitor.total_functions,
            'total_variables': visitor.total_variables,
            'functions_with_hints': visitor.functions_with_hints,
            'variables_with_hints': visitor.variables_with_hints,
            'issues': visitor.type_hint_issues
        }