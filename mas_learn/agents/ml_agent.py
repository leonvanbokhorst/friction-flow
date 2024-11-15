from typing import Dict, Any, List
from .base_agent import BaseAgent
import torch
from torch import nn
import pandas as pd
import json

class MLAgent(BaseAgent):
    def __init__(self, name: str, orchestrator=None):
        super().__init__(
            name=name,
            role="Machine Learning Engineer",
            capabilities=[
                "train_models",
                "evaluate_models",
                "optimize_hyperparameters",
                "data_preprocessing"
            ],
            orchestrator=orchestrator
        )
        self.model_registry = {}
        
    async def train_model(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Train a machine learning model based on the provided specification.
        
        Args:
            specification: Dictionary containing model training parameters
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            return await self._execute_training(specification)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _estimate_training_time(self, specification: Dict[str, Any]) -> float:
        """Estimate training time based on model specification.
        
        Args:
            specification: Dictionary containing model training parameters
            
        Returns:
            float: Estimated training time in minutes
        """
        # Basic estimation logic
        base_time = 5.0  # Base time in minutes
        
        # Adjust for epochs
        epochs = specification.get('training_params', {}).get('epochs', 1)
        time_per_epoch = base_time / 10  # Assume 0.5 minutes per epoch
        
        # Adjust for architecture complexity
        arch_multiplier = {
            'transformer': 2.0,
            'lstm': 1.5,
            'cnn': 1.2,
            'mlp': 1.0
        }.get(specification.get('architecture', 'mlp'), 1.0)
        
        # Calculate total estimated time
        total_time = base_time + (epochs * time_per_epoch * arch_multiplier)
        
        return total_time 

    async def _execute_training(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model training process.
        
        Args:
            specification: Dictionary containing model training parameters
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        model_name = specification.get('model_name', 'unnamed_model')
        architecture = specification.get('architecture', 'unknown')
        training_params = specification.get('training_params', {})
        
        # Simulate training process
        epochs = training_params.get('epochs', 1)
        batch_size = training_params.get('batch_size', 32)
        
        return {
            "status": "success",
            "model_name": model_name,
            "architecture": architecture,
            "training_metrics": {
                "epochs_completed": epochs,
                "final_loss": 0.001,  # Simulated loss
                "accuracy": 0.95,     # Simulated accuracy
                "training_time": self._estimate_training_time(specification)
            },
            "batch_size": batch_size
        }

    async def analyze_architecture_options(self, research_task: Dict) -> Dict:
        try:
            # Define technical requirements schema
            technical_requirements_schema = {
                "compute": "str - CPU/GPU requirements",
                "memory": "str - RAM requirements",
                "storage": "str - Storage requirements",
                "scalability": "str - Scaling considerations",
                "dependencies": "List[str] - Required libraries"
            }
            
            # Create detailed architecture analysis prompt
            prompt = f"""
            Analyze and recommend neural network architectures for the research task:
            {json.dumps(research_task, indent=2)}
            
            Provide a detailed analysis including:
            1. Multiple architecture options with pros/cons
            2. Specific layer configurations
            3. Training strategies
            4. Resource requirements
            5. Scalability considerations
            
            Format the response as JSON with:
            - recommended_architectures: [List of architecture specifications]
            - technical_requirements: {json.dumps(technical_requirements_schema, indent=2)}
            - rationale: Detailed explanation of recommendations
            """
            
            response = await self.llm.ainvoke(prompt)
            analysis = await self._parse_and_validate_analysis(response)
            
            # Store and validate the analysis
            await self.store_artifact(
                "architecture_analysis",
                analysis,
                {"task": research_task.get("objective")}
            )
            
            return analysis
            
        except Exception as e:
            await self.log_activity("architecture_analysis_failed", {"error": str(e)})
            return self._create_fallback_analysis()

    async def _parse_and_validate_analysis(self, response: str) -> Dict:
        """
        Parse and validate the architecture analysis response from LLM
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dict: Validated and structured analysis
        """
        try:
            # Try to parse as JSON first
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # If not valid JSON, create structured format
                analysis = {
                    "recommended_architectures": [],
                    "technical_requirements": {
                        "compute": "Not specified",
                        "memory": "Not specified",
                        "storage": "Not specified",
                        "scalability": "Not specified",
                        "dependencies": []
                    },
                    "rationale": response
                }
            
            # Validate required fields
            required_fields = [
                "recommended_architectures",
                "technical_requirements",
                "rationale"
            ]
            
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = []
                
            # Validate technical requirements structure
            tech_req_fields = [
                "compute", "memory", "storage", 
                "scalability", "dependencies"
            ]
            
            if "technical_requirements" in analysis:
                for field in tech_req_fields:
                    if field not in analysis["technical_requirements"]:
                        analysis["technical_requirements"][field] = "Not specified"
                    
            await self.log_activity("analysis_parsed", {
                "fields": list(analysis.keys()),
                "architectures_count": len(analysis.get("recommended_architectures", []))
            })
            
            return analysis
            
        except Exception as e:
            await self.log_activity("analysis_parsing_failed", {"error": str(e)})
            return self._create_fallback_analysis()

    async def design_architecture(self, inputs: dict) -> dict:
        """Design neural network architecture based on research findings"""
        findings = inputs["research_findings"]
        requirements = inputs["technical_requirements"]
        
        # Generate architecture design
        design = await self._generate_architecture_design(findings, requirements)
        
        return {
            "design_summary": design["summary"],
            "layer_specification": design["layers"],
            "training_configuration": design["training_config"]
        }

    async def _generate_architecture_design(self, findings: Dict, requirements: Dict) -> Dict:
        """
        Generate detailed neural network architecture design
        
        Args:
            findings: Research findings from researcher
            requirements: Technical requirements from analysis
            
        Returns:
            Dict: Complete architecture design specification
        """
        try:
            # Create design prompt using findings and requirements
            prompt = f"""
            Design a detailed neural network architecture based on:
            
            Research Findings:
            {json.dumps(findings, indent=2)}
            
            Technical Requirements:
            {json.dumps(requirements, indent=2)}
            
            Provide a complete architecture specification including:
            1. Layer-by-layer structure
            2. Activation functions
            3. Input/output dimensions
            4. Training configuration
            5. Optimization strategy
            
            Format response as JSON with fields:
            - summary: Overall architecture description
            - layers: Detailed layer specifications
            - training_config: Training parameters
            """
            
            response = await self.llm.ainvoke(prompt)
            
            try:
                design = json.loads(response)
            except json.JSONDecodeError:
                # Create structured format if response isn't valid JSON
                design = {
                    "summary": response,
                    "layers": [],
                    "training_config": {}
                }
                
            await self.log_activity("architecture_design_created", {
                "design_size": len(str(design))
            })
            
            # Store architecture design artifact
            await self.store_artifact(
                "architecture_design",
                design,
                {
                    "findings_summary": findings.get("summary", ""),
                    "requirements": requirements
                }
            )
            
            return design
            
        except Exception as e:
            await self.log_activity("architecture_design_failed", {"error": str(e)})
            raise

    def analyze_architecture(self):
        return {
            'model_selection': self._evaluate_models(),
            'performance_metrics': self._define_metrics(),
            'scalability_analysis': self._analyze_scaling(),
            'resource_requirements': self._estimate_resources()
        }

    def _get_synthesis_format(self) -> Dict:
        """Get the expected format for research synthesis"""
        return {
            "overall_summary": str,
            "key_technical_insights": List[str],
            "implementation_considerations": List[str],
            "potential_challenges": List[str],
            "recommended_approach": str
        }

    def _create_fallback_analysis(self) -> Dict:
        """Create a basic fallback analysis when primary analysis fails"""
        return {
            "recommended_architectures": [],
            "technical_requirements": {
                "compute_requirements": "unknown",
                "memory_requirements": "unknown",
                "storage_requirements": "unknown"
            },
            "rationale": "Fallback analysis due to primary analysis failure"
        }