import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import ollama
from dataclasses import dataclass
from tqdm import tqdm  # For progress bars

@dataclass
class ExperimentParams:
    """Parameters for Monte Carlo experiments."""
    temperature: float
    top_p: float
    model_name: str
    num_iterations: int = 100

class LLMAgent:
    def __init__(self, name: str, role: str, personality: str, params: ExperimentParams):
        self.name = name
        self.role = role
        self.personality = personality
        self.params = params
        self.conversation_history: List[Dict] = []
        self.agent_id = str(uuid.uuid4())

    def _construct_prompt(self, message: str) -> str:
        """Construct a prompt that includes agent context and personality."""
        return f"""You are {self.name}, a {self.role}. Your personality is {self.personality}

Previous conversation context:
{self._format_history()}

Current message to respond to:
{message}

Please provide your response while maintaining your role and personality."""

    def _format_history(self) -> str:
        """Format the conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation."
            
        history = []
        for entry in self.conversation_history[-3:]:  # Only include last 3 turns for context
            history.append(f"Input: {entry['input']}")
            history.append(f"Response: {entry['response']}\n")
        return "\n".join(history)

    async def send_message(self, message: str) -> Dict:
        """Send message to Ollama and get response with metadata."""
        start_time = datetime.now()
        try:
            response = ollama.generate(
                model=self.params.model_name,
                prompt=self._construct_prompt(message),
                options={
                    "temperature": self.params.temperature,
                    "top_p": self.params.top_p,
                }
            )
            
            response_text = response["response"]
            success = True
            error = None
        except Exception as e:
            response_text = ""
            success = False
            error = str(e)

        end_time = datetime.now()
        response_data = {
            "timestamp": start_time.isoformat(),
            "input": message,
            "response": response_text,
            "agent_id": self.agent_id,
            "response_time": (end_time - start_time).total_seconds(),
            "success": success,
            "error": error,
            "parameters": {
                "temperature": self.params.temperature,
                "top_p": self.params.top_p,
                "model": self.params.model_name
            }
        }
        
        self.conversation_history.append(response_data)
        return response_data

class CASAExperiment:
    def __init__(self, params: ExperimentParams):
        self.experiment_id = str(uuid.uuid4())
        self.params = params

        # Initialize all required agents
        self.tutor = LLMAgent(
            "Tutor_AI",
            "academic tutor",
            "Professional, supportive, and slightly formal.",
            params
        )
        self.student = LLMAgent(
            "Student_AI",
            "student",
            "Curious, engaged, and respectful.",
            params
        )
        self.evaluator = LLMAgent(
            "Evaluator_AI",
            "social interaction evaluator",
            "Objective, analytical, and detail-oriented.",
            params
        )

    async def run_direct_evaluation(self, topic: str) -> Dict:
        """Run a direct interaction between tutor and student."""
        # Initial question from student about the topic
        student_query = f"Could you please explain {topic} to me?"
        student_response = await self.student.send_message(student_query)
        
        # Tutor's explanation
        tutor_response = await self.tutor.send_message(student_response["response"])
        
        # Student's follow-up question
        follow_up = await self.student.send_message(tutor_response["response"])
        
        # Tutor's final response
        final_response = await self.tutor.send_message(follow_up["response"])
        
        return {
            "interaction_type": "direct",
            "topic": topic,
            "conversation": [
                student_response,
                tutor_response,
                follow_up,
                final_response
            ],
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id
        }

    async def run_indirect_evaluation(self, topic: str) -> Dict:
        """Run an indirect evaluation where evaluator assesses the interaction."""
        # First, run a direct interaction
        direct_result = await self.run_direct_evaluation(topic)
        
        # Format the conversation for evaluation
        conversation_text = self._format_conversation_for_evaluation(direct_result["conversation"])
        
        # Get evaluator's assessment
        evaluation_prompt = (
            f"Please evaluate the following conversation about {topic} "
            f"in terms of politeness, clarity, and effectiveness:\n\n{conversation_text}"
        )
        evaluation = await self.evaluator.send_message(evaluation_prompt)
        
        return {
            "interaction_type": "indirect",
            "topic": topic,
            "base_conversation": direct_result["conversation"],
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id
        }

    def _format_conversation_for_evaluation(self, conversation: List[Dict]) -> str:
        """Format conversation history into readable text for evaluation."""
        formatted_text = ""
        for turn in conversation:
            formatted_text += f"{turn['agent_id']}: {turn['input']}\n"
            formatted_text += f"Response: {turn['response']}\n\n"
        return formatted_text

    async def run_monte_carlo(self, topics: List[str]) -> List[Dict]:
        """Run Monte Carlo simulation across multiple topics and iterations."""
        all_results = []
        
        for topic in tqdm(topics, desc="Topics"):
            for iteration in tqdm(range(self.params.num_iterations), desc=f"Iterations for {topic}"):
                try:
                    direct_result = await self.run_direct_evaluation(topic)
                    indirect_result = await self.run_indirect_evaluation(topic)
                    
                    # Add Monte Carlo metadata
                    for result in [direct_result, indirect_result]:
                        result.update({
                            "monte_carlo_metadata": {
                                "iteration": iteration,
                                "parameters": self.params.__dict__
                            }
                        })
                    
                    all_results.extend([direct_result, indirect_result])
                    
                    # Save intermediate results periodically
                    #if iteration % 10 == 0:
                    self._save_results(all_results, topic, "intermediate")
                        
                except Exception as e:
                    print(f"Error in iteration {iteration} for topic {topic}: {str(e)}")
                    continue
        
        return all_results

    def _save_results(self, results: List[Dict], topic: str, suffix: str = "") -> None:
        """Save results to JSON file."""
        filename = (
            f'monte_carlo_results_{self.experiment_id}_{topic}'
            f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{suffix}.json'
        )
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

async def main():
    """Run Monte Carlo CASA experiments."""
    # Example parameter configurations
    params = ExperimentParams(
        temperature=0.7,
        top_p=0.9,
        model_name="llama3.2:latest",
        num_iterations=10
    )
    
    experiment = CASAExperiment(params)
    topics = ["photosynthesis", "gravity", "climate change"]
    
    results = await experiment.run_monte_carlo(topics)
    experiment._save_results(results, "all_topics", "final")

if __name__ == "__main__":
    asyncio.run(main())
