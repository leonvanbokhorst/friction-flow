import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List
import ollama
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

MODEL_NAME = "llama3.2:latest"


@dataclass
class ExperimentParams:
    """Parameters for Monte Carlo experiments."""

    temperature: float
    top_p: float
    model_name: str
    num_iterations: int = 100
    metrics_file: str = "casa_metrics_{experiment_id}_{datetime}.csv"
    results_dir: str = "casa/results"


class LLMAgent:
    def __init__(
        self, name: str, role: str, personality: str, params: ExperimentParams
    ):
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
        for entry in self.conversation_history[
            -3:
        ]:  # Only include last 3 turns for context
            history.append(f"Input: {entry['input']}")
            history.append(f"Response: {entry['response']}\n")
        return "\n".join(history)

    async def send_message(self, message: str) -> Dict:
        """Send message to Ollama and get response with metadata."""
        start_time = datetime.now()

        # Count social markers in the message
        social_markers = self._extract_social_markers(message)
        contains_question = "?" in message
        references_previous = any(
            ref in message.lower()
            for ref in ["earlier", "previous", "before", "you said"]
        )

        try:
            response = ollama.generate(
                model=self.params.model_name,
                prompt=self._construct_prompt(message),
                options={
                    "temperature": self.params.temperature,
                    "top_p": self.params.top_p,
                },
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
                "model": self.params.model_name,
            },
            "prompt_length": len(message),
            "response_length": len(response_text),
            "ratio": len(response_text) / len(message) if len(message) > 0 else 0,
            "contains_question": contains_question,
            "references_previous": references_previous,
            "social_markers": social_markers,
        }

        self.conversation_history.append(response_data)
        return response_data

    def _extract_social_markers(self, text: str) -> List[str]:
        """Extract social markers like emotions and gestures from text."""
        markers = []
        # Simple regex or keyword matching could be expanded
        if "*" in text:  # Check for emotes like *smiles*
            markers.extend(word.strip("*") for word in text.split("*")[1::2])
        return markers


class CASAExperiment:
    def __init__(self, params: ExperimentParams):
        self.experiment_id = str(uuid.uuid4())
        self.params = params

        # Create results directory if it doesn't exist
        self.results_dir = Path(params.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Update metrics file path to be in results directory
        self.metrics_file = self.results_dir / params.metrics_file.format(
            experiment_id=self.experiment_id,
            datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        # Initialize all required agents
        self.tutor = LLMAgent(
            "Tutor_AI",
            "academic tutor",
            "Professional, supportive, and slightly formal.",
            params,
        )
        self.student = LLMAgent(
            "Student_AI", "student", "Curious, engaged, and respectful.", params
        )
        self.evaluator = LLMAgent(
            "Evaluator_AI",
            "social interaction evaluator",
            "Objective, analytical, and detail-oriented.",
            params,
        )

        # Initialize DataFrame with explicit dtypes
        self.metrics_df = pd.DataFrame(
            columns=[
                "timestamp",
                "experiment_id",
                "turn_number",
                "speaker_id",
                "prompt_length",
                "response_length",
                "ratio",
                "contains_question",
                "references_previous",
                "social_markers",
                "response_time",
                "temperature",
                "top_p",
            ]
        ).astype(
            {
                "timestamp": str,
                "experiment_id": str,
                "turn_number": int,
                "speaker_id": str,
                "prompt_length": int,
                "response_length": int,
                "ratio": float,
                "contains_question": bool,
                "references_previous": bool,
                "social_markers": str,
                "response_time": float,
                "temperature": float,
                "top_p": float,
            }
        )

    def _log_metrics(self, turn_data: Dict, turn_number: int) -> None:
        """Log conversation metrics to DataFrame and save to CSV."""
        new_row = pd.DataFrame(
            [
                {
                    "timestamp": turn_data["timestamp"],
                    "experiment_id": self.experiment_id,
                    "turn_number": turn_number,
                    "speaker_id": turn_data["agent_id"],
                    "prompt_length": turn_data["prompt_length"],
                    "response_length": turn_data["response_length"],
                    "ratio": turn_data["ratio"],
                    "contains_question": turn_data["contains_question"],
                    "references_previous": turn_data["references_previous"],
                    "social_markers": json.dumps(turn_data["social_markers"]),
                    "response_time": turn_data["response_time"],
                    "temperature": turn_data["parameters"]["temperature"],
                    "top_p": turn_data["parameters"]["top_p"],
                }
            ]
        )

        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        # Save after each update to prevent data loss
        self.metrics_df.to_csv(self.metrics_file, index=False)

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

        # Add metrics logging for each turn
        for turn_number, response in enumerate(
            [student_response, tutor_response, follow_up, final_response], 1
        ):
            self._log_metrics(response, turn_number)

        return {
            "interaction_type": "direct",
            "topic": topic,
            "conversation": [
                student_response,
                tutor_response,
                follow_up,
                final_response,
            ],
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
        }

    async def run_indirect_evaluation(self, topic: str) -> Dict:
        """Run an indirect evaluation where evaluator assesses the interaction."""
        # First, run a direct interaction
        direct_result = await self.run_direct_evaluation(topic)

        # Format the conversation for evaluation
        conversation_text = self._format_conversation_for_evaluation(
            direct_result["conversation"]
        )

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
            "experiment_id": self.experiment_id,
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
            for iteration in tqdm(
                range(self.params.num_iterations), desc=f"Iterations for {topic}"
            ):
                try:
                    direct_result = await self.run_direct_evaluation(topic)
                    indirect_result = await self.run_indirect_evaluation(topic)

                    # Add Monte Carlo metadata
                    for result in [direct_result, indirect_result]:
                        result.update(
                            {
                                "monte_carlo_metadata": {
                                    "iteration": iteration,
                                    "parameters": self.params.__dict__,
                                }
                            }
                        )

                    all_results.extend([direct_result, indirect_result])

                    # Remove intermediate saves
                    # self._save_results(all_results, topic, "intermediate")

                except Exception as e:
                    print(f"Error in iteration {iteration} for topic {topic}: {str(e)}")
                    continue

        return all_results

    def _save_results(self, results: List[Dict], topic: str, suffix: str = "") -> None:
        """Save results to JSON file in results directory."""
        filename = (
            f"monte_carlo_results_{self.experiment_id}_{topic}"
            f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{suffix}.json'
        )
        filepath = self.results_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)


async def main():
    """Run Monte Carlo CASA experiments."""
    # Example parameter configurations
    params = ExperimentParams(
        temperature=0.7,
        top_p=0.9,
        model_name=MODEL_NAME,
        num_iterations=5,
    )

    experiment = CASAExperiment(params)
    topics = ["Unicorns"]

    results = await experiment.run_monte_carlo(topics)
    experiment._save_results(results, "all_topics", "final")


if __name__ == "__main__":
    asyncio.run(main())
