from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
import asyncio
from tqdm import tqdm

import numpy as np
import ollama

logger = logging.getLogger(__name__)


class BiasType(Enum):
    CONFIRMATION = "confirmation_bias"
    STEREOTYPICAL = "stereotypical_bias"
    INGROUP_OUTGROUP = "ingroup_outgroup_bias"
    ANCHORING = "anchoring_bias"
    AVAILABILITY = "availability_bias"


@dataclass
class BiasDetectionResult:
    bias_type: BiasType
    confidence: float
    explanation: str
    affected_segments: List[str]


class BiasDetector:
    def __init__(
        self,
        model_name: str = "hermes3:latest",
        embeddings_model_name: str = "nomic-embed-text",
    ):
        self.model_name = model_name
        self.embeddings_model_name = embeddings_model_name
        self.prompts = self._load_bias_prompts()

    def _load_bias_prompts(self) -> Dict[BiasType, str]:
        """Load prompts for different bias types."""
        return {
            BiasType.CONFIRMATION: (
                "Analyze the following text for confirmation bias. "
                "Look for instances where the author seeks or interprets information "
                "to confirm existing beliefs. Respond in JSON format with fields: "
                "'has_bias' (boolean), 'explanation' (string), 'segments' (list of strings).\n\n"
                "Text: {text}"
            ),
            BiasType.STEREOTYPICAL: (
                "Analyze the following text for stereotypical bias. "
                "Identify any oversimplified beliefs or assumptions about particular groups. "
                "Respond in JSON format with fields: "
                "'has_bias' (boolean), 'explanation' (string), 'segments' (list of strings).\n\n"
                "Text: {text}"
            ),
            BiasType.INGROUP_OUTGROUP: (
                "Analyze the following text for ingroup-outgroup bias. "
                "Look for favoritism towards one's own group or discrimination against other groups. "
                "Respond in JSON format with fields: "
                "'has_bias' (boolean), 'explanation' (string), 'segments' (list of strings).\n\n"
                "Text: {text}"
            ),
            BiasType.ANCHORING: (
                "Analyze the following text for anchoring bias. "
                "Identify instances where initial information disproportionately influences decisions. "
                "Respond in JSON format with fields: "
                "'has_bias' (boolean), 'explanation' (string), 'segments' (list of strings).\n\n"
                "Text: {text}"
            ),
            BiasType.AVAILABILITY: (
                "Analyze the following text for availability bias. "
                "Look for judgments influenced by easily recalled examples rather than complete data. "
                "Respond in JSON format with fields: "
                "'has_bias' (boolean), 'explanation' (string), 'segments' (list of strings).\n\n"
                "Text: {text}"
            ),
        }

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using Ollama."""
        response = ollama.embeddings(model=self.embeddings_model_name, prompt=text)
        return response["embedding"]

    async def detect_bias(
        self, text: str, bias_types: Optional[List[BiasType]] = None
    ) -> List[BiasDetectionResult]:
        """
        Detect various types of bias in the given text.

        Args:
            text: The text to analyze for bias
            bias_types: Optional list of specific bias types to check for

        Returns:
            List of BiasDetectionResult objects
        """
        if bias_types is None:
            bias_types = list(BiasType)

        results = []
        text_embedding = self.get_embedding(text)

        for bias_type in tqdm(bias_types, desc="Checking bias types", leave=False):
            # Get the appropriate prompt
            prompt = self.prompts[bias_type].format(text=text)

            # Generate analysis
            response = ollama.generate(
                model=self.model_name, prompt=prompt, format="json"
            )

            try:
                analysis = json.loads(response["response"])

                # Calculate confidence using embeddings
                confidence = self._calculate_confidence(
                    text_embedding, analysis.get("explanation", "")
                )

                results.append(
                    BiasDetectionResult(
                        bias_type=bias_type,
                        confidence=confidence,
                        explanation=analysis.get("explanation", ""),
                        affected_segments=analysis.get("segments", []),
                    )
                )

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response for {bias_type}")
                continue

        return results

    def _calculate_confidence(
        self, text_embedding: List[float], explanation: str
    ) -> float:
        """Calculate confidence score for bias detection using cosine similarity."""
        # Handle empty explanation
        if not explanation:
            logger.warning("Empty explanation received, returning minimum confidence")
            return 0.5

        try:
            # Get embedding for the explanation
            explanation_embedding = self.get_embedding(explanation)

            # Validate embedding dimensions
            if len(explanation_embedding) == 0:
                logger.warning("Received empty embedding, returning minimum confidence")
                return 0.5

            if len(explanation_embedding) != len(text_embedding):
                logger.error(
                    f"Embedding dimension mismatch: {len(text_embedding)} vs {len(explanation_embedding)}"
                )
                return 0.5

            # Calculate cosine similarity
            dot_product = np.dot(text_embedding, explanation_embedding)
            text_norm = np.linalg.norm(text_embedding)
            explanation_norm = np.linalg.norm(explanation_embedding)

            # Avoid division by zero
            if text_norm == 0 or explanation_norm == 0:
                return 0.5

            similarity = dot_product / (text_norm * explanation_norm)
            confidence = 0.5 + (similarity * 0.5)

            return float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    async def analyze_document(
        self, file_path: Path
    ) -> Dict[BiasType, List[BiasDetectionResult]]:
        """Analyze an entire document for various types of bias."""
        with open(file_path, "r") as f:
            text = f.read()

        # Split into manageable chunks
        chunks = self._split_text(text)

        all_results = {}
        for chunk in tqdm(chunks, desc="Analyzing document chunks"):
            results = await self.detect_bias(chunk)
            for result in results:
                if result.bias_type not in all_results:
                    all_results[result.bias_type] = []
                all_results[result.bias_type].append(result)

        return all_results

    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks for analysis while preserving semantic units.

        Args:
            text: The input text to split
            chunk_size: Target size for each chunk in characters

        Returns:
            List of text chunks that preserve sentence and paragraph boundaries
        """
        # Split into paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            # Split paragraph into sentences (basic splitting)
            sentences = [
                s.strip()
                for s in paragraph.replace("? ", "?|")
                .replace("! ", "!|")
                .replace(". ", ".|")
                .split("|")
            ]

            for sentence in sentences:
                sentence_len = len(sentence) + 1  # +1 for space

                if current_length + sentence_len > chunk_size and current_chunk:
                    # Join completed chunk and start new one
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_len

            # Add paragraph boundary
            if current_chunk:
                current_chunk.append("\n\n")
                current_length += 2

        # Add final chunk if exists
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks

    def save_analysis_report(
        self, results: Dict[BiasType, List[BiasDetectionResult]], output_path: Path
    ) -> None:
        """Generate and save a markdown report of the bias analysis results.

        Args:
            results: Dictionary mapping BiasType to list of detection results
            output_path: Path where the markdown report should be saved
        """
        # Overall statistics
        total_instances = sum(len(bias_results) for bias_results in results.values())
        avg_confidences = {
            bias_type: np.mean([r.confidence for r in bias_results])
            for bias_type, bias_results in results.items()
        }

        report = [
            "# Bias Analysis Report\n",
            *(
                "## Summary\n",
                f"- Total bias instances detected: {total_instances}",
                "- Average confidence by bias type:",
            ),
        ]
        report.extend(
            f"  - {bias_type.value}: {avg_conf:.2f}"
            for bias_type, avg_conf in avg_confidences.items()
        )
        report.append("\n")

        # Detailed results by bias type
        for bias_type, bias_results in results.items():
            report.append(f"## {bias_type.value.replace('_', ' ').title()}\n")

            # Sort results by confidence
            sorted_results = sorted(
                bias_results, key=lambda x: x.confidence, reverse=True
            )

            for result in sorted_results:
                report.extend(
                    (
                        f"### Instance (Confidence: {result.confidence:.2f})",
                        f"\n**Explanation:**\n{result.explanation}\n",
                    )
                )
                if result.affected_segments:
                    report.append("**Affected Segments:**")
                    report.extend(
                        f"- {segment}" for segment in result.affected_segments
                    )
                report.append("\n")

        # Save the report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report))
        logger.info(f"Analysis report saved to {output_path}")


async def main():
    detector = BiasDetector()
    results = await detector.analyze_document(Path("docs/sippe-smiley.txt"))
    detector.save_analysis_report(results, Path("sippe_smiley_analysis_report.md"))


if __name__ == "__main__":
    asyncio.run(main())
