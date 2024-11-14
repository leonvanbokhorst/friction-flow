from typing import Dict, List, Optional, Union, Any
from .base_agent import BaseAgent
import aiohttp
import json
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from langchain_community.llms import Ollama
import logging
import re

# Add logger configuration
logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        name: str = "researcher",
        role: str = "Research Specialist",
        capabilities: List[str] = None,
        orchestrator=None
    ):
        capabilities = capabilities or ["research", "analysis", "ideation"]
        super().__init__(name, role, capabilities, orchestrator=orchestrator)
        # Override with Hermes model specifically for research
        self.llm = Ollama(model="hermes3:latest")  # don't change this
        self.research_history = []
        self.current_research_cycle = None

    async def start_research_cycle(self, topic: str, objectives: List[str]) -> str:
        """
        Initialize a new research cycle with specific objectives

        Args:
            topic: Main research topic
            objectives: List of research objectives

        Returns:
            str: Research cycle ID
        """
        cycle_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_research_cycle = {
            "id": cycle_id,
            "topic": topic,
            "objectives": objectives,
            "findings": [],
            "status": "initiated",
        }

        # Store research cycle in memory
        await self.learn(f"Research cycle {cycle_id} initiated for topic: {topic}")
        self.research_history.append(self.current_research_cycle)

        return cycle_id

    async def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform web search and extract relevant information

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List[Dict]: Processed search results
        """
        # Get user approval for web access
        approval = await self.report_to_user(
            {"type": "web_search", "query": query, "max_results": max_results}
        )

        if approval.lower() != "yes":
            return []

        async with aiohttp.ClientSession() as session:
            results = await self._execute_search(session, query, max_results)
            processed_results = await self._process_search_results(results)

            # Store results in memory
            await self.learn(
                f"Search results for query: {query}\n{json.dumps(processed_results)}"
            )

            return processed_results

    async def generate_ideas(self, objective: str) -> List[str]:
        """Generate research ideas based on context."""
        # Add debug logging
        print(f"[DEBUG] Generating ideas for objective: {objective}")
        
        prompt = self._create_research_prompt(objective)
        ideas = await self._get_llm_response(prompt)
        
        # Add debug logging
        print(f"[DEBUG] Raw LLM response: {ideas}")
        
        parsed_ideas = await self._parse_ideas(ideas)
        
        # Add debug logging
        print(f"[DEBUG] Parsed ideas: {parsed_ideas}")
        
        return parsed_ideas

    async def create_synthetic_data(self, specification: Dict) -> Dict:
        """
        Create synthetic data for training or testing

        Args:
            specification: Data generation specifications

        Returns:
            Dict: Generated synthetic data
        """
        # Get approval for data generation
        approval = await self.report_to_user(
            {"type": "synthetic_data_generation", "specification": specification}
        )

        if approval.lower() != "yes":
            return {}

        data = await self._generate_synthetic_data(specification)

        # Store data generation record
        await self.learn(
            f"Generated synthetic data with specification: {specification}"
        )

        return data

    async def validate_findings(self, findings: List[Dict]) -> Dict:
        """
        Validate research findings through cross-referencing and analysis

        Args:
            findings: List of research findings to validate

        Returns:
            Dict: Validation results
        """
        validation_results = {"validated": [], "uncertain": [], "contradictory": []}

        for finding in findings:
            # Cross-reference with stored knowledge
            supporting_evidence = await self.recall(str(finding))
            validation = await self._analyze_evidence(finding, supporting_evidence)
            validation_results[validation["status"]].append(
                {"finding": finding, "evidence": validation["evidence"]}
            )

        return validation_results

    async def _execute_search(
        self, session: aiohttp.ClientSession, query: str, max_results: int
    ) -> List[Dict]:
        """
        Execute web search using multiple search engines

        Args:
            session: aiohttp client session
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[Dict]: Raw search results from multiple sources
        """
        search_engines = {
            "duckduckgo": "https://api.duckduckgo.com/",
            "github": "https://api.github.com/search/repositories",
            "arxiv": "http://export.arxiv.org/api/query",
        }

        results = []
        for engine, url in search_engines.items():
            try:
                params = self._get_search_params(engine, query)
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.text()
                        parsed_results = self._parse_search_response(engine, data)
                        results.extend(parsed_results[:max_results])

                        # Store search metadata
                        await self.learn(f"Search executed on {engine}: {query}")
            except Exception as e:
                await self.learn(f"Search error on {engine}: {str(e)}")

        return results[:max_results]

    async def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Process and extract relevant information from search results

        Args:
            results: Raw search results

        Returns:
            List[Dict]: Processed and structured results
        """
        processed_results = []

        for result in results:
            # Extract main content using BeautifulSoup if HTML
            if "html_content" in result:
                soup = BeautifulSoup(result["html_content"], "html.parser")
                main_content = self._extract_main_content(soup)
            else:
                main_content = result.get("content", "")

            # Generate summary using LLM
            summary_prompt = f"""
            Summarize the following content concisely:
            {main_content[:1000]}  # Limit content length for LLM
            
            Focus on:
            1. Key findings or insights
            2. Relevant technical details
            3. Novel approaches or solutions
            """

            summary_response = await self.llm.agenerate([summary_prompt])
            summary = summary_response.generations[0].text

            processed_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": result.get("source", ""),
                    "summary": summary,
                    "relevance_score": self._calculate_relevance(main_content),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return sorted(
            processed_results, key=lambda x: x["relevance_score"], reverse=True
        )

    async def _generate_synthetic_data(self, specification: Dict) -> Dict:
        """
        Generate synthetic data based on specifications

        Args:
            specification: Data generation specifications

        Returns:
            Dict: Generated synthetic data with metadata
        """
        data_type = specification.get("type", "tabular")
        size = specification.get("size", 100)
        schema = specification.get("schema", {})

        if data_type == "tabular":
            data = self._generate_tabular_data(schema, size)
        elif data_type == "text":
            data = await self._generate_text_data(schema, size)
        elif data_type == "image":
            data = await self._generate_image_data(schema, size)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        return {
            "data": data,
            "metadata": {
                "type": data_type,
                "size": size,
                "schema": schema,
                "generation_timestamp": datetime.now().isoformat(),
                "validation_metrics": self._validate_synthetic_data(
                    data, specification
                ),
            },
        }

    async def _analyze_evidence(self, finding: Dict, evidence: List[str]) -> Dict:
        """
        Analyze evidence supporting a finding

        Args:
            finding: Research finding to validate
            evidence: Supporting evidence

        Returns:
            Dict: Analysis results
        """
        # Prepare evidence analysis prompt
        analysis_prompt = f"""
        Analyze the following research finding and evidence:
        
        Finding: {finding}
        
        Evidence:
        {evidence}
        
        Determine:
        1. Strength of supporting evidence
        2. Any contradictions
        3. Confidence level
        4. Potential biases
        """

        analysis_response = await self.llm.agenerate([analysis_prompt])
        analysis = analysis_response.generations[0].text

        confidence_score = self._calculate_confidence_score(analysis)

        return {
            "status": self._determine_validation_status(confidence_score),
            "evidence": evidence,
            "confidence_score": confidence_score,
            "analysis": analysis,
        }

    async def _assess_feasibility(self, description: str) -> float:
        prompt = f"""Rate the technical feasibility of this research idea from 0.0 to 1.0:
        
        {description}
        
        Return only a single number between 0.0 and 1.0 representing the feasibility score.
        Score: """
        
        try:
            response = await self.llm.ainvoke(prompt)
            # Extract the first valid float from the response
            matches = re.findall(r'(?:\d*\.)?\d+', response)
            if matches:
                score = float(matches[0])
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            raise ValueError("No valid number found in response")
        except Exception as e:
            logger.warning(f"Failed to parse feasibility score: {e}")
            return 0.5  # Return default score on error

    async def _parse_ideas(self, ideas_text: str) -> list:
        parsed_ideas = []
        
        # Split ideas if multiple are present
        ideas_text = ideas_text.split("IDEA:")[1:]  # Skip empty first split
        
        for idea_text in ideas_text:
            try:
                # Initialize idea dict with required implementation fields
                idea = {
                    "title": "",
                    "description": "",
                    "architecture": {
                        "layers": [],
                        "input_spec": {},
                        "output_spec": {}
                    },
                    "components": [],
                    "challenges": [],
                    "evaluation": [],
                    "implementation": {
                        "libraries": [],
                        "core_classes": [],
                        "data_pipeline": []
                    }
                }
                
                # Parse sections
                current_section = None
                sections = idea_text.split("\n")
                
                # First line is the title
                idea["title"] = sections[0].strip()
                
                for line in sections[1:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Handle section headers
                    if line.startswith("DESCRIPTION:"):
                        current_section = "description"
                        continue
                    elif line.startswith("ARCHITECTURE:"):
                        current_section = "architecture"
                        continue
                    elif line.startswith("COMPONENTS:"):
                        current_section = "components"
                        continue
                    elif line.startswith("CHALLENGES:"):
                        current_section = "challenges"
                        continue
                    elif line.startswith("EVALUATION:"):
                        current_section = "evaluation"
                        continue
                    
                    # Parse content based on section
                    if current_section == "description":
                        idea["description"] += line + " "
                    elif current_section == "architecture":
                        if "layers:" in line.lower():
                            idea["architecture"]["layers"].append(line.split(":", 1)[1].strip())
                        elif "input:" in line.lower():
                            idea["architecture"]["input_spec"] = line.split(":", 1)[1].strip()
                        elif "output:" in line.lower():
                            idea["architecture"]["output_spec"] = line.split(":", 1)[1].strip()
                    elif current_section == "components":
                        # Ensure components are properly captured
                        if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.")):
                            component = re.sub(r'^[-•*\d.]\s*', '', line).strip()
                            if component:  # Only add non-empty components
                                idea["components"].append(component)
                                # Also add to implementation libraries if it looks like a library
                                if any(keyword in component.lower() for keyword in ["library", "framework", "package"]):
                                    idea["implementation"]["libraries"].append(component)
                    elif current_section in ["challenges", "evaluation"]:
                        if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.")):
                            item = re.sub(r'^[-•*\d.]\s*', '', line)
                            idea[current_section].append(item.strip())
                
                # Clean up description
                idea["description"] = idea["description"].strip()
                
                # Add feasibility score
                idea["feasibility"] = await self._assess_feasibility(idea["description"])
                
                # Ensure we have at least some components
                if not idea["components"]:
                    idea["components"] = ["PyTorch", "NumPy", "Scikit-learn"]  # Default components
                
                parsed_ideas.append(idea)
                
            except Exception as e:
                self.logger.error(f"Failed to parse idea: {e}")
                continue
        
        return parsed_ideas

    def _get_search_params(self, engine: str, query: str) -> Dict:
        """
        Get search parameters for different search engines

        Args:
            engine: Search engine name
            query: Search query

        Returns:
            Dict: Search parameters for the specific engine
        """
        if engine == "duckduckgo":
            return {"q": query, "format": "json"}
        elif engine == "github":
            return {"q": query, "sort": "stars", "order": "desc"}
        elif engine == "arxiv":
            return {"search_query": f"all:{query}", "start": 0, "max_results": 10}
        return {}

    def _parse_search_response(self, engine: str, data: str) -> List[Dict]:
        """
        Parse search response from different engines

        Args:
            engine: Search engine name
            data: Raw response data

        Returns:
            List[Dict]: Parsed search results
        """
        if engine == "duckduckgo":
            return self._parse_duckduckgo_response(data)
        elif engine == "github":
            return self._parse_github_response(data)
        elif engine == "arxiv":
            return self._parse_arxiv_response(data)
        return []

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML using heuristics

        Args:
            soup: BeautifulSoup object

        Returns:
            str: Extracted main content
        """
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
        )

        if main_content:
            return main_content.get_text(strip=True)
        return soup.get_text(strip=True)

    def _calculate_relevance(self, content: str) -> float:
        """
        Calculate relevance score for search result

        Args:
            content: Content to analyze

        Returns:
            float: Relevance score between 0 and 1
        """
        if not content:
            return 0.0

        # Get current research objectives
        objectives = (
            self.current_research_cycle.get("objectives", [])
            if self.current_research_cycle
            else []
        )

        # Calculate relevance based on objective matches
        score = 0.0
        for objective in objectives:
            if objective.lower() in content.lower():
                score += 1.0

        # Normalize score
        return min(score / max(len(objectives), 1), 1.0)

    def _calculate_confidence_score(self, analysis: str) -> float:
        """
        Calculate confidence score from analysis text

        Args:
            analysis: Analysis text

        Returns:
            float: Confidence score between 0 and 1
        """
        # Keywords indicating confidence levels
        high_confidence = ["strong evidence", "clearly shows", "demonstrates", "proves"]
        medium_confidence = ["suggests", "indicates", "likely", "probable"]
        low_confidence = ["might", "could", "unclear", "uncertain", "contradictory"]

        analysis_lower = analysis.lower()
        score = 0.5  # Start with neutral score

        # Adjust score based on keyword presence
        for keyword in high_confidence:
            if keyword in analysis_lower:
                score += 0.1
        for keyword in medium_confidence:
            if keyword in analysis_lower:
                score += 0.05
        for keyword in low_confidence:
            if keyword in analysis_lower:
                score -= 0.1

        return max(0.0, min(1.0, score))

    def _determine_validation_status(self, confidence_score: float) -> str:
        """
        Determine validation status based on confidence score

        Args:
            confidence_score: Confidence score

        Returns:
            str: Validation status
        """
        if confidence_score >= 0.7:
            return "validated"
        elif confidence_score <= 0.3:
            return "contradictory"
        return "uncertain"

    def _assess_novelty(self, description: str) -> float:
        """
        Assess novelty of an idea

        Args:
            description: Idea description

        Returns:
            float: Novelty score between 0 and 1
        """
        # Check against existing knowledge in memory
        similar_ideas = self.memory.similarity_search(description, k=5)

        if not similar_ideas:
            return 1.0  # Completely novel

        # Calculate average similarity
        similarity_scores = [result.score for result in similar_ideas]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        # Convert similarity to novelty (inverse relationship)
        return 1.0 - avg_similarity

    async def _get_llm_response(self, prompt):
        try:
            response = await self.llm.ainvoke(prompt)
            # Add debug logging
            print(f"[DEBUG] LLM response type: {type(response)}")
            print(f"[DEBUG] LLM response content: {response}")
            return response
        except Exception as e:
            print(f"[ERROR] Failed to get LLM response: {str(e)}")
            return []  # Consider if this is the best fallback

    def _create_research_prompt(self, objective: str) -> str:
        return f"""As an AI research specialist, generate 3-5 novel research ideas for the following objective:
        
Objective: {objective}

For each idea, provide:
1. A clear title
2. A detailed technical description including:
   - Model architecture details
   - Input/output specifications
   - Key algorithms and methods
3. Implementation components (MUST INCLUDE):
   - Required libraries and frameworks
   - Core classes and functions
   - Data processing pipeline
4. Technical challenges and solutions
5. Evaluation metrics and validation approach

Format each idea EXACTLY as:

IDEA: [Title]
DESCRIPTION: [Technical description]
ARCHITECTURE: 
  Layers: [List of neural network layers]
  Input: [Input specifications]
  Output: [Output specifications]
COMPONENTS:
  - [Required library/framework 1]
  - [Required library/framework 2]
  - [Core implementation component 1]
  - [Core implementation component 2]
CHALLENGES: 
  - [Challenge 1]
  - [Challenge 2]
EVALUATION:
  - [Evaluation metric 1]
  - [Evaluation metric 2]
"""
