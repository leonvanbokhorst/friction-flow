from typing import Dict, List, Optional, Union, Any
from .base_agent import BaseAgent
import aiohttp
import json
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from langchain_ollama import OllamaLLM
import logging
import re
from xml.etree import ElementTree

# Add logger configuration
logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        name: str = "researcher",
        role: str = "Research Specialist",
        capabilities: List[str] = None,
        orchestrator=None,
    ):
        capabilities = capabilities or ["research", "analysis", "ideation"]
        super().__init__(name, role, capabilities, orchestrator=orchestrator)
        # Override with Hermes model specifically for research
        self.llm = OllamaLLM(model="qwen2.5-coder:14b")
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

            # Store search results artifact (this will handle the learning)
            await self.store_artifact(
                "web_search_results",
                processed_results,
                {"query": query, "max_results": max_results},
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
        try:
            # Use direct search implementation
            results = []

            # Execute search using LLM for query enhancement
            enhanced_query = await self._enhance_search_query(query)

            # Perform search using existing capabilities
            response = await self.llm.ainvoke(
                f"""
            Search for relevant information about: {enhanced_query}
            
            Format results as a list of JSON objects with:
            - title: str
            - content: str
            - relevance: float (0-1)
            - source: str
            """
            )

            try:
                search_results = json.loads(response)
                results.extend(search_results)
            except json.JSONDecodeError:
                # Fallback structure if response isn't valid JSON
                results.append(
                    {
                        "title": "Search Results",
                        "content": response,
                        "relevance": 0.5,
                        "source": "llm_direct",
                    }
                )

            return results[:max_results]

        except Exception as e:
            await self.log_activity("search_error", {"error": str(e)})
            return []

    async def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Process and summarize search results

        Args:
            results: Raw search results from different sources

        Returns:
            List[Dict]: Processed and summarized results
        """
        processed_results = []

        for result in results:
            try:
                # Generate summary using LLM
                summary_prompt = f"""
                Summarize the following research finding:
                Title: {result.get('title', 'No title')}
                Content: {result.get('content', 'No content')}
                Source: {result.get('source', 'Unknown')}
                
                Provide a concise summary focusing on key findings and relevance.
                """

                summary = await self.llm.ainvoke(summary_prompt)

                # Calculate relevance score
                relevance_score = self._calculate_relevance(
                    result.get("content", ""),
                    (
                        self.current_research_cycle.get("topic", "")
                        if self.current_research_cycle
                        else ""
                    ),
                )

                processed_results.append(
                    {
                        "title": result.get("title", "Untitled"),
                        "summary": summary.strip(),
                        "source": result.get("source", "Unknown"),
                        "url": result.get("url", ""),
                        "relevance_score": relevance_score,
                        "processed_date": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing result: {str(e)}")
                continue

        # Sort by relevance
        processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return processed_results

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
            matches = re.findall(r"(?:\d*\.)?\d+", response)
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
                    "architecture": {"layers": [], "input_spec": {}, "output_spec": {}},
                    "components": [],
                    "challenges": [],
                    "evaluation": [],
                    "implementation": {
                        "libraries": [],
                        "core_classes": [],
                        "data_pipeline": [],
                    },
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
                            idea["architecture"]["layers"].append(
                                line.split(":", 1)[1].strip()
                            )
                        elif "input:" in line.lower():
                            idea["architecture"]["input_spec"] = line.split(":", 1)[
                                1
                            ].strip()
                        elif "output:" in line.lower():
                            idea["architecture"]["output_spec"] = line.split(":", 1)[
                                1
                            ].strip()
                    elif current_section == "components":
                        # Ensure components are properly captured
                        if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.")):
                            component = re.sub(r"^[-•*\d.]\s*", "", line).strip()
                            if component:  # Only add non-empty components
                                idea["components"].append(component)
                                # Also add to implementation libraries if it looks like a library
                                if any(
                                    keyword in component.lower()
                                    for keyword in ["library", "framework", "package"]
                                ):
                                    idea["implementation"]["libraries"].append(
                                        component
                                    )
                    elif current_section in ["challenges", "evaluation"]:
                        if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.")):
                            item = re.sub(r"^[-*\d.]\s*", "", line)
                            idea[current_section].append(item.strip())

                # Clean up description
                idea["description"] = idea["description"].strip()

                # Add feasibility score
                idea["feasibility"] = await self._assess_feasibility(
                    idea["description"]
                )

                # Ensure we have at least some components
                if not idea["components"]:
                    idea["components"] = [
                        "PyTorch",
                        "NumPy",
                        "Scikit-learn",
                    ]  # Default components

                parsed_ideas.append(idea)

            except Exception as e:
                self.logger.error(f"Failed to parse idea: {e}")
                continue

        return parsed_ideas

    def _get_search_params(self, engine: str, query: str) -> Dict:
        """Get search parameters for different engines"""
        if engine == "duckduckgo":
            return {"q": query, "format": "json"}
        elif engine == "github":
            return {"q": query, "sort": "stars", "order": "desc"}
        elif engine == "arxiv":
            return {"search_query": f"all:{query}", "max_results": 5}
        return {}

    def _parse_duckduckgo_response(self, response: str) -> List[Dict]:
        """Parse DuckDuckGo API response"""
        try:
            data = json.loads(response)
            results = []
            for result in data.get("RelatedTopics", []):
                if "Text" in result:
                    results.append(
                        {
                            "title": result.get("Text", "")[:50],
                            "content": result.get("Text", ""),
                            "url": result.get("FirstURL", ""),
                            "source": "duckduckgo",
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo response: {e}")
            return []

    def _parse_github_response(self, response: str) -> List[Dict]:
        """Parse GitHub API response"""
        try:
            data = json.loads(response)
            results = []
            for item in data.get("items", []):
                results.append(
                    {
                        "title": item.get("full_name", ""),
                        "content": item.get("description", ""),
                        "url": item.get("html_url", ""),
                        "source": "github",
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Error parsing GitHub response: {e}")
            return []

    def _parse_arxiv_response(self, response: str) -> List[Dict]:
        """Parse arXiv API response"""
        try:
            # ArXiv returns XML, we'll need to parse it
            root = ElementTree.fromstring(response)
            results = []

            # ArXiv XML namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)

                results.append(
                    {
                        "title": title.text if title is not None else "",
                        "content": summary.text if summary is not None else "",
                        "url": link.text if link is not None else "",
                        "source": "arxiv",
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
            return []

    def _parse_search_response(self, engine: str, response: str) -> List[Dict]:
        """Route to appropriate parser based on search engine"""
        parsers = {
            "duckduckgo": self._parse_duckduckgo_response,
            "github": self._parse_github_response,
            "arxiv": self._parse_arxiv_response,
        }

        parser = parsers.get(engine)
        if parser:
            return parser(response)
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

    def _calculate_relevance(self, content: str, topic: str) -> float:
        """Calculate relevance score between content and research topic"""
        if not content or not topic:
            return 0.0

        # Simple keyword matching for now
        topic_keywords = set(topic.lower().split())
        content_words = set(content.lower().split())

        matches = len(topic_keywords.intersection(content_words))
        total_keywords = len(topic_keywords)

        return matches / total_keywords if total_keywords > 0 else 0.0

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

    async def synthesize_findings(self, search_results: List[Dict]) -> Dict:
        """
        Synthesize research findings into a coherent summary

        Args:
            search_results: List of processed search results

        Returns:
            Dict: Synthesized findings with summary and key points
        """
        try:
            # Create synthesis prompt using search results
            synthesis_prompt = f"""
            Synthesize these research findings into a coherent summary:
            
            Search Results:
            {json.dumps(search_results, indent=2)}
            
            Provide:
            1. Overall summary
            2. Key technical insights
            3. Implementation considerations
            4. Potential challenges
            5. Recommended approach
            
            Format response as JSON with these fields.
            """

            response = await self.llm.ainvoke(synthesis_prompt)

            try:
                synthesis = json.loads(response)
            except json.JSONDecodeError:
                # Create structured format if response isn't valid JSON
                synthesis = {
                    "summary": response,
                    "technical_insights": [],
                    "implementation_considerations": [],
                    "challenges": [],
                    "recommended_approach": "",
                }

            # Log synthesis completion
            await self.log_activity(
                "findings_synthesized",
                {
                    "result_count": len(search_results),
                    "synthesis_length": len(str(synthesis)),
                },
            )

            # Store synthesis artifact
            await self.store_artifact(
                "research_synthesis", synthesis, {"result_count": len(search_results)}
            )

            return synthesis

        except Exception as e:
            await self.log_activity("synthesis_failed", {"error": str(e)})
            raise

    def search(self):
        return {
            "sources": ["arxiv", "scholar", "papers_with_code", "github"],
            "result_processing": {
                "extract_abstracts": True,
                "relevance_scoring": "cosine_similarity",
                "citation_tracking": True,
            },
        }

    async def _enhance_search_query(self, query: str) -> str:
        """
        Enhance the search query using LLM to make it more specific and relevant

        Args:
            query: Original search query

        Returns:
            str: Enhanced search query
        """
        try:
            # Create prompt for query enhancement
            enhancement_prompt = f"""
            Enhance this search query to be more specific and technical:
            {query}
            
            Consider:
            1. Technical terminology
            2. Specific concepts
            3. Recent developments
            4. Key requirements
            
            Return only the enhanced query text without any explanation.
            """

            enhanced_query = await self.llm.ainvoke(enhancement_prompt)

            # Clean up response
            enhanced_query = enhanced_query.strip().replace("\n", " ")

            # Log the enhancement
            await self.log_activity(
                "query_enhanced", {"original": query, "enhanced": enhanced_query}
            )

            return enhanced_query

        except Exception as e:
            await self.log_activity("query_enhancement_failed", {"error": str(e)})
            return query  # Return original query if enhancement fails
