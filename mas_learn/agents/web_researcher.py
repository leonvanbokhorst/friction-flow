from langchain.tools import DuckDuckGoSearchRun
from .base_agent import BaseAgent

class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="web_researcher",
            role="information_gatherer",
            capabilities=["web_search", "information_synthesis"]
        )
        self.search_tool = DuckDuckGoSearchRun()
    
    async def search_web(self, query: str) -> str:
        """Performs web search and synthesizes results"""
        search_results = self.search_tool.run(query)
        
        # Store in memory
        await self.learn(search_results)
        
        return search_results 