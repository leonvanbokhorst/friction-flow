from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4
import asyncio
import json
import chromadb
import ollama

@dataclass
class Story:
    """A narrative element in the field with rich context"""
    content: str
    context: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resonances: List[str] = field(default_factory=list)
    field_effects: List[Dict] = field(default_factory=list)

@dataclass
class FieldState:
    """Represents the current state of the narrative field"""
    description: str
    patterns: List[Dict] = field(default_factory=list)
    active_resonances: List[Dict] = field(default_factory=list)
    emergence_points: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class OllamaInterface:
    """Interface to Ollama LLM"""
    
    def __init__(self, model_name: str = "mistral-nemo", embed_model_name: str = "mxbai-embed-large"):
        self.model = model_name
        self.embed_model = embed_model_name
    
    async def analyze(self, prompt: str) -> str:
        """Get LLM analysis of narrative"""
        response = await asyncio.to_thread(
            ollama.chat,
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        return response['message']['content']
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        response = await asyncio.to_thread(
            ollama.embeddings,
            model=self.embed_model,
            prompt=text
        )
        return response['embedding']

class ChromaStore:
    """Local vector store using ChromaDB"""
    
    def __init__(self, collection_name: str = "narrative_field"):
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def store(self, id: str, embedding: List[float], metadata: Dict) -> None:
        """Store embedding and metadata"""
        await asyncio.to_thread(
            self.collection.add,
            documents=[json.dumps(metadata)],
            embeddings=[embedding],
            ids=[id],
            metadatas=[metadata]
        )
    
    async def find_similar(self, embedding: List[float], threshold: float = 0.8, limit: int = 5) -> List[Dict]:
        """Find similar narratives"""
        count = self.collection.count()
        if count == 0:
            return []
            
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=min(limit, count)
        )
        
        similar = []
        for idx, id in enumerate(results['ids'][0]):
            metadata = json.loads(results['documents'][0][idx])
            similar.append({
                'id': id,
                'similarity': results['distances'][0][idx],
                'metadata': metadata
            })
        
        return [s for s in similar if s['similarity'] <= threshold]

class FieldAnalyzer:
    """Handles analysis of narrative field dynamics"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
    
    async def analyze_impact(self, story: Story, current_state: FieldState) -> Dict:
        """Analyze how a story impacts the field"""
        prompt = f"""
        Current field state: {current_state.description}
        Active patterns: {current_state.patterns}
        
        New narrative entering field:
        Content: {story.content}
        Context: {story.context}
        
        Analyze field impact:
        1. Immediate resonance effects
        2. Pattern interactions/disruptions
        3. Potential emergence points
        4. Field state transformations
        
         Do not make up anything. Just use the information provided. Use the context to determine the impact. Do not use markdown or code blocks.
        """
        
        analysis = await self.llm.analyze(prompt)
        return {
            'analysis': analysis,
            'timestamp': datetime.now(),
            'story_id': story.id
        }

    async def detect_patterns(self, stories: List[Story], current_state: FieldState) -> List[Dict]:
        """Identify emergent patterns in the narrative field"""
        story_contexts = [
            {'content': s.content, 'context': s.context, 'effects': s.field_effects}
            for s in stories
        ]
        
        prompt = f"""
        Analyze narrative collection for emergent patterns:
        Stories: {story_contexts}
        Current Patterns: {current_state.patterns}
        Active Resonances: {current_state.active_resonances}
        
        Identify:
        1. New pattern formation
        2. Pattern evolution/dissolution
        3. Resonance networks
        4. Critical transition points
        5. Emergence phenomena
        
        Do not make up anything. Just use the information provided. Use the context to determine the impact. Do not use markdown or code blocks.
        """
        
        return await self.llm.analyze(prompt)

class ResonanceDetector:
    """Handles semantic detection and analysis of narrative resonances"""
    
    def __init__(self, vector_store, llm_interface):
        self.vector_store = vector_store
        self.llm = llm_interface
    
    async def find_resonances(self, story: Story, limit: int = 3) -> List[Dict]:
        """Find and analyze resonating stories using semantic understanding"""
        embedding = await self.llm.generate_embedding(story.content + " " + story.context)
        similar_stories = await self.vector_store.find_similar(embedding, limit=limit)
        
        resonances = []
        for similar in similar_stories:
            similar_metadata = similar['metadata']
            similar_story = Story(
                id=similar['id'],
                content=similar_metadata['content'],
                context=similar_metadata['context'],
                timestamp=datetime.fromisoformat(similar_metadata['timestamp']) 
                    if isinstance(similar_metadata['timestamp'], str) 
                    else similar_metadata['timestamp']
            )
            
            resonance = await self.determine_resonance_type(story, similar_story)
            resonances.append({
                'story_id': similar['id'],
                'resonance': resonance,
                'timestamp': datetime.now()
            })
        
        return resonances

    async def determine_resonance_type(self, story1: Story, story2: Story) -> Dict:
        """Analyze the semantic relationship between stories"""
        prompt = f"""
        Analyze the resonance between these two narratives in the context of a research lab environment:
        
        Story 1: {story1.content}
        Context 1: {story1.context}
        
        Story 2: {story2.content}
        Context 2: {story2.context}
        
        Provide a detailed analysis:
        1. Type of Resonance:
           - How do these narratives interact?
           - What kind of relationship exists between them?
           - Are they reinforcing, conflicting, or transforming each other?
        
        2. Meaning Evolution:
           - How do they influence each other's interpretation?
           - What new meanings emerge from their interaction?
           - How might this change the overall narrative field?
        
        3. Pattern Formation:
           - What patterns might emerge from their interaction?
           - How might these patterns influence future narratives?
           - What potential developments could this resonance trigger?
           
        Do not make up anything. Just use the information provided. Use the context to determine the impact. Do not use markdown or code blocks.    
        """
        
        analysis = await self.llm.analyze(prompt)
        
        return {
            'type': 'semantic_resonance',
            'analysis': analysis,
            'stories': {
                'source': {
                    'id': story1.id,
                    'content': story1.content,
                    'context': story1.context
                },
                'resonant': {
                    'id': story2.id,
                    'content': story2.content,
                    'context': story2.context
                }
            },
            'timestamp': datetime.now()
        }

class NarrativeField:
    """Core system for managing narrative field dynamics"""
    
    def __init__(self, llm_interface, vector_store):
        self.analyzer = FieldAnalyzer(llm_interface)
        self.resonance_detector = ResonanceDetector(vector_store, llm_interface)
        self.vector_store = vector_store
        self.state = FieldState(description="Initial empty narrative field")
        self.stories: Dict[str, Story] = {}
    
    async def add_story(self, content: str, context: str) -> Story:
        """Add a new story and analyze its field effects"""
        story = Story(content=content, context=context)
        
        # Analyze field impact
        impact = await self.analyzer.analyze_impact(story, self.state)
        story.field_effects.append(impact)
        
        # Find resonances
        resonances = await self.resonance_detector.find_resonances(story)
        story.resonances.extend([r['story_id'] for r in resonances])
        
        # Store story and update field
        await self._store_story(story)
        await self._update_field_state(story, impact, resonances)
        
        return story
    
    async def _store_story(self, story: Story) -> None:
        """Store story and its embeddings"""
        embedding = await self.resonance_detector.llm.generate_embedding(
            story.content + " " + story.context
        )
        
        metadata = {
            'content': story.content,
            'context': story.context,
            'field_effects': json.dumps([{
                'analysis': effect['analysis'],
                'timestamp': effect['timestamp'].isoformat(),
                'story_id': effect['story_id']
            } for effect in story.field_effects]),
            'resonances': json.dumps(story.resonances),
            'timestamp': story.timestamp.isoformat()
        }
        
        await self.vector_store.store(story.id, embedding, metadata)
        self.stories[story.id] = story
    
    async def _update_field_state(self, story: Story, impact: Dict, resonances: List[Dict]) -> None:
        """Update field state with enhanced resonance understanding"""
        patterns = await self.analyzer.detect_patterns(
            list(self.stories.values()),
            self.state
        )
        
        self.state = FieldState(
            description=impact['analysis'],
            patterns=patterns,
            active_resonances=resonances,
            emergence_points=[{
                'story_id': story.id,
                'timestamp': datetime.now(),
                'type': 'new_narrative',
                'resonance_context': [r['resonance']['analysis'] for r in resonances]
            }]
        )

async def demo_scenario():
    """Demonstrate the narrative field system with a simple scenario"""
    
    # Initialize components
    llm = OllamaInterface(model_name="llama3_q8", embed_model_name="mxbai-embed-large")
    vector_store = ChromaStore(collection_name="research_lab")
    field = NarrativeField(llm, vector_store)
    
    # Example research lab scenario
    stories = [
        {
            "content": "Leon really want to go to lunch without having to wait for the others.",
            "context": "After a long meeting with the others, Leon is frustrated. It's noisy and he can't hear himself think."
        },
        {
            "content": "Leon discusses his concerns about the AI for Society minor with Coen. Coen is supportive but thinks Leon should talk to manager of the minor, Danny.",
            "context": "After lunch, Leon and Coen are walking back to the lab."
        },
        {
            "content": "Danny fell of his bike and is hurt. He is going to the hospital. He is not sure if he will be able to work on the AI for Society minor in the near future.",
            "context": "Leon is worried about Danny. He is also worried about the lab and the AI for Society minor. He is also worried about his own research. Leon talks to his manager, Robbert."
        },
        {
            "content": "Robbert is very worried about Danny. He is not interested in the AI for Society minor. He is also worried about his own research. Robbert talks to Leon. He thinks Leon should man up and stop whining.",
            "context": "After work, Robbert and Leon are walking back to the lab."
        }
    ]
    
    # Process stories
    print("Processing stories and analyzing field effects...")
    for story in stories:
        try:
            result = await field.add_story(story['content'], story['context'])
            print(f"\n---\nAdded story:\n{story['content']}")
            print(f"\nField effects:\n{result.field_effects[-1]['analysis']}")
            print("\nCurrent field state:\n", field.state.description)
            
            if result.resonances:
                print("\nResonances detected:")
                for r_id in result.resonances:
                    r_story = field.stories.get(r_id)
                    if r_story:
                        print(f"- Resonates with: {r_story.content}")
        
        except Exception as e:
            print(f"Error processing story: {e}")
            continue

if __name__ == "__main__":
    print("Starting narrative field demonstration...")
    asyncio.run(demo_scenario())