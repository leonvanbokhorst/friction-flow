from dataclasses import dataclass
from typing import List
import ollama
import time

@dataclass
class StoryState:
    """Represents the current state of a narrative in the field"""
    content: str
    context: str
    resonances: List[str]
    field_effects: List[str]

class NarrativeFieldSimulator:
    """Pure narrative-driven simulator using LLM for field evolution"""
    
    story_line: List[str] = []  
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.field_state = "Empty narrative field awaiting stories"
        self.active_stories: List[StoryState] = []
        
    def simulate_story_evolution(self, initial_setup: str) -> str:
        """Simulates natural story evolution without mechanical state tracking"""
        
        # Initial field formation prompt
        field_prompt = f"""
        A new story enters the narrative field:
        {initial_setup}
        
        Considering narrative field dynamics, describe how this story naturally begins 
        to evolve. Focus on:
        - Natural narrative flows
        - Character perspective resonances
        - Emerging story patterns
        - Potential narrative tensions
        
        Describe this purely through story, avoiding any mechanical state descriptions. Short sentences no line breaks. No markdown.
        """
        
        # Get initial field state
        field_response = self.llm.generate(field_prompt)
        self.story_line.append(field_response)
        print(f"\n---\nInitial field state:\n{field_response}")
        
        self.field_state = field_response
        
        # Simulate evolution through multiple phases
        for _ in range(5):  # Three evolution phases
            evolution_prompt = f"""
            Current story field:
            {self.field_state}
            
            Allow this narrative field to naturally evolve to its next state. Consider:
            - How character perspectives influence each other
            - Where stories naturally want to flow
            - What patterns are emerging
            - How tensions resolve or transform
            
            Describe the next state of the story field, maintaining pure narrative focus. Short sentences no line breaks. 
            """
            
            # Get next evolution state
            next_state = self.llm.generate(evolution_prompt)
            print(f"\n---\nNext field state:\n{next_state}")
            
            # Look for emergent patterns
            pattern_prompt = f"""
            Previous field state:
            {self.field_state}
            
            New field state:
            {next_state}
            
            What narrative patterns and resonances are naturally emerging? 
            Describe any:
            - Story convergence
            - Character alignment
            - Resolution patterns
            - New tensions
            
            Express this purely through story, not technical analysis. Short sentences no line breaks. 
            """
            
            patterns = self.llm.generate(pattern_prompt)
            print(f"\n---\nEmerging patterns:\n{patterns}")
            
            # Update field state with new patterns
            self.field_state = f"""
            {next_state}
            
            Emerging patterns:
            {patterns}
            """
            
        return self.field_state

    def introduce_narrative_force(self, new_element: str) -> str:
        """Introduces a new narrative element and observes field effects"""
        
        force_prompt = f"""
        Current narrative field:
        {self.field_state}
        
        A new force enters the field:
        {new_element}
        
        How does this new element interact with the existing story?
        Describe the natural narrative reactions and adjustments,
        focusing on story flow rather than mechanics. Short sentences no line breaks. 
        """
        
        field_response = self.llm.generate(force_prompt)
        self.story_line.append(field_response)
        print(f"\n---\nNew field state:\n{field_response}")   
        self.field_state = field_response
        return field_response
    
    def evaluate_story_state(self, initial_story_state: str) -> str:
        """Evaluates the state of a story"""
        
        evaluation_prompt = f"""
        Initial story state:
        {initial_story_state}
        
        Story line:
        {self.story_line}
        
        Use the initial story state and the evolving story line to tell a new story, on how their biases have evolved. 
        """     
        print(f"\n---\nStory evaluation prompt:\n{evaluation_prompt}")
        evaluation = self.llm.generate(evaluation_prompt)
        print(f"\n---\nStory evaluation:\n{evaluation}")
        return evaluation

class LLMInterface:
    def __init__(self, model: str = "llama3"): # "mistral-nemo" "nemotron-mini"
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']

def simulate_road_trip_planning():
    """Simulate the evolution of a bias through a narrative field"""
    
    # Create an LLM interface 
    llm_interface = LLMInterface()  
    
    # Initialize simulator with the LLM interface
    simulator = NarrativeFieldSimulator(llm_interface)
    
    # Initial setup
    initial_bias = """
    Leon is a 55yo educator and researcher in the field of AI, especially conversational AI and human-machine interaction. Marleen is a 45yo former nurse and now a researcher in the field of transdisciplinary research and cooperation. The both work at Fontys University of Applied Sciences. 
    Leon and Marleen challenge each other to research their own biases and to understand each other better. They use Claude 3.5 Sonnet to write stories about each other and understand each others language.
    """
    
    # Simulate natural evolution
    simulator.simulate_story_evolution(initial_bias)
    
    # Optionally introduce new force
    narrative_force = """
    Leon learns from Marleen that transdisciplinary research is about collaboration and cooperation. He recognizes that his peers are not aware of this. He sees that his field of AI is changing towards this. He tells everybody for years that it's not about the technology, but about the people and people's needs. Peopleproblems, he calls them.
    Marleen learns from Leon that AI can be used to write stories. She is excited about this new development. She designed a Marleen assistant that acts like her, just by prompting the LLM. She has mixed feelings about the new assistant. What do I want to give away? The machine is not a human, but feels like one. Marleen thinks that AI experts are technical people who don't understand people.   
    """
    
    simulator.introduce_narrative_force(narrative_force)
    
    return initial_bias, simulator

# Example output would show natural story evolution through
# narrative field dynamics, without explicit state tracking

if __name__ == "__main__":
    initial_bias, simulator = simulate_road_trip_planning()
    
    simulator.evaluate_story_state(initial_bias)
