import asyncio
from language_models import LlamaInterface
import numpy as np

# Initialize the language model
language_model = LlamaInterface()

# Sample story elements: events, characters, and setting
story_elements = {
    "events": [
        "A young hero sets out on a journey to reclaim a stolen treasure.",
        "The hero faces numerous challenges in a dark forest.",
        "An encounter with a wise elder changes the hero's perspective.",
    ],
    "characters": [
        "Hero: a brave and determined young person",
        "Elder: a wise and experienced guide",
    ],
    "setting": [
        "The story takes place in a mystical forest filled with ancient secrets."
    ],
}

# Narrative elements: tone, perspective, and purpose
narrative_modifiers = {
    "tone": {
        "optimistic": "The story is told with an uplifting, hopeful perspective.",
        "tragic": "The story has a dark, sorrowful undertone.",
        "ironic": "The story is narrated with a sense of irony and humor.",
    },
    "perspective": {
        "first-person": "The hero narrates the events from a personal viewpoint.",
        "omniscient": "The story is told by an all-knowing narrator.",
    },
    "purpose": {
        "to entertain": "The story is crafted to captivate and amuse.",
        "to inform": "The story provides insights and lessons from the hero's journey.",
    },
}

# Embedding each element
async def embed_elements(elements_dict):
    embedded_dict = {}
    for key, value in elements_dict.items():
        if isinstance(value, str):
            embedded_dict[key] = await language_model.generate_embedding(value)
        else:
            embedded_dict[key] = [await language_model.generate_embedding(text) for text in value]
    return embedded_dict

# Function to apply a narrative modifier to a story element
def apply_narrative(story_embedding, narrative_embedding, weight=0.5):
    return (1 - weight) * np.array(story_embedding) + weight * np.array(narrative_embedding)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def main():
    # Embed story and narrative elements
    embedded_story = await embed_elements(story_elements)
    embedded_narratives = {
        category: await embed_elements(mods) for category, mods in narrative_modifiers.items()
    }

    # Modify and observe how the narrative affects the story
    modified_story = {}
    for event_name, event_embedding in zip(story_elements["events"], embedded_story["events"]):
        for tone_name, tone_embedding in embedded_narratives["tone"].items():
            modified_embedding = apply_narrative(event_embedding, tone_embedding, weight=0.6)
            similarity = cosine_similarity(event_embedding, modified_embedding)
            modified_story[(event_name, tone_name)] = {
                "modified_embedding": modified_embedding,
                "similarity": similarity,
            }

    # Output the modified stories with similarity scores to the original
    for (event_text, tone_name), mod_data in modified_story.items():
        print(f"Original Event: {event_text}")
        print(f"Applied Tone: {tone_name}")
        print(f"Similarity to Original: {mod_data['similarity']:.4f}")
        print("Modified Embedding:", mod_data['modified_embedding'], "\n")

    # Clean up resources
    await language_model.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
