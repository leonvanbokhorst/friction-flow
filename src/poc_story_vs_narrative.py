import asyncio
from language_models import OllamaInterface
import numpy as np
from embedding_cache import EmbeddingCache
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import logging
from logging_config import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize the language model and embedding cache
language_model = OllamaInterface()
embedding_cache = EmbeddingCache()

# Sample story elements: events, characters, and setting
story_elements = {
    "events": [
        "A young woman sets out on a journey to reclaim a stolen treasure.",
        "The hero faces numerous challenges in a dark forest.",
        "The boy is bullied by his classmates.",
    ],
    "characters": [
        "Hero: a brave, strong, and angry young person",
        "Woman of his dreams: a beautiful, chubby, and courageous woman",
        "Boy: a shy, anxious, and bullied boy",
    ],
    "setting": [
        "The story takes place in a bustling office filled with people and noise.",
        "The story takes place in a dark forest filled with ancient creatures.",
    ],
}

# Narrative elements: tone, perspective, and purpose
narrative_modifiers = {
    "tone": {
        "optimistic": "The story is told with an uplifting, hopeful perspective.",
        "ironic": "The story is narrated with a sense of irony and humor.",
        "sarcastic": "The story is told with a sarcastic, humorous tone.",
    },
    "perspective": {
        "first-person": "The hero narrates the events from a personal viewpoint.",
        "childish": "The story is told from a childish perspective.",
    },
    "purpose": {
        "to inspire": "The story is designed to motivate and uplift.",
        "to shock": "The story is crafted to shock and surprise.",
    },
}


# Modified embed_elements function to use the cache
async def embed_elements(elements_dict):
    embedded_dict = {}
    for key, value in elements_dict.items():
        if isinstance(value, str):
            embedding = embedding_cache.get(value)
            if embedding is None:
                logger.debug(f"Generating embedding for: {value[:50]}...")
                embedding = await language_model.generate_embedding(value)
                embedding_cache.set(value, embedding)
            embedded_dict[key] = embedding
        else:
            embedded_dict[key] = []
            for text in value:
                embedding = embedding_cache.get(text)
                if embedding is None:
                    logger.debug(f"Generating embedding for: {text[:50]}...")
                    embedding = await language_model.generate_embedding(text)
                    embedding_cache.set(text, embedding)
                embedded_dict[key].append(embedding)
    return embedded_dict


# Function to apply a narrative modifier to a story element
def apply_narrative(story_embedding, narrative_embedding, weight=0.5):
    return (1 - weight) * np.array(story_embedding) + weight * np.array(
        narrative_embedding
    )


async def generate_modified_text(
    original_text: str, modified_embedding: List[float], narrative_modifier: str
) -> str:
    logger.info(
        f"Generating modified text for: {original_text[:50]}... with modifier: {narrative_modifier}"
    )
    prompt = f"""
    Original text: "{original_text}"
    Narrative modifier: {narrative_modifier}

    Rewrite the text above applying the narrative modifier. 
    Keep the main events and characters, but adjust the tone, perspective, and style to match the modifier.
    Respond with only the modified text, without any additional explanation.
    """

    response = await language_model.generate(prompt)
    return response.strip()


async def main():
    logger.info("Starting main process")
    # Embed story and narrative elements
    logger.info("Embedding story elements")
    embedded_story = await embed_elements(story_elements)
    logger.info("Embedding narrative elements")
    embedded_narratives = {
        category: await embed_elements(mods)
        for category, mods in narrative_modifiers.items()
    }

    # Modify and observe how the narrative affects the story
    modified_story = {}
    for event_name, event_embedding in zip(
        story_elements["events"], embedded_story["events"]
    ):
        for category, modifiers in narrative_modifiers.items():
            for modifier_name, modifier_embedding in embedded_narratives[
                category
            ].items():
                logger.debug(
                    f"Applying narrative modifier: {category} - {modifier_name}"
                )
                modified_embedding = apply_narrative(
                    event_embedding, modifier_embedding, weight=0.6
                )
                similarity = cosine_similarity([event_embedding], [modified_embedding])[
                    0
                ][0]

                # Generate modified text
                modified_text = await generate_modified_text(
                    event_name, modified_embedding, f"{category}: {modifier_name}"
                )

                modified_story[(event_name, category, modifier_name)] = {
                    "modified_embedding": modified_embedding,
                    "similarity": similarity,
                    "modified_text": modified_text,
                }

    # Output the modified stories with similarity scores and modified text
    logger.info("Outputting modified stories")
    for (event_text, category, modifier_name), mod_data in modified_story.items():
        print(f"Original Event: {event_text}")
        print(f"Applied {category.capitalize()}: {modifier_name}")
        print(f"Similarity to Original: {mod_data['similarity']:.4f}")
        print(f"Modified Text: {mod_data['modified_text']}")
        print()

    # Clean up resources
    logger.info("Cleaning up resources")
    await language_model.cleanup()
    embedding_cache.clear()  # Clear the cache after use
    logger.info("Main process completed")


if __name__ == "__main__":
    asyncio.run(main())
