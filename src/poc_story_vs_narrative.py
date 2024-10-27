"""
Proof of Concept: Story vs Narrative Modification

This module demonstrates the application of narrative modifiers to story elements
using language models and embedding techniques. It explores how different narrative
aspects (tone, perspective, purpose) can alter the presentation of a story while
maintaining its core elements.

Key Components:
    - Story Elements: Events, characters, and settings that form the base narrative.
    - Narrative Modifiers: Tone, perspective, and purpose modifiers to alter the story.
    - Embedding Generation: Using a language model to create vector representations of text.
    - Narrative Application: Combining story and narrative embeddings to create modified stories.
    - Text Generation: Using a language model to generate modified story text based on embeddings.

The module uses asynchronous programming for efficient processing and includes
caching mechanisms to optimize embedding generation.

Main Functions:
    - embed_elements: Generate embeddings for story or narrative elements.
    - apply_narrative: Combine story and narrative embeddings.
    - generate_modified_text: Create new text based on original story and narrative modifier.
    - main: Orchestrate the entire process of story modification and output generation.

Dependencies:
    - asyncio: For asynchronous programming.
    - numpy: For numerical operations on embeddings.
    - sklearn: For computing cosine similarity between embeddings.
    - logging: For structured logging throughout the module.

Usage:
    Run this module directly to see a demonstration of story modification:
    ```
    python src/poc_story_vs_narrative.py
    ```

Note:
    This is a proof of concept and may require further optimization for large-scale use.
"""

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


async def embed_elements(elements_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embed story or narrative elements using the language model and cache the results.

    This function takes a dictionary of elements (either story elements or narrative modifiers)
    and generates embeddings for each element. It uses a cache to avoid regenerating embeddings
    for previously seen elements.

    Args:
        elements_dict (Dict[str, Any]): A dictionary containing story elements or narrative modifiers.

    Returns:
        Dict[str, Any]: A dictionary with the same structure as the input, but with embeddings instead of text.

    Note:
        This function is asynchronous and requires an active event loop to run.
    """
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


def apply_narrative(story_embedding: List[float], narrative_embedding: List[float], weight: float = 0.5) -> np.ndarray:
    """
    Apply a narrative modifier to a story element embedding.

    This function combines the story embedding and narrative embedding using a weighted sum.

    Args:
        story_embedding (List[float]): The embedding of the original story element.
        narrative_embedding (List[float]): The embedding of the narrative modifier.
        weight (float, optional): The weight given to the narrative modifier. Defaults to 0.5.

    Returns:
        np.ndarray: The modified embedding combining the story and narrative.
    """
    return (1 - weight) * np.array(story_embedding) + weight * np.array(narrative_embedding)


async def generate_modified_text(
    original_text: str, modified_embedding: List[float], narrative_modifier: str
) -> str:
    """
    Generate modified text based on the original text and a narrative modifier.

    This function uses the language model to rewrite the original text applying the specified
    narrative modifier. It adjusts the tone, perspective, and style while keeping the main
    events and characters.

    Args:
        original_text (str): The original story text.
        modified_embedding (List[float]): The modified embedding (not used in the current implementation).
        narrative_modifier (str): A description of the narrative modifier to apply.

    Returns:
        str: The modified text after applying the narrative modifier.

    Note:
        This function is asynchronous and requires an active event loop to run.
    """
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
    """
    Main function to demonstrate the story modification process.

    This function performs the following steps:
    1. Embeds story elements and narrative modifiers.
    2. Applies narrative modifiers to story events.
    3. Generates modified text for each combination of event and modifier.
    4. Outputs the results, including similarity scores and modified text.
    5. Cleans up resources.

    Note:
        This function is asynchronous and requires an active event loop to run.
    """
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
            for modifier_name, modifier_embedding in embedded_narratives[category].items():
                logger.debug(f"Applying narrative modifier: {category} - {modifier_name}")
                modified_embedding = apply_narrative(
                    event_embedding, modifier_embedding, weight=0.6
                )
                similarity = cosine_similarity([event_embedding], [modified_embedding])[0][0]

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
        logger.info(f"Original Event: {event_text}")
        logger.info(f"Applied {category.capitalize()}: {modifier_name}")
        logger.info(f"Similarity to Original: {mod_data['similarity']:.4f}")
        logger.info(f"Modified Text: {mod_data['modified_text']}")
        logger.info("")  # Empty line for readability

    # Clean up resources
    logger.info("Cleaning up resources")
    await language_model.cleanup()
    embedding_cache.clear()  # Clear the cache after use
    logger.info("Main process completed")


if __name__ == "__main__":
    asyncio.run(main())
