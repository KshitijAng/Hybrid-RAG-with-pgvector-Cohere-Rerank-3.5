"""Wrap the GitHub Models embedding API.

GitHub Models exposes an OpenAI-compatible endpoint, so the standard `openai`
SDK works by overriding base_url + api_key.

Auth: GITHUB_TOKEN from .env (loaded via python-dotenv).
"""

import os

from dotenv import load_dotenv

# every .embeddings.create() / .chat.completions.create() call is automatically
# traced — model name, token counts, cost, prompt, response, all captured without 
# you doing anything extra.
from langfuse.openai import OpenAI


load_dotenv()

# text-embedding-3-small → 1536 dims, cheap, matches our pgvector(1536) column.
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunks per API call — kept well under OpenAI's input cap.
EMBED_BATCH_SIZE = 100


_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many strings. Returns one 1536-dim vector per input.

    Auto-batches so the caller doesn't have to think about API limits.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
