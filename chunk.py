"""Chunk markdown files from data/corpus/<source>/ into 800-token pieces.

Strategy:
  1. MarkdownHeaderTextSplitter — split on # / ## / ### boundaries first.
  2. RecursiveCharacterTextSplitter — if a section is still too big, split
     again on paragraph / sentence / word boundaries.

Token-based sizing via tiktoken so chunk sizes match what the embedding
model and LLM actually count.
"""

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# Built once and reused — they're stateless.
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,   # keep the heading inside the chunk text
)
_char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
    disallowed_special=(),   # don't error on `<|endoftext|>` literals in docs
)


def chunk_text(source: str, doc_id: str, text: str) -> list[dict]:
    """Split one markdown document into chunks. Returns ready-to-insert dicts.

    Sample call:
        chunk_text(source="fastapi", doc_id="tutorial__first-steps.md", text=raw_md)
    """
    sections = _md_splitter.split_text(text)

    chunks: list[dict] = []
    chunk_idx = 0
    for section in sections:
        # Pick the deepest heading as the chunk's title (h3 > h2 > h1).
        # `section.metadata` is a dict like {"h1": "...", "h2": "...", "h3": "..."}
        # — whichever headings sat above this section in the source.
        meta = section.metadata
        title = meta.get("h3") or meta.get("h2") or meta.get("h1")

        # If the section is still too big, recursively char-split it.
        sub_texts = _char_splitter.split_text(section.page_content)
        for sub in sub_texts:
            # Skip empty chunks — they'd embed to useless vectors.
            if not sub.strip():
                continue
            chunks.append({
                "source": source,
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "title": title,
                "text": sub,
            })
            chunk_idx += 1
    return chunks


def chunk_file(source: str, filename: str) -> list[dict]:
    """Read a corpus file and chunk it."""
    path = CORPUS_DIR / source / filename
    return chunk_text(source, filename, path.read_text(encoding="utf-8"))
