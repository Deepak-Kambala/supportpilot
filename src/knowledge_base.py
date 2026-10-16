import io
import os
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader


@lru_cache(maxsize=1)
def get_embedder():
    import torch
    # Avoid a reproducible macOS ARM crash in the CPU embedding path when
    # PyTorch starts its default-sized thread pool inside Streamlit.
    torch.set_num_threads(max(1, int(os.getenv("SUPPORTPILOT_TORCH_THREADS", "1"))))
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def read_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    raise ValueError("Only PDF, TXT, and Markdown are supported.")


def chunk_text(text: str, size: int = 100, overlap: int = 20):
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunk = " ".join(words[start:start + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_markdown(text: str, size: int = 100, overlap: int = 20):
    """Keep Markdown heading sections (such as FAQ entries) together as chunks."""
    sections = re.split(r"(?m)(?=^#{2,6}\s)", text)
    chunks = []
    for section in sections:
        words = section.split()
        if not words:
            continue
        if len(words) <= size:
            chunks.append(" ".join(words))
        else:
            chunks.extend(chunk_text(section, size=size, overlap=overlap))
    return chunks


def build_index(documents):
    """documents: list of {"filename": str, "text": str}"""
    chunks = []
    for doc in documents:
        chunker = chunk_markdown if Path(doc["filename"]).suffix.lower() == ".md" else chunk_text
        for i, chunk in enumerate(chunker(doc["text"])):
            chunks.append({"text": chunk, "source": doc["filename"], "chunk": i})
    if not chunks:
        raise ValueError("No readable text found in uploaded documents.")
    vectors = get_embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def retrieve(question: str, index, chunks, top_k: int = 4):
    q = get_embedder().encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, ids = index.search(q, min(top_k, len(chunks)))
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0]) if i >= 0
    ]
