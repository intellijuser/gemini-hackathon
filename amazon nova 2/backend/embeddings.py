"""
Nova DueDiligence — Semantic Search
Google Gemini text-embedding-004 + pgvector-ready chunk metadata.
Auto-switching: parallel for short docs, sequential for large docs.
"""
import logging
import math
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

logger = logging.getLogger(__name__)
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIMENSIONS = 768       # text-embedding-004 native — reduce to 512 with truncate_dim if needed
PARALLEL_THRESHOLD = 30
PARALLEL_WORKERS = 5
PARALLEL_BATCH = 5
PARALLEL_DELAY = 0.2
SEQUENTIAL_DELAY = 0.05
MAX_CHUNKS = 150


def embed_text(text: str) -> list[float] | None:
    """Embed text using Gemini text-embedding-004. No bedrock client needed."""
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text[:8000],
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as e:
        logger.warning(f"Gemini embedding failed: {e}")
        return None


def embed_query(text: str) -> list[float] | None:
    """Embed a query with query-specific task type for better retrieval."""
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text[:2000],
            task_type="retrieval_query",
        )
        return result["embedding"]
    except Exception as e:
        logger.warning(f"Gemini query embedding failed: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


def _extract_section_heading(text: str) -> str | None:
    patterns = [
        r'^((?:ARTICLE|SECTION|SCHEDULE|EXHIBIT)\s+[\dIVXA-Z]+[.\s].*?)$',
        r'^(\d+\.\s+[A-Z][A-Z\s]+)$',
        r'^([A-Z][A-Z\s]{3,40})$',
    ]
    for line in text[:300].split('\n'):
        line = line.strip()
        for pattern in patterns:
            m = re.match(pattern, line, re.MULTILINE)
            if m:
                return m.group(1).strip()[:80]
    return None


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 3000,
    overlap: int = 200,
    chars_per_page: int = 2500,
) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk": chunk,
                "page_estimate": max(1, start // chars_per_page + 1),
                "section_heading": _extract_section_heading(chunk),
                "char_start": start,
                "char_end": end,
            })
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """In-memory vector store — swap for pgvector in production via crud.py."""
    def __init__(self):
        self._entries: list[tuple[str, str, list[float], dict]] = []

    def add_document(self, doc_id: str, text: str) -> dict:
        """bedrock param removed — Gemini SDK uses global genai.configure()."""
        chunks_with_meta = chunk_text_with_metadata(text)
        total = len(chunks_with_meta)
        to_embed = chunks_with_meta[:MAX_CHUNKS]
        capped = len(to_embed) < total
        truncation_pct = round((1 - len(to_embed) / max(total, 1)) * 100, 1)

        if len(to_embed) <= PARALLEL_THRESHOLD:
            mode = "parallel"
            message = f"Indexing {len(to_embed)} chunks using parallel Gemini embedding."
            indexed = self._parallel(doc_id, to_embed)
        else:
            mode = "sequential"
            message = (
                f"Document is large ({total} chunks). "
                f"Switching to sequential embedding. "
                f"Indexing {len(to_embed)} of {total} chunks."
            )
            indexed = self._sequential(doc_id, to_embed)

        if capped:
            message += f" Coverage: {100-truncation_pct:.0f}% of document."

        logger.info(f"Indexed {indexed}/{total} chunks for doc {doc_id} ({mode})")
        return {
            "indexed": indexed, "total": total, "mode": mode, "message": message,
            "truncation_pct": truncation_pct if capped else 0,
            "coverage_pct": round(100 - truncation_pct if capped else 100, 1),
        }

    def _parallel(self, doc_id: str, chunks: list[dict]) -> int:
        ordered = [None] * len(chunks)

        def embed_one(idx_item):
            idx, item = idx_item
            emb = embed_text(item["chunk"])
            return idx, item, emb

        for batch_start in range(0, len(chunks), PARALLEL_BATCH):
            batch = [(i, chunks[i]) for i in range(batch_start, min(batch_start + PARALLEL_BATCH, len(chunks)))]
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = [executor.submit(embed_one, item) for item in batch]
                for future in as_completed(futures):
                    idx, item, emb = future.result()
                    if emb is not None:
                        ordered[idx] = (item, emb)
            if batch_start + PARALLEL_BATCH < len(chunks):
                time.sleep(PARALLEL_DELAY)

        indexed = 0
        for item in ordered:
            if item is not None:
                chunk_item, emb = item
                self._entries.append((doc_id, chunk_item["chunk"], emb, {
                    "page_estimate": chunk_item["page_estimate"],
                    "section_heading": chunk_item["section_heading"],
                }))
                indexed += 1
        return indexed

    def _sequential(self, doc_id: str, chunks: list[dict]) -> int:
        indexed = 0
        for item in chunks:
            emb = embed_text(item["chunk"])
            if emb is not None:
                self._entries.append((doc_id, item["chunk"], emb, {
                    "page_estimate": item["page_estimate"],
                    "section_heading": item["section_heading"],
                }))
                indexed += 1
            time.sleep(SEQUENTIAL_DELAY)
        return indexed

    def remove_document(self, doc_id: str):
        self._entries = [(d, c, e, m) for d, c, e, m in self._entries if d != doc_id]

    def search(self, query: str, doc_ids: list[str] | None = None, top_k: int = 6) -> list[dict]:
        """Use query-specific task type for better retrieval relevance."""
        query_emb = embed_query(query)
        if not query_emb:
            logger.warning("Query embedding unavailable — returning unranked fallback")
            return [
                {"doc_id": d, "chunk": c, "score": 0.0,
                 "page_estimate": m.get("page_estimate"),
                 "section_heading": m.get("section_heading"), "fallback": True}
                for d, c, _, m in self._entries if not doc_ids or d in doc_ids
            ][:top_k]

        scored = [
            {"doc_id": d, "chunk": c, "score": cosine_similarity(query_emb, e),
             "page_estimate": m.get("page_estimate"),
             "section_heading": m.get("section_heading"), "fallback": False}
            for d, c, e, m in self._entries if not doc_ids or d in doc_ids
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def doc_count(self, doc_id: str) -> int:
        return sum(1 for d, _, _, _ in self._entries if d == doc_id)

    def total_chunks(self) -> int:
        return len(self._entries)