from pathlib import Path
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi
import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "ga4gh_chunks"
INGEST_BATCH = 64 


def _build_search_text(chunk: Dict[str, Any]) -> str:
    #enrich content with document name, title, and keywords
    parts = []
    if chunk.get("document_name"):
        parts.append(f"Document: {chunk['document_name']}")
    # avoid duplicating topic if content already starts with it
    if chunk.get("title") and not chunk["content"].startswith("Topic:"):
        parts.append(f"Topic: {chunk['title']}")
    parts.append(chunk["content"])
    if chunk.get("keywords"):
        parts.append(f"Keywords: {', '.join(chunk['keywords'])}")
    return "\n".join(parts)

# Deduplicate clauses for retrieval
def deduplicate_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate clauses while preserving:
    - Different parts of the same clause (e.g., 5_part1, 5_part2)
    - Different text under the same clause_number (important for tables or version differences)
    """
    seen_texts = set()
    deduped = []

    for c in clauses:
        # Normalize text for comparison
        text_key = (c.get("text") or "").strip().lower()
        if not text_key:
            continue

        if text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped.append(c)
        else:
            # Already seen exact text; skip
            continue

    return deduped

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

class VectorStore:

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
  
        self._bm25 = None
        self._bm25_chunks = []
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        # clear old data before re-ingesting
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # I had memory running out issue so processing chunks in batches
        stored = 0
        for i in range(0, len(chunks), INGEST_BATCH):
            batch = chunks[i : i + INGEST_BATCH]
            ids = [
                f"{c.get('document_name', '')}_{c['chunk_id']}_{i + j}"
                for j, c in enumerate(batch)
            ]
            docs = [_build_search_text(c) for c in batch]
            metas = [
                {
                    "document_name": c.get("document_name", ""),
                    "clause_id": c.get("clause_id", c["chunk_id"]),
                    "chunk_id": c["chunk_id"],
                    "title": c.get("title", ""),
                    "content": c["content"],
                    "level": c.get("level", ""),
                    "source_url": c.get("source_url", ""),
                    "page": c.get("page") or 0,
                    "keywords": ",".join(c.get("keywords", [])),
                }
                for c in batch
            ]
            embeddings = self.embedding_model.encode(
                docs,
                batch_size=32,
                show_progress_bar=False
            ).tolist()
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            stored += len(batch)
        
        # build BM25 index once after ingest
        if chunks:
            tokenized = [_tokenize(_build_search_text(c)) for c in chunks]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_chunks = chunks

        return stored

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        #top_k similar clauses for query
        if self.collection.count() == 0:
            return []

        # Use BGE-style instruction prefix for queries
        query_embedding = self.embedding_model.encode(
            "Represent this sentence for searching relevant passages: " + query_text
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k*3,   #added more results for reranker
            include=["documents", "metadatas", "distances"],
        )

        clauses: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            #since only one query currently, might expand on this later 
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            clauses.append({
                "document_name": meta.get("document_name", ""),
                "clause_number": meta.get("clause_id", meta.get("chunk_id", "")),
                "title": meta.get("title", ""),
                "text": meta.get("content", doc),
                "similarity": round(1 - dist, 4),
                "source": meta.get("source_url", ""),
                "page": meta.get("page", 0),
            })

        if self._bm25:
            scores = self._bm25.get_scores(_tokenize(query_text))
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k*2]
            for i in top_indices:
                c = self._bm25_chunks[i]
                clauses.append({
                    "document_name": c.get("document_name", ""),
                    "clause_number": c.get("clause_id", c.get("chunk_id", "")),
                    "title": c.get("title", ""),
                    "text": c["content"],
                    "similarity": None,
                    "bm25_score": round(float(scores[i]), 4),
                    "source": c.get("source_url", ""),
                    "page": c.get("page", 0),
                })

        #change 2: added reranker for better relevance
        if self.reranker:
            scores = self.reranker.predict([
                (query_text, f"{c.get('title', '')}\n{c.get('text', '')}")
                for c in clauses
            ])
            for c, score in zip(clauses, scores):
                c["rerank_score"] = round(score, 4)
            clauses.sort(key=lambda x: x["rerank_score"], reverse=True)

        #return clauses[:top_k]

        #deduplicate: keep highest-ranked chunk per doc+clause_number
        #avoid dropping distinct clauses that share titles
        
        deduped_clauses = deduplicate_clauses(clauses)

        return deduped_clauses[:top_k]
