from pathlib import Path
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "ga4gh_chunks"
INGEST_BATCH = 64  # documents per ChromaDB add call


def _build_search_text(chunk: Dict[str, Any]) -> str:
    """Enrich content with document name, title, and keywords for better embedding."""
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

class VectorStore:

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.reranker = CrossEncoder('sentence-transformers/msmarco-MiniLM-L-12-v3')
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
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
            embeddings = self.embedding_model.encode(docs).tolist()
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            stored += len(batch)
        return stored

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        #top_k similar clauses for query
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(query_text).tolist()
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
                "clause_number": meta.get("chunk_id", ""),
                "title": meta.get("title", ""),
                "text": meta.get("content", doc),
                "similarity": round(1 - dist, 4),
                "source": meta.get("source_url", ""),
                "page": meta.get("page", 0),
            })

        #change 2: added reranker for better relevance
        if self.reranker:
            scores = self.reranker.predict([(query_text, c["text"]) for c in clauses])
            for c, score in zip(clauses, scores):
                c["rerank_score"] = round(score, 4)
            clauses.sort(key=lambda x: x["rerank_score"], reverse=True)

        #return clauses[:top_k]

        #deduplicate: keep highest-ranked chunk per doc
        #future improvement dedup using jacard similarity
        seen = set()
        deduped = []
        for c in clauses:
            key = (c["document_name"], c["title"])
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped[:top_k]