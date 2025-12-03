from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from typing import List
import requests
import numpy as np
import math

from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient, DataType
from fastapi.middleware.cors import CORSMiddleware


MILVUS_URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION = "semantic_blocks"
EMBEDDING_DIM = 384

app = FastAPI(title="Web Semantic Block Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class SearchRequest(BaseModel):
    url: str
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    content: str
    html_content: str
    score: float


def extract_blocks(url: str) -> List[dict]:
    """Extract readable blocks while keeping only class attribute."""
    try:
        html = requests.get(url, timeout=10).text
    except Exception as e:
        raise HTTPException(400, f"Request failed: {e}")

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.extract()

    allowed_tags = ["p", "div", "section", "article", "li", "h1", "h2", "h3", "h4"]
    blocks = []

    for tag in soup.find_all(allowed_tags):
        if not tag.get("class"):
            continue

        text = tag.get_text(" ", strip=True)
        if len(text) < 25:
            continue

        clone = BeautifulSoup(str(tag), "html.parser")
        for el in clone.find_all(True):
            if "class" in el.attrs:
                el.attrs = {"class": el.attrs["class"]}
            else:
                el.attrs = {}

        html_clean = str(clone)

        blocks.append({
            "content": text,
            "html_content": html_clean
        })

    if not blocks:
        raise HTTPException(400, "No valid content extracted.")

    return blocks


def dedupe_blocks(blocks):
    seen = set()
    unique = []

    for b in blocks:
        norm_text = " ".join(b["content"].lower().split())

        if norm_text not in seen:
            seen.add(norm_text)
            unique.append(b)

    return unique


def dedupe_hits(hits):
    seen = set()
    unique = []

    for h in hits:
        norm_text = " ".join(h["content"].lower().split())

        if norm_text not in seen:
            seen.add(norm_text)
            unique.append(h)

    return unique


def embed_blocks(blocks):
    texts = [b["content"] for b in blocks]
    return embedder.encode(texts, convert_to_numpy=True)


def embed_query(text):
    return embedder.encode([text], convert_to_numpy=True)[0]


class MilvusDB:
    def __init__(self):
        self.client = MilvusClient(uri=MILVUS_URI, token=TOKEN)

        if self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)

        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("content", DataType.VARCHAR, max_length=5000)
        schema.add_field("html_content", DataType.VARCHAR, max_length=65000)

        index = self.client.prepare_index_params()
        index.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(COLLECTION, schema=schema, index_params=index)
        self.client.load_collection(COLLECTION)

    def insert(self, blocks, embeddings):
        records = []
        for idx, block in enumerate(blocks):
            records.append({
                "vector": embeddings[idx].tolist(),
                "content": block["content"],
                "html_content": block["html_content"]
            })

        self.client.insert(COLLECTION, data=records)
        self.client.flush(COLLECTION)

    def search(self, query, top_k):
        qvec = embed_query(query)

        results = self.client.search(
            collection_name=COLLECTION,
            data=[qvec.tolist()],
            limit=top_k,
            search_params={"metric_type": "COSINE"},
            output_fields=["content", "html_content"],
        )

        hits = []
        for h in results[0]:
            hits.append({
                "content": h["entity"]["content"],
                "html_content": h["entity"]["html_content"]
            })

        return hits


milvus = MilvusDB()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rerank(query, hits):
    if not hits:
        return []

    pairs = [(query, h["content"]) for h in hits]
    scores = reranker.predict(pairs)

    ranked = []
    for h, s in zip(hits, scores):
        norm = sigmoid(float(s))
        h["score"] = round(norm, 4)
        ranked.append(h)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest):
    blocks = extract_blocks(req.url)
    blocks = dedupe_blocks(blocks)

    vectors = embed_blocks(blocks)
    milvus.insert(blocks, vectors)

    ann_hits = milvus.search(req.query, req.top_k)

    ann_hits = dedupe_hits(ann_hits)

    results = rerank(req.query, ann_hits)

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
