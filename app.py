import json
import re
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

KB = json.loads((Path(__file__).parent / "kb.json").read_text(encoding="utf-8"))
if not isinstance(KB, list) or len(KB) != 50 or not all(isinstance(p, str) for p in KB):
    raise RuntimeError("kb.json must contain 50 passage strings in original order")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
lock = Lock()
chunks = []
chunk_ids = []
for passage_id, passage in enumerate(KB):
    for sentence in re.split(r"(?<=[.!?])\\s+", passage.strip()):
        if sentence:
            chunks.append(sentence)
            chunk_ids.append(passage_id)
passage_vecs = encoder.encode(KB, normalize_embeddings=True, convert_to_numpy=True)
chunk_vecs = encoder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)

app = FastAPI(title="RAG Colosseum")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    top_k: int = Field(default=3, ge=1, le=50)

def abstain():
    return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}

@app.get("/health")
def health():
    return {"status": "ok", "passages": len(KB)}

@app.post("/query")
def query(request: Query):
    question = request.question.strip()
    if not question:
        return abstain()
    with lock:
        qvec = encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = passage_vecs @ qvec
    ranked = np.argsort(-scores)
    if float(scores[ranked[0]]) < 0.39:
        return abstain()
    sentence_scores = chunk_vecs @ qvec
    selected_ids = [int(i) for i in ranked[:request.top_k]]
    best_chunk = None
    for idx in np.argsort(-sentence_scores):
        idx = int(idx)
        if chunk_ids[idx] in selected_ids and sentence_scores[idx] >= 0.42:
            best_chunk = idx
            break
    if best_chunk is None:
        return abstain()
    result_scores = [round(float(np.clip(scores[i], 0, 1)), 4) for i in selected_ids]
    return {
        "answer": chunks[best_chunk],
        "sources": selected_ids,
        "scores": result_scores,
        "confidence": round(float(np.clip(sentence_scores[best_chunk], 0, 1)), 4),
    }
