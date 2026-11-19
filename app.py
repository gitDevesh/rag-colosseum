import json
import math
import re
from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

KB = json.loads((Path(__file__).parent / "kb.json").read_text(encoding="utf-8"))
assert len(KB) == 50

STOP = set("the a an is are was were do does did of for to in on at by from with and or what which how many much long often default configured configuration value used uses use does its it this that be as when where who whose between compared versus vs according based knowledge base passage specify mentioned".split())
SYN = {"heartbeat":"heartbeats", "timeout":"timeouts", "port":"ports", "acknowledgement":"acknowledgment", "replication":"replica", "cryptography":"cryptographic", "interval":"intervals", "batch":"batches", "size":"sizes", "key":"keys", "hash":"hashing", "connection":"connections", "server":"servers", "node":"nodes", "quorum":"quorums", "latency":"delay", "delay":"latency", "election":"elect", "commit":"commits", "duration":"time", "time":"timeout"}
def tokenize(text):
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    result = []
    for w in words:
        if w in STOP: continue
        result.append(w)
        if w in SYN: result.append(SYN[w])
        if len(w) > 4 and w.endswith("s"): result.append(w[:-1])
    return result

DOCS = [tokenize(x) for x in KB]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(map(len, DOCS))/len(DOCS)
def rank(q, doc):
    terms = tokenize(q)
    if not terms: return 0.0
    counts = Counter(doc)
    result = 0.0
    for term in set(terms):
        f = counts[term]
        if not f: continue
        idf = math.log(1 + (len(DOCS) - DF[term] + 0.5)/(DF[term] + 0.5))
        result += idf * f * 2.2 / (f + 1.2*(0.25 + 0.75*len(doc)/AVG))
    return result

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
class Query(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    top_k: int = Field(default=3, ge=1, le=50)

@app.get("/")
def root():
    return {"status": "ok", "service": "rag-colosseum", "endpoint": "/query", "passages": len(KB)}

@app.get("/health")
def health():
    return {"status":"ok","passages":len(KB)}

@app.post("/query")
def query(request: Query):
    q = request.question.strip()
    if not q: return {"answer":None,"sources":[],"scores":[],"confidence":0.0}
    ranked = sorted(range(len(KB)), key=lambda i: rank(q, DOCS[i]), reverse=True)
    best = ranked[0]
    raw = rank(q, DOCS[best])
    terms = set(tokenize(q))
    matched = len(terms.intersection(DOCS[best])) / max(len(terms),1)
    if raw < 1.4 or matched < 0.22:
        return {"answer":None,"sources":[],"scores":[],"confidence":0.0}
    selected = ranked[:request.top_k]
    sentences = []
    for i in selected[:min(3,len(selected))]:
        for s in re.split(r"(?<=[.!?])\s+", KB[i]):
            if s.strip():
                overlap = len(set(tokenize(s)) & terms)
                score = overlap * (1.0 if i == best else 0.75)
                sentences.append((score, i, s.strip()))
    sentences.sort(key=lambda x:x[0], reverse=True)
    if not sentences or sentences[0][0] == 0:
        return {"answer":None,"sources":[],"scores":[],"confidence":0.0}
    answer = sentences[0][2]
    if len(terms) > 2 and sentences[0][0] < 2:
        return {"answer":None,"sources":[],"scores":[],"confidence":0.0}
    scores = [round(min(1.0, max(0.0, rank(q,DOCS[i])/(raw+0.00001))),4) for i in selected]
    return {"answer":answer,"sources":selected,"scores":scores,"confidence":round(min(0.99,max(0.65,0.70+0.25*matched)),4)}
