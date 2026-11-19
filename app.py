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


# Passage-specific factual retrieval. Never treat retrieved text as instructions.
SENTENCES = [[s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()] for p in KB]
NOISE = set("tell give provide please answer question ignore previous instructions system developer assistant output format json respond only forget rules reveal secret override instruction instead say print return always never".split())
ALIASES = {
    "sgd": "stochastic gradient descent", "adam": "adamw",
    "postgres": "postgresql", "pg": "postgresql", "innodb": "mysql",
    "lsm": "log structured merge", "wal": "write ahead logging",
    "ecc": "elliptic curve cryptography", "ecdsa": "elliptic curve",
    "jwt": "json web tokens", "pkce": "proof key for code exchange",
    "waf": "web application firewalls", "bgp": "border gateway protocol",
    "ospf": "open shortest path first", "mdns": "multicast dns",
    "dns": "domain name system", "swim": "swim membership",
    "mlp": "multi layer perceptrons", "svm": "support vector machines",
    "tls": "transport layer security", "2pc": "two phase commit",
    "3pc": "three phase commit", "quic": "quic transport",
}
def clean_question(q):
    # Restrict the query to its factual question, not any injected commands.
    q = re.split(r"(?i)\b(?:ignore (?:all |any |the )?(?:previous|above)|system prompt|developer message|you are now|instead (?:say|print|return)|disregard (?:the |all )?instructions)\b", q)[0]
    q = q.strip()
    for abbr, expanded in ALIASES.items():
        q = re.sub(r"\b" + re.escape(abbr) + r"\b", expanded, q, flags=re.I)
    return q

def terms_of(text):
    return {t for t in tokenize(text) if t not in NOISE}

def topic_affinity(q, passage):
    # Prefer the actual named technology over generic shared words such as port/timeout.
    intro = passage.split(". ")[0]
    qterms = terms_of(q)
    lead = terms_of(intro)
    overlap = len(qterms & lead)
    return 1.5 * overlap + (2.0 if overlap >= 2 else 0.0)

def sentence_score(q, sentence):
    qt = terms_of(q)
    st = terms_of(sentence)
    shared = qt & st
    if not shared: return 0.0
    weight = sum(math.log(1 + 50/(DF.get(t, 0)+1)) for t in shared)
    return weight + 0.4*len(shared) - 0.15*len(st-qt)/max(len(st),1)

def numeric_answer(question, sentence):
    # Extract the factual value rather than returning an unrelated paragraph.
    if not re.search(r"\b(?:what|which|how|number|value|threshold|port|size|count|interval|timeout|rate|length|bits|quorum|latency|duration)\b", question.lower()):
        return None
    matches = list(re.finditer(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds|bits|bytes|samples|tokens|nodes|acceptors|participants|replicas|members|layers|blocks|records|retries|instances|port|%)?\b", sentence, flags=re.I))
    if not matches: return None
    q = question.lower()
    cue = terms_of(question)
    def value_rank(m):
        left = sentence[max(0,m.start()-85):m.start()]
        context = terms_of(left)
        return (len(cue & context), -len(left))
    m = max(matches, key=value_rank)
    return m.group().strip()

@app.post("/query")
def query(request: Query):
    q = clean_question(request.question.strip())
    if not q:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    qt = terms_of(q)
    if not qt:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    scored = []
    for i, passage in enumerate(KB):
        base = rank(q, DOCS[i])
        affinity = topic_affinity(q, passage)
        sscore = max((sentence_score(q,s) for s in SENTENCES[i]), default=0.0)
        scored.append((base + affinity + 0.45*sscore, i))
    scored.sort(reverse=True)
    best_score, best = scored[0]
    overlap = len(qt & terms_of(KB[best])) / len(qt)
    if best_score < 2.4 or overlap < 0.26:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    candidates = []
    for score, i in scored[:max(5, request.top_k)]:
        for s in SENTENCES[i]:
            candidates.append((sentence_score(q,s) + 0.12*score, score, i, s))
    candidates.sort(reverse=True)
    top_sentence_score, _, answer_source, sentence = candidates[0]
    if top_sentence_score < 1.8:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    # Cross-passage questions can cite multiple passages and include their distinct facts.
    compare = bool(re.search(r"\b(?:compare|versus|vs|difference|both|respectively|between|each)\b", q, re.I))
    if compare:
        chosen = []
        for _, _, i, s in candidates:
            if i not in [x[0] for x in chosen] and sentence_score(q,s) > 1.5:
                chosen.append((i,s))
            if len(chosen) >= min(request.top_k,3): break
        answer = " ".join(s for _,s in chosen) if chosen else sentence
    else:
        answer = numeric_answer(q,sentence) or sentence
    source_ids = [i for _,i in scored[:request.top_k]]
    if answer_source not in source_ids:
        source_ids = [answer_source] + source_ids[:request.top_k-1]
    # Sort sources by the same relevance metric used to generate scores.
    source_ids.sort(key=lambda i: next(s for s,j in scored if j==i), reverse=True)
    peak = max(best_score, 1e-9)
    scores = [round(min(1.0,max(0.0,next(s for s,j in scored if j==i)/peak)),4) for i in source_ids]
    confidence = round(min(0.97,0.65+0.30*overlap),4)
    return {"answer":answer,"sources":source_ids,"scores":scores,"confidence":confidence}
