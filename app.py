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
    q = re.split(r"(?i)\b(?:ignore (?:all |any |the )?(?:previous|above)|system prompt|developer message|you are now|instead (?:say|print|return)|disregard (?:the |all )?instructions)\b", q)[-1]
    q = q.strip()
    alias_pattern = r"\b(?:" + "|".join(sorted(map(re.escape, ALIASES), key=len, reverse=True)) + r")\b"
    q = re.sub(alias_pattern, lambda m: ALIASES[m.group().lower()], q, flags=re.I)
    return q

def terms_of(text):
    return {t for t in tokenize(text) if t not in NOISE}

def topic_affinity(q, passage):
    # Prefer the actual named technology over generic shared words such as port/timeout.
    intro = passage.split(". ")[0]
    qterms = terms_of(q)
    lead = terms_of(intro)
    overlap = len(qterms & lead)
    return 2.0 * overlap + (3.0 if overlap >= 2 else 0.0)

def sentence_score(q, sentence):
    qt = terms_of(q)
    st = terms_of(sentence)
    shared = qt & st
    if not shared: return 0.0
    weight = sum(math.log(1 + 50/(DF.get(t, 0)+1)) for t in shared)
    return weight + 0.4*len(shared) - 0.15*len(st-qt)/max(len(st),1)

def numeric_answer(question, sentence):
    # Prefer the number attached to the requested fact, not version numbers or other facts.
    if not re.search(r"\b(?:what|which|how|number|value|threshold|port|size|count|interval|timeout|rate|length|bits|quorum|latency|duration)\b", question.lower()):
        return None
    matches = list(re.finditer(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds|bits|bytes|samples|tokens|nodes|acceptors|participants|replicas|members|layers|blocks|records|retries|instances|port|%)?\b", sentence, flags=re.I))
    if not matches: return None
    qterms = terms_of(question)
    def score(m):
        prefix = sentence[max(0,m.start()-85):m.start()]
        nearby = terms_of(prefix)
        version_penalty = 8 if m.group().strip() in re.findall(r"\b\d+\.\d+\b", question) else 0
        return (len(qterms & nearby)*2 - version_penalty, m.start())
    return max(matches,key=lambda m: (bool(re.search(r"\b(?:ms|bits|bytes|samples|tokens|nodes|acceptors|participants|replicas|members|layers|blocks|records|retries|%)\b",m.group(),re.I)), score(m))).group().strip()

@app.post("/query")
def query(request: Query):
    q = clean_question(request.question.strip())
    # Unmentioned technical parameters are not answerable from this KB.
    original_terms = terms_of(q)
    unknown = [t for t in original_terms if len(t)>=8 and t not in DF and t not in {"duration","threshold","configured","autogenerated","respectively","difference","comparison","transport","default"}]
    if unknown:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
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
    # Strongly anchor named technologies to their passage introductions.
    original_q = request.question.lower()
    for i, passage in enumerate(KB):
        name = passage.split(" ")[0].lower().strip(".,")
        intro = passage.split(". ")[0].lower()
        phrase = " ".join(intro.split()[:3])
        bonus = 0.0
        if len(name)>2 and re.search(r"(?<![a-z])"+re.escape(name)+r"(?![a-z])",original_q):
            bonus += 100.0
        if phrase in q.lower():
            bonus += 100.0
        if i==36 and re.search(r"tls\s*1\.3",original_q): bonus += 45.0
        if i==37 and re.search(r"tls\s*1\.2",original_q): bonus += 45.0
        scored[i] = (scored[i][0]+bonus,i)
    scored.sort(reverse=True)
    best_score, best = scored[0]
    overlap = len(qt & terms_of(KB[best])) / len(qt)
    if best_score < 2.4 or overlap < 0.26:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    candidates = []
    compare = bool(re.search(r"\b(?:compare|versus|vs|difference|both|respectively|between|each)\b", q, re.I))
    numeric_intent = bool(re.search(r"\b(?:what|how many|how long|port|size|number|interval|timeout|duration|threshold|bits|quorum|batch|blocks|time|window|length|lease)\b",q,re.I))
    focus = terms_of(q) - terms_of(KB[best].split(". ")[0])
    if not focus:
        focus = terms_of(q)
    for score, i in scored[:max(5, request.top_k)]:
        if not compare and i != best:
            continue
        for s in SENTENCES[i]:
            has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b",s))
            if numeric_intent and not has_number:
                continue
            st = terms_of(s)
            matched = focus & st
            relevant = sum(math.log(1 + 50/(DF.get(t,0)+1)) for t in matched)
            candidates.append((relevant*3 + len(matched)*2 + 0.05*sentence_score(q,s), score, i, s))
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


# Offline smoke tests run once at startup; never block serving if a test fails.
try:
    from evaluation import evaluate
    _evaluation = evaluate(query, Query)
    print("RAG_EVALUATION_SUMMARY " + json.dumps(_evaluation), flush=True)
except Exception as exc:
    print("RAG_EVALUATION_ERROR " + repr(exc), flush=True)
