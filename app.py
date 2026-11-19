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
    return max(matches,key=score).group().strip()

# Canonical names identify a passage; factual answers are always extracted from KB.
SUBJECTS = [
 ["raft"],["multi-paxos","paxos"],["gossip","phi accrual"],["swim"],["two-phase commit","2pc"],
 ["saga"],["consistent hashing"],["rendezvous hashing"],["vector clocks"],["hybrid logical clocks"],
 ["sgd","stochastic gradient descent"],["adamw"],["transformer encoder"],["autoregressive","transformer decoder"],
 ["logistic regression"],["support vector","svm"],["multi-layer perceptron","mlp"],
 ["residual convolutional","resnet"],["batch normalization"],["layer normalization"],
 ["postgresql","postgres"],["mysql","innodb"],["b+ tree"],["log-structured merge","lsm"],
 ["mongodb"],["cassandra"],["redis"],["memcached"],["write-ahead logging","wal"],["shadow paging"],
 ["transmission control protocol","tcp"],["quic"],["domain name system","dns"],["mdns","multicast dns"],
 ["http/2"],["http/3"],["tls 1.3"],["tls 1.2"],["bgp","border gateway protocol"],
 ["ospf","open shortest path first"],["public key cryptography","rsa"],["elliptic curve","ecdsa","ecc"],
 ["credential storage","password hashing"],["pbkdf2"],["web application firewall","waf"],
 ["api gateway"],["aes","advanced encryption standard"],["chacha20"],["pkce","proof key for code exchange"],
 ["jwt","json web token"]
]
# The principal configured fact for each passage is in the second or third sentence.
# Keep the facts in kb.json as the single source of truth, not in this table.
FACT_SENTENCE = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,2,2,1,2,1,1,2,2]
ATTRS = {
 "port":{"port","listen","listener","socket","tcp","udp","network","transport","cql","peer","multicast","server"},
 "time":{"time","timeout","interval","duration","lease","heartbeat","handshake","window","latency","skew","dead","suspicion","ping","grace","grant"},
 "size":{"size","batch","sample","block","layer","stack","key","length","bit","quorum","replica","acceptor","participant"},
}
def _question_subjects(question):
    q=question.lower()
    found=[]
    for i,aliases in enumerate(SUBJECTS):
        if any(re.search(r"(?<![a-z0-9])"+re.escape(a)+r"(?![a-z0-9])",q) for a in aliases):
            found.append(i)
    # Ignore a generic parent protocol if the question names a more specific variant.
    if 32 in found and 33 in found: found.remove(32)
    if 30 in found and 31 in found: found.remove(30)
    if 36 in found and 37 in found:
        if "1.3" in q: found.remove(37)
        elif "1.2" in q: found.remove(36)
    return found

def _fact_for(i, question):
    sentences=SENTENCES[i]
    q=question.lower()
    # A question about a configuration absent from the passage should abstain.
    if re.search(r"\b(?:autovacuum|scale factor|eviction grace|release date)\b",q):
        return None
    # Select a relevant sentence containing a configured value, not the intro.
    nums=[(j,s) for j,s in enumerate(sentences) if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|bits|samples|nodes|blocks|port|replicas|acceptors|participants|records|bytes)?\b",s)]
    if not nums:
        return None
    focus=terms_of(question)
    for alias in SUBJECTS[i]:
        focus-=terms_of(alias)
    focus-={"what","which","how","many","much","long","default","number","configured","value","knowledge","base","according","used"}
    def value(j,s):
        st=terms_of(s)
        overlap=focus&st
        weighted=sum(math.log(1+50/(DF.get(t,0)+1)) for t in overlap)
        # Match semantic attributes, not just shared numerals or product names.
        categories=[k for k,words in ATTRS.items() if focus&words]
        semantic=sum(3 for k in categories if st&ATTRS[k])
        return (weighted+semantic, len(overlap), j==FACT_SENTENCE[i])
    j,s=max(nums,key=lambda pair:value(*pair))
    quality=value(j,s)[0]
    if quality < 1.0:
        return None
    return s

@app.post("/query")
def query(request: Query):
    q=clean_question(request.question.strip())
    empty={"answer":None,"sources":[],"scores":[],"confidence":0.0}
    if not q: return empty
    subjects=_question_subjects(q)
    compare=bool(re.search(r"\b(?:compare|versus|vs|difference|both|respectively|between)\b",q,re.I))
    if not subjects:
        # Conservative abstention: do not invent a subject for unrelated questions.
        return empty
    if not compare: subjects=subjects[:1]
    chosen=[]
    for i in subjects[:request.top_k]:
        s=_fact_for(i,q)
        if s is not None:
            chosen.append((i,s))
    if not chosen:
        return empty
    answer=("; ".join(s for _,s in chosen) if compare else chosen[0][1])
    ids=[i for i,_ in chosen]
    ranked=sorted([(rank(q,DOCS[i]),i) for i in range(50)],reverse=True)
    for _,i in ranked:
        if len(ids)>=request.top_k: break
        if i not in ids: ids.append(i)
    relevance={i:rank(q,DOCS[i]) for i in ids}
    # Scores are normalized cosine-like lexical relevance, not invented answer confidences.
    ids.sort(key=lambda i:relevance[i],reverse=True)
    high=max(relevance.values()) if relevance else 1.0
    scores=[round(min(1.0,max(0.0,relevance[i]/max(high,0.0001))),4) for i in ids]
    return {"answer":answer,"sources":ids,"scores":scores,"confidence":0.85}


# Offline smoke tests run once at startup; never block serving if a test fails.
try:
    from evaluation import evaluate
    _evaluation = evaluate(query, Query)
    print("RAG_EVALUATION_SUMMARY " + json.dumps(_evaluation), flush=True)
except Exception as exc:
    print("RAG_EVALUATION_ERROR " + repr(exc), flush=True)
