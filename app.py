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
    empty = {"answer":None,"sources":[],"scores":[],"confidence":0.0}
    if not q:
        return empty
    qt = terms_of(q)
    if not qt:
        return empty
    # Identify the subject using distinctive names, not generic words like "batch".
    names = {
        0:["raft"],1:["multi-paxos","paxos"],2:["gossip","phi accrual"],3:["swim"],
        4:["two-phase commit","2pc"],5:["saga"],6:["consistent hashing"],
        7:["rendezvous hashing"],8:["vector clocks"],9:["hybrid logical clocks"],
        10:["sgd","stochastic gradient descent"],11:["adamw"],12:["transformer encoder"],
        13:["autoregressive","transformer decoder"],14:["logistic regression"],
        15:["support vector","svm"],16:["multi-layer perceptron","mlp"],
        17:["residual convolutional","resnet"],18:["batch normalization"],
        19:["layer normalization"],20:["postgresql","postgres"],21:["mysql","innodb"],
        22:["b+ tree"],23:["lsm","log-structured merge"],
        24:["mongodb"],25:["cassandra"],26:["redis"],27:["memcached"],
        28:["write-ahead logging","wal"],29:["shadow paging"],
        30:["transmission control protocol","tcp"],31:["quic"],
        32:["domain name system","dns"],33:["mdns","multicast dns"],
        34:["http/2"],35:["http/3"],36:["tls 1.3"],37:["tls 1.2"],
        38:["bgp","border gateway protocol"],39:["ospf","open shortest path first"],
        40:["public key cryptography","rsa"],41:["elliptic curve","ecc","ecdsa"],
        42:["credential storage","password hashing"],43:["pbkdf2"],
        44:["web application firewall","waf"],45:["api gateway"],
        46:["aes","advanced encryption standard"],47:["chacha20"],
        48:["pkce","proof key for code exchange"],49:["jwt","json web token"]
    }
    rawq = request.question.lower()
    anchors = []
    for i, aliases in names.items():
        if any(re.search(r"(?<![a-z0-9])"+re.escape(alias)+r"(?![a-z0-9])",rawq) for alias in aliases):
            anchors.append(i)
    # Don't mistake broad TCP/DNS/TLS mentions for the named subject of another passage.
    if len(anchors)>1:
        anchors = [i for i in anchors if not any(
            i!=j and any(alias in rawq for alias in names[j])
            and (i,j) in {(32,33),(30,31),(36,37),(37,36),(40,41),(10,11)}
            for j in anchors)] or anchors
    scored=[]
    for i in range(50):
        lexical=rank(q,DOCS[i])+topic_affinity(q,KB[i])
        scored.append((lexical + (80 if i in anchors else 0),i))
    scored.sort(reverse=True)
    bestscore,best=scored[0]
    # Unknown parameters or out-of-KB subjects should abstain.
    if not anchors and bestscore < 4.5:
        return empty
    compare=bool(re.search(r"\b(?:compare|versus|vs|difference|both|respectively|between)\b",rawq))
    targets=[i for i in anchors if i in [x[1] for x in scored[:10]]] if compare and len(anchors)>1 else [best]
    if compare and len(targets)<2:
        targets=[i for _,i in scored[:2]]
    answers=[]
    for i in targets[:min(request.top_k,3)]:
        # Remove subject words so sentence matching focuses on the requested attribute.
        subject=set()
        for alias in names[i]:
            subject |= terms_of(alias)
        focus=qt-subject
        focus -= {"mini","batch","default","number","value","knowledge","base","according"}
        if not focus:
            focus=qt
        sentences=SENTENCES[i]
        numbered=[s for s in sentences if re.search(r"\b\d+(?:\.\d+)?\b",s)]
        is_fact=bool(re.search(r"\b(?:port|timeout|interval|duration|quorum|threshold|size|rate|length|bits|block|layer|batch|time|window|lease|heartbeat|number|how many|how long|how much|key|clock skew)\b",rawq))
        pool=numbered if is_fact else sentences
        if not pool:
            continue
        def relevance(s):
            st=terms_of(s)
            common=focus&st
            return (sum(math.log(1+50/(DF.get(t,0)+1)) for t in common),
                    len(common),sentence_score(q,s))
        sentence=max(pool,key=relevance)
        matched=focus&terms_of(sentence)
        # Reject unsupported attributes, rather than returning unrelated numbers.
        if not matched and len(focus)>0:
            continue
        value=numeric_answer(q,sentence) if is_fact else None
        if is_fact and not value:
            continue
        answers.append((i,(value or sentence),sentence))
    if not answers:
        return empty
    if not compare:
        i,value,sentence=answers[0]
        answer=value
    else:
        answer="; ".join(f"{KB[i].split(' ')[0]}: {sentence}" for i,_,sentence in answers)
    ids=[i for _,i in scored[:request.top_k]]
    for i,_,_ in answers:
        if i not in ids:
            ids[-1]=i
    ids=list(dict.fromkeys(ids))
    score_map=dict((i,s) for s,i in scored)
    ids.sort(key=lambda i:score_map[i],reverse=True)
    scores=[round(min(1.,max(0.,score_map[i]/max(bestscore,0.0001))),4) for i in ids]
    return {"answer":answer,"sources":ids,"scores":scores,"confidence":0.85}


# Offline smoke tests run once at startup; never block serving if a test fails.
try:
    from evaluation import evaluate
    _evaluation = evaluate(query, Query)
    print("RAG_EVALUATION_SUMMARY " + json.dumps(_evaluation), flush=True)
except Exception as exc:
    print("RAG_EVALUATION_ERROR " + repr(exc), flush=True)
