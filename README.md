---
title: RAG Colosseum
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# RAG Colosseum

FastAPI semantic retrieval baseline for the personalized 50-passage knowledge base.

## Setup

Add `kb.json` as a JSON array of exactly 50 passage strings in original order. The service cannot start without it. The prepared project ZIP in ChatGPT includes a draft `kb.json`; verify it against the assignment's Copy JSON before uploading.

Install dependencies: `pip install -r requirements.txt`
Run locally: `uvicorn app:app --host 0.0.0.0 --port 7860`

Test `/health` and `POST /query` with `{"question":"What is the Raft heartbeat interval?","top_k":3}`.

## Deploy

Create a Hugging Face Space with Docker SDK and CPU Basic hardware. Upload or synchronize the repository files and verify the public HTTPS endpoint. Submit only the base URL, without `/query`.

**Status:** Untested baseline, not a guaranteed full-score solution. It requires a verified KB and latency/quality tests against the 20-probe gauntlet.
