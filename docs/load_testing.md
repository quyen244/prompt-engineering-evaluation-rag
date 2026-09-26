# Local RAG Query Load Test

## TechQA 1-user and 100-user comparison

Run this only after the TechQA retrieval and generation evaluations are complete.
The script validates the 100-document corpus and 20 questions, creates a dedicated Qdrant collection, starts the API against that collection, then runs one user followed by 100 users.
The one-user profile sends each question once (20 sequential requests).
The 100-user profile sends 100 concurrent requests, cycling through the 20 questions five times.
Both profiles use the same Top K, embedding model, generation model, API, and collection.
The live API currently serves dense Qdrant retrieval, so this is the serving-stack dense/Qwen baseline even if a different retrieval configuration wins the offline ablation.
The second run starts only if the first completes without API failures.

Prerequisites are Docker Compose, Python 3.11, Ollama running locally with `qwen3-embedding:0.6b` and `qwen3.5:4b` installed, and enough local capacity for 100 live generation requests.
The script creates `.venv-loadtest` and installs only the load-test dependencies if it is absent.

```powershell
./scripts/run_techqa_load.ps1
```

Each run is bounded by 900 seconds, a 600-second request timeout, and its request count.
Failed or incomplete runs are marked FAILED in MLflow and cause the script to stop.
The MLflow experiment `rag-query-concurrency` records the corpus and question SHA-256 values, exact collection, dense retrieval strategy, local model names, user count, response time p50/p95/p99, throughput, failure rate, and server-side stage times.
The API must use the generated `QDRANT_COLLECTION`; the script passes it into Compose for `rag-api`.
This comparison includes API queuing and Ollama generation time, so it measures the full live query path.

For a read-only collection check after indexing:

```powershell
.venv-loadtest\Scripts\python.exe scripts/techqa_index.py check
```

`scripts/techqa_index.py name` prints the collection name without network calls.
The index command never deletes a collection or mutates the default application collection.

This Locust scenario models concurrent users asking questions against documents that are already indexed.
It calls the same `POST /api/query` endpoint as the Ask UI.
The API reports separate server timings for question embedding, Qdrant retrieval, and Ollama answer generation in the `Server-Timing` response header.
Locust aggregates those timings into the MLflow run.
The scenario does not upload documents.

Each run performs API health and indexed-document preflight checks before starting.
By default, all currently indexed documents are selected; set `RAG_LOAD_DOCUMENT_IDS` to a comma-separated subset to pin a run to specific documents.
The request text is fixed for the run and its SHA-256 is logged instead of the question itself.
Document IDs are also represented by a fingerprint in MLflow, not logged as plaintext.

## Start Locust

Use Python 3.11, matching the project's documented Python version.
The load-test dependencies are isolated from the application environment:

```powershell
py -3.11 -m venv .venv-loadtest
.venv-loadtest\Scripts\python.exe -m pip install -r requirements/load-test.txt
```

Make sure the Compose stack is running and at least one document is indexed.
Then start the Locust web interface:

```powershell
$env:RAG_LOAD_BASE_URL = "http://localhost:8000"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
.venv-loadtest\Scripts\python.exe -m locust -f scripts/locust_rag.py --host http://localhost:8000
```

Open `http://localhost:8089`.
For the next five-user stage-timing run, enter 5 users and a spawn rate of 1 user/second, then start the test.
The profile automatically stops after 120 seconds or 20 requests, whichever happens first.
Each request also has a 60-second client timeout.
The limits can be changed with `RAG_LOAD_MAX_DURATION_SECONDS`, `RAG_LOAD_MAX_REQUESTS`, and `RAG_LOAD_REQUEST_TIMEOUT_SECONDS` before starting Locust.
Review the Locust total latency and the per-stage mean/p50/p95 values in MLflow before increasing concurrency.

The same run can be executed headlessly:

```powershell
.venv-loadtest\Scripts\python.exe -m locust -f scripts/locust_rag.py --host http://localhost:8000 --headless -u 5 -r 1
```

## Results

Every completed run is recorded in MLflow experiment `rag-query-concurrency` by default.
The run logs the target user count, spawn rate, Top K, selected-document fingerprint, configured local models, request cap, duration cap, request count, failure rate, throughput, and mean/p50/p95/p99/max response times.
It also records embedding, retrieval, and generation sample counts and mean/p50/p95 latencies in `load-test/summary.json`.

The API query path generates an answer for every request, so local Ollama inference capacity may become the bottleneck before Qdrant does.
These are single-machine learning measurements, not a production capacity claim.
Do not start at 100 users; increase load gradually and stop if the machine becomes unresponsive or resource pressure becomes unsafe.
