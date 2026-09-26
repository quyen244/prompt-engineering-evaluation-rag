param(
    [string]$BaseUrl = 'http://localhost:8000',
    [string]$QdrantUrl = 'http://localhost:6333',
    [string]$OllamaUrl = 'http://localhost:11434',
    [string]$OllamaContainerUrl = 'http://host.docker.internal:11434',
    [string]$MlflowUrl = 'http://localhost:5000'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$venvPython = Join-Path $repoRoot '.venv-loadtest\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3.11 -m venv .venv-loadtest
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
    & $venvPython -m pip install -r requirements/load-test.txt
    if ($LASTEXITCODE -ne 0) { throw 'Load test dependency installation failed' }
}

function Wait-Endpoint([string]$Url) {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            Invoke-RestMethod -Uri $Url -TimeoutSec 3 | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    throw "Service did not become ready: $Url"
}

$nameLines = & $venvPython scripts/techqa_index.py name
if ($LASTEXITCODE -ne 0) { throw 'TechQA dataset validation failed' }
$collectionLine = $nameLines | Where-Object { $_ -match '^collection: ' } | Select-Object -First 1
if (-not $collectionLine) { throw 'TechQA collection name was not returned' }
$env:QDRANT_COLLECTION = ($collectionLine -replace '^collection: ', '').Trim()
$env:QDRANT_URL = $QdrantUrl
$env:OLLAMA_BASE_URL = $OllamaUrl
$env:OLLAMA_EMBED_MODEL = 'qwen3-embedding:0.6b'
$env:OLLAMA_CHAT_MODEL = 'qwen3.5:4b'
$env:RAG_LOAD_DATASET = 'techqa'
$env:RAG_LOAD_BASE_URL = $BaseUrl
$env:MLFLOW_TRACKING_URI = $MlflowUrl
$env:RAG_LOAD_TOP_K = '5'
$env:RAG_LOAD_REQUEST_TIMEOUT_SECONDS = '600'
$env:RAG_LOAD_MAX_DURATION_SECONDS = '900'

docker compose up -d qdrant
if ($LASTEXITCODE -ne 0) { throw 'Qdrant startup failed' }
Wait-Endpoint "$QdrantUrl/collections"
& $venvPython scripts/techqa_index.py index --qdrant-url $QdrantUrl --ollama-url $OllamaUrl
if ($LASTEXITCODE -ne 0) { throw 'TechQA collection indexing failed' }

$env:OLLAMA_BASE_URL = $OllamaContainerUrl
docker compose up -d --build rag-api rag-mlflow
if ($LASTEXITCODE -ne 0) { throw 'FastAPI or MLflow startup failed' }
Wait-Endpoint "$BaseUrl/api/health"
Wait-Endpoint $MlflowUrl

$env:RAG_LOAD_MAX_REQUESTS = '20'
$env:RAG_LOAD_RUN_NAME = 'techqa-1-user'
& $venvPython -m locust -f scripts/locust_rag.py --host $BaseUrl --headless -u 1 -r 1 -t 900s --exit-code-on-error 1
if ($LASTEXITCODE -ne 0) { throw 'One-user load run failed; the 100-user run was not started' }

$env:RAG_LOAD_MAX_REQUESTS = '100'
$env:RAG_LOAD_RUN_NAME = 'techqa-100-users'
& $venvPython -m locust -f scripts/locust_rag.py --host $BaseUrl --headless -u 100 -r 100 -t 900s --exit-code-on-error 1
if ($LASTEXITCODE -ne 0) { throw '100-user load run failed; inspect the failed MLflow run' }

Write-Output 'status: complete'
Write-Output "collection: $env:QDRANT_COLLECTION"
Write-Output "mlflow: $MlflowUrl"
