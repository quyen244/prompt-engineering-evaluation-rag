param(
  [string]$BaseUrl = 'http://localhost:11434',
  [string]$ChatModel = 'qwen3.5:4b',
  [string]$EmbeddingModel = 'qwen3-embedding:0.6b'
)

$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')

try {
  $tags = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/tags" -TimeoutSec 15
} catch {
  throw "Cannot reach Ollama at $BaseUrl. Start the existing Windows Ollama app or run 'ollama serve', then retry. Details: $($_.Exception.Message)"
}

$installedModels = @($tags.models | ForEach-Object { [string]$_.name })
foreach ($modelName in @($ChatModel, $EmbeddingModel)) {
  if ($installedModels -notcontains $modelName) {
    throw "Model '$modelName' is not installed. Pull it with: ollama pull $modelName"
  }
}

$chatBody = @{
  model = $ChatModel
  messages = @(@{ role = 'user'; content = 'Reply with a short greeting.' })
  think = $false
  stream = $false
} | ConvertTo-Json -Depth 10

try {
  $chat = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/chat" `
    -ContentType 'application/json' -Body $chatBody -TimeoutSec 300
} catch {
  throw "Ollama chat call failed for '$ChatModel'. Details: $($_.Exception.Message)"
}

if ([string]::IsNullOrWhiteSpace([string]$chat.message.content)) {
  throw 'Chat call returned an empty message.content.'
}
if (-not [string]::IsNullOrWhiteSpace([string]$chat.message.thinking)) {
  throw 'Chat call returned non-empty message.thinking despite think=false.'
}
Write-Host "PASS chat: $ChatModel returned content with think=false"

$embedBody = @{
  model = $EmbeddingModel
  input = 'A short example sentence to embed.'
} | ConvertTo-Json -Depth 10

try {
  $embed = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/embed" `
    -ContentType 'application/json' -Body $embedBody -TimeoutSec 300
} catch {
  throw "Ollama embedding call failed for '$EmbeddingModel'. Details: $($_.Exception.Message)"
}

$vector = $embed.embeddings[0]
if ($null -eq $vector -or @($vector).Count -eq 0) {
  throw 'Embedding call returned no vector in embeddings[0].'
}
Write-Host "PASS embedding: $EmbeddingModel returned a $(@($vector).Count)-value vector"
