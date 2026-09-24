# Use the existing Windows Ollama server

The project uses Ollama already installed on Windows. It does not start a
second Ollama server or download another copy of `qwen3.5:4b` through Docker.
Keep the Windows Ollama app or `ollama serve` running at
`http://localhost:11434`.

`qwen3.5:4b` is the chat model. If the embedding model is not installed yet,
open PowerShell and run:

```powershell
ollama pull qwen3-embedding:0.6b
```

Run the smoke check from Windows PowerShell:

```powershell
.\scripts\smoke-local-ollama.ps1
```

The script checks that both model tags are available, sends a chat request with
`think: false`, confirms a non-empty answer without returned thinking content,
then requests an embedding and checks that its vector is non-empty.

## Manual API calls

Chat with thinking disabled:

```powershell
$body = @{
  model = 'qwen3.5:4b'
  messages = @(@{ role = 'user'; content = 'Say hello in one sentence.' })
  think = $false
  stream = $false
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri 'http://localhost:11434/api/chat' `
  -ContentType 'application/json' -Body $body
```

Generate an embedding:

```powershell
$body = @{
  model = 'qwen3-embedding:0.6b'
  input = 'A short example sentence to embed.'
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri 'http://localhost:11434/api/embed' `
  -ContentType 'application/json' -Body $body
```

Both models run through the existing Windows Ollama process, which chooses GPU
or CPU placement according to its own settings and available hardware. The
Docker Compose stack does not manage Ollama or its model files.
