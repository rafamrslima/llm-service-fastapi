# llm-service-fastapi

A small FastAPI service with one real feature: turn a blob of free text into structured
JSON (entity name, sentiment, topics), backed by either Gemini or a local Ollama model's
structured-output mode. The rest of the routes are leftover FastAPI-tutorial scaffolding
kept around as reference.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) — **required
  to start the app at all**, even if you only intend to call the Ollama endpoint. The
  Gemini client is created at import time and raises immediately if the key is missing.
- [Ollama](https://ollama.com/download) installed and running locally, with the
  `gemma3:270m` model pulled (`ollama pull gemma3:270m`) — only needed for the
  `/analyze/ollama` endpoint.
- Docker, if you want to run it containerized

## Setup

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

Do **not** wrap the value in quotes. `python-dotenv` (used for local runs) strips quotes,
but `docker run --env-file` does not — quoting the value here works locally and silently
breaks inside Docker, since Google then receives the quote characters as part of the key.

## Running locally

```bash
uv sync
uv run fastapi dev src/llm_service_fastapi/api.py
```

The API is served at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## Running with Docker

```bash
docker build -t llm-service-fastapi .
docker run -p 8000:8000 --env-file .env llm-service-fastapi
```

The Ollama client defaults to `http://127.0.0.1:11434`, which inside a container is the
container itself, not your host machine — so `/analyze/ollama` won't reach an Ollama
server running on the host unless you point it there explicitly, e.g.:

```bash
docker run -p 8000:8000 --env-file .env -e OLLAMA_HOST=http://host.docker.internal:11434 llm-service-fastapi
```

(`host.docker.internal` resolves to the host on Docker Desktop for Mac/Windows; on Linux
you may instead need `--add-host=host.docker.internal:host-gateway` or the host's LAN IP.)

## Running tests

```bash
uv run pytest
```

## Endpoints

| Method | Path                  | Description                                                              |
|--------|-----------------------|---------------------------------------------------------------------------|
| GET    | `/`                   | Health-check-style hello world.                                          |
| GET    | `/items/{item_id}`    | Tutorial route: echoes a path param and optional query param.            |
| PUT    | `/Items/{item_id}`    | Tutorial route: echoes a request body validated against `Item`.          |
| POST   | `/analyze/gemini`     | The real feature, via Gemini — see below.                                |
| POST   | `/analyze/ollama`     | The real feature, via a local Ollama model — see below.                  |

### `POST /analyze/gemini` and `POST /analyze/ollama`

Both endpoints take the same request shape and return the same response shape — they
differ only in which model backs them.

Request:

```json
{
  "text": "The new phone has an amazing camera but terrible battery life."
}
```

Response (`200`):

```json
{
  "name": "phone",
  "sentiment": "negative",
  "topics": ["camera", "battery life"]
}
```

`sentiment` is always one of `positive`, `negative`, or `neutral`. On any failure
(invalid input, model/API error, malformed model output) the endpoint returns a `500`
with an error `detail` string rather than a partially-filled response.

`/analyze/ollama` runs against `gemma3:270m`, a very small model — expect noticeably
lower-quality output than Gemini, including occasional degenerate repetition inside the
`topics` array under near-zero temperature.

## Design choices

- **Structured output via a JSON schema, not prompt engineering.** `TextAnalyzes` (a
  Pydantic model) is passed straight to the model as its output schema — `response_schema`
  for Gemini, `TextAnalyzes.model_json_schema()` as `format` for Ollama — and the returned
  JSON is parsed back with `TextAnalyzes.model_validate_json(...)`. This means the response
  is validated against the schema on the way in — a shape mismatch fails loudly as a `500`
  rather than surfacing as a downstream `KeyError` or silently wrong data.
- **Each model call is a plain function, not a route.** `call_gemini` and `call_ollama`
  aren't decorated with `@app.post` — they're internal helpers that `/analyze/gemini` and
  `/analyze/ollama` call directly. That keeps the public API surface to two purpose-built
  endpoints instead of exposing the raw prompt interface.
