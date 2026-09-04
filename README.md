# llm-service-fastapi

A small FastAPI service with one real feature: turn a blob of free text into structured
JSON (entity name, sentiment, topics) using Gemini's structured-output mode. The rest of
the routes are leftover FastAPI-tutorial scaffolding kept around as reference.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
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

## Running tests

```bash
uv run pytest
```

## Endpoints

| Method | Path                | Description                                                              |
|--------|---------------------|---------------------------------------------------------------------------|
| GET    | `/`                 | Health-check-style hello world.                                          |
| GET    | `/items/{item_id}`  | Tutorial route: echoes a path param and optional query param.            |
| PUT    | `/Items/{item_id}`  | Tutorial route: echoes a request body validated against `Item`.          |
| POST   | `/analyze`          | The real feature — see below.                                            |

### `POST /analyze`

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
(invalid input, Gemini error, malformed model output) the endpoint returns a `500` with
an error `detail` string rather than a partially-filled response.

## Design choices

- **Structured output via `response_schema`, not prompt engineering.** `TextAnalyzes` (a
  Pydantic model) is passed straight to Gemini as `response_schema` with
  `response_mime_type="application/json"`, and the returned JSON is parsed back with
  `TextAnalyzes.model_validate_json(...)`. This means the response is validated against
  the schema on the way in — a shape mismatch fails loudly as a `500` rather than
  surfacing as a downstream `KeyError` or silently wrong data.
- **The Gemini call is a plain function, not a route.** `callGemini` isn't decorated with
  `@app.post` — it's an internal helper that `/analyze` calls directly. That keeps the
  public API surface to one purpose-built endpoint instead of exposing the raw prompt
  interface.
