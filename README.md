# llm-service-fastapi

A small FastAPI service with one real feature: turn a blob of free text into structured
JSON (entity name, sentiment, topics), backed by Gemini, Claude, or a local Ollama model's
structured-output mode. The rest of the routes are leftover FastAPI-tutorial scaffolding
kept around as reference.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) — **required
  to start the app at all**, even if you only intend to call the Ollama endpoint. The
  Gemini client is created at import time and raises immediately if the key is missing.
- An Anthropic API key from the [Claude Console](https://console.anthropic.com/) — only
  needed for `/analyze/anthropic` and `/ask/anthropic`. Unlike the Gemini client, the
  Anthropic client doesn't check for the key at startup; a missing key only fails the
  requests that call Claude.
- [Ollama](https://ollama.com/download) installed and running locally, with the
  `gemma3:270m` model pulled (`ollama pull gemma3:270m`) — only needed for the
  `/analyze/ollama` endpoint.
- Docker, if you want to run it containerized

## Setup

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

Do **not** wrap the values in quotes. `python-dotenv` (used for local runs) strips quotes,
but `docker run --env-file` does not — quoting a value here works locally and silently
breaks inside Docker, since the provider then receives the quote characters as part of the
key.

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
| POST   | `/analyze/anthropic`  | The real feature, via Claude — see below.                                |
| POST   | `/analyze/ollama`     | The real feature, via a local Ollama model — see below.                  |
| POST   | `/ask/gemini`         | Free-form question answered by Gemini with time tools — see below.       |
| POST   | `/ask/anthropic`      | Free-form question answered by Claude with time tools — see below.       |
| GET    | `/time-mcp/tools`     | Lists the tools of the time MCP server — see [Time MCP server](#time-mcp-server). |
| POST   | `/time-mcp/call`      | Calls a tool on the time MCP server — see [Time MCP server](#time-mcp-server).     |

### `POST /analyze/gemini`, `POST /analyze/anthropic`, and `POST /analyze/ollama`

All three endpoints take the same request shape and return the same response shape — they
differ only in which model backs them (`gemini-3.6-flash`, `claude-opus-5`, or
`gemma3:270m`).

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

### `POST /ask/gemini` and `POST /ask/anthropic`

Both endpoints take a free-form question and let the model call the time tools below to
answer it. The model decides which tools to call, and it can chain them: for example, get
the current time in one zone, then check business hours in another.

Request:

```json
{
  "prompt": "If it's 15:00 in São Paulo, what time is it in Tokyo? Is that business hours there?"
}
```

Response (`200`):

```json
{
  "answer": "15:00 in São Paulo is 03:00 the next day in Tokyo, which is outside business hours."
}
```

If the model calls a tool that doesn't exist, or passes arguments that fail validation,
the error is sent back to it as the tool result instead of failing the request. Models
usually correct the call on the next round.

Each request allows at most 5 tool-call rounds. If the model is still calling tools after
that, it's told to answer with the results it already has, tool calls are disabled for
that last request, and the endpoint returns that partial answer. A warning is logged when
this happens. `/ask/anthropic` also requires `ANTHROPIC_API_KEY` in `.env`.

## Tools

The tools are exposed to both Gemini and Claude. Timezones are real IANA zone names,
resolved with Python's `zoneinfo`, so daylight saving time is handled. The supported zones
are `America/Sao_Paulo`, `Europe/London`, and `Asia/Tokyo`.

| Tool                | Arguments                                   | Returns                                        |
|---------------------|---------------------------------------------|------------------------------------------------|
| `get_current_time`  | `tz`                                        | Current local date and time in `tz`.           |
| `convert_time`      | `time` (`HH:MM`, 24-hour), `from_tz`, `to_tz` | That time converted from `from_tz` to `to_tz`. |
| `is_business_hours` | `tz`                                        | `true` if it's 09:00–17:00, Monday–Friday, in `tz`. |

- **`convert_time`** treats `time` as today's date in `from_tz`, so the conversion uses the
  offsets in effect today. The result can land on a different calendar day. For example,
  15:00 in São Paulo is 03:00 the next day in Tokyo.
- **`is_business_hours`** uses fixed hours (`BUSINESS_HOURS_START` / `BUSINESS_HOURS_END` in
  `time_tools.py`). It doesn't know about public holidays.

The tool functions live in `time_tools.py`, and both the Gemini/Claude loops and the MCP
server below use them.

### Time MCP server

The same three tools are also exposed as an MCP server, built with
[FastMCP](https://gofastmcp.com) in `time_mcp_server.py`. The API starts it as a child
process over stdio when it boots, next to the filesystem MCP server, and stops it on
shutdown. Nothing needs to be started by hand.

| Method | Path               | Description                                                       |
|--------|--------------------|-------------------------------------------------------------------|
| GET    | `/time-mcp/tools`  | Lists the server's tools with their descriptions and schemas.      |
| POST   | `/time-mcp/call`   | Calls a tool by name, e.g. `{"name": "convert_time", "arguments": {"time": "15:00", "from_tz": "America/Sao_Paulo", "to_tz": "Asia/Tokyo"}}`. |

Response (`200`): `{"status": "success", "result": ["2026-09-22T03:00:00+09:00"]}`. An
unknown tool name or invalid arguments (for example `"time": "3pm"`) returns a `400` with
the server's error message.

You can also run the server on its own and point any stdio MCP client at it:

```bash
uv run python -m llm_service_fastapi.time_mcp_server
```

Stdout is the protocol channel for stdio servers, so code in `time_tools.py` and
`time_mcp_server.py` must not `print()`. Log to stderr instead.

## Design choices

- **Structured output via a JSON schema, not prompt engineering.** `TextAnalyzes` (a
  Pydantic model) is passed straight to the model as its output schema — `response_schema`
  for Gemini, `output_format` on `messages.parse(...)` for Claude, and
  `TextAnalyzes.model_json_schema()` as `format` for Ollama. The response is validated
  against the schema on the way in (`TextAnalyzes.model_validate_json(...)` for Gemini and
  Ollama; the Anthropic SDK does it and returns `parsed_output`), so a shape mismatch fails
  loudly as a `500` rather than surfacing as a downstream `KeyError` or silently wrong data.
- **Each model call is a plain function, not a route.** `call_gemini`, `call_anthropic`,
  and `call_ollama` aren't decorated with `@app.post` — they're internal helpers that the
  `/analyze/*` endpoints call directly. That keeps the public API surface to purpose-built
  endpoints instead of exposing the raw prompt interface.
