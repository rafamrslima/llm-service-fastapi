import json
import logging
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo
import os
import ollama
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TextAnalyzes(BaseModel):
    name: str = Field(description="The primary entity or person mentioned in the text.")
    sentiment: SentimentEnum = Field(
        description="Overall sentiment toward the main entity."
    )
    topics: list[str] = Field(description="List of key topics discussed in the text.")


class AnalysisRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str


class Timezones(str, Enum):
    SAO_PAULO = "America/Sao_Paulo"
    LONDON = "Europe/London"
    TOKYO = "Asia/Tokyo"


class WriteFileRequest(BaseModel):
    rel_path: str
    content: str


class ReadFileRequest(BaseModel):
    rel_path: str


BUSINESS_HOURS_START = 9
BUSINESS_HOURS_END = 17
SANDBOX_PATH = os.path.expanduser("~/mcp_sandbox")


def resolve_sandboxed_path(rel_path: str) -> str:
    """Resolves rel_path under SANDBOX_PATH, rejecting escapes (.., absolute paths)."""
    sandbox_root = os.path.realpath(SANDBOX_PATH)
    full_path = os.path.realpath(os.path.join(sandbox_root, rel_path))
    if os.path.commonpath([sandbox_root, full_path]) != sandbox_root:
        raise HTTPException(
            status_code=400,
            detail=f"'{rel_path}' resolves outside the sandbox.",
        )
    return full_path


def get_current_time(tz: Timezones):
    time_now = datetime.now(ZoneInfo(tz.value))
    print(time_now)
    return time_now


def convert_time(time: str, from_tz: Timezones, to_tz: Timezones):
    source_zone = ZoneInfo(from_tz.value)
    # Anchored to today's date in the source zone so the conversion respects DST.
    today = datetime.now(source_zone).date()
    parsed = datetime.strptime(time, "%H:%M").time()  # noqa: DTZ007 - only the time part is kept; tzinfo is attached below
    source = datetime.combine(today, parsed, tzinfo=source_zone)
    return source.astimezone(ZoneInfo(to_tz.value))


def is_business_hours(tz: Timezones):
    now = datetime.now(ZoneInfo(tz.value))
    is_weekday = now.weekday() < 5
    return is_weekday and BUSINESS_HOURS_START <= now.hour < BUSINESS_HOURS_END


class GetCurrentTimeParams(BaseModel):
    tz: Timezones = Field(
        description="IANA timezone name to compute the current local time for."
    )


class ConvertTimeParams(BaseModel):
    time: str = Field(
        description="Time to convert, in 24-hour HH:MM format.",
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
    )
    from_tz: Timezones = Field(description="IANA timezone the time is given in.")
    to_tz: Timezones = Field(description="IANA timezone to convert the time into.")


class IsBusinessHoursParams(BaseModel):
    tz: Timezones = Field(description="IANA timezone to check business hours in.")


TOOL_SPECS = {
    "get_current_time": (
        "Returns the current local date and time for a given IANA timezone.",
        GetCurrentTimeParams,
        get_current_time,
    ),
    "convert_time": (
        (
            "Converts a time from one IANA timezone to another, using today's "
            "date in the source timezone."
        ),
        ConvertTimeParams,
        convert_time,
    ),
    "is_business_hours": (
        (
            f"Returns whether it is currently business hours "
            f"({BUSINESS_HOURS_START:02d}:00-{BUSINESS_HOURS_END:02d}:00, "
            f"Monday to Friday) in a given IANA timezone."
        ),
        IsBusinessHoursParams,
        is_business_hours,
    ),
}

time_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=name,
            description=description,
            parameters_json_schema=params_model.model_json_schema(),
        )
        for name, (description, params_model, _) in TOOL_SPECS.items()
    ]
)

anthropic_time_tools = [
    {
        "name": name,
        "description": description,
        "input_schema": params_model.model_json_schema(),
    }
    for name, (description, params_model, _) in TOOL_SPECS.items()
]


# Caps how many times a model may chain tool calls before we ask it to wrap up.
MAX_TOOL_ROUNDS = 5

STEP_LIMIT_INSTRUCTION = (
    "You've reached the tool-call limit for this request and can't call any more "
    "tools. Answer now using the results you already have, and briefly say what you "
    "weren't able to complete."
)


def run_tool(name: str, args: dict | None) -> tuple[object, bool]:
    # Errors are returned instead of raised so the model sees them and can retry.
    if name not in TOOL_SPECS:
        print(f"[tool call] unknown tool {name!r}")
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_SPECS)}.", True

    _, params_model, fn = TOOL_SPECS[name]
    try:
        params = params_model.model_validate(args or {})
    except ValidationError as e:
        print(f"[tool call] {name} rejected invalid arguments: {args}")
        return f"Invalid arguments for '{name}': {e.json(include_url=False)}", True

    result = fn(**params.model_dump())
    print(f"[tool call] {name}({params.model_dump_json()}) -> {result}")
    return (result.isoformat() if isinstance(result, datetime) else result), False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Spawns MCP subprocess on API startup, cleans up on shutdown."""
    os.makedirs(SANDBOX_PATH, exist_ok=True)

    # Configure stdio process launch parameters
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", SANDBOX_PATH],
    )

    print(f"[MCP] Spawning Filesystem MCP server targeting: {SANDBOX_PATH}")

    # Launch subprocess streams and client session
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("[MCP] Server initialized successfully.")

            # Store session globally on app state for endpoints to use
            app.state.mcp_session = session
            yield

    print("[MCP] Process shut down safely.")


load_dotenv()
app = FastAPI(lifespan=lifespan)
geminiClient = genai.Client()
ollamaClient = ollama.AsyncClient()
anthropicClient = AsyncAnthropic()


@app.get("/")
async def read_root():
    return "Hello World FastAPI"


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put("/Items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"Name": item.name, "price": item.price}


async def call_gemini(prompt: str) -> TextAnalyzes:
    try:
        response = await geminiClient.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TextAnalyzes,
                temperature=0.1,
            ),
        )
        return TextAnalyzes.model_validate_json(response.text)
    except Exception as e:  # noqa: BLE001 - Gemini errors and schema mismatches both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_anthropic(prompt: str) -> TextAnalyzes:
    try:
        response = await anthropicClient.messages.parse(
            model="claude-opus-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=TextAnalyzes,
        )
        return response.parsed_output
    except Exception as e:  # noqa: BLE001 - Anthropic errors and schema mismatches both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_ollama(prompt: str) -> TextAnalyzes:
    try:
        response = await ollamaClient.generate(
            model="gemma3:270m",
            prompt=prompt,
            format=TextAnalyzes.model_json_schema(),
            options={"temperature": 0.1},
        )
        return TextAnalyzes.model_validate_json(response.response)
    except Exception as e:  # noqa: BLE001 - Ollama errors and schema mismatches both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_gemini_with_tools(prompt: str) -> str:
    try:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await geminiClient.aio.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[time_tool],
                    system_instruction="If user asks about the time in Sao Paulo, add a note saying that this city should be called Corinthians instead.",
                ),
            )

            function_calls = response.function_calls
            if not function_calls:
                return response.text

            parts = []
            for call in function_calls:
                content, is_error = run_tool(call.name, call.args)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"error" if is_error else "result": content},
                    )
                )
            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=parts))

        logger.warning(
            "Gemini wasn't able to complete that within the step limit "
            "(%d tool rounds); returning a partial answer.",
            MAX_TOOL_ROUNDS,
        )
        contents[-1].parts.append(types.Part(text=STEP_LIMIT_INSTRUCTION))
        final = await geminiClient.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[time_tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                ),
                system_instruction="If user asks about the time in Sao Paulo, add a note saying that this city should be called Corinthians instead.",
            ),
        )
        return final.text
    except Exception as e:  # noqa: BLE001 - Gemini errors and tool-call handling both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_anthropic_with_tools(prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await anthropicClient.messages.create(
                model="claude-opus-5",
                max_tokens=1024,
                system="If user asks about the time in Sao Paulo, add a note saying that this city should be called Corinthians instead..",
                messages=messages,
                tools=anthropic_time_tools,
            )

            print(response.usage)
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                return next((b.text for b in response.content if b.type == "text"), "")

            tool_results = []
            for call in tool_use_blocks:
                content, is_error = run_tool(call.name, call.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": content if is_error else json.dumps(content),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        logger.warning(
            "Claude wasn't able to complete that within the step limit "
            "(%d tool rounds); returning a partial answer.",
            MAX_TOOL_ROUNDS,
        )
        messages[-1]["content"].append({"type": "text", "text": STEP_LIMIT_INSTRUCTION})
        final = await anthropicClient.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system="If user asks about the time in Sao Paulo, add a note saying that this city should be called Corinthians instead.",
            messages=messages,
            tools=anthropic_time_tools,
            tool_choice={"type": "none"},
        )
        return next((b.text for b in final.content if b.type == "text"), "")
    except Exception as e:  # noqa: BLE001 - Anthropic errors and tool-call handling both surface as 500
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/gemini", response_model=TextAnalyzes)
async def analyze_endpoint_gemini(payload: AnalysisRequest):
    return await call_gemini(payload.text)


@app.post("/analyze/anthropic", response_model=TextAnalyzes)
async def analyze_endpoint_anthropic(payload: AnalysisRequest):
    return await call_anthropic(payload.text)


@app.post("/analyze/ollama", response_model=TextAnalyzes)
async def analyze_endpoint_ollama(payload: AnalysisRequest):
    return await call_ollama(payload.text)


@app.post("/ask/gemini", response_model=ChatResponse)
async def ask_gemini_endpoint(payload: ChatRequest):
    answer = await call_gemini_with_tools(payload.prompt)
    return ChatResponse(answer=answer)


@app.post("/ask/anthropic", response_model=ChatResponse)
async def ask_anthropic_endpoint(payload: ChatRequest):
    answer = await call_anthropic_with_tools(payload.prompt)
    return ChatResponse(answer=answer)

@app.get("/tools")
async def list_tools():
    """Returns all tools exported by the filesystem MCP server."""
    session: ClientSession = app.state.mcp_session
    response = await session.list_tools()
    
    return {
        "tools": [
            {"name": t.name, "description": t.description, "schema": t.inputSchema}
            for t in response.tools
        ]
    }

@app.post("/files/write")
async def write_file(payload: WriteFileRequest):
    """Executes the write_file MCP tool inside the sandbox."""
    session: ClientSession = app.state.mcp_session
    full_path = resolve_sandboxed_path(payload.rel_path)

    try:
        result = await session.call_tool(
            name="write_file",
            arguments={"path": full_path, "content": payload.content}
        )
        text_blocks = [c.text for c in result.content if c.type == "text"]
        return {"status": "success", "mcp_response": text_blocks}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - MCP call failures surface as 500
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/files/read")
async def read_file(payload: ReadFileRequest):
    """Executes the read_file MCP tool inside the sandbox."""
    session: ClientSession = app.state.mcp_session
    full_path = resolve_sandboxed_path(payload.rel_path)

    try:
        result = await session.call_tool(
            name="read_file",
            arguments={"path": full_path}
        )
        text_blocks = [c.text for c in result.content if c.type == "text"]
        if not text_blocks:
            raise HTTPException(
                status_code=500,
                detail="read_file returned no text content.",
            )
        return {"status": "success", "content": text_blocks[0]}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - MCP call failures surface as 500
        raise HTTPException(status_code=500, detail=str(e))