import json
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

import ollama
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


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


BUSINESS_HOURS_START = 9
BUSINESS_HOURS_END = 17


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
    time: str = Field(description="Time to convert, in 24-hour HH:MM format.")
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


# Caps how many times a model may chain tool calls before we stop asking it.
MAX_TOOL_ROUNDS = 5


def run_tool(name: str, args: dict):
    _, params_model, fn = TOOL_SPECS[name]
    params = params_model.model_validate(args)
    result = fn(**params.model_dump())
    print(f"[tool call] {name}({params.model_dump_json()}) -> {result}")
    return result.isoformat() if isinstance(result, datetime) else result


load_dotenv()
app = FastAPI()
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
                config=types.GenerateContentConfig(tools=[time_tool]),
            )

            function_calls = response.function_calls
            if not function_calls:
                return response.text

            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": run_tool(call.name, call.args)},
                        )
                        for call in function_calls
                    ],
                )
            )

        raise HTTPException(
            status_code=500,
            detail=f"Gemini still requested tools after {MAX_TOOL_ROUNDS} rounds.",
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - Gemini errors and tool-call handling both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_anthropic_with_tools(prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]

        for _ in range(MAX_TOOL_ROUNDS):
            response = await anthropicClient.messages.create(
                model="claude-opus-5",
                max_tokens=1024,
                messages=messages,
                tools=anthropic_time_tools,
            )

            print(response.usage)
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                return next((b.text for b in response.content if b.type == "text"), "")

            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": json.dumps(run_tool(call.name, call.input)),
                        }
                        for call in tool_use_blocks
                    ],
                }
            )

        raise HTTPException(
            status_code=500,
            detail=f"Claude still requested tools after {MAX_TOOL_ROUNDS} rounds.",
        )
    except HTTPException:
        raise
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
