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


def get_current_time(tz: Timezones):
    time_now = datetime.now(ZoneInfo(tz.value))
    print(time_now)
    return time_now


class GetCurrentTimeParams(BaseModel):
    tz: Timezones = Field(
        description="IANA timezone name to compute the current local time for."
    )


get_current_time_declaration = types.FunctionDeclaration(
    name="get_current_time",
    description=(
        "Returns the current local date and time for a given IANA timezone "
        "(America/Sao_Paulo, Europe/London, or Asia/Tokyo)."
    ),
    parameters_json_schema=GetCurrentTimeParams.model_json_schema(),
)

time_tool = types.Tool(function_declarations=[get_current_time_declaration])

anthropic_time_tool = {
    "name": "get_current_time",
    "description": (
        "Returns the current local date and time for a given IANA timezone "
        "(America/Sao_Paulo, Europe/London, or Asia/Tokyo)."
    ),
    "input_schema": GetCurrentTimeParams.model_json_schema(),
}


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
        response = await geminiClient.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[time_tool]),
        )

        function_calls = response.function_calls
        if not function_calls:
            return response.text

        call = function_calls[0]
        print(f"[tool call] {call.name}(tz={call.args.get('tz')})")
        params = GetCurrentTimeParams.model_validate(call.args)
        result = get_current_time(params.tz)

        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=call.name,
                        response={"current_time": result.isoformat()},
                    )
                ],
            )
        )

        follow_up = await geminiClient.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[time_tool]),
        )
        return follow_up.text
    except Exception as e:  # noqa: BLE001 - Gemini errors and tool-call handling both surface as 500
        raise HTTPException(status_code=500, detail=str(e))


async def call_anthropic_with_tools(prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = await anthropicClient.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            messages=messages,
            tools=[anthropic_time_tool],
        )

        print(response.usage)
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            return next((b.text for b in response.content if b.type == "text"), "")

        call = tool_use_blocks[0]
        print(f"[tool call] {call.name}(tz={call.input.get('tz')})")
        params = GetCurrentTimeParams.model_validate(call.input)
        result = get_current_time(params.tz)

        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result.isoformat(),
                    }
                ],
            }
        )

        follow_up = await anthropicClient.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            messages=messages,
            tools=[anthropic_time_tool],
        )
        return next((b.text for b in follow_up.content if b.type == "text"), "")
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
