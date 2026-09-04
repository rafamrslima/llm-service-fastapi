import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from llm_service_fastapi import api

client = TestClient(api.app)


class _FakeClient:
    """Stand-in for genai.Client() so tests never hit the real Gemini API."""

    def __init__(self, generate_content: AsyncMock):
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_analyze_endpoint_returns_structured_data(monkeypatch):
    fake_json = '{"name": "Rafael", "sentiment": "positive", "topics": ["testing", "fastapi"]}'
    monkeypatch.setattr(
        api, "client", _FakeClient(AsyncMock(return_value=SimpleNamespace(text=fake_json)))
    )

    response = client.post("/analyze", json={"text": "Rafael loves testing FastAPI."})

    assert response.status_code == 200
    assert response.json() == {
        "name": "Rafael",
        "sentiment": "positive",
        "topics": ["testing", "fastapi"],
    }


def test_call_gemini_parses_llm_json_into_text_analyzes(monkeypatch):
    fake_json = (
        '{"name": "Acme Corp", "sentiment": "negative", "topics": ["support", "billing"]}'
    )
    monkeypatch.setattr(
        api, "client", _FakeClient(AsyncMock(return_value=SimpleNamespace(text=fake_json)))
    )

    result = asyncio.run(api.call_gemini("Some angry review about Acme Corp support."))

    assert isinstance(result, api.TextAnalyzes)
    assert result.name == "Acme Corp"
    assert result.sentiment == api.SentimentEnum.NEGATIVE
    assert result.topics == ["support", "billing"]


def test_call_gemini_raises_500_when_llm_output_is_not_valid_json(monkeypatch):
    monkeypatch.setattr(
        api, "client", _FakeClient(AsyncMock(return_value=SimpleNamespace(text="not valid json")))
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.call_gemini("irrelevant prompt"))

    assert exc_info.value.status_code == 500
