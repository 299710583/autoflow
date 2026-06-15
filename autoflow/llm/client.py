from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from autoflow.settings import settings


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    streaming: bool = True

    @classmethod
    def from_settings(cls) -> "LLMConfig":
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required")
        return cls(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            streaming=settings.llm_streaming,
        )


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_settings()
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(messages, max_tokens=max_tokens)

    def complete_messages(self, messages: list[dict[str, str]], max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
            )
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls,
            "finish_reason": response.choices[0].finish_reason,
        }

    def ping(self) -> str:
        return self.complete("Reply with exactly: ok", max_tokens=10).strip()

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        content = self.complete(prompt=prompt, system=system, max_tokens=max_tokens)
        return parse_json_object(content)

    def complete_json_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        content = self.complete_messages(messages=messages, max_tokens=max_tokens)
        return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response does not contain a JSON object")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def create_chat_openai_model(config: LLMConfig | None = None):
    cfg = config or LLMConfig.from_settings()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain_openai is not installed in the active environment") from exc

    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        streaming=cfg.streaming,
    )
