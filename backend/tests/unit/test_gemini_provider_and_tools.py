from types import SimpleNamespace

import pytest
from app.providers.gemini_provider import GeminiProvider
from app.services.orchestrator import Orchestrator


def test_convert_messages_only_uses_leading_system_messages() -> None:
    provider = GeminiProvider(api_key=None)
    contents, system_instruction = provider._convert_messages(
        [
            {"role": "system", "content": "trusted"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "untrusted"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert system_instruction == "trusted"
    assert [c.role for c in contents] == ["user", "user", "model"]
    assert [c.parts[0].text for c in contents] == ["hello", "untrusted", "answer"]


@pytest.mark.asyncio
async def test_gemini_tool_call_with_google_genai_client() -> None:
    class FakeProvider:
        api_key = "test"

        def __init__(self) -> None:
            self._client = SimpleNamespace(
                models=SimpleNamespace(generate_content=self._generate_content)
            )

        @staticmethod
        def _convert_messages(messages: list[dict[str, str]]) -> tuple[list[str], str]:
            return ["converted"], "system"

        @staticmethod
        def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            return 0.0

        @staticmethod
        async def generate(**kwargs: object) -> None:
            raise AssertionError("Should not fall back to provider.generate")

        @staticmethod
        def _generate_content(**kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == "gemini-2.0-flash"
            assert kwargs["contents"] == ["converted"]
            assert kwargs["config"].tools
            part = SimpleNamespace(
                text="",
                function_call=SimpleNamespace(name="echo", args={"value": "ok"}),
            )
            candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
            usage = SimpleNamespace(prompt_token_count=3, candidates_token_count=5)
            return SimpleNamespace(text="", candidates=[candidate], usage_metadata=usage)

    orchestrator = Orchestrator.__new__(Orchestrator)
    provider = FakeProvider()
    response = await orchestrator._gemini_tool_call(
        provider=provider,
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "name": "echo",
                "description": "Echo value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ],
    )

    assert response.finish_reason == "tool_calls"
    assert response.raw_response["tool_calls"][0]["name"] == "echo"
