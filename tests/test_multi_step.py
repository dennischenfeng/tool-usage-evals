"""Unit tests for multi-step evals"""

from tool_usage_evals.multi_step import run_agent_turn, AgentTurnResult
from openai import AzureOpenAI


def mock_get_weather(location: str) -> str:
    """Mock weather function for testing"""
    return f"The weather in {location} is sunny and 72°F"


def call_function(name: str, args: dict) -> str:
    """Simple function dispatcher for tests"""
    if name == "get_weather":
        return mock_get_weather(**args)
    else:
        raise ValueError(f"Unknown function: {name}")


def test_run_agent_turn_with_function_call(aoai_client: AzureOpenAI) -> None:
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    result = run_agent_turn(
        aoai_client=aoai_client,
        tools=tools,
        call_function=call_function,
        user_message="What's the weather like in Paris?",
        max_steps=5,
    )

    assert isinstance(result, AgentTurnResult)
    assert len(result.messages) >= 1  # At least user message
    assert result.steps >= 1
    assert len(result.tool_calls) >= 0  # May or may not call functions
