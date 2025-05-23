"""
Evaluate tool selection on a single LLM turn
"""

from openai.types.chat import ChatCompletionMessageToolCall
import os
from typing import Any
from pydantic import BaseModel, Field


class MatchingToolNameResult(BaseModel):
    accuracy: float = Field(description="Accuracy of matching correct tool names")
    n_trials: int


def evaluate_matching_tool_name(
    aoai_client: Any,
    tools: list[dict],
    user_message: str,
    expected_tool_names: list[str],
    n_trials: int = 1,
) -> MatchingToolNameResult:
    """
    Given model + tools and initial user message, take a single LLM turn, checking if the response is a tool call and
    matches the expected tool name(s).
    Will do n trials to gather statistics.
    """
    n_successes = 0
    for i in range(n_trials):
        messages = [dict(role="user", content=user_message)]
        completion = aoai_client.chat.completions.create(
            model=os.environ["AOAI_MODEL"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = completion.choices[0].message

        actual_tool_calls = response_message.tool_calls

        success = (
            len(actual_tool_calls) > 0
            and isinstance(actual_tool_calls[0], ChatCompletionMessageToolCall)
            and actual_tool_calls[0].function.name in expected_tool_names
        )
        if success:
            n_successes += 1

    return MatchingToolNameResult(
        accuracy=n_successes / n_trials,
        n_trials=n_trials,
    )
