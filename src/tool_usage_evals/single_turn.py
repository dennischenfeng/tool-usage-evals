"""
Evaluate tool selection on a single LLM turn
"""

from openai import AzureOpenAI
import os
from pydantic import BaseModel, Field


class MatchingToolNameResult(BaseModel):
    accuracy: float = Field(description="Accuracy of matching correct tool names")
    n_trials: int


def evaluate_matching_tool_name(
    aoai_client: AzureOpenAI,
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
        input_messages = [{"role": "user", "content": user_message}]
        response = aoai_client.responses.create(
            model=os.environ["AOAI_MODEL"],
            input=input_messages,
            tools=tools,
            tool_choice="auto",
        )

        # TODO deprecate
        success = len(response.tools) > 0 and response.tools[0].name in expected_tool_names
        if success:
            n_successes += 1

    return MatchingToolNameResult(
        accuracy=n_successes / n_trials,
        n_trials=n_trials,
    )
