"""
Evaluate tool selection on a single LLM turn
"""

from openai import AzureOpenAI
import os
from pydantic import BaseModel, Field


class MatchingToolNameResult(BaseModel):
    accuracy: float = Field(description="Accuracy of matching correct tool names")
    n_trials: int


def evaluate_tool_name_was_selected(
    aoai_client: AzureOpenAI,
    tools: list[dict],
    user_message: str,
    expected_tool_names: str,
    n_trials: int = 1,
) -> MatchingToolNameResult:
    """
    Given model + tools and initial user message, take a single LLM turn, checking if the selected tool calls contains
    the expected tool name or not.
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

        selected_tool_names = [t.name for t in response.tools]
        success = expected_tool_names in selected_tool_names
        if success:
            n_successes += 1

    return MatchingToolNameResult(
        accuracy=n_successes / n_trials,
        n_trials=n_trials,
    )
