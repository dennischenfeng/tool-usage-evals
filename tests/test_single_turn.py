"""Unit tests for single turn evals"""

from tool_usage_evals.single_turn import evaluate_matching_tool_name
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import pytest
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def aoai_client() -> AzureOpenAI:
    """Azure OpenAI client"""
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.environ["AOAI_ENDPOINT"],
        api_version=os.environ["AOAI_API_VERSION"],
    )
    return client


def test_evaluate_matching_tool_name(aoai_client: AzureOpenAI) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    result = evaluate_matching_tool_name(
        aoai_client=aoai_client,
        tools=tools,
        user_message="What is the current time in New York?",
        expected_tool_names=["get_current_time"],
        n_trials=3,
    )
    assert result.accuracy == 1.0
    assert result.n_trials == 3
