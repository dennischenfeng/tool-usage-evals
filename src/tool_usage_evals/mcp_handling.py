"""
Utils for handling tools that live inside MCP servers
"""

from typing import AsyncIterator, Awaitable, Callable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def mcp_session_generator(command: str, args: list[str]) -> AsyncIterator[ClientSession]:
    """
    Yield the session, within the mcp stdio session context.
    Example arguments:
        command="python"
        args=["./mcp.py"]
    """
    server_params = StdioServerParameters(
        command=command,
        args=args,
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            yield session


async def extract_tool_definitions(session: ClientSession) -> list[dict]:
    """Extracts the tool definitions object (ingestable by openai chat completion)  from the MCP client session"""
    mcp_tools = (await session.list_tools()).tools

    openai_tools = [
        {
            "type": "function",
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
            "strict": True,
        }
        for mcp_tool in mcp_tools
    ]
    return openai_tools


async def build_mcp_tool_caller(session: ClientSession) -> Callable[..., Awaitable[str]]:
    """Returns a call_tool function, which will call the tool functions from the specified mcp session"""

    async def call_mcp_tool_fn(name: str, args: dict) -> str:
        response = await session.call_tool(name=name, arguments=args)
        return response

    return call_mcp_tool_fn
