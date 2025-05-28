"""
Utils for handling tools that live inside MCP servers
"""

from typing import AsyncIterator
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
