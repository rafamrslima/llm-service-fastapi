"""MCP server exposing the time tools over stdio.

The API spawns this as a child process (see `lifespan` in api.py). You can also run it
on its own: `python -m llm_service_fastapi.time_mcp_server`. Stdout is the protocol
channel, so never print() here; log to stderr instead.
"""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from llm_service_fastapi import time_tools
from llm_service_fastapi.time_tools import TOOL_DESCRIPTIONS, TimeOfDay, Timezones

mcp = FastMCP("time-tools")


@mcp.tool(description=TOOL_DESCRIPTIONS["get_current_time"])
def get_current_time(
    tz: Annotated[
        Timezones,
        Field(description="IANA timezone name to compute the current local time for."),
    ],
) -> str:
    return time_tools.get_current_time(tz).isoformat()


@mcp.tool(description=TOOL_DESCRIPTIONS["convert_time"])
def convert_time(
    time: TimeOfDay,
    from_tz: Annotated[Timezones, Field(description="IANA timezone the time is given in.")],
    to_tz: Annotated[Timezones, Field(description="IANA timezone to convert the time into.")],
) -> str:
    return time_tools.convert_time(time, from_tz, to_tz).isoformat()


@mcp.tool(description=TOOL_DESCRIPTIONS["is_business_hours"])
def is_business_hours(
    tz: Annotated[Timezones, Field(description="IANA timezone to check business hours in.")],
) -> bool:
    return time_tools.is_business_hours(tz)


if __name__ == "__main__":
    mcp.run(show_banner=False)
