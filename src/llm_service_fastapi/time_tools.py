"""Timezone helpers shared by the API's tool-calling loops and the time MCP server.

Nothing here may write to stdout: the MCP server runs over stdio, so stdout is its
protocol channel.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import Field

BUSINESS_HOURS_START = 9
BUSINESS_HOURS_END = 17

TIME_OF_DAY_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"


class Timezones(str, Enum):
    SAO_PAULO = "America/Sao_Paulo"
    LONDON = "Europe/London"
    TOKYO = "Asia/Tokyo"


TimeOfDay = Annotated[
    str,
    Field(
        description="Time to convert, in 24-hour HH:MM format.",
        pattern=TIME_OF_DAY_PATTERN,
    ),
]

TOOL_DESCRIPTIONS = {
    "get_current_time": (
        "Returns the current local date and time for a given IANA timezone."
    ),
    "convert_time": (
        "Converts a time from one IANA timezone to another, using today's "
        "date in the source timezone."
    ),
    "is_business_hours": (
        f"Returns whether it is currently business hours "
        f"({BUSINESS_HOURS_START:02d}:00-{BUSINESS_HOURS_END:02d}:00, "
        f"Monday to Friday) in a given IANA timezone."
    ),
}


def get_current_time(tz: Timezones) -> datetime:
    return datetime.now(ZoneInfo(tz.value))


def convert_time(time: str, from_tz: Timezones, to_tz: Timezones) -> datetime:
    source_zone = ZoneInfo(from_tz.value)
    # Anchored to today's date in the source zone so the conversion respects DST.
    today = datetime.now(source_zone).date()
    parsed = datetime.strptime(time, "%H:%M").time()  # noqa: DTZ007 - only the time part is kept; tzinfo is attached below
    source = datetime.combine(today, parsed, tzinfo=source_zone)
    return source.astimezone(ZoneInfo(to_tz.value))


def is_business_hours(tz: Timezones) -> bool:
    now = datetime.now(ZoneInfo(tz.value))
    is_weekday = now.weekday() < 5
    return is_weekday and BUSINESS_HOURS_START <= now.hour < BUSINESS_HOURS_END
