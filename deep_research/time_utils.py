"""Lightweight date/time helpers.

Kept separate from utils.py (which pulls in the heavy search-tool, prompt, and
model stack) so the platform agent engine (agents/base.py) can import the date
helper without dragging in the production pipeline's imports. This keeps
sub-agents runnable independently of the supervisor/report-writing code.
"""

from datetime import datetime


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")
