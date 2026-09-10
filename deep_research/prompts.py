"""
Dynamic Prompt Loader for Deep Research Agent.

This module acts as a router to select specific prompt versions based on the
PROMPT_VERSION setting in src/config.py.
"""

from deep_research.config import PROMPT_VERSION

_active = PROMPT_VERSION.upper()

if _active == "OPEN":
    from deep_research.prompts_open import *
elif _active == "LEGACY":
    from deep_research.prompts_legacy import *
else:
    # Fallback to OPEN if a version is unknown.
    from deep_research.prompts_open import *
