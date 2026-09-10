"""Credential lookup for tools and search providers.

Tools historically read API keys from ``os.getenv`` at call time. Under a
per-run runtime (API), keys must come from the run's ``Credentials`` without
mutating ``os.environ``. ``get_secret`` resolves a secret from the active
runtime's credentials first, then falls back to the process environment — so
standalone/CLI behaviour is unchanged.
"""

from __future__ import annotations

import os
from typing import Optional


def get_secret(env_name: str) -> Optional[str]:
    try:
        from deep_research.runtime import get_runtime
        rt = get_runtime()
        value = rt.credentials.get(env_name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(env_name)
