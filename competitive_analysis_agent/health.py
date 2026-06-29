"""CLI health check for the Stage 0 project shell."""

from __future__ import annotations

import json
import platform
from typing import Any

from competitive_analysis_agent.config import Settings


def build_health_report(settings: Settings | None = None) -> dict[str, Any]:
    current_settings = settings or Settings.from_env()
    return {
        "status": "ok",
        "python_version": platform.python_version(),
        "configuration": {
            "llm": "configured"
            if current_settings.llm_configured
            else "not_configured",
            "search": "configured"
            if current_settings.search_configured
            else "not_configured",
        },
    }


def main() -> None:
    print(json.dumps(build_health_report(), indent=2))


if __name__ == "__main__":
    main()
