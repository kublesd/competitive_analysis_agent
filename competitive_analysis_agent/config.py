"""Environment-backed application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from typing import Mapping


def _optional_value(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration values read at the application boundary."""

    llm_api_key: str | None = field(repr=False)
    llm_base_url: str | None
    llm_model: str | None
    tavily_api_key: str | None = field(repr=False)

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> "Settings":
        source = environ if values is None else values
        return cls(
            llm_api_key=_optional_value(source, "LLM_API_KEY"),
            llm_base_url=_optional_value(source, "LLM_BASE_URL"),
            llm_model=_optional_value(source, "LLM_MODEL"),
            tavily_api_key=_optional_value(source, "TAVILY_API_KEY"),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @property
    def search_configured(self) -> bool:
        return bool(self.tavily_api_key)
