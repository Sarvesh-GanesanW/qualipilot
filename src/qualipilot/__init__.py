"""Configurable data quality checks for tabular data."""

from importlib.metadata import version

from qualipilot.checker import DataQualityChecker
from qualipilot.models.config import CheckConfig, LLMConfig, QualipilotConfig
from qualipilot.models.results import CheckResult, QualityReport

__all__ = [
    "CheckConfig",
    "CheckResult",
    "DataQualityChecker",
    "LLMConfig",
    "QualipilotConfig",
    "QualityReport",
]

__version__ = version("qualipilot")
