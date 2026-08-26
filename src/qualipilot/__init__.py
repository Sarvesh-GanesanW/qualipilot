"""Configurable data quality checks for tabular data."""

from importlib.metadata import version

from qualipilot.checker import DataQualityChecker
from qualipilot.models.config import CheckConfig, LLMConfig, QualipilotConfig
from qualipilot.models.results import CheckResult, QualityReport
from qualipilot.ner import NamedEntity, SpacyEntityRecognizer

__all__ = [
    "CheckConfig",
    "CheckResult",
    "DataQualityChecker",
    "LLMConfig",
    "NamedEntity",
    "QualipilotConfig",
    "QualityReport",
    "SpacyEntityRecognizer",
]

__version__ = version("qualipilot")
