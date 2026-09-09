"""Configurable data quality checks for tabular data."""

from importlib.metadata import version

from qualipilot.checker import DataQualityChecker
from qualipilot.checks.registry import register_check
from qualipilot.models.config import CheckConfig, LLMConfig, QualipilotConfig
from qualipilot.models.results import CheckResult, Metric, QualityReport
from qualipilot.ner import NamedEntity, SpacyEntityRecognizer

__all__ = [
    "CheckConfig",
    "CheckResult",
    "DataQualityChecker",
    "LLMConfig",
    "Metric",
    "NamedEntity",
    "QualipilotConfig",
    "QualityReport",
    "SpacyEntityRecognizer",
    "register_check",
]

__version__ = version("qualipilot")
