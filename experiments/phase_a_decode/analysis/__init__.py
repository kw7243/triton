"""Validation and descriptive analysis for Phase A result artifacts."""

from .schema import SCHEMA_VERSION, ValidationError, validate_result_directory

__all__ = ["SCHEMA_VERSION", "ValidationError", "validate_result_directory"]
