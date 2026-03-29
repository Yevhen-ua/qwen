from __future__ import annotations

from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator


EXISTS_OUTPUT_VALIDATOR = Draft202012Validator(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "comment"],
        "properties": {
            "status": {"enum": ["found", "not_found", "ambiguous"]},
            "comment": {"type": "string"},
        },
    }
)


def _coord_schema(coord_max: int) -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "integer",
                "minimum": 0,
                "maximum": coord_max,
            },
            {"type": "null"},
        ]
    }


@lru_cache(maxsize=None)
def _point_output_validator(coord_max: int) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "x", "y", "comment"],
            "properties": {
                "status": {"enum": ["found", "not_found", "ambiguous"]},
                "x": _coord_schema(coord_max),
                "y": _coord_schema(coord_max),
                "comment": {"type": "string"},
            },
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "found"}}},
                    "then": {
                        "properties": {
                            "x": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                            "y": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                        }
                    },
                    "else": {
                        "properties": {
                            "x": {"type": "null"},
                            "y": {"type": "null"},
                        }
                    },
                }
            ],
        }
    )


@lru_cache(maxsize=None)
def _drag_output_validator(coord_max: int) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "x", "y", "x2", "y2", "comment"],
            "properties": {
                "status": {"enum": ["found", "not_found", "ambiguous"]},
                "x": _coord_schema(coord_max),
                "y": _coord_schema(coord_max),
                "x2": _coord_schema(coord_max),
                "y2": _coord_schema(coord_max),
                "comment": {"type": "string"},
            },
            "allOf": [
                {
                    "if": {"properties": {"status": {"const": "found"}}},
                    "then": {
                        "properties": {
                            "x": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                            "y": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                            "x2": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                            "y2": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": coord_max,
                            },
                        }
                    },
                    "else": {
                        "properties": {
                            "x": {"type": "null"},
                            "y": {"type": "null"},
                            "x2": {"type": "null"},
                            "y2": {"type": "null"},
                        }
                    },
                }
            ],
        }
    )


def _format_error_path(error_path: Any) -> str:
    parts = [str(part) for part in error_path]
    return ".".join(parts) if parts else "$"


def _validate_with_schema(
    validator: Draft202012Validator,
    data: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    errors = sorted(validator.iter_errors(data), key=lambda err: list(err.absolute_path))
    if not errors:
        return data

    first_error = errors[0]
    error_path = _format_error_path(first_error.absolute_path)
    raise ValueError(f"{label}: {error_path}: {first_error.message}")


def validate_exists_output(data: dict[str, Any]) -> dict[str, Any]:
    return _validate_with_schema(EXISTS_OUTPUT_VALIDATOR, data, "invalid exists output")


def validate_point_output(data: dict[str, Any], coord_max: int) -> dict[str, Any]:
    return _validate_with_schema(
        _point_output_validator(coord_max),
        data,
        "invalid point output",
    )


def validate_drag_output(data: dict[str, Any], coord_max: int) -> dict[str, Any]:
    return _validate_with_schema(
        _drag_output_validator(coord_max),
        data,
        "invalid drag output",
    )
