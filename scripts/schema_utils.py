#!/usr/bin/env python3
"""Dependency-free validator for the JSON-Schema subset used by this repository."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def resolve_ref(root: dict, reference: str) -> dict:
    if not reference.startswith("#/"):
        raise ValueError(f"only local JSON Schema references are supported: {reference}")
    current = root
    for token in reference[2:].split("/"):
        current = current[token.replace("~1", "/").replace("~0", "~")]
    return current


def type_matches(value, expected: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }[expected]()


def validate_instance(instance, schema: dict, *, root: dict | None = None, path: str = "$") -> list[str]:
    root = schema if root is None else root
    if not schema:
        return []
    if "$ref" in schema:
        return validate_instance(instance, resolve_ref(root, schema["$ref"]), root=root, path=path)
    errors: list[str] = []
    if "oneOf" in schema:
        matches = [not validate_instance(instance, candidate, root=root, path=path) for candidate in schema["oneOf"]]
        if sum(matches) != 1:
            errors.append(f"{path}: must match exactly one oneOf branch")
    if "anyOf" in schema:
        if not any(not validate_instance(instance, candidate, root=root, path=path) for candidate in schema["anyOf"]):
            errors.append(f"{path}: must match at least one anyOf branch")
    if "allOf" in schema:
        for candidate in schema["allOf"]:
            errors.extend(validate_instance(instance, candidate, root=root, path=path))
    if "not" in schema and not validate_instance(instance, schema["not"], root=root, path=path):
        errors.append(f"{path}: must not match the forbidden schema")
    if "if" in schema:
        condition_matches = not validate_instance(instance, schema["if"], root=root, path=path)
        selected = schema.get("then") if condition_matches else schema.get("else")
        if selected is not None:
            errors.extend(validate_instance(instance, selected, root=root, path=path))
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} is not in enum")
    expected_types = schema.get("type")
    if expected_types:
        expected_types = [expected_types] if isinstance(expected_types, str) else expected_types
        if not any(type_matches(instance, expected) for expected in expected_types):
            errors.append(f"{path}: expected type {expected_types}, got {type(instance).__name__}")
            return errors
    if isinstance(instance, dict):
        for field in schema.get("required", []):
            if field not in instance:
                errors.append(f"{path}: missing required property {field}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_path = f"{path}.{key}"
            if key in properties:
                errors.extend(validate_instance(value, properties[key], root=root, path=child_path))
            elif additional is False:
                errors.append(f"{child_path}: additional property is forbidden")
            elif isinstance(additional, dict):
                errors.extend(validate_instance(value, additional, root=root, path=child_path))
    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: requires at least {schema['minItems']} items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in instance]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items must be unique")
        if "items" in schema:
            for index, value in enumerate(instance):
                errors.extend(validate_instance(value, schema["items"], root=root, path=f"{path}[{index}]"))
        if "contains" in schema and not any(not validate_instance(value, schema["contains"], root=root, path=path) for value in instance):
            errors.append(f"{path}: no item satisfies contains")
    if isinstance(instance, str):
        if len(instance) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: string is shorter than minLength")
        if schema.get("pattern") and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: string does not match pattern")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{path}: invalid date-time")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: value is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: value is above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: value is not above exclusiveMinimum")
    return errors


def validate_json_file(instance_path: Path, schema_path: Path) -> list[str]:
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return validate_instance(instance, schema)
