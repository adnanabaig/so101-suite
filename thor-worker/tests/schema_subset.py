"""Small Draft 2020-12 subset used to test checked-in schema parity without deps."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


class SchemaViolation(AssertionError):
    pass


class SchemaSet:
    def __init__(self, directory: Path) -> None:
        self.documents = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in directory.glob("*.schema.json")
        }

    def validate(self, instance: object, schema_name: str) -> None:
        self._validate(instance, self.documents[schema_name], schema_name, "$")

    def accepts(self, instance: object, schema_name: str) -> bool:
        try:
            self.validate(instance, schema_name)
        except SchemaViolation:
            return False
        return True

    def _resolve(self, reference: str, document_name: str) -> tuple[object, str]:
        file_name, separator, fragment = reference.partition("#")
        target_name = file_name or document_name
        try:
            target: object = self.documents[target_name]
        except KeyError as exc:
            raise SchemaViolation(f"unknown schema reference: {reference}") from exc
        if separator and fragment:
            if not fragment.startswith("/"):
                raise SchemaViolation(f"unsupported schema fragment: {reference}")
            for raw_part in fragment[1:].split("/"):
                part = raw_part.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or part not in target:
                    raise SchemaViolation(f"missing schema fragment: {reference}")
                target = target[part]
        return target, target_name

    def _matches(self, instance: object, schema: object, document_name: str, path: str) -> bool:
        try:
            self._validate(instance, schema, document_name, path)
        except SchemaViolation:
            return False
        return True

    def _validate(
        self, instance: object, schema: object, document_name: str, path: str
    ) -> None:
        if schema is True:
            return
        if schema is False:
            raise SchemaViolation(f"{path}: rejected by false schema")
        if not isinstance(schema, dict):
            raise SchemaViolation(f"{path}: schema must be an object or boolean")
        if "$ref" in schema:
            target, target_name = self._resolve(schema["$ref"], document_name)
            self._validate(instance, target, target_name, path)
        for child in schema.get("allOf", []):
            self._validate(instance, child, document_name, path)
        if "anyOf" in schema and not any(
            self._matches(instance, child, document_name, path) for child in schema["anyOf"]
        ):
            raise SchemaViolation(f"{path}: no anyOf branch matched")
        if "oneOf" in schema:
            matches = sum(
                self._matches(instance, child, document_name, path)
                for child in schema["oneOf"]
            )
            if matches != 1:
                raise SchemaViolation(f"{path}: expected one oneOf match, got {matches}")
        if "if" in schema:
            condition = self._matches(instance, schema["if"], document_name, path)
            branch = schema.get("then" if condition else "else")
            if branch is not None:
                self._validate(instance, branch, document_name, path)

        if "type" in schema and not self._has_type(instance, schema["type"]):
            raise SchemaViolation(f"{path}: wrong type")
        if "const" in schema and not self._json_equal(instance, schema["const"]):
            raise SchemaViolation(f"{path}: value differs from const")
        if "enum" in schema and not any(
            self._json_equal(instance, option) for option in schema["enum"]
        ):
            raise SchemaViolation(f"{path}: value is outside enum")

        if isinstance(instance, str):
            if len(instance) < schema.get("minLength", 0):
                raise SchemaViolation(f"{path}: string is too short")
            if "pattern" in schema and re.search(schema["pattern"], instance) is None:
                raise SchemaViolation(f"{path}: string does not match pattern")
        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if not math.isfinite(float(instance)):
                raise SchemaViolation(f"{path}: number is not finite")
            if "minimum" in schema and instance < schema["minimum"]:
                raise SchemaViolation(f"{path}: number is below minimum")
            if "maximum" in schema and instance > schema["maximum"]:
                raise SchemaViolation(f"{path}: number is above maximum")
            if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
                raise SchemaViolation(f"{path}: number is not above exclusiveMinimum")

        if isinstance(instance, dict):
            required = schema.get("required", [])
            missing = [key for key in required if key not in instance]
            if missing:
                raise SchemaViolation(f"{path}: missing required keys {missing}")
            if len(instance) > schema.get("maxProperties", math.inf):
                raise SchemaViolation(f"{path}: too many properties")
            properties = schema.get("properties", {})
            for key, child in properties.items():
                if key in instance:
                    self._validate(instance[key], child, document_name, f"{path}.{key}")
            extras = set(instance) - set(properties)
            additional = schema.get("additionalProperties", True)
            if additional is False and extras:
                raise SchemaViolation(f"{path}: additional properties {sorted(extras)}")
            if isinstance(additional, dict):
                for key in extras:
                    self._validate(instance[key], additional, document_name, f"{path}.{key}")

        if isinstance(instance, list):
            if len(instance) < schema.get("minItems", 0):
                raise SchemaViolation(f"{path}: too few items")
            if len(instance) > schema.get("maxItems", math.inf):
                raise SchemaViolation(f"{path}: too many items")
            if schema.get("uniqueItems"):
                rendered = [json.dumps(item, sort_keys=True) for item in instance]
                if len(rendered) != len(set(rendered)):
                    raise SchemaViolation(f"{path}: items are not unique")
            prefixes = schema.get("prefixItems", [])
            for index, child in enumerate(prefixes[: len(instance)]):
                self._validate(instance[index], child, document_name, f"{path}[{index}]")
            items = schema.get("items")
            if items is False and len(instance) > len(prefixes):
                raise SchemaViolation(f"{path}: extra tuple items")
            if isinstance(items, dict):
                start = len(prefixes) if prefixes else 0
                for index in range(start, len(instance)):
                    self._validate(instance[index], items, document_name, f"{path}[{index}]")

    @staticmethod
    def _json_equal(left: object, right: object) -> bool:
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            return float(left) == float(right)
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

    @staticmethod
    def _has_type(instance: object, expected: object) -> bool:
        choices = expected if isinstance(expected, list) else [expected]
        for choice in choices:
            if choice == "null" and instance is None:
                return True
            if choice == "boolean" and isinstance(instance, bool):
                return True
            if choice == "object" and isinstance(instance, dict):
                return True
            if choice == "array" and isinstance(instance, list):
                return True
            if choice == "string" and isinstance(instance, str):
                return True
            if choice == "integer" and isinstance(instance, int) and not isinstance(instance, bool):
                return True
            if choice == "number" and isinstance(instance, (int, float)) and not isinstance(
                instance, bool
            ):
                return math.isfinite(float(instance))
        return False
