"""Tolerant per-row Pydantic model and type defaults derived from the dataset manifest."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, create_model

# (python type, default value) per manifest type string.
_TYPE_TABLE: dict[str, tuple[type, Any]] = {
    "string":       (str,   ""),
    "large_string": (str,   ""),
    "utf8":         (str,   ""),
    "bool":         (bool,  False),
    "int8":         (int,   0),
    "int16":        (int,   0),
    "int32":        (int,   0),
    "int64":        (int,   0),
    "uint32":       (int,   0),
    "uint64":       (int,   0),
    "float":        (float, 0.0),
    "float32":      (float, 0.0),
    "float64":      (float, 0.0),
    "double":       (float, 0.0),
}


def _python_type_and_default(manifest_type: str) -> tuple[type, Any]:
    t = manifest_type.strip().lower()
    if t in _TYPE_TABLE:
        return _TYPE_TABLE[t]
    if t.startswith("list<") and "double" in t:
        return list[float], []
    raise ValueError(f"Unsupported manifest type: {manifest_type!r}")


def build_input_row_model(
    manifest: dict,
    field_aliases: dict[str, str] | None = None,
) -> type[BaseModel]:
    """Pydantic row model derived from ``manifest['features']``.

    - Unknown keys are dropped (``extra='ignore'``) so upstream schema growth
      does not break the service.
    - Each known field is ``Optional[T]`` with default ``None`` -> missing keys
      and explicit ``null`` are tolerated.
    - Wrong types raise ``ValidationError`` -> HTTP 422 with a clear message.
    - ``field_aliases`` maps ``source_key -> manifest_feature_name``. The model
      will accept either the manifest name or the source alias on input.
    """
    field_aliases = field_aliases or {}
    inv: dict[str, list[str]] = {}
    for src, dst in field_aliases.items():
        inv.setdefault(dst, []).append(src)

    type_by_name = {c["name"]: c["type"] for c in manifest["schema"]}
    fields: dict[str, Any] = {}
    for name in manifest["features"]:
        if name not in type_by_name:
            raise KeyError(f"Feature {name!r} has no entry in manifest['schema']")
        py_t, _ = _python_type_and_default(type_by_name[name])
        if name in inv:
            field = Field(default=None, validation_alias=AliasChoices(name, *inv[name]))
        else:
            field = Field(default=None)
        fields[name] = (py_t | None, field)
    return create_model(
        "InputRow",
        __config__=ConfigDict(extra="ignore", populate_by_name=True),
        **fields,
    )


def manifest_defaults(manifest: dict) -> dict[str, Any]:
    """Map ``feature_name -> default_value`` for filling missing/null inputs."""
    type_by_name = {c["name"]: c["type"] for c in manifest["schema"]}
    return {
        name: _python_type_and_default(type_by_name[name])[1]
        for name in manifest["features"]
    }
