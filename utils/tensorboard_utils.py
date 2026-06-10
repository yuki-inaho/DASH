from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from beartype import beartype


@beartype
def write_training_metadata(
    writer: Any | None,
    args: Namespace,
    optimizer_names: Mapping[str, str],
) -> None:
    if writer is None:
        return

    writer.add_text("config/args", _to_markdown_json(vars(args)), 0)
    writer.add_text("config/optimizers", _to_markdown_json(dict(optimizer_names)), 0)
    if hasattr(args, "iterations"):
        writer.add_scalar("config/iterations", int(args.iterations), 0)


@beartype
def _to_markdown_json(value: Mapping[str, Any]) -> str:
    return "```json\n" + json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n```"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
