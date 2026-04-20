import yaml
from typing import Any


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"expected YAML object at root, got {type(config)}")
    return config