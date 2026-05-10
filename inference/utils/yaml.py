import yaml


def read_yaml_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)