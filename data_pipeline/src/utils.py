import yaml


def load_yaml_config(path: str) -> dict:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        return config