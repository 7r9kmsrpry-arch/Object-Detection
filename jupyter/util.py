import json
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)