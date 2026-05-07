import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_phrase_config(config_path: str = "config/phrases_50.json") -> Dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_id_maps(config_path: str = "config/phrases_50.json") -> Tuple[Dict[int, str], Dict[str, int], Dict[str, Dict[str, str]]]:
    payload = load_phrase_config(config_path)
    id_to_key: Dict[int, str] = {}
    key_to_id: Dict[str, int] = {}
    display_map: Dict[str, Dict[str, str]] = {}
    for item in payload["phrases"]:
        pid = int(item["id"])
        key = item["key"]
        id_to_key[pid] = key
        key_to_id[key] = pid
        display_map[key] = item["display"]
    return id_to_key, key_to_id, display_map


def class_count(config_path: str = "config/phrases_50.json") -> int:
    return len(load_phrase_config(config_path)["phrases"])


def phrase_keys(config_path: str = "config/phrases_50.json") -> List[str]:
    return [p["key"] for p in load_phrase_config(config_path)["phrases"]]
