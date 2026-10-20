"""Настройки, которые переживают перезапуск: лежат в JSON рядом с exe (или со скриптом)."""
import json
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "headtracker_settings.json")

DEFAULTS = {"cam": 0, "width": 640, "height": 480, "fps": 30, "port": 5005, "depth_scale": 1.0, "ipd_mm": 63.1,
            "camera_latency_ms": 70.0, "exposure_ms": None, "gain": None}


def load() -> dict:
    s = dict(DEFAULTS)
    try:
        with open(PATH, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save(s: dict):
    try:
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except OSError:
        pass  # например, exe лежит в Program Files — просто не сохраняем
