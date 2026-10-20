"""
Онлайн-трекинг положения головы и глаз через вебкамеру (MediaPipe Face Landmarker)
с окном настроек и UDP-сервером, раздающим результат.

Запуск:
    python main.py [--cam 0] [--width 640 --height 480] [--fps 30] [--port 5005] [--host 127.0.0.1]

Установка, модель, протокол и сборка exe — см. README.md.
"""
import argparse
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # без спама при переборе несуществующих камер

import gui  # noqa: E402
import settings  # noqa: E402


def main():
    s = settings.load()  # аргументы командной строки важнее сохранённых настроек
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=s["cam"], help="индекс камеры по умолчанию")
    ap.add_argument("--width", type=int, default=s["width"])
    ap.add_argument("--height", type=int, default=s["height"])
    ap.add_argument("--fps", type=int, default=s["fps"])
    ap.add_argument("--port", type=int, default=s["port"])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — принимать подключения из сети")
    args = ap.parse_args()
    gui.run(cam=args.cam, width=args.width, height=args.height, fps=args.fps, port=args.port, host=args.host,
            depth_scale=s["depth_scale"], ipd_mm=s["ipd_mm"], camera_latency_ms=s["camera_latency_ms"],
            exposure_ms=s["exposure_ms"], gain=s["gain"])


if __name__ == "__main__":
    main()
