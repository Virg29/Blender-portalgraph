"""
Захват с вебкамеры и трекинг головы/глаз (MediaPipe Face Landmarker) в отдельном потоке.

Система координат (как у MediaPipe): x вправо, y вверх, z на зрителя,
то есть лицо перед камерой имеет z < 0. Единицы — сантиметры.
"""
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_NAME = "face_landmarker.task"

# MediaPipe считает facial_transformation_matrix для перспективной камеры
# с вертикальным FOV 63°. Чтобы обратная проекция глаз совпадала по масштабу
# с позой головы, используем то же фокусное расстояние.
MP_VFOV_DEG = 63.0

# Шире этого кадр для модели уменьшается: MediaPipe всё равно вырезает лицо и ужимает его
# до ~200 px, а кадр 1080p втрое дороже переводить в RGB и готовить.
PROC_MAX_W = 640

# Индексы точек: центры радужек (468 — правый глаз человека, 473 — левый)
IRIS_R, IRIS_L = 468, 473

# Центры глазных яблок в координатах канонической модели лица MediaPipe (см).
# Середина между уголками глаза (вершины 33/133 и 362/263 из
# geometry_pipeline_metadata_landmarks.binarypb), сдвинутая на 1.2 см вглубь головы.
# Привязаны к черепу: в отличие от радужки, не смещаются при переводе взгляда.
EYE_R_CANON = np.array([-3.155, 2.625, 2.27])
EYE_L_CANON = np.array([3.155, 2.625, 2.27])
CANON_IPD_MM = 63.1  # расстояние между ними

if sys.platform == "win32":
    CAP_BACKEND = cv2.CAP_DSHOW
elif sys.platform.startswith("linux"):
    CAP_BACKEND = cv2.CAP_V4L2
else:
    CAP_BACKEND = cv2.CAP_ANY


def resource_path(name: str) -> str:
    """Ищет файл внутри PyInstaller-бандла, рядом с exe или рядом со скриптом."""
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, name))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), name))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Не найден {name}. Скачайте модель (см. README.md) и положите рядом.")


def rotation_to_euler(R: np.ndarray):
    """Матрица поворота -> (yaw, pitch, roll) в градусах.

    yaw — поворот вокруг вертикальной оси Y, pitch — вокруг X, roll — вокруг оси взгляда Z.
    """
    sy = math.hypot(R[0, 0], R[1, 0])
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        roll = 0.0
    return yaw, pitch, roll


def _camera_names() -> dict[int, str]:
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph  # порядок совпадает с индексами DSHOW
            return dict(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            return {}
    names = {}
    if sys.platform.startswith("linux"):
        for i in range(16):
            try:
                with open(f"/sys/class/video4linux/video{i}/name") as f:
                    names[i] = f.read().strip()
            except OSError:
                pass
    return names


def list_cameras(max_index: int = 8, assume_ok: tuple = ()) -> list[tuple[int, str]]:
    """Перебирает индексы и возвращает камеры, с которых реально читается кадр.

    assume_ok — индексы, которые уже открыты (повторно их открыть может не получиться).
    """
    names = _camera_names()
    found = []
    for i in range(max_index):
        if i in assume_ok:
            found.append((i, names.get(i, f"Камера {i}")))
            continue
        cap = cv2.VideoCapture(i, CAP_BACKEND)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append((i, names.get(i, f"Камера {i}")))
        finally:
            cap.release()
    return found


def apply_exposure(cap, exposure_ms: float | None, gain: int | None):
    """Выдержка (None — авто) и усиление (None — не трогать).

    Короткая выдержка — меньше смазки и на полвыдержки меньше задержки, но темнее:
    компенсируется усилением или светом.
    """
    if CAP_BACKEND == cv2.CAP_DSHOW:
        # DirectShow: 0.75 — авто, 0.25 — ручная; выдержка задаётся как log2 секунд
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if exposure_ms is None else 0.25)
        if exposure_ms is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, round(math.log2(exposure_ms / 1000)))
    else:
        # V4L2: 3 — авто, 1 — ручная; выдержка в единицах по 100 мкс
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if exposure_ms is None else 1)
        if exposure_ms is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, round(exposure_ms * 10))
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, gain)


class FrameGrabber:
    """Читает камеру в своём потоке и держит только самый свежий кадр.

    Драйвер копит очередь кадров (у V4L2 обычно 4). Если читать по одному в темпе обработки,
    после любой заминки очередь уже не рассасывается и каждый кадр приходит на ~100 мс устаревшим.
    """

    def __init__(self, cap, on_frame=None):
        self.on_frame = on_frame  # (время, кадр) на каждый прочитанный кадр, из потока захвата
        # CAP_PROP_BUFFERSIZE=1 не ставим: V4L2 с одним буфером теряет каждый второй кадр
        self.cap = cap
        self.failed = False
        self._pending_controls = None  # применяются в потоке захвата: VideoCapture не потокобезопасен
        self._cond = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fails = 0
        while not self._stop:
            controls, self._pending_controls = self._pending_controls, None
            if controls is not None:
                controls(self.cap)
            ok, frame = self.cap.read()
            t = time.time()
            with self._cond:
                if ok:
                    fails = 0
                    self.failed = False
                    self._frame, self._t = frame, t
                    if self.on_frame:
                        self.on_frame(t, frame)
                    self._seq += 1
                else:
                    fails += 1
                    self.failed = fails > 5
                self._cond.notify_all()
            if not ok:
                time.sleep(0.05)

    def request_controls(self, fn):
        """fn(cap) выполнится в потоке захвата перед следующим чтением."""
        self._pending_controls = fn

    def latest(self, last_seq: int, timeout: float = 0.5):
        """Ждёт кадр новее last_seq. Возвращает (seq, кадр, время захвата) или None."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self._stop, timeout)
            if self._seq == last_seq:
                return None
            return self._seq, self._frame, self._t

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)
        self.cap.release()


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int


@dataclass
class Snapshot:
    frame: np.ndarray | None  # BGR-кадр с отрисованным трекингом
    data: dict | None  # последний результат трекинга (то же, что уходит клиентам)
    status: str
    fps: float  # реальная частота обработки
    actual: tuple[int, int, float] | None  # что камера выдала на самом деле: w, h, fps


class Tracker:
    def __init__(self, on_result=None):
        """on_result(dict) вызывается из рабочего потока на каждом обработанном кадре."""
        self.on_result = on_result
        self._lock = threading.Lock()
        self._pending: CameraConfig | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snap = Snapshot(None, None, "Камера не выбрана", 0.0, None)
        # MediaPipe оценивает глубину для камеры с VFOV 63°, у реальной вебки угол другой,
        # поэтому z (и только z) домножается на калибровочный коэффициент
        self.depth_scale = 1.0
        # Модель лица — «средняя голова». Если у человека другое межзрачковое расстояние,
        # вся поза (и расстояние до камеры) пропорционально врёт — это масштаб всех осей.
        self.ipd_mm = CANON_IPD_MM
        # Задержка камеры и экрана: от события перед камерой до кадра в программе. Её не видно
        # изнутри (кадр получает метку, только когда OpenCV его отдал), поэтому она меряется
        # отдельно миганием экрана и прибавляется к latency_ms, чтобы клиенты могли её компенсировать.
        self.camera_latency_ms = 50.0
        self.frame_probe = None  # (время, средняя яркость) — для замера задержки
        self.exposure_ms: float | None = None  # None — автоэкспозиция
        self.gain: int | None = None  # None — не трогать
        self._grabber: FrameGrabber | None = None
        self._raw_depth: tuple[float, float] | None = None  # (z головы от MediaPipe, z-смещение глаз)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def configure(self, cfg: CameraConfig):
        """Переоткрыть камеру с новыми параметрами (применяется в рабочем потоке)."""
        with self._lock:
            self._pending = cfg

    def set_exposure(self, exposure_ms: float | None, gain: int | None):
        """Меняет выдержку и усиление на лету, без переоткрытия камеры."""
        self.exposure_ms, self.gain = exposure_ms, gain
        grabber = self._grabber
        if grabber is not None:
            grabber.request_controls(lambda cap: apply_exposure(cap, exposure_ms, gain))

    def _on_raw_frame(self, t, frame):
        probe = self.frame_probe
        if probe is not None:
            probe(t, float(frame[::8, ::8].mean()))

    def calibrate_depth(self, distance_cm: float) -> bool:
        """Подбирает depth_scale так, чтобы глаза оказались на distance_cm от камеры."""
        raw = self._raw_depth
        if raw is None or distance_cm <= 0:
            return False
        head_z, eye_dz = raw
        self.depth_scale = (-distance_cm - eye_dz) / head_z
        return True

    def snapshot(self) -> Snapshot:
        with self._lock:
            return replace(self._snap)

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self._snap, k, v)

    def _open(self, cfg: CameraConfig):
        cap = cv2.VideoCapture(cfg.index, CAP_BACKEND)
        if not cap.isOpened():
            return None
        # До 640x480 несжатый YUYV обычно идёт на полной частоте и на ~5 мс быстрее: камере не надо
        # сжимать кадр, а нам — распаковывать. Выше — MJPG, иначе по USB 2.0 не пролезает 30 fps.
        small = cfg.width * cfg.height <= 640 * 480
        fourcc = "YUYV" if small and CAP_BACKEND == cv2.CAP_V4L2 else "MJPG"
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        # Настройки камеры живут в драйвере между запусками: если другая программа оставила
        # ручную длинную выдержку, кадры смазаны и запаздывают. Всегда выставляем свою.
        apply_exposure(cap, self.exposure_ms, self.gain)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        return cap

    def _run(self):
        try:
            model = resource_path(MODEL_NAME)
        except FileNotFoundError as e:
            self._set(status=str(e))
            return
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,  # 6DoF поза головы
            output_face_blendshapes=False,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        grabber = None
        seq = 0
        t0 = time.time()
        last_ts = -1
        n, t_fps, fps = 0, time.monotonic(), 0.0
        try:
            while not self._stop.is_set():
                with self._lock:
                    cfg, self._pending = self._pending, None
                if cfg is not None:
                    if grabber is not None:
                        self._grabber = None
                        grabber.close()
                        grabber = None
                    self._set(status=f"Открываю камеру {cfg.index}…", actual=None)
                    cap = self._open(cfg)
                    if cap is None:
                        self._set(status=f"Не удалось открыть камеру {cfg.index}", frame=None, data=None)
                        continue
                    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              cap.get(cv2.CAP_PROP_FPS))
                    grabber, seq = FrameGrabber(cap, on_frame=self._on_raw_frame), 0
                    self._grabber = grabber
                    self._set(status="Работает", actual=actual)
                    n, t_fps = 0, time.monotonic()
                if grabber is None:
                    time.sleep(0.1)
                    continue

                got = grabber.latest(seq)
                if got is None:
                    if grabber.failed:
                        self._set(status="Камера не отдаёт кадры")
                    continue
                seq, frame, t_cap = got

                # detect_for_video требует строго возрастающих меток времени
                ts = max(int((t_cap - t0) * 1000), last_ts + 1)
                last_ts = ts
                full_w = frame.shape[1]
                if full_w > PROC_MAX_W:
                    frame = cv2.resize(frame, (PROC_MAX_W, round(frame.shape[0] * PROC_MAX_W / full_w)),
                                       interpolation=cv2.INTER_AREA)
                data = self._process(landmarker, frame, ts, px_scale=full_w / frame.shape[1])
                if not data["face"]:
                    self._raw_depth = None

                n += 1
                now = time.monotonic()
                if now - t_fps >= 0.5:
                    fps = n / (now - t_fps)
                    n, t_fps = 0, now
                data["fps"] = round(fps, 1)
                # t — оценка момента, когда кадр реально снят; latency_ms — его возраст к отправке
                cam_lat = self.camera_latency_ms / 1000
                data["t"] = t_cap - cam_lat
                data["latency_ms"] = round((time.time() - t_cap + cam_lat) * 1000, 1)
                data["camera_latency_ms"] = round(self.camera_latency_ms, 1)

                self._set(frame=frame, data=data, fps=fps, status="Работает")
                if self.on_result:
                    self.on_result(data)
        finally:
            if grabber is not None:
                grabber.close()
            landmarker.close()

    def _process(self, landmarker, frame: np.ndarray, ts_ms: int, px_scale: float = 1.0) -> dict:
        """Трекинг одного кадра; рисует разметку прямо в frame и возвращает данные.

        px_scale — во сколько раз исходный кадр камеры больше переданного: пиксельные
        координаты в данных отдаются в разрешении камеры.
        """
        h, w = frame.shape[:2]
        focal = (h / 2.0) / math.tan(math.radians(MP_VFOV_DEG) / 2.0)
        cx, cy = w / 2.0, h / 2.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        data = {"t": time.time(), "face": False, "frame_size": [round(w * px_scale), round(h * px_scale)]}
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return data
        lm = res.face_landmarks[0]

        # --- 2D-положение глаз в пикселях ---
        eyes_px = {}
        for name, idx in (("right", IRIS_R), ("left", IRIS_L)):
            px, py = lm[idx].x * w, lm[idx].y * h
            eyes_px[name] = (px, py)
            cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)
            for k in range(1, 5):  # контур радужки (4 точки вокруг центра)
                cv2.circle(frame, (int(lm[idx + k].x * w), int(lm[idx + k].y * h)), 1, (0, 200, 255), -1)

        # --- 6DoF поза головы ---
        M = np.array(res.facial_transformation_matrixes[0])
        R = M[:3, :3]
        face = self.ipd_mm / CANON_IPD_MM
        t_raw = M[:3, 3] * face
        yaw, pitch, roll = rotation_to_euler(R)
        # Поперечные x/y от фокусного не зависят (оно сокращается), а z ему пропорционален
        dscale = np.array([1.0, 1.0, self.depth_scale])
        t = t_raw * dscale
        M[:3, 3] = t

        def project(p):
            p = p / dscale  # обратно в «пиксельную» глубину MediaPipe (масштаб лица на проекцию не влияет)
            d = max(-p[2], 1e-3)
            return int(cx + focal * p[0] / d), int(cy - focal * p[1] / d)

        origin = project(t)
        for axis, color in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):  # X красная, Y зелёная, Z синяя
            cv2.line(frame, origin, project(t + R[:, axis] * 6.0), color, 2, cv2.LINE_AA)

        # --- 3D-координаты глаз: центры глазных яблок через позу головы ---
        eyes_cm = {"right": t + R @ (EYE_R_CANON * face), "left": t + R @ (EYE_L_CANON * face)}
        mid = (eyes_cm["right"] + eyes_cm["left"]) / 2
        self._raw_depth = (float(t_raw[2]), float(mid[2] - t[2]))
        for p in eyes_cm.values():
            cv2.circle(frame, project(p), 4, (255, 0, 255), 1, cv2.LINE_AA)

        def r2(v):
            return [round(float(c), 2) for c in v]

        eyes = {name: {"px": [round(px * px_scale, 1), round(py * px_scale, 1)], "cm": r2(eyes_cm[name])}
                for name, (px, py) in eyes_px.items()}

        data.update(
            face=True,
            head={
                "pos_cm": [round(float(v), 2) for v in t],
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
                "matrix": [[round(float(v), 5) for v in row] for row in M],
            },
            eyes=eyes,
            eye_mid_cm=r2(mid),
            depth_scale=round(self.depth_scale, 4),
        )
        return data
