"""Окно: превью с камеры, данные трекинга, настройки камеры и запуск сервера."""
import math
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

import settings
from server import BroadcastServer
from tracker import CANON_IPD_MM, MP_VFOV_DEG, CameraConfig, Tracker, list_cameras

PREVIEW_W = 480
RESOLUTIONS = ["320x240", "640x480", "800x600", "1280x720", "1920x1080"]
FPS_OPTIONS = ["15", "24", "30", "60"]
AUTO = "авто"
KEEP = "как есть"
EXPOSURE_OPTIONS = [AUTO, "16", "8", "4", "2", "1"]
GAIN_OPTIONS = [KEEP, "64", "128", "192", "255"]
REFRESH_MS = 33


class App:
    def __init__(self, root: tk.Tk, cam: int, width: int, height: int, fps: int, port: int, host: str,
                 depth_scale: float = 1.0, ipd_mm: float = 63.1, camera_latency_ms: float = 50.0,
                 exposure_ms: float | None = None, gain: int | None = None):
        self.root = root
        self.host = host
        self.server = BroadcastServer()
        self.tracker = Tracker(on_result=self.server.broadcast)
        self.tracker.depth_scale = depth_scale
        self.tracker.ipd_mm = ipd_mm
        self.tracker.camera_latency_ms = camera_latency_ms
        self.tracker.exposure_ms, self.tracker.gain = exposure_ms, gain
        self.cameras: list[tuple[int, str]] = []
        self.current_cam = cam
        self._found_cams = None  # результат поиска из фонового потока: (cams, first)
        self._photo = None

        root.title("Head tracking server")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.close)

        # --- превью ---
        self.preview = ttk.Label(root, text="Нет изображения", anchor="center", width=60)
        self.preview.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        side = ttk.Frame(root)
        side.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="n")

        # --- камера ---
        cam_box = ttk.LabelFrame(side, text="Камера", padding=6)
        cam_box.grid(row=0, column=0, sticky="ew")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(cam_box, textvariable=self.cam_var, state="readonly", width=30)
        self.cam_combo.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.cam_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_camera())
        self.refresh_btn = ttk.Button(cam_box, text="Обновить", command=self.refresh_cameras)
        self.refresh_btn.grid(row=0, column=3, padx=(4, 0))

        ttk.Label(cam_box, text="Разрешение").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.res_var = tk.StringVar(value=f"{width}x{height}")
        res_combo = ttk.Combobox(cam_box, textvariable=self.res_var, values=RESOLUTIONS, width=10)
        res_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))
        res_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_camera())
        res_combo.bind("<Return>", lambda e: self.apply_camera())

        ttk.Label(cam_box, text="FPS").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.fps_var = tk.StringVar(value=str(fps))
        fps_combo = ttk.Combobox(cam_box, textvariable=self.fps_var, values=FPS_OPTIONS, width=10)
        fps_combo.grid(row=2, column=1, sticky="w", pady=(4, 0))
        fps_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_camera())
        fps_combo.bind("<Return>", lambda e: self.apply_camera())

        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cam_box, text="Превью", variable=self.preview_var).grid(row=2, column=2, columnspan=2, sticky="w")
        ttk.Label(cam_box, text="Выдержка, мс").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.exp_var = tk.StringVar(value=AUTO if exposure_ms is None else f"{exposure_ms:g}")
        exp_combo = ttk.Combobox(cam_box, textvariable=self.exp_var, values=EXPOSURE_OPTIONS, width=10)
        exp_combo.grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Label(cam_box, text="Усиление").grid(row=3, column=2, sticky="e", pady=(4, 0))
        self.gain_var = tk.StringVar(value=KEEP if gain is None else str(gain))
        gain_combo = ttk.Combobox(cam_box, textvariable=self.gain_var, values=GAIN_OPTIONS, width=8)
        gain_combo.grid(row=3, column=3, sticky="w", padx=(4, 0), pady=(4, 0))
        for combo in (exp_combo, gain_combo):
            combo.bind("<<ComboboxSelected>>", lambda e: self.apply_exposure())
            combo.bind("<Return>", lambda e: self.apply_exposure())
        ttk.Label(cam_box, text="640x480 оптимально: выше точность не растёт, а CPU — да.\n"
                                "Короче выдержка — меньше смазки и задержки, но нужен свет или усиление.",
                  foreground="gray").grid(row=5, column=0, columnspan=4, sticky="w")
        self.cam_status = ttk.Label(cam_box, text="Поиск камер…", foreground="gray")
        self.cam_status.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # --- данные трекинга ---
        data_box = ttk.LabelFrame(side, text="Трекинг", padding=6)
        data_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.data_label = ttk.Label(data_box, text="", font="TkFixedFont", justify="left")
        self.data_label.grid(row=0, column=0, sticky="w")

        # --- калибровка ---
        cal_box = ttk.LabelFrame(side, text="Калибровка", padding=6)
        cal_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.cal_summary = ttk.Label(cal_box, text="")
        self.cal_summary.grid(row=0, column=0, sticky="w")
        ttk.Button(cal_box, text="Калибровка…", command=self.open_calibration).grid(row=0, column=1, padx=(8, 0))
        cal_box.columnconfigure(0, weight=1)
        self._cal_dialog = None

        # --- сервер ---
        srv_box = ttk.LabelFrame(side, text="Сервер (UDP, JSON)", padding=6)
        srv_box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(srv_box, text=f"{host} : ").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value=str(port))
        self.port_entry = ttk.Entry(srv_box, textvariable=self.port_var, width=7)
        self.port_entry.grid(row=0, column=1, sticky="w")
        self.srv_btn = ttk.Button(srv_box, text="Старт", command=self.toggle_server)
        self.srv_btn.grid(row=0, column=2, padx=(8, 0))
        self.srv_status = ttk.Label(srv_box, text="Остановлен", foreground="gray")
        self.srv_status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.tracker.start()
        self.refresh_cameras()
        self.root.after(REFRESH_MS, self.tick)

    # ---------- камера ----------

    def refresh_cameras(self):
        self.refresh_btn.state(["disabled"])
        self.cam_status.configure(text="Поиск камер…", foreground="gray")
        opened = (self.current_cam,) if self.cameras else ()

        def work():
            # tkinter нельзя трогать из другого потока — результат заберёт tick()
            self._found_cams = (list_cameras(assume_ok=opened), not opened)

        threading.Thread(target=work, daemon=True).start()

    def _on_cameras(self, cams, first: bool):
        self.refresh_btn.state(["!disabled"])
        self.cameras = cams
        self.cam_combo.configure(values=[f"{i}: {name}" for i, name in cams])
        if not cams:
            self.cam_status.configure(text="Камеры не найдены", foreground="red")
            return
        indices = [i for i, _ in cams]
        if self.current_cam not in indices:
            self.current_cam = indices[0]
        self.cam_combo.current(indices.index(self.current_cam))
        if first:
            self.apply_camera()

    def apply_camera(self):
        if not self.cameras:
            return
        try:
            w, h = (int(v) for v in self.res_var.get().lower().split("x"))
            fps = int(self.fps_var.get())
        except ValueError:
            self.cam_status.configure(text="Неверное разрешение или FPS", foreground="red")
            return
        self.current_cam = self.cameras[self.cam_combo.current()][0]
        self.tracker.configure(CameraConfig(self.current_cam, w, h, fps))

    def apply_exposure(self):
        try:
            exp = self.exp_var.get().strip()
            exposure_ms = None if exp in ("", AUTO) else float(exp.replace(",", "."))
            gain_text = self.gain_var.get().strip()
            gain = None if gain_text in ("", KEEP) else int(gain_text)
            if exposure_ms is not None and not 0.1 <= exposure_ms <= 1000:
                raise ValueError
        except ValueError:
            self.cam_status.configure(text="Выдержка — «авто» или мс, усиление — число", foreground="red")
            return
        self.tracker.set_exposure(exposure_ms, gain)

    # ---------- калибровка ----------

    def open_calibration(self):
        if self._cal_dialog is not None and self._cal_dialog.winfo_exists():
            self._cal_dialog.lift()
            return
        self._cal_dialog = CalibrationDialog(self)

    # ---------- сервер ----------

    def toggle_server(self):
        if self.server.running:
            self.server.stop()
            self.srv_btn.configure(text="Старт")
            self.port_entry.state(["!disabled"])
            self.srv_status.configure(text="Остановлен", foreground="gray")
            return
        try:
            port = int(self.port_var.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self.srv_status.configure(text="Порт должен быть числом 1–65535", foreground="red")
            return
        try:
            self.server.start(self.host, port)
        except OSError as e:
            self.srv_status.configure(text=f"Ошибка: {e.strerror or e}", foreground="red")
            return
        self.srv_btn.configure(text="Стоп")
        self.port_entry.state(["disabled"])

    # ---------- обновление окна ----------

    def tick(self):
        if self._found_cams is not None:
            found, self._found_cams = self._found_cams, None
            self._on_cameras(*found)

        snap = self.tracker.snapshot()

        if snap.frame is not None and not self.preview_var.get():
            self._photo = None
            self.preview.configure(image="", text="Превью выключено")
        elif snap.frame is not None:
            h, w = snap.frame.shape[:2]
            ph = round(h * PREVIEW_W / w)
            small = cv2.resize(snap.frame, (PREVIEW_W, ph), interpolation=cv2.INTER_AREA)
            self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.preview.configure(image=self._photo, text="")
        else:
            self._photo = None
            self.preview.configure(image="", text=snap.status)

        if snap.actual:
            aw, ah, afps = snap.actual
            ok = snap.status == "Работает"
            d = snap.data
            # обработку показываем отдельно: задержка камеры — константа из калибровки
            lat_text = (f"\nзадержка: обработка {d['latency_ms'] - d.get('camera_latency_ms', 0):.0f} мс"
                        f" + камера {d.get('camera_latency_ms', 0):.0f} мс (из калибровки)") if d and "latency_ms" in d else ""
            self.cam_status.configure(
                text=f"{snap.status}: {aw}x{ah}, камера {afps:.0f} fps, обработано {snap.fps:.1f} кадр/с{lat_text}",
                foreground="green" if ok else "red")
        elif self.cameras:
            self.cam_status.configure(text=snap.status, foreground="gray")

        self.data_label.configure(text=format_data(snap.data))

        self.cal_summary.configure(
            text=f"Межзрачковое {self.tracker.ipd_mm:g} мм, глубина ×{self.tracker.depth_scale:.3f}")

        if self.server.running:
            self.srv_status.configure(text=f"Работает, клиентов: {self.server.client_count}", foreground="green")

        self.root.after(REFRESH_MS, self.tick)

    def close(self):
        # калибровка сохраняется всегда; поля, где введена ерунда, остаются как были
        s = settings.load()
        s.update(cam=self.current_cam, depth_scale=self.tracker.depth_scale, ipd_mm=self.tracker.ipd_mm,
                 camera_latency_ms=self.tracker.camera_latency_ms,
                 exposure_ms=self.tracker.exposure_ms, gain=self.tracker.gain)
        try:
            s["width"], s["height"] = (int(v) for v in self.res_var.get().lower().split("x"))
        except ValueError:
            pass
        for key, var in (("fps", self.fps_var), ("port", self.port_var)):
            try:
                s[key] = int(var.get())
            except ValueError:
                pass
        settings.save(s)
        self.server.stop()
        self.tracker.stop()
        self.root.destroy()


class LatencyTest(tk.Toplevel):
    """Мигает экраном чёрным/белым и по яркости кадров камеры меряет задержку «экран → программа»."""

    FLIPS = 14
    PERIOD_MS = 700

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self.tracker = parent.tracker
        self.on_done = on_done
        self.samples: list[tuple[float, float]] = []
        self.flips: list[float] = []
        self.configure(bg="black", cursor="none")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.attributes("-topmost", True)
        self.attributes("-fullscreen", True)
        self.lift()
        self.focus_force()
        self.bind("<Escape>", lambda e: self._finish(cancel=True))
        self.tracker.frame_probe = lambda t, b: self.samples.append((t, b))
        self._i = 0
        self.after(1500, self._step)  # дать автоэкспозиции устояться на чёрном

    def _step(self):
        if self._i >= self.FLIPS:
            self._finish()
            return
        self.configure(bg="white" if self._i % 2 == 0 else "black")
        self.update_idletasks()
        self.update()
        self.flips.append(time.time())
        self._i += 1
        self.after(self.PERIOD_MS, self._step)

    def _finish(self, cancel=False):
        self.tracker.frame_probe = None
        self.destroy()
        if cancel:
            return
        s = list(self.samples)
        swing = (max(b for _, b in s) - min(b for _, b in s)) if s else 0.0
        values = []
        bounds = self.flips + [float("inf")]
        for t0, t_next in zip(bounds, bounds[1:]):
            before = [b for t, b in s if t0 - 0.25 < t <= t0]
            after = [b for t, b in s if t0 + 0.35 < t < t_next]
            if not before or not after:
                continue
            lo, hi = sum(before) / len(before), sum(after) / len(after)
            if abs(hi - lo) < 1.5:  # изменения освещённости не видно
                continue
            mid = (lo + hi) / 2
            hit = next((t for t, b in s if t > t0 and ((b > mid) if hi > lo else (b < mid))), None)
            if hit is not None:
                values.append((hit - t0) * 1000)
        self.on_done(values, swing)


class CalibrationDialog(tk.Toplevel):
    """Калибровка расстояния: автоматически по рулетке или вручную, подбирая параметры на глаз."""

    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.app = app
        self.tracker = app.tracker
        self.title("Калибровка")
        self.resizable(False, False)
        self.transient(app.root)
        self._syncing = False
        f = ttk.Frame(self, padding=10)
        f.grid(sticky="nsew")

        auto = ttk.LabelFrame(f, text="Автоматически", padding=6)
        auto.grid(row=0, column=0, sticky="ew")
        ttk.Label(auto, text="Сядьте прямо напротив вебки, измерьте рулеткой расстояние\n"
                             "от переносицы до объектива и нажмите «Калибровать».\n"
                             "Сначала задайте межзрачковое расстояние ниже.",
                  foreground="gray").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(auto, text="Глаза → вебка, см").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dist_var = tk.StringVar(value="50")
        ttk.Entry(auto, textvariable=self.dist_var, width=7).grid(row=1, column=1, padx=6, pady=(6, 0))
        ttk.Button(auto, text="Калибровать", command=self.calibrate).grid(row=1, column=2, pady=(6, 0))

        manual = ttk.LabelFrame(f, text="Вручную", padding=6)
        manual.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.ipd_var = tk.StringVar(value=f"{self.tracker.ipd_mm:g}")
        self.scale_var = tk.StringVar()
        self.fov_var = tk.StringVar()
        rows = [
            ("Межзрачковое расстояние, мм", self.ipd_var, 45, 85, 0.5,
             "Масштабирует всю позу: меряется линейкой у зеркала или в оптике"),
            ("Масштаб глубины", self.scale_var, 0.3, 3.0, 0.01,
             "Поправка расстояния до камеры (то же, что подбирает автокалибровка)"),
            ("Угол обзора вебки по горизонтали, °", self.fov_var, 30, 140, 1,
             "Другой способ задать масштаб глубины, если угол камеры известен"),
        ]
        for i, (label, var, lo, hi, step, hint) in enumerate(rows):
            ttk.Label(manual, text=label).grid(row=2 * i, column=0, sticky="w", pady=(6 if i else 0, 0))
            ttk.Spinbox(manual, textvariable=var, from_=lo, to=hi, increment=step, width=8).grid(
                row=2 * i, column=1, padx=(8, 0), pady=(6 if i else 0, 0))
            ttk.Label(manual, text=hint, foreground="gray").grid(row=2 * i + 1, column=0, columnspan=2, sticky="w")
        ttk.Button(manual, text="Сбросить всё", command=self.reset).grid(row=6, column=0, sticky="w", pady=(8, 0))

        lat = ttk.LabelFrame(f, text="Задержка камеры и экрана", padding=6)
        lat.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(lat, text="Сколько проходит от события перед камерой до кадра в программе.\n"
                            "Клиенты компенсируют её предсказанием. «Измерить» мигает экраном\n"
                            "~10 с и смотрит, когда камера увидит изменение освещения\n"
                            "(сядьте перед камерой; в тёмной комнате точнее).",
                  foreground="gray").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(lat, text="Задержка, мс").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lat_var = tk.StringVar(value=f"{self.tracker.camera_latency_ms:.0f}")
        ttk.Spinbox(lat, textvariable=self.lat_var, from_=0, to=300, increment=5, width=8).grid(
            row=1, column=1, padx=8, pady=(6, 0))
        ttk.Button(lat, text="Измерить", command=self.measure_latency).grid(row=1, column=2, pady=(6, 0))
        self.lat_var.trace_add("write", lambda *_: self._on_latency())

        self.live = ttk.Label(f, text="", font="TkFixedFont")
        self.live.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.status = ttk.Label(f, text="", foreground="gray")
        self.status.grid(row=4, column=0, sticky="w")

        self._load_from_tracker()
        self.ipd_var.trace_add("write", lambda *_: self._on_ipd())
        self.scale_var.trace_add("write", lambda *_: self._on_scale())
        self.fov_var.trace_add("write", lambda *_: self._on_fov())
        self._tick()

    # --- пересчёт масштаба глубины <-> угла обзора ---
    # MediaPipe считает глубину для фокусного f_mp (VFOV 63°); реальная глубина = z · f_real / f_mp

    def _frame_size(self):
        snap = self.tracker.snapshot()
        return (snap.actual[0], snap.actual[1]) if snap.actual else (640, 480)

    def _scale_to_fov(self, scale: float) -> float:
        w, h = self._frame_size()
        f_mp = (h / 2) / math.tan(math.radians(MP_VFOV_DEG) / 2)
        return math.degrees(2 * math.atan((w / 2) / (scale * f_mp)))

    def _fov_to_scale(self, fov: float) -> float:
        w, h = self._frame_size()
        f_mp = (h / 2) / math.tan(math.radians(MP_VFOV_DEG) / 2)
        return ((w / 2) / math.tan(math.radians(fov) / 2)) / f_mp

    def _load_from_tracker(self):
        self._syncing = True
        self.ipd_var.set(f"{self.tracker.ipd_mm:g}")
        if hasattr(self, "lat_var"):
            self.lat_var.set(f"{self.tracker.camera_latency_ms:.0f}")
        self.scale_var.set(f"{self.tracker.depth_scale:.3f}")
        self.fov_var.set(f"{self._scale_to_fov(self.tracker.depth_scale):.1f}")
        self._syncing = False

    @staticmethod
    def _num(var, lo, hi):
        try:
            v = float(var.get().replace(",", "."))
        except ValueError:
            return None
        return v if lo <= v <= hi else None

    def _on_ipd(self):
        if self._syncing:
            return
        v = self._num(self.ipd_var, 40, 90)
        if v is not None:
            self.tracker.ipd_mm = v

    def _on_scale(self):
        if self._syncing:
            return
        v = self._num(self.scale_var, 0.1, 10)
        if v is not None:
            self.tracker.depth_scale = v
            self._syncing = True
            self.fov_var.set(f"{self._scale_to_fov(v):.1f}")
            self._syncing = False

    def _on_fov(self):
        if self._syncing:
            return
        v = self._num(self.fov_var, 10, 170)
        if v is not None:
            self.tracker.depth_scale = self._fov_to_scale(v)
            self._syncing = True
            self.scale_var.set(f"{self.tracker.depth_scale:.3f}")
            self._syncing = False

    def _on_latency(self):
        v = self._num(self.lat_var, 0, 500)
        if v is not None:
            self.tracker.camera_latency_ms = v

    def measure_latency(self):
        self.status.configure(text="Замер: смотрите на экран…", foreground="gray")
        LatencyTest(self, self._latency_done)

    def _latency_done(self, values: list[float], swing: float):
        if len(values) < 4:
            self.status.configure(
                text=f"Не удалось: камера почти не видит мигание (перепад яркости {swing:.1f}).\n"
                     f"Приглушите свет, сядьте ближе к экрану или задайте задержку вручную.",
                foreground="red")
            return
        values.sort()
        med = values[len(values) // 2]
        self.lat_var.set(f"{med:.0f}")
        self.status.configure(
            text=f"Задержка {med:.0f} мс (разброс {values[0]:.0f}–{values[-1]:.0f}: зависит от того, "
                 f"в какой момент кадра случилось событие)", foreground="green")

    def calibrate(self):
        d = self._num(self.dist_var, 5, 300)
        if d is None:
            self.status.configure(text="Расстояние — число в см", foreground="red")
            return
        if not self.tracker.calibrate_depth(d):
            self.status.configure(text="Лицо не найдено — сядьте перед камерой", foreground="red")
            return
        self._load_from_tracker()
        self.status.configure(text=f"Готово: масштаб глубины {self.tracker.depth_scale:.3f}", foreground="green")

    def reset(self):
        self.tracker.depth_scale = 1.0
        self.tracker.ipd_mm = CANON_IPD_MM
        self._load_from_tracker()
        self.status.configure(text="Сброшено", foreground="gray")

    def _tick(self):
        if not self.winfo_exists():
            return
        d = self.tracker.snapshot().data
        if d and d.get("face"):
            m = d["eye_mid_cm"]
            dist = math.sqrt(sum(c * c for c in m))
            self.live.configure(text=f"Сейчас глаза: {dist:5.1f} см от вебки (по оси z {-m[2]:5.1f})")
        else:
            self.live.configure(text="Лицо не найдено")
        self.after(150, self._tick)


def format_data(d: dict | None) -> str:
    if not d:
        return "Нет данных\n\n\n\n\n\n"
    if not d["face"]:
        return "Лицо не найдено\n\n\n\n\n\n"
    hd, er, el = d["head"], d["eyes"]["right"], d["eyes"]["left"]

    def xyz(v):
        return f"{v[0]:+7.1f} {v[1]:+7.1f} {v[2]:+7.1f}"

    return "\n".join([
        f"               x       y       z",
        f"Голова, см  {xyz(hd['pos_cm'])}",
        f"Глаз П, см  {xyz(er['cm'])}",
        f"Глаз Л, см  {xyz(el['cm'])}",
        f"Глаза до плоскости камеры: {-d['eye_mid_cm'][2]:.1f} см",
        f"yaw {hd['yaw']:+6.1f}°  pitch {hd['pitch']:+6.1f}°  roll {hd['roll']:+6.1f}°",
        f"Глаз П px {er['px'][0]:6.0f} {er['px'][1]:6.0f}",
        f"Глаз Л px {el['px'][0]:6.0f} {el['px'][1]:6.0f}",
    ])


def run(**kw):
    root = tk.Tk()
    App(root, **kw)
    root.mainloop()
