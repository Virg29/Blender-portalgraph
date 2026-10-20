"""
Head Tracking — управление 3D-вьюпортом положением головы по данным headtracking-server.

Режимы:
- «Окно»: монитор работает как окно в сцену (head-coupled perspective). Камера стоит в точке
  глаз, смотрит перпендикулярно экрану, а сдвиг объектива (shift) и фокусное подбираются так,
  чтобы рамка окна оставалась на месте — сцена «за монитором» видна под правильным углом.
- «Камера на голове»: вид повторяет движения и повороты головы.

Аддон не меняет камеры пользователя: на время работы создаётся временная камера и
назначается локальной камерой вьюпорта. Всё, что меняется (вид, оверлеи, разрешение рендера),
запоминается при старте и возвращается при остановке, перед сохранением файла,
а после аварийного выхода — при следующем открытии файла.
"""
import atexit
import json
import math
import socket
import time

import bpy
from bpy.app.handlers import persistent
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Quaternion, Vector

bl_info = {
    "name": "Head Tracking",
    "author": "virg",
    "version": (0, 10, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > N-панель > View > Head Tracking",
    "description": "Управление видом вьюпорта положением головы с вебкамеры (headtracking-server)",
    "category": "3D View",
}

TICK_S = 1 / 120  # чаще кадров камеры: между ними положение экстраполируется
UI_REDRAW_S = 0.25  # панель со статусом обновляется реже, чем вьюпорт
MAX_PREDICT_S = 0.2
MEAS_SIGMA_CM = 0.1  # типичный шум положения глаза от трекера
NOMINAL_DT = 1 / 30  # период кадров камеры
TEMP_NAME = "HeadTracking Camera"
TEMP_FLAG = "head_tracking_temp"
BACKUP_KEY = "head_tracking_backup"
SENSOR = 36.0
# Ширина «сенсора» вида вьюпорта: Blender считает его как 36 мм с зумом ×2
VIEWPORT_SENSOR = 72.0

# Камера трекера смотрит на человека, а нам нужны оси со стороны зрителя:
# x — вправо от зрителя, y — вверх, z — от экрана к зрителю. Это поворот на 180° вокруг Y.
TRACKER_TO_VIEWER = Matrix.Diagonal((-1, 1, -1))


# ---------------------------------------------------------------- сеть

class Receiver:
    """UDP-клиент headtracking-server. Без потоков: сокет неблокирующий и опрашивается в таймере."""

    SUBSCRIBE_EVERY = 1.0  # сервер забывает клиентов, молчащих дольше 3 с

    def __init__(self, host: str, port: int):
        addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.connect(addr)  # принимать датаграммы только от сервера
        self.latest: dict | None = None
        self.recv_time = 0.0  # time.monotonic() прихода последней датаграммы
        self.refused = False
        self._last_sub = 0.0
        self._subscribe()

    def _subscribe(self):
        self._last_sub = time.monotonic()
        try:
            self.sock.send(b"sub")
        except OSError:
            pass

    def poll(self) -> bool:
        """Вычитывает всё накопившееся, оставляет самое свежее. True — пришло новое."""
        if time.monotonic() - self._last_sub > self.SUBSCRIBE_EVERY:
            self._subscribe()
        packet = None
        while True:
            try:
                packet = self.sock.recv(65536)
            except BlockingIOError:
                break
            except ConnectionError:  # ICMP «порт недоступен»: сервер не запущен
                self.refused = True
                break
            except OSError:
                break
        if packet is None:
            return False
        try:
            self.latest = json.loads(packet)
        except ValueError:
            return False
        self.recv_time = time.monotonic()
        self.refused = False
        return True

    def close(self):
        try:
            self.sock.send(b"bye")
        except OSError:
            pass
        self.sock.close()


class MotionKalman:
    """Фильтр Калмана с моделью постоянного ускорения, независимо по каждой оси.

    Оценивает положение, скорость и ускорение головы. Ускорение нужно для предсказания на
    полную задержку (камера + обработка + экран, ~100 мс): при торможении экстраполяция
    учитывает его и не уходит дальше точки остановки. Одиночные дикие кадры отбрасываются.
    """

    OUTLIER_CM = 3.0  # скачок больше этого и больше 6σ — подозрение на выброс

    def __init__(self):
        self.x: list[Vector] | None = None  # по осям: (положение, скорость, ускорение)
        self.P: list[Matrix] = []
        self.t = 0.0
        self._rejected = 0

    @property
    def p(self) -> Vector:
        return Vector([x[0] for x in self.x])

    @property
    def v(self) -> Vector:
        return Vector([x[1] for x in self.x])

    def update(self, z: Vector, t: float, meas_sigma: Vector, lam: float):
        """lam — индекс манёвренности: больше — быстрее реакция, меньше — сильнее сглаживание."""
        if self.x is None:
            self.x = [Vector((z[i], 0.0, 0.0)) for i in range(3)]
            self.P = [Matrix.Diagonal((meas_sigma[i] ** 2, 1e4, 1e4)) for i in range(3)]
            self.t = t
            return
        dt = min(max(t - self.t, 1e-3), 0.5)
        F = Matrix(((1, dt, dt * dt / 2), (0, 1, dt), (0, 0, 1)))
        G = Vector((dt ** 3 / 6, dt * dt / 2, dt))
        pred_x, pred_P, innov, S = [], [], [], []
        for i in range(3):
            q = (lam * meas_sigma[i] / NOMINAL_DT ** 3) ** 2  # дисперсия «рывка»
            Q = Matrix([[q * G[r] * G[c] for c in range(3)] for r in range(3)])
            x = F @ self.x[i]
            P = F @ self.P[i] @ F.transposed() + Q
            pred_x.append(x)
            pred_P.append(P)
            innov.append(z[i] - x[0])
            S.append(P[0][0] + meas_sigma[i] ** 2)
        jump = math.sqrt(sum(d * d for d in innov))
        if jump > self.OUTLIER_CM and any(abs(innov[i]) > 6 * math.sqrt(S[i]) for i in range(3)) \
                and self._rejected < 2:
            self._rejected += 1  # пропускаем, но если повторится 3 раза подряд — это реальное движение
            return
        self._rejected = 0
        for i in range(3):
            P = pred_P[i]
            K = Vector((P[0][0], P[1][0], P[2][0])) / S[i]
            self.x[i] = pred_x[i] + K * innov[i]
            self.P[i] = Matrix([[P[r][c] - K[r] * P[0][c] for c in range(3)] for r in range(3)])
        self.t = t

    def predict(self, ahead: float) -> Vector:
        out = []
        for p, v, a in self.x:
            h = ahead
            if a * v < 0:  # торможение: не дальше точки, где скорость обнулится
                h = min(h, -v / a)
                out.append(p + v * h + 0.5 * a * h * h)
            else:  # разгон не экстраполируем — ускорение шумное, проскок хуже отставания
                out.append(p + v * h)
        return Vector(out)


class _State:
    receiver: Receiver | None = None
    scene_name: str = ""
    area_ptr: int = 0  # вьюпорт, в котором нажали «Старт»
    attached: bool = False
    region_size: tuple = (0, 0)
    measure_in: int = 0  # через сколько тиков измерить рамку камеры (ждём перерисовку)
    frame: tuple | None = None  # рамка камеры на экране: (центр от центра экрана, ширина, высота), см
    eye: Vector | None = None  # глаз относительно центра рамки, см — для панели
    window: Matrix | None = None  # «окно» в мире: центр и ориентация (z — к зрителю)
    window_width: float = 1.0  # видимая ширина вида на старте в плоскости центра вращения, единиц мира
    scaled: bool = False  # масштаб мира уже выставлен после первого замера рамки
    scene_depth_units: float | None = None  # глубина подогнанных объектов, единиц Blender
    head_base: Matrix | None = None  # матрица глаза вида на момент старта
    head_lens: float = 50.0
    clip: tuple = (0.01, 1000.0)
    ref_pos: Vector | None = None  # «ноль» головы для режима камеры на голове
    ref_rot: Matrix | None = None
    pos: Vector | None = None  # сглаженные данные трекера
    rot: Quaternion | None = None
    kalman: MotionKalman = MotionKalman()
    neutral: Vector | None = None  # нейтральное положение глаза для «силы параллакса»
    last_t: float = -1.0  # метка времени последнего обработанного кадра трекера
    latency_ms: float = 0.0  # от захвата кадра до отрисовки, по последнему кадру
    last_ui: float = 0.0
    resume_after_save: bool = False
    status: str = "Остановлен"


state = _State()


# ---------------------------------------------------------------- поиск UI

def find_area(ptr: int):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.as_pointer() == ptr:
                return window, area
    return None, None


def window_region(area):
    for region in area.regions:
        if region.type == "WINDOW":
            return region
    return None


def redraw_viewports():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ---------------------------------------------------------------- сохранение и восстановление

def make_backup(scene, window, area) -> dict:
    space = area.spaces.active
    rv3d = space.region_3d
    r = scene.render
    return {
        "scene": scene.name,
        "scene_camera": scene.camera.name if scene.camera else None,
        "screen": window.screen.name,
        "area_index": list(window.screen.areas).index(area),
        "render": {
            "resolution_x": r.resolution_x, "resolution_y": r.resolution_y,
            "pixel_aspect_x": r.pixel_aspect_x, "pixel_aspect_y": r.pixel_aspect_y,
        },
        "space": {
            "use_local_camera": space.use_local_camera,
            "camera": space.camera.name if space.camera else None,
            "show_overlays": space.overlay.show_overlays,
        },
        "rv3d": {
            "view_perspective": rv3d.view_perspective,
            "view_location": list(rv3d.view_location),
            "view_rotation": list(rv3d.view_rotation),
            "view_distance": rv3d.view_distance,
            "view_camera_zoom": rv3d.view_camera_zoom,
            "view_camera_offset": list(rv3d.view_camera_offset),
        },
    }


def restore_backup(b: dict):
    """Возвращает всё, что аддон поменял, и удаляет временную камеру."""
    scene = bpy.data.scenes.get(b["scene"])
    if scene is not None:
        for k, v in b["render"].items():
            setattr(scene.render, k, v)
        cam = b.get("scene_camera")
        scene.camera = bpy.data.objects.get(cam) if cam else None

    screen = bpy.data.screens.get(b["screen"])
    areas = list(screen.areas) if screen else []
    if b["area_index"] < len(areas) and areas[b["area_index"]].type == "VIEW_3D":
        space = areas[b["area_index"]].spaces.active
        rv3d = space.region_3d
        s = b["space"]
        # use_local_camera включаем раньше camera: иначе Blender назначит её камерой сцены
        space.use_local_camera = True
        space.camera = bpy.data.objects.get(s["camera"]) if s["camera"] else None
        space.use_local_camera = s["use_local_camera"]
        space.overlay.show_overlays = s["show_overlays"]
        for k, v in b["rv3d"].items():
            setattr(rv3d, k, v)

    remove_temp_camera()
    if scene is not None and BACKUP_KEY in scene:
        del scene[BACKUP_KEY]


def remove_temp_camera():
    for obj in [o for o in bpy.data.objects if o.get(TEMP_FLAG)]:
        data = obj.data
        bpy.data.objects.remove(obj)
        if data is not None and data.users == 0:
            bpy.data.cameras.remove(data)


def ensure_temp_camera(scene):
    """Временная камера; создаётся и настраивается только если её нет.

    Вызывается каждый тик, поэтому ничего не пишет в существующую камеру: любая запись
    в свойства — пересчёт сцены и перерисовка вьюпорта, даже если значение то же.
    """
    obj = bpy.data.objects.get(TEMP_NAME)
    if obj is None or not obj.get(TEMP_FLAG):
        data = bpy.data.cameras.new(TEMP_NAME)
        data.sensor_fit = "AUTO"  # сенсор и сдвиг считаются по большей стороне кадра
        data.passepartout_alpha = 1.0
        data.show_passepartout = True
        data.clip_start, data.clip_end = state.clip
        obj = bpy.data.objects.new(TEMP_NAME, data)
        obj[TEMP_FLAG] = True
        obj.hide_select = True
    if obj.name not in scene.collection.objects:
        scene.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------- захват вьюпорта

def capture_view(props, area):
    """Запоминает, откуда смотрел вид на момент старта: отсюда берутся окно и база для головы."""
    space = area.spaces.active
    rv3d = space.region_3d
    eye = rv3d.view_matrix.inverted()
    dist = rv3d.view_distance
    # окно — плоскость, перпендикулярная взгляду и проходящая через центр вращения вида;
    # её ширина равна видимой ширине вида на этом расстоянии
    state.window = eye @ Matrix.Translation((0, 0, -dist))
    state.window_width = dist * VIEWPORT_SENSOR / space.lens
    state.head_base = eye
    state.head_lens = space.lens
    state.clip = (space.clip_start, space.clip_end)


def attach(scene, window, area, props):
    """Перехватывает вьюпорт. Исходные настройки кладутся в сцену, чтобы пережить аварийный выход."""
    space = area.spaces.active
    rv3d = space.region_3d
    scene[BACKUP_KEY] = json.dumps(make_backup(scene, window, area))

    # кадр камеры по пропорциям вьюпорта — тогда он заполняет вьюпорт почти целиком
    region = window_region(area)
    aspect = region.height / max(region.width, 1)
    r = scene.render
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.resolution_y = max(4, round(r.resolution_x * aspect))

    cam = ensure_temp_camera(scene)
    # пока рамка не измерена, камера показывает ровно то, что было во вьюпорте
    cam.data.sensor_width, cam.data.lens = VIEWPORT_SENSOR, state.head_lens
    cam.data.shift_x = cam.data.shift_y = 0.0
    cam.data.clip_start, cam.data.clip_end = state.clip
    cam.matrix_world = state.head_base
    space.use_local_camera = True  # до назначения camera, иначе Blender сменит камеру сцены
    space.camera = cam
    rv3d.view_perspective = "CAMERA"
    if props.hide_overlays:
        space.overlay.show_overlays = False
    state.attached = True
    fit_camera(window, area)


def detach():
    """Возвращает всё как было. Безопасно вызывать повторно."""
    if not state.attached:
        return
    state.attached = False
    scene = bpy.data.scenes.get(state.scene_name)
    if scene is not None and BACKUP_KEY in scene:
        restore_backup(json.loads(scene[BACKUP_KEY]))
    else:
        remove_temp_camera()
    redraw_viewports()


def fit_camera(window, area):
    """Растягивает рамку камеры на весь вьюпорт."""
    region = window_region(area)
    if region is None:
        return
    state.region_size = (region.width, region.height)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_center_camera()
    state.frame = None
    state.measure_in = 3


def measure_frame(props, scene, window, area, cam_obj):
    """Где на мониторе реально нарисована рамка камеры, в см относительно центра экрана.

    Считается, что окно Blender развёрнуто на весь экран: тогда ширина окна в пикселях
    соответствует физической ширине экрана.
    """
    region = window_region(area)
    rv3d = area.spaces.active.region_3d
    pts = [location_3d_to_region_2d(region, rv3d, cam_obj.matrix_world @ v)
           for v in cam_obj.data.view_frame(scene=scene)]
    if region is None or None in pts:
        return None
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    cm_per_px = props.screen_width / window.width
    screen_h_px = props.screen_height / cm_per_px
    # координаты окна Blender — от левого нижнего угла; низ окна считаем низом экрана
    cx = region.x + (min(xs) + max(xs)) / 2 - window.width / 2
    cy = region.y + (min(ys) + max(ys)) / 2 - screen_h_px / 2
    return (Vector((cx * cm_per_px, cy * cm_per_px, 0.0)),
            (max(xs) - min(xs)) * cm_per_px, (max(ys) - min(ys)) * cm_per_px)


# ---------------------------------------------------------------- масштаб мира

FIT_FILL = 0.8  # какую долю окна занимают объекты после подгонки


def fit_objects(scene):
    """Выделенные объекты, а если ничего не выделено — все видимые меши сцены."""
    objs = [o for o in scene.objects if o.select_get() and not o.get(TEMP_FLAG)]
    if not objs:
        objs = [o for o in scene.objects if o.type == "MESH" and o.visible_get() and not o.get(TEMP_FLAG)]
    return objs


def fit_to_objects(props, scene) -> bool:
    """Стекло — по передней грани объектов, масштаб — чтобы они заняли FIT_FILL окна.

    Ось z окна смотрит на зрителя, поэтому «передняя грань» — максимальный z.
    """
    objs = fit_objects(scene)
    if not objs or state.frame is None:
        return False
    rot = state.window.to_3x3()
    origin = state.window.translation
    inv = rot.transposed()
    pts = [inv @ (o.matrix_world @ Vector(c) - origin) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    size = hi - lo
    _, fw, fh = state.frame
    cpu = min(fw * FIT_FILL / max(size.x, 1e-6), fh * FIT_FILL / max(size.y, 1e-6))
    front = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, hi.z))
    state.window = Matrix.Translation(origin + rot @ front) @ rot.to_4x4()
    state.scene_depth_units = size.z
    props.cm_per_unit = cpu
    props.glass_offset = 0.0
    return True


def fit_to_portal(props, scene) -> bool:
    """Окно — плоскость-портал из сцены: её центр, ориентация и размер.

    Портал — прямоугольник в локальной плоскости XY объекта, локальная ось Z смотрит на зрителя.
    Масштаб подбирается так, чтобы портал целиком вписался в рамку на экране.
    """
    portal = props.portal
    if portal is None or state.frame is None:
        return False
    lo = Vector((min(c[0] for c in portal.bound_box), min(c[1] for c in portal.bound_box), 0.0))
    hi = Vector((max(c[0] for c in portal.bound_box), max(c[1] for c in portal.bound_box), 0.0))
    m = portal.matrix_world
    _, rot, scale = m.decompose()
    pw, ph = (hi.x - lo.x) * scale.x, (hi.y - lo.y) * scale.y
    if pw <= 1e-6 or ph <= 1e-6:
        return False
    _, fw, fh = state.frame
    state.window = Matrix.Translation(m @ ((lo + hi) / 2)) @ rot.to_matrix().to_4x4()
    props.cm_per_unit = min(fw / pw, fh / ph)
    props.glass_offset = 0.0
    # глубина сцены за стеклом — для панели
    inv = state.window.inverted()
    zs = [(inv @ (o.matrix_world @ Vector(c))).z for o in fit_objects(scene) if o != portal for c in o.bound_box]
    state.scene_depth_units = max(0.0, -min(zs)) if zs else None
    return True


def fit_scene(props, scene) -> bool:
    """Портал, если задан, иначе — по объектам."""
    if props.portal is not None:
        return fit_to_portal(props, scene)
    return fit_to_objects(props, scene)


def on_portal_update(props, context):
    if state.receiver is not None and state.frame is not None:
        fit_scene(props, context.scene)


def init_scale(props, scene):
    """После первого замера рамки: портал, подгонка под объекты или вид как был на старте."""
    state.scaled = True
    if props.portal is not None and fit_to_portal(props, scene):
        return
    if props.fit_on_start and fit_to_objects(props, scene):
        return
    _, fw, fh = state.frame
    props.cm_per_unit = max(fw, fh) / state.window_width
    state.scene_depth_units = None


# ---------------------------------------------------------------- математика

def tracker_head(d: dict):
    """(точка между глаз в см, поворот головы 3×3) в системе камеры трекера, или None."""
    if not d or not d.get("face"):
        return None
    if "eye_mid_cm" in d:
        pos = Vector(d["eye_mid_cm"])  # центр между глазными яблоками, не зависит от взгляда
    elif d.get("eyes"):
        pos = (Vector(d["eyes"]["left"]["cm"]) + Vector(d["eyes"]["right"]["cm"])) / 2
    else:
        pos = Vector(d["head"]["pos_cm"])
    rot = Matrix([row[:3] for row in d["head"]["matrix"][:3]])
    return pos, rot


def mirror_matrix(props) -> Matrix:
    return Matrix.Diagonal((
        -1 if props.mirror_x else 1,
        -1 if props.mirror_y else 1,
        -1 if props.mirror_z else 1,
    ))


def set_camera(cam_obj, matrix: Matrix, lens: float, sensor: float, shift_x: float, shift_y: float) -> bool:
    """Пишет в камеру, только если что-то поменялось: любая запись заставляет вьюпорт перерисоваться."""
    cam = cam_obj.data
    same = (abs(cam.lens - lens) < 1e-4 and cam.sensor_width == sensor
            and abs(cam.shift_x - shift_x) < 1e-5 and abs(cam.shift_y - shift_y) < 1e-5
            and all(abs(a - b) < 1e-6 for ra, rb in zip(cam_obj.matrix_world, matrix) for a, b in zip(ra, rb)))
    if same:
        return False
    cam.sensor_width, cam.lens = sensor, lens
    cam.shift_x, cam.shift_y = shift_x, shift_y
    cam_obj.matrix_world = matrix
    return True


def real_cm_per_unit(scene) -> float:
    """Сколько см в одной единице Blender в реальном размере: единица = scale_length метров."""
    return 100.0 * scene.unit_settings.scale_length


def viewer_eye(props, pos: Vector) -> Vector:
    """Положение глаза (данные трекера) в см относительно центра рамки камеры на экране."""
    webcam = Vector((props.webcam_offset_x, props.screen_height / 2 + props.webcam_gap, props.webcam_offset_z))
    # вебка может быть повёрнута относительно экрана (монитор лежит, вебка на подставке смотрит на зрителя)
    tilt = Matrix.Rotation(math.radians(props.webcam_tilt), 3, "X")
    eye = tilt @ (mirror_matrix(props) @ (TRACKER_TO_VIEWER @ pos)) + webcam  # от центра экрана
    return eye - state.frame[0]  # от центра рамки — это и есть «окно»


def effective_cm_per_unit(props) -> float:
    """Масштаб с учётом «Вглубь» в режиме «Сохранять размер».

    Объект, отодвинутый с расстояния E на E+g, выглядит в E/(E+g) раз меньше. Чтобы из
    нейтральной позы он остался того же размера на экране, сцена увеличивается в (E+g)/E раз.
    """
    cpu = props.cm_per_unit
    g = props.glass_offset
    if not props.glass_keep_size or g == 0.0 or state.neutral is None or state.frame is None:
        return cpu
    e = max(viewer_eye(props, state.neutral).z, 5.0)
    return cpu * max(e + g, 1.0) / e


def apply_window(props, cam_obj, pos: Vector):
    """Off-axis проекция: рамка кадра всегда совпадает с окном, камера стоит в точке глаза."""
    eye = viewer_eye(props, pos)
    eye.z = max(eye.z, 5.0)
    state.eye = eye

    _, fw, fh = state.frame
    big = max(fw, fh)  # sensor_fit AUTO: сенсор и сдвиг считаются по большей стороне
    k = 1.0 / effective_cm_per_unit(props)  # единиц мира на 1 см
    glass = state.window @ Matrix.Translation((0, 0, props.glass_offset * k))
    return set_camera(cam_obj, glass @ Matrix.Translation(eye * k),
                      SENSOR * eye.z / big, SENSOR, -eye.x / big, -eye.y / big)


def apply_head(props, cam_obj, pos: Vector, rot: Quaternion):
    """Вид повторяет смещение и поворот головы относительно «нуля»."""
    to_view = mirror_matrix(props) @ TRACKER_TO_VIEWER
    m = Matrix.Translation(to_view @ ((pos - state.ref_pos) * props.distance_scale))
    if props.use_rotation:
        d_rot = to_view @ (rot.to_matrix() @ state.ref_rot.transposed()) @ to_view
        axis, angle = d_rot.to_quaternion().to_axis_angle()
        m = m @ Matrix.Rotation(angle * props.angle_scale, 4, axis)
    return set_camera(cam_obj, state.head_base @ m, state.head_lens, VIEWPORT_SENSOR, 0.0, 0.0)


# ---------------------------------------------------------------- цикл

def tick():
    if state.receiver is None:
        return None
    if not state.attached:  # временно отпущено на время сохранения файла
        return TICK_S
    rcv = state.receiver
    fresh = rcv.poll()
    scene = bpy.data.scenes.get(state.scene_name)
    window, area = find_area(state.area_ptr)
    if scene is None or area is None:
        stop("Вьюпорт закрыт")
        return None
    props = scene.head_tracking
    space = area.spaces.active

    # пользователь мог выйти из вида камеры, отменить действие (Ctrl+Z) и т.п. — возвращаем
    cam = ensure_temp_camera(scene)
    if space.camera != cam or not space.use_local_camera:
        space.use_local_camera = True
        space.camera = cam
    if space.region_3d.view_perspective != "CAMERA":
        space.region_3d.view_perspective = "CAMERA"
    region = window_region(area)
    if region is not None and (region.width, region.height) != state.region_size:
        fit_camera(window, area)
    if state.measure_in > 0:  # камера заморожена, пока вьюпорт не перерисуется
        state.measure_in -= 1
        if state.measure_in == 0:
            state.frame = measure_frame(props, scene, window, area, cam)
            if state.frame is None:
                state.measure_in = 3
            elif not state.scaled:
                init_scale(props, scene)
        state.status = "Измеряю окно…"
        redraw_viewports()
        return TICK_S

    now = time.monotonic()
    head = tracker_head(rcv.latest)
    if fresh and head is not None:
        pos, rot3 = head
        rot = rot3.to_quaternion()
        t = rcv.latest.get("t", 0.0)
        if state.ref_pos is None:
            state.ref_pos, state.ref_rot = pos.copy(), rot3.copy()
        if state.pos is None:
            state.kalman = MotionKalman()
            state.rot = rot
        if t != state.last_t:  # фильтруем только новые кадры трекера
            # Поведение фильтра задаёт одно число — индекс манёвренности λ (ожидаемый «рывок»
            # головы к шуму измерений); сглаживание отображается в проверенный диапазон λ.
            lam = 0.1 + 1.9 * (1.0 - props.smoothing) ** 2
            meas = Vector((MEAS_SIGMA_CM, MEAS_SIGMA_CM, MEAS_SIGMA_CM * 2))  # глубина шумнее вдвое
            state.kalman.update(pos, t, meas, lam)
            state.pos = state.kalman.p
            state.rot = state.rot.slerp(rot, 1.0 - 0.9 * props.smoothing)
        if state.neutral is None:
            state.neutral = state.pos.copy()
        state.last_t = t

    moved = False
    if state.pos is not None and head is not None:
        # возраст данных к этому моменту: захват -> отправка (меряет трекер) + сколько лежит у нас
        age = rcv.latest.get("latency_ms", 0.0) / 1000 + (now - rcv.recv_time)
        state.latency_ms = age * 1000
        ahead = 0.0
        if props.predict:
            ahead = age * props.predict_amount + props.predict_extra_ms / 1000
            ahead = min(max(ahead, 0.0), MAX_PREDICT_S)
        pos = state.kalman.predict(ahead)
        if props.mode == "WINDOW" and props.parallax != 1.0:
            pos = state.neutral + (pos - state.neutral) * props.parallax
        if props.mode == "WINDOW":
            moved = apply_window(props, cam, pos)
        else:
            moved = apply_head(props, cam, pos, state.rot)

    if rcv.latest is None:
        state.status = "Сервер не отвечает" if rcv.refused else "Жду данные от сервера…"
    elif now - rcv.recv_time > 2.0:
        state.status = "Данные не приходят — сервер остановлен?"
    elif head is None:
        state.status = "Лицо не найдено"
    else:
        state.status = f"Работает, {rcv.latest.get('fps', 0):.0f} fps, задержка {state.latency_ms:.0f} мс"

    if moved:
        area.tag_redraw()
    if now - state.last_ui > UI_REDRAW_S:
        state.last_ui = now
        redraw_viewports()
    return TICK_S


def stop(status: str = "Остановлен"):
    if state.receiver:
        state.receiver.close()
    state.receiver = None
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    detach()
    state.status = status
    redraw_viewports()


# ---------------------------------------------------------------- обработчики файла

@persistent
def on_save_pre(*_):
    # в файл не должны попасть временная камера и подменённые настройки
    state.resume_after_save = state.attached
    detach()


@persistent
def on_save_post(*_):
    if not state.resume_after_save or state.receiver is None:
        return
    state.resume_after_save = False
    scene = bpy.data.scenes.get(state.scene_name)
    window, area = find_area(state.area_ptr)
    if scene is None or area is None:
        stop("Вьюпорт закрыт")
        return
    attach(scene, window, area, scene.head_tracking)


@persistent
def on_load_pre(*_):
    # старый файл выгружается — восстанавливать уже нечего, просто отключаемся
    state.attached = False
    stop()


@persistent
def on_load_post(*_):
    # файл мог быть сохранён аварийно (autosave, quit.blend) во время работы аддона
    for scene in bpy.data.scenes:
        if BACKUP_KEY in scene:
            try:
                restore_backup(json.loads(scene[BACKUP_KEY]))
            except (ValueError, KeyError, TypeError):
                del scene[BACKUP_KEY]
    remove_temp_camera()


def _on_exit():
    # отписаться от сервера при закрытии Blender
    if state.receiver:
        state.receiver.close()


atexit.register(_on_exit)

HANDLERS = (
    (bpy.app.handlers.save_pre, on_save_pre),
    (bpy.app.handlers.save_post, on_save_post),
    (bpy.app.handlers.load_pre, on_load_pre),
    (bpy.app.handlers.load_post, on_load_post),
)


# ---------------------------------------------------------------- UI

class HeadTrackingProps(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(name="Хост", default="127.0.0.1")
    port: bpy.props.IntProperty(name="Порт", default=5005, min=1, max=65535)
    mode: bpy.props.EnumProperty(
        name="Режим",
        items=[
            ("WINDOW", "Окно", "Монитор — окно в сцену: сдвигаешь голову — заглядываешь за край"),
            ("HEAD", "Камера на голове", "Вид повторяет движения и повороты головы"),
        ],
    )
    # --- окно ---
    screen_width: bpy.props.FloatProperty(
        name="Ширина экрана", default=34.5, min=1.0, precision=1,
        description="Физическая ширина видимой области монитора, см")
    screen_height: bpy.props.FloatProperty(
        name="Высота экрана", default=19.4, min=1.0, precision=1,
        description="Физическая высота видимой области монитора, см")
    webcam_gap: bpy.props.FloatProperty(
        name="Вебка над экраном", default=1.0, precision=1,
        description="Насколько объектив вебки выше верхнего края экрана, см")
    webcam_offset_x: bpy.props.FloatProperty(
        name="Вебка по X", default=0.0, precision=1,
        description="Смещение вебки вправо от центра экрана, см")
    webcam_offset_z: bpy.props.FloatProperty(
        name="Вебка перед экраном", default=0.0, precision=1,
        description="Насколько объектив вебки выдвинут от плоскости экрана к зрителю, см "
                    "(вебка на подставке над лежащим монитором)")
    webcam_tilt: bpy.props.FloatProperty(
        name="Наклон вебки", default=0.0, soft_min=-90.0, soft_max=90.0, precision=0,
        description="На сколько градусов вебка повёрнута относительно экрана: 0 — смотрит "
                    "перпендикулярно экрану, плюс — повёрнута в сторону нижнего края "
                    "(монитор лежит, вебка у дальнего края смотрит на зрителя)")
    cm_per_unit: bpy.props.FloatProperty(
        name="1 единица, см", default=10.0, min=0.001, soft_max=1000.0, precision=2,
        description="Сколько реальных сантиметров на экране соответствует одной единице Blender. "
                    "Выставляется при старте, можно крутить вживую")
    glass_offset: bpy.props.FloatProperty(
        name="Вглубь, см", default=0.0, soft_min=-50.0, soft_max=50.0, precision=1,
        description="Отодвинуть сцену за стекло на столько см (отрицательное — вытащить из монитора)")
    glass_keep_size: bpy.props.BoolProperty(
        name="Сохранять размер", default=True,
        description="Отодвигая сцену вглубь, увеличивать её так, чтобы из нейтральной позы "
                    "она осталась того же размера на экране. Без галочки — как настоящий предмет: "
                    "дальше — значит меньше")
    portal: bpy.props.PointerProperty(
        name="Портал", type=bpy.types.Object, update=on_portal_update,
        description="Плоскость в сцене, которая совпадёт с экраном: окно ставится по ней "
                    "(локальная Z — к зрителю), масштаб — чтобы она целиком влезла в экран. "
                    "Если не задан — подгонка по объектам")
    fit_on_start: bpy.props.BoolProperty(
        name="Подогнать при старте", default=True,
        description="При старте поставить стекло по передней грани выделенных объектов "
                    "(или всей сцены) и подобрать масштаб, чтобы они заняли окно")
    # --- камера на голове ---
    use_rotation: bpy.props.BoolProperty(name="Вращение", default=True)
    distance_scale: bpy.props.FloatProperty(
        name="Расстояние", default=0.01, min=0.0, soft_max=1.0, precision=4,
        description="Сколько единиц мира на 1 см движения головы")
    angle_scale: bpy.props.FloatProperty(
        name="Угол", default=1.0, min=0.0, soft_max=5.0,
        description="Множитель угла поворота головы")
    # --- общие ---
    mirror_x: bpy.props.BoolProperty(name="X", default=False, description="Зеркалировать по X")
    mirror_y: bpy.props.BoolProperty(name="Y", default=False, description="Зеркалировать по Y")
    mirror_z: bpy.props.BoolProperty(name="Z", default=False, description="Зеркалировать по Z")
    smoothing: bpy.props.FloatProperty(
        name="Сглаживание", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
        description="Больше — меньше дрожания, но оценка скорости медленнее подстраивается "
                    "к разгону и торможению")
    predict: bpy.props.BoolProperty(
        name="Компенсировать задержку", default=True,
        description="Экстраполировать положение головы по скорости на время задержки трекера "
                    "и между кадрами камеры — меньше отставание, плавнее движение")
    predict_amount: bpy.props.FloatProperty(
        name="Компенсация", default=0.8, min=0.0, max=1.0, subtype="FACTOR",
        description="Какую долю задержки (камера + обработка, её сообщает трекер) компенсировать. "
                    "1 — полностью, но при резкой остановке картинка проскакивает на 1–2 см")
    predict_extra_ms: bpy.props.FloatProperty(
        name="Сверх, мс", default=0.0, soft_min=-50.0, soft_max=100.0,
        description="Дополнительное упреждение на задержку отрисовки и монитора")
    parallax: bpy.props.FloatProperty(
        name="Сила параллакса", default=1.0, min=0.0, soft_max=2.0, subtype="FACTOR",
        description="Множитель отклонения глаза от нейтральной позы. 1 — физически честное окно, "
                    "меньше — далёкие объекты ездят слабее")
    hide_overlays: bpy.props.BoolProperty(
        name="Скрыть оверлеи", default=True,
        description="Скрыть сетку, гизмо и т.п. на время работы")


class HEADTRACK_OT_start(bpy.types.Operator):
    bl_idname = "head_tracking.start"
    bl_label = "Старт"
    bl_description = "Подключиться к серверу и перехватить этот вьюпорт"

    def execute(self, context):
        props = context.scene.head_tracking
        area = context.area
        if area is None or area.type != "VIEW_3D":
            self.report({"ERROR"}, "Нажмите «Старт» в 3D-вьюпорте")
            return {"CANCELLED"}
        try:
            state.receiver = Receiver(props.host, props.port)
        except OSError as e:  # например, не резолвится хост
            state.status = f"Не подключиться: {e}"
            self.report({"ERROR"}, state.status)
            return {"CANCELLED"}
        state.scene_name = context.scene.name
        state.area_ptr = area.as_pointer()
        state.ref_pos = state.pos = state.neutral = None
        state.last_t = -1.0
        state.scaled = False
        state.scene_depth_units = None
        capture_view(props, area)
        attach(context.scene, context.window, area, props)
        state.status = "Жду данные…"
        bpy.app.timers.register(tick)
        return {"FINISHED"}


class HEADTRACK_OT_stop(bpy.types.Operator):
    bl_idname = "head_tracking.stop"
    bl_label = "Стоп"
    bl_description = "Отключиться и вернуть вьюпорт как было"

    def execute(self, context):
        stop()
        return {"FINISHED"}


class HEADTRACK_OT_fit(bpy.types.Operator):
    bl_idname = "head_tracking.fit"
    bl_label = "Подогнать"
    bl_description = "По порталу, если задан. Иначе стекло — по передней грани выделенных объектов (или всей сцены), масштаб — чтобы они заняли окно"

    def execute(self, context):
        if not fit_scene(context.scene.head_tracking, context.scene):
            self.report({"WARNING"}, "Нечего подгонять или окно ещё не измерено")
            return {"CANCELLED"}
        return {"FINISHED"}


class HEADTRACK_OT_real_scale(bpy.types.Operator):
    bl_idname = "head_tracking.real_scale"
    bl_label = "Реальный размер"
    bl_description = "Масштаб 1:1 — единица Blender показывается своим настоящим размером (по Unit Scale сцены)"

    def execute(self, context):
        props = context.scene.head_tracking
        props.cm_per_unit = real_cm_per_unit(context.scene)
        props.glass_offset = 0.0
        return {"FINISHED"}


class HEADTRACK_OT_recenter(bpy.types.Operator):
    bl_idname = "head_tracking.recenter"
    bl_label = "Сбросить ноль"
    bl_description = "Считать текущее положение головы нулевым (нейтралью для силы параллакса)"

    def execute(self, context):
        state.ref_pos = None  # следующий кадр станет новым «нулём»
        state.neutral = state.pos.copy() if state.pos is not None else None
        return {"FINISHED"}


class HEADTRACK_PT_panel(bpy.types.Panel):
    bl_label = "Head Tracking"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "View"

    def draw(self, context):
        props = context.scene.head_tracking
        running = state.receiver is not None
        layout = self.layout

        col = layout.column()
        col.enabled = not running
        row = col.row(align=True)
        row.prop(props, "host", text="")
        row.prop(props, "port", text="")
        col.prop(props, "mode", text="")

        if props.mode == "WINDOW":
            col = layout.column(align=True)
            col.enabled = not running  # рамка измеряется при старте
            col.prop(props, "screen_width", text="Экран, ширина см")
            col.prop(props, "screen_height", text="Экран, высота см")
            col = layout.column(align=True)
            col.prop(props, "webcam_gap", text="Вебка над экраном, см")
            col.prop(props, "webcam_offset_x", text="Вебка вправо, см")
            col.prop(props, "webcam_offset_z", text="Вебка перед экраном, см")
            col.prop(props, "webcam_tilt", text="Наклон вебки, °")
            col = layout.column(align=True)
            row = col.row(align=True)
            row.prop(props, "cm_per_unit")
            row.operator("head_tracking.real_scale", text="", icon="WORLD")
            col.prop(props, "glass_offset")
            col.prop(props, "glass_keep_size")
            layout.prop(props, "portal", icon="MESH_PLANE")
            row = layout.row()
            row.enabled = not running and props.portal is None
            row.prop(props, "fit_on_start")
            if running:
                layout.operator("head_tracking.fit", icon="SHADING_BBOX")
        else:
            layout.prop(props, "use_rotation")
            col = layout.column(align=True)
            col.prop(props, "distance_scale")
            col.prop(props, "angle_scale")

        row = layout.row(align=True)
        row.label(text="Зеркало")
        row.prop(props, "mirror_x", toggle=True)
        row.prop(props, "mirror_y", toggle=True)
        row.prop(props, "mirror_z", toggle=True)
        if props.mode == "WINDOW":
            layout.prop(props, "parallax")
        layout.prop(props, "smoothing")
        col = layout.column(align=True)
        col.prop(props, "predict")
        sub = col.column(align=True)
        sub.enabled = props.predict
        sub.prop(props, "predict_amount")
        sub.prop(props, "predict_extra_ms")
        row = layout.row()
        row.enabled = not running
        row.prop(props, "hide_overlays")

        if running:
            row = layout.row(align=True)
            row.operator("head_tracking.stop", icon="PAUSE")
            row.operator("head_tracking.recenter", icon="PIVOT_CURSOR")
        else:
            layout.operator("head_tracking.start", icon="PLAY")
        layout.label(text=state.status)
        if running and props.mode == "WINDOW" and state.frame:
            c, fw, fh = state.frame
            col = layout.column(align=True)
            col.label(text=f"Окно: {fw:.1f} × {fh:.1f} см, центр {c.x:+.1f} {c.y:+.1f}")
            if state.eye:
                e = state.eye
                col.label(text=f"Глаз от окна: {e.x:+.1f} {e.y:+.1f} {e.z:.1f} см")
            cpu = effective_cm_per_unit(props)
            ratio = cpu / real_cm_per_unit(context.scene)
            scale = f"1:{1 / ratio:.3g}" if ratio < 1 else f"{ratio:.3g}:1"
            col.label(text=f"Масштаб {scale}: 1 единица = {cpu:.3g} см на экране")
            if state.scene_depth_units is not None:
                g = props.glass_offset
                col.label(text=f"Объекты за стеклом: {g:.1f}…{g + state.scene_depth_units * cpu:.1f} см")
        elif not running and props.mode == "WINDOW":
            col = layout.column(align=True)
            col.label(text="Окно Blender — на весь экран,")
            col.label(text="вьюпорт — Ctrl+Alt+Space.")


classes = (HeadTrackingProps, HEADTRACK_OT_start, HEADTRACK_OT_stop, HEADTRACK_OT_fit, HEADTRACK_OT_real_scale,
           HEADTRACK_OT_recenter,
           HEADTRACK_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.head_tracking = bpy.props.PointerProperty(type=HeadTrackingProps)
    for handlers, fn in HANDLERS:
        if fn not in handlers:
            handlers.append(fn)


def unregister():
    stop()
    for handlers, fn in HANDLERS:
        if fn in handlers:
            handlers.remove(fn)
    del bpy.types.Scene.head_tracking
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
