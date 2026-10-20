"""
Собирает демо-сцену для режима «Окно» аддона Head Tracking: монитор — окно в комнату
с неоновой сеткой на стенах, внутри танцует девочка, вокруг летают шары, которые
вылетают за стекло — к зрителю.

Две сцены в одном файле:
- «Портал сбоку» — экран как окно в стене комнаты;
- «Портал сверху» — экран как люк в полу: смотришь вниз, в колодец.

Запуск:
    blender --background --factory-startup --python make_portal_demo.py -- portal_demo.blend
"""
import math
import random
import sys

from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "portal_demo.blend"

# Проём портала — 16:9, как экран. На мониторе 34.5 см 1 единица ≈ 4.3 см: девочка (1.4) — около 6 см,
# комната уходит вглубь на ~26 см. Чем меньше объект относительно окна, тем правдоподобнее глубина.
W, H = 8.0, 4.5
DEPTH = 6.0
PIT = 4.0  # глубина колодца в сцене «сверху»: смотришь на макушку, глубже — совсем мелко
BEAT = "frame*0.2094395"  # π/15: доля 120 BPM при 30 fps
BAR = "frame*0.1047198"  # вдвое медленнее — такт из двух долей


# ---------------------------------------------------------------- материалы

def principled(name, color, rough=0.5, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*color, 1.0)
    return m


def grid_material(name, base, line, strength, density=2.0, width=0.04):
    """Тёмная поверхность со светящейся сеткой в мировых координатах.

    Линии рисуются только вдоль поверхности: ось, перпендикулярная грани, исключается
    по нормали — иначе грань, попавшая ровно на линию сетки, засветилась бы целиком.
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    geo = nodes.new("ShaderNodeNewGeometry")

    def vmath(op, a, b=None):
        n = nodes.new("ShaderNodeVectorMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, tuple):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    def math(op, a, b=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, float):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    f = vmath("FRACTION", vmath("MULTIPLY", geo.outputs["Position"], (density,) * 3))
    d = vmath("MINIMUM", f, vmath("SUBTRACT", (1.0, 1.0, 1.0), f))  # до ближайшей линии, в клетках
    sd = nodes.new("ShaderNodeSeparateXYZ")
    links.new(d, sd.inputs[0])
    sn = nodes.new("ShaderNodeSeparateXYZ")
    links.new(vmath("ABSOLUTE", geo.outputs["Normal"]), sn.inputs[0])
    lines = None
    for i in range(3):
        on = math("LESS_THAN", sd.outputs[i], width)
        along = math("SUBTRACT", 1.0, sn.outputs[i])
        v = math("MULTIPLY", on, along)
        lines = v if lines is None else math("MAXIMUM", lines, v)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*line, 1.0)
    em.inputs["Strength"].default_value = strength
    mix = nodes.new("ShaderNodeMixShader")
    links.new(lines, mix.inputs[0])
    links.new(bsdf.outputs[0], mix.inputs[1])
    links.new(em.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs[0])
    m.diffuse_color = (*line, 1.0)
    return m


MAT = {}


def materials():
    MAT.update(
        skin=principled("Кожа", (0.95, 0.76, 0.66), 0.6),
        hair=principled("Волосы", (0.95, 0.38, 0.62), 0.4),
        navy=principled("Синяя ткань", (0.04, 0.06, 0.2), 0.7),
        white=principled("Белая ткань", (0.92, 0.92, 0.95), 0.6),
        red=principled("Бант", (0.85, 0.05, 0.12), 0.4),
        sock=principled("Гольфы", (0.02, 0.02, 0.025), 0.5),
        shoe=principled("Туфли", (0.12, 0.05, 0.03), 0.3),
        iris=principled("Радужка", (0.05, 0.45, 0.55), 0.2, emit=(0.1, 0.6, 0.7), strength=0.3),
        pupil=principled("Зрачок", (0.01, 0.01, 0.02), 0.2),
        shine=principled("Блик", (1, 1, 1), 0.2, emit=(1, 1, 1), strength=2.0),
        mouth=principled("Рот", (0.55, 0.08, 0.12), 0.5),
        blush=principled("Румянец", (1.0, 0.45, 0.5), 0.6),
        bezel=principled("Рамка", (0.0, 0.0, 0.0), 1.0),
        stage=principled("Сцена", (0.08, 0.08, 0.12), 0.25),
        walls=grid_material("Сетка стен", (0.004, 0.006, 0.02), (0.0, 0.75, 1.0), 2.5),
        floor=grid_material("Сетка пола", (0.01, 0.004, 0.02), (1.0, 0.15, 0.75), 2.5),
    )
    for i, c in enumerate([(1.0, 0.85, 0.2), (0.2, 1.0, 0.6), (0.3, 0.6, 1.0), (1.0, 0.3, 0.3), (0.85, 0.4, 1.0)]):
        MAT[f"orb{i}"] = principled(f"Шар {i}", c, 0.2, emit=c, strength=1.2)
    for i, c in enumerate([(1.0, 0.9, 0.35), (0.7, 1.0, 0.35), (0.4, 1.0, 0.8)]):
        MAT[f"fly{i}"] = principled(f"Светлячок {i}", c, 0.5, emit=c, strength=1.3)


# ---------------------------------------------------------------- геометрия

def mesh_obj(name, build, mat, coll, parent=None, loc=(0, 0, 0), smooth=True):
    bm = bmesh.new()
    build(bm)
    for f in bm.faces:
        f.smooth = smooth
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj.parent = parent
    obj.location = loc
    return obj


def sphere(r, center=(0, 0, 0), scale=(1, 1, 1), seg=24):
    def build(bm):
        m = Matrix.Translation(center) @ Matrix.Diagonal((*scale, 1))
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r, matrix=m)
    return build


def cone(r1, r2, length, center=(0, 0, 0), seg=20, rot=None):
    """Усечённый конус вдоль -Z: радиус r1 в center, r2 — на length ниже."""
    def build(bm):
        m = Matrix.Translation(center)
        if rot is not None:
            m = m @ rot
        m = m @ Matrix.Translation((0, 0, -length / 2))
        bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r2, radius2=r1, depth=length, matrix=m)
    return build


def limb(r1, r2, length):
    """Капсула от начала координат вниз на length."""
    def build(bm):
        cone(r1, r2, length)(bm)
        sphere(r1, seg=16)(bm)
        sphere(r2, center=(0, 0, -length), seg=16)(bm)
    return build


def combine(*builds):
    def build(bm):
        for b in builds:
            b(bm)
    return build


def drive(obj, path, index, expr):
    fc = obj.driver_add(path, index)
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = expr


# ---------------------------------------------------------------- танцовщица

def build_dancer():
    """Девочка в матроске из примитивов. Смотрит на -Y, анимация — драйверами от номера кадра."""
    coll = bpy.data.collections.new("Танцовщица")
    root = bpy.data.objects.new("Танцовщица", None)
    root.empty_display_size = 0.3
    root.location = (0, 0, 0.06)  # стоит на подиуме
    coll.objects.link(root)
    drive(root, "rotation_euler", 2, "0.7*sin(frame*0.0131)")  # медленно поворачивается

    hips = mesh_obj("Таз", sphere(0.1, scale=(1.2, 0.8, 0.7)), MAT["navy"], coll, root, (0, 0, 0.70))
    drive(hips, "location", 2, f"0.70-0.05*abs(sin({BEAT}))")
    drive(hips, "location", 0, f"0.05*sin({BAR})")
    drive(hips, "rotation_euler", 2, f"0.25*sin({BAR})")
    mesh_obj("Юбка", cone(0.11, 0.26, 0.2, center=(0, 0, 0.04), seg=14), MAT["navy"], coll, hips, smooth=False)

    for s, side in ((1, "L"), (-1, "R")):
        thigh = mesh_obj(f"Бедро.{side}", limb(0.055, 0.045, 0.32), MAT["skin"], coll, hips, (s * 0.07, 0, -0.02))
        drive(thigh, "rotation_euler", 0, f"-0.45*abs(sin({BEAT}))")
        drive(thigh, "rotation_euler", 1, f"{-s}*0.06")
        shin = mesh_obj(f"Голень.{side}", limb(0.046, 0.034, 0.32), MAT["sock"], coll, thigh, (0, 0, -0.32))
        drive(shin, "rotation_euler", 0, f"0.85*abs(sin({BEAT}))")
        mesh_obj(f"Туфля.{side}", sphere(0.045, center=(0, -0.04, -0.33), scale=(0.9, 1.6, 0.7)),
                 MAT["shoe"], coll, shin)

    torso = mesh_obj("Тело", cone(0.13, 0.095, 0.30, center=(0, 0, 0.30), rot=Matrix.Diagonal((1, 0.75, 1, 1))),
                     MAT["white"], coll, hips, (0, 0, 0.04))
    drive(torso, "rotation_euler", 2, f"-0.18*sin({BAR})")
    drive(torso, "rotation_euler", 1, f"0.08*sin({BAR})")
    mesh_obj("Воротник", cone(0.07, 0.165, 0.07, center=(0, 0.01, 0.315), seg=16,
                              rot=Matrix.Diagonal((1, 0.8, 1, 1))), MAT["navy"], coll, torso)
    mesh_obj("Бант", combine(sphere(0.035, (-0.035, -0.105, 0.23), (1.2, 0.5, 0.8)),
                             sphere(0.035, (0.035, -0.105, 0.23), (1.2, 0.5, 0.8)),
                             sphere(0.018, (0, -0.115, 0.23))), MAT["red"], coll, torso)
    mesh_obj("Шея", cone(0.035, 0.035, 0.07, center=(0, 0, 0.37), seg=12), MAT["skin"], coll, torso)

    def face(x, z, out=0.0):
        """Точка на поверхности лица (голова — сфера R 0.15 с центром на высоте 0.13, сжатая по Y)."""
        return (x, -0.95 * math.sqrt(max(0.15 ** 2 - x * x - (z - 0.13) ** 2, 0.0)) - out, z)

    head = mesh_obj("Голова", sphere(0.15, center=(0, 0, 0.13), scale=(1, 0.95, 1.02)), MAT["skin"], coll,
                    torso, (0, 0, 0.36))
    drive(head, "rotation_euler", 1, f"0.15*sin({BAR}+0.5)")
    drive(head, "rotation_euler", 0, f"0.1*abs(sin({BEAT}))")
    for s, side in ((1, "L"), (-1, "R")):
        x = s * 0.058
        mesh_obj(f"Глаз.{side}", sphere(0.036, face(x, 0.115, 0.004), (0.85, 0.35, 1.25)), MAT["iris"], coll, head)
        mesh_obj(f"Зрачок.{side}", sphere(0.018, face(x, 0.11, 0.011), (0.9, 0.3, 1.3)), MAT["pupil"], coll, head)
        mesh_obj(f"Блик.{side}", sphere(0.011, face(x + s * 0.012, 0.135, 0.016), (1, 0.4, 1)), MAT["shine"],
                 coll, head)
        mesh_obj(f"Румянец.{side}", sphere(0.025, face(s * 0.09, 0.065, -0.003), (1, 0.3, 0.5)), MAT["blush"],
                 coll, head)
    mesh_obj("Рот", sphere(0.016, face(0, 0.045, -0.001), (1, 0.3, 0.45)), MAT["mouth"], coll, head)

    def hair_cap(bm):
        sphere(0.163, center=(0, 0.012, 0.14), seg=32)(bm)
        # вырез под лицо: убрать переднюю нижнюю часть шапки
        cut = [v for v in bm.verts if v.co.y < -0.04 and v.co.z < 0.165]
        bmesh.ops.delete(bm, geom=cut, context="VERTS")
        sphere(0.15, center=(0, 0.06, 0.06), scale=(1.0, 0.75, 1.25))(bm)  # волосы сзади до плеч
        for x in (-0.09, -0.045, 0.0, 0.045, 0.09):  # чёлка
            cone(0.035, 0.004, 0.11, center=(x, -0.115, 0.255), seg=8,
                 rot=Matrix.Rotation(-0.35, 4, "X"))(bm)
    mesh_obj("Волосы", hair_cap, MAT["hair"], coll, head)

    for s, side in ((1, "L"), (-1, "R")):
        tail = mesh_obj(f"Хвостик.{side}", combine(
            cone(0.055, 0.012, 0.62, center=(0, 0, -0.02), seg=14),
            sphere(0.06, center=(0, 0, -0.03), scale=(1, 1, 0.8))), MAT["hair"], coll, head, (s * 0.15, 0.04, 0.22))
        mesh_obj(f"Резинка.{side}", sphere(0.03, scale=(1, 1, 0.6)), MAT["red"], coll, tail, (0, 0, 0.02))
        drive(tail, "rotation_euler", 1, f"{-s}*(0.25+0.15*sin({BAR}-1.0))")
        drive(tail, "rotation_euler", 0, f"0.2*abs(sin({BEAT}-0.6))")

        sh = mesh_obj(f"Плечо.{side}", combine(limb(0.04, 0.034, 0.24), sphere(0.058, scale=(1, 1, 0.9))),
                      MAT["skin"], coll, torso, (s * 0.15, 0, 0.27))
        mesh_obj(f"Рукав.{side}", sphere(0.06, center=(0, 0, -0.03), scale=(1, 1, 1.2)), MAT["white"], coll, sh)
        ph = "0" if s > 0 else "3.14159"
        drive(sh, "rotation_euler", 1, f"{-s}*(0.35+0.9*(1+sin({BAR}+{ph})))")
        drive(sh, "rotation_euler", 0, f"-0.35*abs(sin({BEAT}))")
        fore = mesh_obj(f"Предплечье.{side}", combine(limb(0.033, 0.028, 0.22), sphere(0.04, (0, 0, -0.25))),
                        MAT["skin"], coll, sh, (0, 0, -0.24))
        drive(fore, "rotation_euler", 0, f"-(0.3+0.9*abs(sin({BEAT}+0.4)))")
    return coll


# ---------------------------------------------------------------- комнаты

def quads(name, faces, mat, coll):
    """Меш из прямоугольников, заданных четырьмя углами."""
    verts, polys = [], []
    for f in faces:
        polys.append(tuple(range(len(verts), len(verts) + 4)))
        verts.extend(f)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], polys)
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    return obj


def portal_plane(name, matrix, coll, w=None, h=None):
    """Плоскость, которую аддон совместит с экраном. Локальная Z — к зрителю."""
    a, b = (w or W) / 2, (h or H) / 2
    obj = quads(name, [[(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0)]],
                MAT["bezel"], coll)
    obj.matrix_world = matrix
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def bezel(name, matrix, coll, margin=6.0, w=None, h=None):
    """Чёрная рамка вокруг проёма в плоскости стекла: если пропорции окна Blender
    не совпали с порталом, по краям видна она, а не пустота вокруг комнаты."""
    a, b = (w or W) / 2, (h or H) / 2
    M = a + margin
    faces = [
        [(-M, b, 0), (M, b, 0), (M, M, 0), (-M, M, 0)],
        [(-M, -M, 0), (M, -M, 0), (M, -b, 0), (-M, -b, 0)],
        [(-M, -b, 0), (-a, -b, 0), (-a, b, 0), (-M, b, 0)],
        [(a, -b, 0), (M, -b, 0), (M, b, 0), (a, b, 0)],
    ]
    obj = quads(name, faces, MAT["bezel"], coll)
    obj.matrix_world = matrix
    return obj


def box_room(prefix, coll, x, y, z, open_side):
    """Комната-коробка нормалями внутрь, без грани open_side."""
    (x0, x1), (y0, y1), (z0, z1) = x, y, z
    walls = {
        "back": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        "front": [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
        "left": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        "right": [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
        "top": [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
    }
    walls.pop(open_side)
    quads(f"{prefix} стены", walls.values(), MAT["walls"], coll)
    quads(f"{prefix} пол", [[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]], MAT["floor"], coll)
    stage = mesh_obj(f"{prefix} подиум", cone(0.55, 0.6, 0.06, center=(0, 0, 0.06), seg=48), MAT["stage"], coll)
    return stage


def orbs(prefix, coll, center, radius, bob):
    """Светящиеся шары по кругу; часть пути — перед стеклом."""
    pivot = bpy.data.objects.new(f"{prefix} орбита", None)
    coll.objects.link(pivot)
    pivot.location = center
    drive(pivot, "rotation_euler", 2, "frame*0.025")
    for i in range(5):
        a = i * 2 * math.pi / 5
        o = mesh_obj(f"{prefix} шар {i}", sphere(0.13 + 0.04 * (i % 2)), MAT[f"orb{i}"], coll, pivot,
                     (radius * math.cos(a), radius * math.sin(a), 0))
        drive(o, "location", 2, f"{bob}*sin(frame*0.05+{i * 1.3:.2f})")


def fireflies(prefix, coll, back, front, n_back, n_front, seed):
    """Светлячки по всему объёму: мелкие, на всех глубинах, часть — перед стеклом.

    back/front — коробки (lo, hi) за стеклом и перед ним. Каждый медленно дрейфует
    по своей фигуре Лиссажу — драйверы с простыми выражениями, без Python.
    """
    rnd = random.Random(seed)
    meshes = []
    for i in range(3):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        for f in bm.faces:
            f.smooth = True
        me = bpy.data.meshes.new(f"{prefix} светлячок {i}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(MAT[f"fly{i}"])
        meshes.append(me)
    boxes = [back] * n_back + [front] * n_front
    for k, (lo, hi) in enumerate(boxes):
        o = bpy.data.objects.new(f"{prefix} светлячок {k}", meshes[k % 3])
        coll.objects.link(o)
        base = [rnd.uniform(lo[a], hi[a]) for a in range(3)]
        o.scale = (rnd.uniform(0.025, 0.05),) * 3
        for a in range(3):
            amp = rnd.uniform(0.15, 0.4)
            freq = rnd.uniform(0.008, 0.025)
            ph = rnd.uniform(0, 6.283)
            drive(o, "location", a, f"{base[a]:.3f}+{amp:.3f}*sin(frame*{freq:.4f}+{ph:.2f})")


def light(coll, name, kind, loc, energy, color=(1, 1, 1), size=0.5, rot=(0, 0, 0), shadow=False):
    """Тени по умолчанию выключены: каждая лампа с тенью заметно роняет FPS вьюпорта EEVEE."""
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = color
    data.use_shadow = shadow
    if kind == "AREA":
        data.size = size
    else:
        data.shadow_soft_size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = loc
    obj.rotation_euler = rot
    coll.objects.link(obj)
    return obj


def setup_scene(scene, dancer):
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 9000
    try:
        scene.sync_mode = "FRAME_DROP"
    except (AttributeError, TypeError):
        pass
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            pass
    scene.eevee.taa_samples = 4
    # Скорость вьюпорта (замерено на встроенной Intel, вьюпорт 2361×1518, «Материалы»):
    # тень только у верхнего света, без ламп в шарах и с дешёвыми тенями — 8 → ~25 fps,
    # с половинным разрешением вьюпорта — ~57 fps. Мягкой тени от площадного света хватает.
    scene.eevee.shadow_resolution_scale = 0.25
    scene.eevee.shadow_ray_count = 1
    scene.eevee.shadow_step_count = 1
    scene.render.preview_pixel_size = "2"
    scene.eevee.taa_render_samples = 32
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    world = bpy.data.worlds.new(f"{scene.name} мир")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world
    scene.collection.children.link(dancer)


def build_side(scene, dancer):
    """Экран — окно в стене: комната за стеклом, проём — передняя грань (y = -DEPTH/2)."""
    coll = scene.collection
    y0, y1 = -DEPTH / 2, DEPTH / 2
    box_room("Сбоку:", coll, (-W / 2, W / 2), (y0, y1), (0, H), "front")
    glass = Matrix.Translation((0, y0, H / 2)) @ Matrix.Rotation(math.pi / 2, 4, "X")
    portal = portal_plane("Портал сбоку", glass, coll)
    bezel("Сбоку: рамка", glass, coll)
    orbs("Сбоку:", coll, (0, y0, 1.3), 1.6, 0.5)
    # перед стеклом — до ~15 см (3.5 единицы) и ближе к центру, чтобы реже обрезались рамкой
    fireflies("Сбоку:", coll, ((-W / 2 + 0.4, y0 + 0.3, 0.3), (W / 2 - 0.4, y1 - 0.4, H - 0.4)),
              ((-W * 0.35, y0 - 3.5, H * 0.15), (W * 0.35, y0 - 0.3, H * 0.85)), 110, 40, seed=1)
    light(coll, "Сбоку: верхний свет", "AREA", (0, 0, H - 0.05), 250, (1, 0.92, 0.95), 2.5, shadow=True)
    light(coll, "Сбоку: голубой", "POINT", (-2.8, 2.0, 2.2), 150, (0.2, 0.7, 1))
    light(coll, "Сбоку: розовый", "POINT", (2.8, 2.0, 2.2), 150, (1, 0.3, 0.7))
    light(coll, "Сбоку: заполняющий", "AREA", (0, y0 + 0.1, 1.5), 60, (1, 1, 1), 3.0,
          rot=(math.pi / 2, 0, 0))
    setup_scene(scene, dancer)
    return portal


def build_top(scene, dancer):
    """Экран — люк в полу: смотришь сверху в колодец, проём — верхняя грань (z = PIT)."""
    coll = scene.collection
    box_room("Сверху:", coll, (-W / 2, W / 2), (-H / 2, H / 2), (0, PIT), "top")
    glass = Matrix.Translation((0, 0, PIT))
    portal = portal_plane("Портал сверху", glass, coll)
    bezel("Сверху: рамка", glass, coll)
    orbs("Сверху:", coll, (0, 0, PIT), 1.5, 0.8)
    fireflies("Сверху:", coll, ((-W / 2 + 0.4, -H / 2 + 0.4, 0.3), (W / 2 - 0.4, H / 2 - 0.4, PIT - 0.3)),
              ((-W * 0.35, -H * 0.35, PIT + 0.3), (W * 0.35, H * 0.35, PIT + 3.5)), 90, 40, seed=2)
    light(coll, "Сверху: верхний свет", "AREA", (0, 0, PIT - 0.1), 300, (1, 0.92, 0.95), 2.5, shadow=True)
    light(coll, "Сверху: голубой", "POINT", (-3.0, 1.5, 1.5), 150, (0.2, 0.7, 1))
    light(coll, "Сверху: розовый", "POINT", (3.0, -1.5, 1.5), 150, (1, 0.3, 0.7))
    setup_scene(scene, dancer)
    return portal


def setup_addon_props(scene, portal):
    if not hasattr(scene, "head_tracking"):
        return
    p = scene.head_tracking
    p.mode = "WINDOW"
    p.portal = portal
    p.hide_overlays = True


def setup_view(scene):
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            # RENDERED Blender при открытии файла сбрасывает в Solid; Material Preview со светом
            # и миром сцены выглядит так же и сохраняется
            space.shading.type = "MATERIAL"
            space.shading.use_scene_lights = True
            space.shading.use_scene_world = True
            r3d = space.region_3d
            r3d.view_perspective = "PERSP"
            r3d.view_location = (0, 0, H / 2)
            r3d.view_rotation = Matrix.Rotation(math.radians(80), 3, "X").to_quaternion()
            r3d.view_distance = 18.0


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # свойства аддона нужны, чтобы прописать портал в сцены
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import head_tracking
    head_tracking.register()
    materials()
    dancer = build_dancer()
    side = bpy.data.scenes[0]
    side.name = "Портал сбоку"
    top = bpy.data.scenes.new("Портал сверху")
    setup_addon_props(side, build_side(side, dancer))
    setup_addon_props(top, build_top(top, dancer))
    for scene in (side, top):
        for obj in scene.objects:
            if obj.name.startswith("Портал"):
                obj.hide_set(True, view_layer=scene.view_layers[0])
    setup_view(side)
    bpy.ops.wm.save_as_mainfile(filepath=OUT)
    print("saved", OUT)


if __name__ == "__main__":  # make_floor_demo.py импортирует отсюда материалы, девочку и т.п.
    main()
