"""
Демо «как у Portalgraph»: содержимое стоит прямо на стекле, а не прячется в глубине за ним.

Точки в плоскости экрана от ошибок трекера не сдвигаются вовсе, поэтому ноги девочки,
подиум и пол стоят намертво, а дрожит и отстаёт только то, что выше, и то несильно.
Девочка, материалы, светлячки и шары берутся из make_portal_demo.py.

Две сцены:
- «Экран лёжа» — монитор (или ноутбук) лежит на столе, экран — это пол сцены, девочка
  стоит на нём и торчит вверх. Ближе всего к Portalgraph. Вебку ставят отдельно у дальнего
  края экрана и в аддоне задают её наклон и высоту;
- «Экран стоит» — обычный монитор: пол на уровне нижней кромки экрана, девочка в 3–4 см
  за стеклом, комната неглубокая.

Запуск:
    blender --background --factory-startup --python make_floor_demo.py -- portal_floor_demo.blend
"""
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_portal_demo as pd  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "portal_floor_demo.blend"

# Проём 16:9. На мониторе 34.5 см 1 единица ≈ 5.75 см: девочка (1.4) — около 8 см.
W, H = 6.0, 3.375
STAND_GAP = 0.6  # «Экран стоит»: на сколько единиц девочка за стеклом
STAND_DEPTH = 2.6  # и глубина комнаты
STAND_LIFT = 0.45  # пол выше нижней кромки экрана, чтобы ноги не упирались в край


def crystals(prefix, coll, spots, seed):
    """Светящиеся кристаллы на полу: стоят в плоскости стекла — неподвижные опорные точки."""
    rnd = random.Random(seed)
    for i, (x, y) in enumerate(spots):
        h = rnd.uniform(0.35, 0.75)
        r = rnd.uniform(0.1, 0.17)
        o = pd.mesh_obj(f"{prefix} кристалл {i}",
                        pd.combine(pd.cone(0.0, r, h * 0.35, center=(0, 0, h), seg=4),
                                   pd.cone(r, r * 0.4, h * 0.65, center=(0, 0, h * 0.65), seg=4)),
                        pd.MAT[f"orb{i % 5}"], coll, loc=(x, y, 0), smooth=False)
        pd.drive(o, "rotation_euler", 2, f"frame*{rnd.uniform(0.01, 0.03):.3f}+{rnd.uniform(0, 6.28):.2f}")


def build_lying(scene, dancer):
    """Экран — пол. Портал лежит в z = 0, локальная Y — к дальнему краю экрана."""
    coll = scene.collection
    glass = Matrix.Identity(4)
    portal = pd.portal_plane("Портал лёжа", glass, coll, W, H)
    pd.bezel("Лёжа: рамка", glass, coll, w=W, h=H)
    # пол чуть ниже стекла и ровно по проёму: за пределами проёма — рамка
    pd.quads("Лёжа: пол", [[(-W / 2, -H / 2, -0.003), (W / 2, -H / 2, -0.003), (W / 2, H / 2, -0.003),
                            (-W / 2, H / 2, -0.003)]], pd.MAT["floor"], coll)
    pd.mesh_obj("Лёжа: подиум", pd.cone(0.55, 0.6, 0.06, center=(0, 0, 0.06), seg=48), pd.MAT["stage"], coll)
    crystals("Лёжа:", coll, [(-2.3, -1.0), (2.2, -1.1), (-2.1, 1.1), (2.4, 1.0), (-0.9, 1.25), (1.1, -1.3),
                              (0.2, 1.35), (-1.5, -0.2)], seed=3)
    pd.orbs("Лёжа:", coll, (0, 0, 1.1), 1.3, 0.35)
    pd.fireflies("Лёжа:", coll, ((-W * 0.42, -H * 0.42, 0.15), (W * 0.42, H * 0.42, 0.9)),
                 ((-W * 0.35, -H * 0.35, 0.9), (W * 0.35, H * 0.35, 3.0)), 35, 55, seed=4)
    pd.light(coll, "Лёжа: верхний свет", "AREA", (0, 0, 4.0), 150, (1, 0.92, 0.95), 3.0, shadow=True)
    pd.light(coll, "Лёжа: голубой", "POINT", (-2.5, 1.5, 1.5), 60, (0.2, 0.7, 1))
    pd.light(coll, "Лёжа: розовый", "POINT", (2.5, -1.0, 1.5), 60, (1, 0.3, 0.7))
    pd.setup_scene(scene, dancer)
    return portal


def build_standing(scene, dancer):
    """Обычный монитор: нижняя кромка экрана — передний край пола, комната неглубокая."""
    coll = scene.collection
    y0, y1 = -STAND_GAP, STAND_DEPTH - STAND_GAP
    pd.box_room("Стоя:", coll, (-W / 2, W / 2), (y0, y1), (0, H - STAND_LIFT), "front")
    glass = Matrix.Translation((0, y0, H / 2 - STAND_LIFT)) @ Matrix.Rotation(math.pi / 2, 4, "X")
    # передняя кромка подиума — ровно в плоскости стекла, под полом
    pd.quads("Стоя: кромка", [[(-W / 2, y0, -STAND_LIFT), (W / 2, y0, -STAND_LIFT), (W / 2, y0, 0),
                               (-W / 2, y0, 0)]], pd.MAT["stage"], coll)
    portal = pd.portal_plane("Портал стоя", glass, coll, W, H)
    pd.bezel("Стоя: рамка", glass, coll, w=W, h=H)
    crystals("Стоя:", coll, [(-2.2, y0 + 0.4), (2.3, y0 + 0.5), (-2.4, y1 - 0.5), (2.2, y1 - 0.4),
                              (-1.2, y0 + 0.3), (1.3, y1 - 0.3)], seed=5)
    pd.orbs("Стоя:", coll, (0, y0, 1.2), 1.3, 0.35)
    pd.fireflies("Стоя:", coll, ((-W / 2 + 0.3, y0 + 0.2, 0.2), (W / 2 - 0.3, y1 - 0.2, H - 0.3)),
                 ((-W * 0.35, y0 - 1.5, H * 0.2), (W * 0.35, y0 - 0.2, H * 0.8)), 70, 25, seed=6)
    pd.light(coll, "Стоя: верхний свет", "AREA", (0, 0.4, H - 0.05), 120, (1, 0.92, 0.95), 2.5, shadow=True)
    pd.light(coll, "Стоя: голубой", "POINT", (-2.3, y1 - 0.4, 2.0), 60, (0.2, 0.7, 1))
    pd.light(coll, "Стоя: розовый", "POINT", (2.3, y1 - 0.4, 2.0), 60, (1, 0.3, 0.7))
    pd.light(coll, "Стоя: заполняющий", "AREA", (0, y0 + 0.1, 1.2), 30, (1, 1, 1), 2.5,
             rot=(math.pi / 2, 0, 0))
    pd.setup_scene(scene, dancer)
    return portal


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import head_tracking
    head_tracking.register()
    pd.materials()
    dancer = pd.build_dancer()
    lying = bpy.data.scenes[0]
    lying.name = "Экран лёжа"
    standing = bpy.data.scenes.new("Экран стоит")
    pd.setup_addon_props(lying, build_lying(lying, dancer))
    pd.setup_addon_props(standing, build_standing(standing, dancer))
    for scene in (lying, standing):
        for obj in scene.objects:
            if obj.name.startswith("Портал"):
                obj.hide_set(True, view_layer=scene.view_layers[0])
    pd.setup_view(lying)
    bpy.ops.wm.save_as_mainfile(filepath=OUT)
    print("saved", OUT)


main()
