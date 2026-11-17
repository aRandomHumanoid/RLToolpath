"""Test parts a little more complex than the bracket and link: STL + loads.yaml for each.

- hook: an eye at the top, a shank, and a C-shaped hook pulled down in its throat.
- frame: a rounded plate with a window cut out and four bolt holes; bolted at the left
  pair, the right pair pushed down (shear and bending around the window).
- wrench: an open-end spanner; the jaw grips a nut (held) and the hand pushes at the end
  of the handle, with a ring end that is free.
- plate: a wide mounting plate, 40 x 28 mm, four bolt holes and a centre bore; bolted at
  the left pair, a shaft in the bore pushes down (added Nov 16, 2026).

Each primary load is sized so the slicer start's FI_agg is about 1 on configs/printer.yaml.
Random load cases are left to the command line (--load-cases 10 --load-moments 10).

The bracket and link predate this script. Their outlines are kept: their STLs are stretched
to the thickness T and their loads.yaml forces resized the same way (`thicken`).

usage: python examples/make_parts.py [OUT_DIR] [--only plate,hook]  (default: examples/, every part)
"""
import re
import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box

T = 5.0                                     # thickness, mm (25 layers of 0.2)


def rounded_box(x0, y0, x1, y1, r):
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def hook():
    eye = Point(0, 34).buffer(6, quad_segs=24)
    shank = box(-3, 16, 3, 34)
    c = (7.0, 9.0)
    ring = Point(c).buffer(11, quad_segs=32).difference(Point(c).buffer(5.5, quad_segs=32))
    gap = Polygon([c] + [(c[0] + 30 * np.cos(np.radians(a)), c[1] + 30 * np.sin(np.radians(a)))
                         for a in (20, 45, 70, 95)])          # open the hook at the upper right
    body = ring.difference(gap)
    tip = Point(c[0] + 8.25 * np.cos(np.radians(20)), c[1] + 8.25 * np.sin(np.radians(20))).buffer(2.75, quad_segs=16)
    g = shapely.union_all([eye, shank, body, tip])
    g = g.buffer(1.0, quad_segs=8).buffer(-1.0, quad_segs=8)   # fillet the inside corners
    g = g.difference(Point(0, 34).buffer(2.5, quad_segs=24))
    loads = {"supports": [("circle", [0, 34, 3.6])],
             "load": ("circle", [7.0, 3.5, 2.0]), "dir": (0.0, -1.0), "pz": (7.0, 3.5)}
    return g, loads


def frame():
    g = rounded_box(0, 0, 44, 20, 4).difference(rounded_box(11, 6, 33, 14, 2))
    holes = [(4.5, 4.5), (4.5, 15.5), (39.5, 4.5), (39.5, 15.5)]
    g = g.difference(shapely.union_all([Point(h).buffer(1.6, quad_segs=24) for h in holes]))
    loads = {"supports": [("circle", [4.5, 4.5, 2.8]), ("circle", [4.5, 15.5, 2.8])],
             "load": ("box", [36.5, 1.5, 42.5, 18.5]), "dir": (0.0, -1.0), "pz": (39.5, 10.0)}
    return g, loads


def wrench():
    head = Point(0, 0).buffer(9, quad_segs=32)
    jaw = affinity.rotate(box(-1, -4, 12, 4), 15, origin=(0, 0))   # 8 mm across the flats
    handle = LineString([(6, 0), (46, 0)]).buffer(3.5, quad_segs=16)
    ring = Point(50, 0).buffer(6.5, quad_segs=32)
    g = shapely.union_all([head.difference(jaw), handle, ring])
    g = g.buffer(1.0, quad_segs=8).buffer(-1.0, quad_segs=8)
    g = g.difference(Point(50, 0).buffer(3.5, quad_segs=32))
    flats = affinity.rotate(box(-1.5, -5.0, 6.5, 5.0), 15, origin=(0, 0)).difference(
        affinity.rotate(box(-1.5, -3.6, 6.5, 3.6), 15, origin=(0, 0)))
    loads = {"supports": [("polygon", [list(map(float, xy)) for xy in flats.exterior.coords[:-1]])],
             "load": ("circle", [40.0, 0.0, 2.5]), "dir": (0.0, -1.0), "pz": (40.0, 0.0)}
    if flats.geom_type != "Polygon" or len(flats.interiors):          # the jaw band as two strips
        upper = affinity.rotate(box(-1.5, 3.6, 6.5, 5.0), 15, origin=(0, 0))
        lower = affinity.rotate(box(-1.5, -5.0, 6.5, -3.6), 15, origin=(0, 0))
        loads["supports"] = [("polygon", [list(map(float, xy)) for xy in s.exterior.coords[:-1]])
                             for s in (upper, lower)]
    return g, loads


def plate():
    """A wide part: most of it lies more than 2 mm from any edge (the others are mostly narrow
    members), so whether material gathers along edges or in the middle matters."""
    g = rounded_box(0, 0, 40, 28, 4)
    holes = [(5, 5), (5, 23), (35, 5), (35, 23)]
    g = g.difference(shapely.union_all([Point(h).buffer(1.6, quad_segs=24) for h in holes]
                                       + [Point(22, 14).buffer(4.0, quad_segs=32)]))
    loads = {"supports": [("circle", [5, 5, 2.8]), ("circle", [5, 23, 2.8])],
             "load": ("circle", [22, 14, 5.0]), "dir": (0.0, -1.0), "pz": (22.0, 14.0)}
    return g, loads


def region_yaml(kind, v):
    if kind == "polygon":
        return "{polygon: [" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in v) + "]}"
    return "{" + kind + ": [" + ", ".join(f"{x:g}" for x in v) + "]}"


def loads_yaml(name, doc, ld, F):
    fx, fy = F * ld["dir"][0], F * ld["dir"][1]
    sup = "\n".join(f"  - region: {region_yaml(k, v)}" for k, v in ld["supports"])
    px, py = ld["pz"]
    return (f"# {doc}\n# Generated by examples/make_parts.py; the force puts the slicer start's FI_agg near 1.\n"
            f"supports:\n{sup}\nloads:\n  - region: {region_yaml(*ld['load'])}\n"
            f"    force: [{fx:.2f}, {fy:.2f}]                 # N on the whole part\n"
            f"point_loads:\n  - pos: [{px:g}, {py:g}, {T:g}]\n"
            f"    force: [{fx:.2f}, {fy:.2f}, {round(4.0 / 25.0 * F, 2):g}]\n")


PARTS = {"hook": (hook, "Hook: held by a pin in the eye, pulled down in the throat of the hook."),
         "frame": (frame, "Window frame: bolted at the two left holes, the right end pushed down."),
         "wrench": (wrench, "Open-end wrench: the jaw holds a nut, the hand pushes the handle down."),
         "plate": (plate, "Mounting plate: bolted at the two left holes, a shaft in the centre bore pushes down.")}


def fi_at(stl, loads_path):
    """The slicer start's FI_agg on configs/printer.yaml (primary case)."""
    from rltp.config import PartSpec, load_loads, load_printer
    from rltp.io.mesh import load_mesh
    from rltp.part import build_part
    pr = load_printer(Path(__file__).resolve().parents[1] / "configs" / "printer.yaml")
    spec = PartSpec(pr, load_loads(loads_path)[0], terms=("strength", "time", "cont"))
    return build_part(load_mesh(stl), spec, baseline="zigzag").breakdown()["fi_agg"]


def main(out: Path, only=None):
    for name, (fn, doc) in PARTS.items():
        if only and name not in only:
            continue
        g, ld = fn()
        assert g.geom_type == "Polygon" and g.is_valid, name
        mesh = trimesh.creation.extrude_polygon(g, T)
        mesh.apply_translation([-mesh.bounds[0, 0], -mesh.bounds[0, 1], 0])
        dx, dy = -float(g.bounds[0]), -float(g.bounds[1])
        for i, (k, v) in enumerate(ld["supports"]):
            ld["supports"][i] = (k, shift(k, v, dx, dy))
        ld["load"] = (ld["load"][0], shift(*ld["load"], dx, dy))
        ld["pz"] = (ld["pz"][0] + dx, ld["pz"][1] + dy)
        assert mesh.is_watertight, name
        stl = out / f"{name}.stl"
        mesh.export(stl)
        lp = out / f"{name}_loads.yaml"
        F = 100.0
        lp.write_text(loads_yaml(name, doc, ld, F), encoding="utf-8")
        fi = fi_at(stl, lp)
        F = round(F / fi, 1)
        lp.write_text(loads_yaml(name, doc, ld, F), encoding="utf-8")
        print(f"{name}: area {g.area:.0f} mm^2, {mesh.bounds[1, 0]:.1f} x {mesh.bounds[1, 1]:.1f} mm, "
              f"{len(g.interiors)} holes, F {F} N (FI_agg {fi:.3f} at 100 N)")
    for name in ("bracket", "link"):
        if not only or name in only:
            thicken(out, name)


def thicken(out: Path, name: str):
    """Stretch an older part (examples/NAME.stl) to thickness T, move its point loads to the
    new top face and scale every force in its loads.yaml so the slicer start's FI_agg is
    about 1; the yaml's comments stay."""
    src = Path(__file__).resolve().parent
    mesh = trimesh.load(src / f"{name}.stl")
    z0, z1 = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    mesh.apply_translation([0, 0, -z0])
    mesh.apply_transform(np.diag([1.0, 1.0, T / (z1 - z0), 1.0]))
    assert mesh.is_watertight, name
    stl, lp = out / f"{name}.stl", out / f"{name}_loads.yaml"
    mesh.export(stl)
    text = re.sub(r"(pos: \[[^,\]]+,[^,\]]+,)[^\]]+\]", lambda m: f"{m.group(1)} {T:g}]",
                  (src / f"{name}_loads.yaml").read_text(encoding="utf-8"))
    lp.write_text(text, encoding="utf-8")
    first = re.search(r"force: \[([^\]]*)\]", text).group(1).split(",")
    F0 = float(np.hypot(float(first[0]), float(first[1])))
    fi = fi_at(stl, lp)
    k = round(F0 / fi, 1) / F0
    text = re.sub(r"force: \[([^\]]*)\]",
                  lambda m: "force: [" + ", ".join(f"{k * float(v) + 0.0:.2f}" for v in m.group(1).split(",")) + "]",
                  text)
    lp.write_text(text, encoding="utf-8")
    print(f"{name}: {z1 - z0:g} -> {T:g} mm, F {F0:.1f} -> {k * F0:.1f} N (FI_agg {fi:.3f} at {F0:.1f} N)")


def shift(kind, v, dx, dy):
    if kind == "circle":
        return [round(v[0] + dx, 3), round(v[1] + dy, 3), v[2]]
    if kind == "box":
        return [round(v[0] + dx, 3), round(v[1] + dy, 3), round(v[2] + dx, 3), round(v[3] + dy, 3)]
    return [[x + dx, y + dy] for x, y in v]


if __name__ == "__main__":
    a = sys.argv[1:]
    only = None
    if "--only" in a:
        i = a.index("--only")
        only = set(a[i + 1].split(","))
        a = a[:i] + a[i + 2:]
    main(Path(a[0]) if a else Path(__file__).resolve().parent, only)
