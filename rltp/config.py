"""Part spec: printer profile, material constants, bond model, loads and score settings.

All lengths are mm, forces N, stresses MPa, times s.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import yaml
from shapely.geometry import Point, Polygon, box as sbox

TERMS = ("strength", "time", "cont", "inter", "infill", "cover")
INFILL_RULES = ("both", "band", "penalty", "free")


def _from_dict(cls, d: dict | None):
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


@dataclass
class Material:
    """Orthotropic lamina constants from printed coupons (defaults: generic PLA)."""
    E1: float = 3300.0   # along the bead
    E2: float = 2600.0   # across beads, in plane
    G12: float = 1000.0
    nu12: float = 0.33
    Xt: float = 50.0
    Xc: float = 65.0
    Yt: float = 28.0
    Yc: float = 55.0
    S: float = 30.0
    Zt: float = 22.0     # interlayer tensile strength (v1.5)
    Sz: float = 26.0     # interlayer shear strength (v1.5)


@dataclass
class BondModel:
    """q = max(phi(dt) * s, q_min), phi(dt) = 1 / (1 + (dt / tau)^n)."""
    tau: float = 30.0
    n: float = 2.0
    s_end: float = 0.7
    q_min: float = 0.05


@dataclass
class Printer:
    name: str = "generic-0.4"
    nozzle_diameter: float = 0.4
    bead_width: float = 0.4          # cell pitch p
    filament_diameter: float = 1.75
    layer_height: float = 0.2
    first_layer_height: float = 0.2
    print_speed: float = 40.0
    first_layer_speed: float = 20.0
    travel_speed: float = 150.0
    accel: float = 1500.0
    travel_accel: float = 3000.0
    junction_deviation: float = 0.013
    retract_length: float = 0.8
    retract_speed: float = 35.0
    z_hop: float = 0.0
    z_speed: float = 10.0
    layer_change_time: float = 0.5
    # Shell + infill. All at their defaults, every layer is fully filled (the original mode).
    walls: int = 0                   # wall rings (bead widths) along every outline, kept as fixed loops
    solid_layers: int = 0            # layers of solid skin at every up- and down-facing surface
    infill: float = 1.0              # target density of the optional cells (0.4 = remove 60% of the inside)
    infill_rule: str = "both"        # band: stay within +-infill_band of the target; penalty: score term
                                     # with weight infill_weight; both: penalty inside a hard band;
                                     # free: the target only sets the start
    infill_band: float = 0.02        # band rule: allowed deviation of the part's infill fraction
    infill_weight: float = 1.0       # penalty rule: cost of 1 percentage point off target, in % of a term
    max_bridge: float = 4.0          # mm: longest unsupported span between filled cells of the layer below
    fea_element: float = 0.0         # mm: strength-analysis element size; 0 = one element per cell
    cell: float = 0.0                # mm: grid cell; below bead_width, beads sit anywhere on a finer
                                     # grid (rltp.cover.beads); 0 = one cell per bead width
    cover_weight: float = 1.0        # fine grid: weight of the gap/over-extrusion term
    start: str = "slicer"            # starting layout: "slicer" (fixed directions), "stress" (beads
                                     # along each area's main stress, 0/90; rltp.eval.stressmap),
                                     # "stress4" (fine grid: 45/135-degree areas too), "gradient"
                                     # (bead-direction and infill-density fields optimised for in-plane
                                     # strength, laid out 0/90; rltp.search.fields) or "gradient4"
                                     # (fine grid: 45/135 too)
    infill_pattern: str = "alternate"  # starting infill lines: alternate 0/90 by layer, or stacked (one
                                       # direction, so adjacent layers' lines overlap); skins always alternate
    bed_center: tuple = (110.0, 110.0)
    nozzle_temp: float = 210.0
    bed_temp: float = 60.0
    fan_speed: int = 255
    start_gcode: str = (
        "G28 ; home\n"
        "M140 S{bed_temp}\nM104 S{nozzle_temp}\nM190 S{bed_temp}\nM109 S{nozzle_temp}\n"
        "G92 E0\nG1 Z0.3 F600\nG1 X5 Y5 F6000\nG1 X5 Y100 E10 F1200 ; purge line\nG92 E0\n"
    )
    end_gcode: str = "M107\nM104 S0\nM140 S0\nG91\nG1 Z10 F600\nG90\nM84\n"
    material: Material = field(default_factory=Material)
    bond: BondModel = field(default_factory=BondModel)

    @property
    def p(self) -> float:
        return self.bead_width

    @property
    def kb(self) -> int:
        """Grid cells per bead width (1 unless `cell` is smaller than the bead)."""
        if 0 < self.cell < self.bead_width - 1e-9:
            return max(int(round(self.bead_width / self.cell)), 1)
        return 1

    @property
    def pitch(self) -> float:
        """Grid cell size (mm)."""
        return self.bead_width / self.kb

    @property
    def sparse(self) -> bool:
        """Walls, solid skins and optional infill instead of fully filled layers (the fine
        bead grid always uses this machinery)."""
        return self.walls > 0 or self.solid_layers > 0 or self.infill < 1.0 or self.kb > 1

    @staticmethod
    def from_dict(d: dict | None) -> "Printer":
        d = dict(d or {})
        mat = _from_dict(Material, d.pop("material", None))
        bond = _from_dict(BondModel, d.pop("bond", None))
        pr = _from_dict(Printer, d)
        pr.material, pr.bond = mat, bond
        pr.bed_center = tuple(pr.bed_center)
        return pr


# ---------------------------------------------------------------- regions

@dataclass
class Region:
    """A 2D region in world XY, optionally limited to a z range.

    Forms (exactly one): box [xmin, ymin, xmax, ymax], circle [x, y, r],
    polygon [[x, y], ...], or side {xmin|xmax|ymin|ymax} with `width` (mm),
    measured from the part's XY bounds.
    """
    box: list | None = None
    circle: list | None = None
    polygon: list | None = None
    side: str | None = None
    width: float = 1.0
    zmin: float = -np.inf
    zmax: float = np.inf
    _geom: Any = field(default=None, repr=False, compare=False)

    def resolve(self, bounds) -> "Region":
        """bounds = (xmin, ymin, xmax, ymax) of the part."""
        eps = 1e-6
        if self.box is not None:
            g = sbox(*self.box)
        elif self.circle is not None:
            x, y, r = self.circle
            g = Point(x, y).buffer(r, quad_segs=32)
        elif self.polygon is not None:
            g = Polygon(self.polygon)
        elif self.side is not None:
            x0, y0, x1, y1 = bounds
            big = 1e3
            w = self.width
            g = {
                "xmin": sbox(x0 - big, y0 - big, x0 + w, y1 + big),
                "xmax": sbox(x1 - w, y0 - big, x1 + big, y1 + big),
                "ymin": sbox(x0 - big, y0 - big, x1 + big, y0 + w),
                "ymax": sbox(x0 - big, y1 - w, x1 + big, y1 + big),
            }[self.side]
        else:
            raise ValueError("region needs one of box, circle, polygon, side")
        self._geom = g.buffer(eps)
        shapely.prepare(self._geom)
        return self

    def contains(self, x, y, z: float | None = None) -> np.ndarray:
        if self._geom is None:
            raise RuntimeError("region not resolved")
        inside = shapely.contains_xy(self._geom, np.asarray(x, float), np.asarray(y, float))
        if z is not None and not (self.zmin <= z <= self.zmax):
            inside = np.zeros_like(inside)
        return inside

    @staticmethod
    def from_dict(d) -> "Region":
        d = dict(d)
        if "region" in d:  # supports may be written like loads: {region: {...}, zmin: ...}
            d = {**d.pop("region"), **d}
        return _from_dict(Region, d)


@dataclass
class Load2D:
    region: Region
    force: tuple  # total (Fx, Fy) on the part, split across layers by h / height


@dataclass
class PointLoad:
    pos: tuple    # (x, y, z) world
    force: tuple  # (Fx, Fy, Fz)


@dataclass
class LoadSpec:
    """Supports and 2D load cases. `loads` is the primary case. `cases` holds extra
    cases given explicitly, and `random_cases` asks build_part for that many random
    ones (stored in `generated`). Every case uses the same supports. With more than
    one case, the strength term scores each case against its own baseline and takes
    a p-norm over all of them (see PartState)."""
    supports: list = field(default_factory=list)      # list[Region]
    loads: list = field(default_factory=list)         # list[Load2D]: the primary case
    point_loads: list = field(default_factory=list)   # list[PointLoad] (v1.5)
    split_by_height: bool = True                      # v0 prismatic load share
    cases: list = field(default_factory=list)         # extra cases: list[list[Load2D]]
    random_cases: int = 0                             # random extra cases built per part
    random_seed: int = 0
    generated: list = field(default_factory=list)     # the random cases, set by add_random_cases

    @property
    def all_cases(self) -> list:
        """Case 0 is the primary `loads` (possibly empty), then explicit, then random cases."""
        return [self.loads] + [c for c in self.cases if c] + [c for c in self.generated if c]

    @property
    def n_cases(self) -> int:
        return len(self.all_cases)

    @property
    def has_2d(self) -> bool:
        return bool(self.supports) and (bool(self.loads) or any(self.cases) or self.random_cases > 0)

    def resolve(self, bounds) -> "LoadSpec":
        for r in self.supports:
            r.resolve(bounds)
        for case in [self.loads] + self.cases + self.generated:
            for ld in case:
                ld.region.resolve(bounds)
        return self

    def add_random_cases(self, geom, p: float, bounds) -> list:
        """Build `random_cases` random load cases for a part outline `geom`: each is
        one disc load (radius 4 cells) on the outline, holes included, placed away
        from the supports, pushing in a uniformly random in-plane direction. The force
        equals the primary case's total; it only sets units, since each case is scored
        against its own baseline. Deterministic in `random_seed`."""
        self.generated = []
        if self.random_cases <= 0:
            return self.generated
        rng = np.random.default_rng(self.random_seed)
        mag = float(sum(np.hypot(*ld.force[:2]) for ld in self.loads)) or 1.0
        r = 4 * p
        line = geom.boundary
        sup = shapely.union_all([s._geom for s in self.supports]) if self.supports else None
        for _ in range(self.random_cases):
            best = None
            for _try in range(64):
                pt = line.interpolate(float(rng.uniform(0, line.length)))
                dist = sup.distance(pt) if sup is not None else np.inf
                if best is None or dist > best[0]:
                    best = (dist, pt)
                if dist > r + 2 * p:
                    break
            pt = best[1]
            ang = float(rng.uniform(0, 2 * np.pi))
            reg = Region(circle=[float(pt.x), float(pt.y), r]).resolve(bounds)
            self.generated.append([Load2D(reg, (mag * np.cos(ang), mag * np.sin(ang)))])
        return self.generated

    def describe_cases(self) -> list:
        """JSON-friendly list of every case (for metrics.json)."""
        def reg(r: Region):
            for k in ("box", "circle", "polygon"):
                v = getattr(r, k)
                if v is not None:
                    return {k: [list(map(float, x)) if isinstance(x, (list, tuple)) else float(x) for x in v]}
            return {"side": r.side, "width": r.width}
        return [[{"region": reg(ld.region), "force": [float(f) for f in ld.force]} for ld in case]
                for case in self.all_cases]

    @staticmethod
    def from_dict(d: dict | None) -> "LoadSpec":
        d = dict(d or {})
        sup = [Region.from_dict(r) for r in d.get("supports", [])]

        def loads(xs):
            if isinstance(xs, dict):     # a case written as a single load
                xs = [xs]
            return [Load2D(Region.from_dict(x["region"]), tuple(x["force"])) for x in xs]

        lds = loads(d.get("loads", []))
        pls = [PointLoad(tuple(x["pos"]), tuple(x["force"])) for x in d.get("point_loads", [])]
        cases = [loads(c) for c in d.get("load_cases", [])]
        return LoadSpec(sup, lds, pls, bool(d.get("split_by_height", True)), cases=cases,
                        random_cases=int(d.get("random_cases", 0)), random_seed=int(d.get("random_seed", 0)))


@dataclass
class ScoreConfig:
    p_norm: float = 8.0
    c_stroke: float = 1.0      # per stroke (one retraction each)
    c_travel: float = 0.05     # per cell pitch of travel
    c_boundary_end: float = 0.5
    sigma_floor: float = 1e-6
    norm: str = "relative"     # term scales: "relative" (to the part's baseline) or "std" (per-edit std)
    stack: str = "auto"        # in-plane stack model: "split", "laminate", or "auto" (laminate when sparse)
    fea_refine: int = 1        # FEA elements per cell side (toolpath cells stay one bead wide)
    solver: str = "fast"       # FEA linear solver: "fast" (symmetric LU, CG to 1e-7) or "legacy"


@dataclass
class PartSpec:
    printer: Printer = field(default_factory=Printer)
    loads: LoadSpec = field(default_factory=LoadSpec)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    part_height: float = 1.0
    terms: tuple = ("strength", "time", "cont")   # enabled score terms

    @property
    def material(self) -> Material:
        return self.printer.material

    @property
    def bond(self) -> BondModel:
        return self.printer.bond


def load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_printer(path) -> Printer:
    return Printer.from_dict(load_yaml(path))


def load_loads(path) -> tuple[LoadSpec, dict]:
    """Returns the load spec plus optional overrides (material, bond) from loads.yaml."""
    d = load_yaml(path)
    extra = {k: d.pop(k) for k in ("material", "bond") if k in d}
    return LoadSpec.from_dict(d), extra


def printer_to_dict(pr: Printer) -> dict:
    d = asdict(pr)
    d["bed_center"] = list(pr.bed_center)
    return d


def parse_weights(s: str | None, n: int = len(TERMS)) -> np.ndarray:
    """'1.0,0.3,0.3' -> [1.0, 0.3, 0.3, 0.0, 0.0]: strength, time, continuity, interlayer
    (v1.5), infill target penalty (sparse layers, usually set from infill_weight)."""
    if s is None:
        w = [1.0, 0.3, 0.3, 0.0]
    else:
        w = [float(x) for x in str(s).split(",")]
    w = (w + [0.0] * n)[:n]
    return np.asarray(w, dtype=np.float64)
