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
from shapely.geometry import LineString, Point, Polygon, box as sbox

TERMS = ("strength", "time", "cont", "inter", "infill", "cover", "stiff", "bulk")
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
    """q = max(phi(dt) * s, q_min), phi(dt) = 1 / (1 + (dt / tau)^n). fi: "quadratic" scores
    an interface cell by (sigma / q Zt)^2 + (tau / q Sz)^2 (1 at failure, quadratic in load);
    "ratio" by its square root, the inverse strength ratio, linear in load like the in-plane
    FI, so the p-norm weighs Z peaks as it weighs in-plane ones."""
    tau: float = 30.0
    n: float = 2.0
    s_end: float = 0.7
    q_min: float = 0.05
    fi: str = "quadratic"


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
                                     # strength, laid out 0/90; rltp.search.fields), "gradient4"
                                     # (fine grid: 45/135 too) or "gradient16" (true directions,
                                     # ScoreConfig.directions = "path": every 11.25 degrees)
    infill_pattern: str = "alternate"  # starting infill lines: alternate 0/90 by layer, or stacked (one
                                       # direction, so adjacent layers' lines overlap); skins always alternate
    infill_lines: str = "cut"          # gradient starts: infill lines start and stop with the density
                                       # field (ends anchored by T-joints), or "whole": each run whole
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
    polygon [[x, y], ...], side {xmin|xmax|ymin|ymax} with `width` (mm),
    measured from the part's XY bounds, or edge [[x, y], ...]: a polyline (one edge
    of the outline, see rltp.slicing.edges) and the points within `width` of it.
    """
    box: list | None = None
    circle: list | None = None
    polygon: list | None = None
    side: str | None = None
    edge: list | None = None
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
        elif self.edge is not None:
            g = LineString(self.edge).buffer(self.width, cap_style="flat")   # a closed edge gives a band
        else:
            raise ValueError("region needs one of box, circle, polygon, side, edge")
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
    force: tuple         # total (Fx, Fy) on the part, split across layers by h / height
    moment: float = 0.0  # total moment about +z (N mm), as forces about the region's loaded nodes


@dataclass
class PointLoad:
    pos: tuple    # (x, y, z) world
    force: tuple  # (Fx, Fy, Fz)


RANDOM_MODES = ("edge", "disc")


@dataclass
class LoadSpec:
    """Supports and 2D load cases. `loads` is the primary case. `cases` holds extra
    cases given explicitly, and `random_cases` asks build_part for that many random
    ones (stored in `generated`). The primary and explicit cases use `supports`. A
    random case of the default `edge` mode has its own (`generated_supports`); a `disc`
    case uses `supports`. With more than one case, the strength term scores each case
    against its own baseline and takes a p-norm over all of them (see PartState).

    Without a primary case (no `loads`), the cases are only the explicit and random ones:
    the load-agnostic mode, where no load is special."""
    supports: list = field(default_factory=list)      # list[Region]
    loads: list = field(default_factory=list)         # list[Load2D]: the primary case
    point_loads: list = field(default_factory=list)   # list[PointLoad] (v1.5)
    split_by_height: bool = True                      # v0 prismatic load share
    cases: list = field(default_factory=list)         # extra cases: list[list[Load2D]]
    random_cases: int = 0                             # random force cases built per part
    random_moments: int = 0                           # random moment cases (edge mode)
    random_seed: int = 0
    random_mode: str = "edge"                         # edge or disc: see add_random_cases
    # edge mode: relative odds of the edge kind (hole, side, arc) a random case fixes or
    # loads, then an edge of that kind by length; None picks every edge alike (before Nov 15, 2026)
    random_weights: tuple | None = (3.0, 2.0, 1.0)
    generated: list = field(default_factory=list)     # the random cases, set by add_random_cases
    generated_supports: list = field(default_factory=list)   # per random case: list[Region], or None = `supports`

    @property
    def has_primary(self) -> bool:
        """Whether case 0 is the primary `loads`. Without them it is left out whenever other
        cases exist (load-agnostic: every case comes from the same random draw)."""
        return bool(self.loads) or not ([c for c in self.cases if c] or [c for c in self.generated if c])

    @property
    def all_cases(self) -> list:
        """The primary `loads` (see has_primary), then explicit, then random cases."""
        return (([self.loads] if self.has_primary else [])
                + [c for c in self.cases if c] + [c for c in self.generated if c])

    @property
    def case_supports(self) -> list:
        """Each case's supports, in all_cases order. Cases that share supports share the list
        object, which is how the FEA groups them into one stiffness matrix."""
        own = self.generated_supports + [None] * (len(self.generated) - len(self.generated_supports))
        return (([self.supports] if self.has_primary else []) + [self.supports for c in self.cases if c]
                + [s if s is not None else self.supports for c, s in zip(self.generated, own) if c])

    @property
    def n_cases(self) -> int:
        return len(self.all_cases)

    @property
    def has_2d(self) -> bool:
        n = self.random_cases + self.random_moments
        shared = bool(self.supports) and (bool(self.loads) or any(self.cases) or n > 0)
        return shared or (n > 0 and self.random_mode == "edge")

    def resolve(self, bounds) -> "LoadSpec":
        for r in self.supports:
            r.resolve(bounds)
        for case in [self.loads] + self.cases + self.generated:
            for ld in case:
                ld.region.resolve(bounds)
        for sups in self.generated_supports:
            for r in sups or []:
                r.resolve(bounds)
        return self

    def add_random_cases(self, geom, p: float, bounds) -> list:
        """Build `random_cases` random force cases and `random_moments` random moment cases
        for a part outline `geom`, deterministic in `random_seed`. A force case pushes in a
        uniformly random in-plane direction with the primary case's total force (1 N without
        one); a moment case twists, either way, by that force times 10 mm. The size only sets
        units, since each case is scored against its own baseline.

        edge (the default): the outline is split into edges (rltp.slicing.edges: straight
        sides, arcs, round holes), keeping those at least 2 bead widths `p` long. Each case
        fixes one random edge and loads a different one. Both regions are the points within
        one bead width of their edge. A moment on an edge is a couple: forces about the
        loaded nodes' centroid, bending a straight edge and twisting a hole. Each edge is
        drawn in two steps: its kind, with odds `random_weights` (hole, side, arc) over the
        kinds the part has, as parts are mostly held and loaded at bolt holes and flat
        faces; then an edge of that kind in proportion to its length, so short fillets
        are rarely picked. `random_weights` None draws every edge alike.

        disc (force cases only): one disc load, 4 bead widths in radius, on the outline (holes included) and
        away from the shared supports."""
        self.generated, self.generated_supports = [], []
        if self.random_cases <= 0 and self.random_moments <= 0:
            return self.generated
        if self.random_mode not in RANDOM_MODES:
            raise ValueError(f"unknown random_mode {self.random_mode!r}; expected one of {RANDOM_MODES}")
        if self.random_mode == "disc" and self.random_moments > 0:
            raise ValueError("random moments need random_mode: edge")
        if self.random_mode == "disc" and not self.supports:
            raise ValueError("random_mode disc loads the outline against the loads.yaml supports; "
                             "without supports use random_mode: edge")
        rng = np.random.default_rng(self.random_seed)
        mag = float(sum(np.hypot(*ld.force[:2]) for ld in self.loads)) or 1.0
        if self.random_mode == "disc":
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
                self.generated_supports.append(None)
            return self.generated
        from rltp.slicing.edges import EDGE_KINDS, edge_kind, outline_edges
        edges = [e for e in outline_edges(geom) if LineString(e).length >= 2 * p]
        if len(edges) < 2:
            raise ValueError(f"random edge cases need two outline edges at least {2 * p:g} mm long, "
                             f"and this part has {len(edges)}; use random_mode: disc")
        regions = {}

        def region(i):        # one Region per edge, so cases fixing the same edge share supports
            if i not in regions:
                regions[i] = Region(edge=np.round(edges[i], 4).tolist(), width=p).resolve(bounds)
            return regions[i]

        if self.random_weights is None:
            def pick():
                s = int(rng.integers(len(edges)))
                ld = int(rng.integers(len(edges) - 1))
                return s, ld + (ld >= s)
        else:
            wk = dict(zip(EDGE_KINDS, map(float, self.random_weights)))
            if len(self.random_weights) != len(EDGE_KINDS) or min(wk.values()) < 0:
                raise ValueError(f"random_weights: three odds >= 0 for {', '.join(EDGE_KINDS)}; "
                                 f"got {self.random_weights}")
            kind = [edge_kind(e, geom) for e in edges]
            length = [LineString(e).length for e in edges]

            def draw(skip=None):
                idx = [i for i in range(len(edges)) if i != skip and wk[kind[i]] > 0]
                kinds = [k for k in EDGE_KINDS if any(kind[i] == k for i in idx)]
                if not kinds:
                    raise ValueError(f"random_weights {self.random_weights} leave no edge of this part "
                                     f"({', '.join(sorted(set(kind)))}) to fix and another to load")
                odds = np.array([wk[k] for k in kinds])
                k = kinds[int(rng.choice(len(kinds), p=odds / odds.sum()))]
                cand = [i for i in idx if kind[i] == k]
                L = np.array([length[i] for i in cand])
                return cand[int(rng.choice(len(cand), p=L / L.sum()))]

            def pick():
                s = draw()
                return s, draw(skip=s)

        for _ in range(self.random_cases):
            s, ld = pick()
            ang = float(rng.uniform(0, 2 * np.pi))
            self.generated.append([Load2D(region(ld), (mag * np.cos(ang), mag * np.sin(ang)))])
            self.generated_supports.append([region(s)])
        for _ in range(self.random_moments):
            s, ld = pick()
            sign = 1.0 if rng.random() < 0.5 else -1.0
            self.generated.append([Load2D(region(ld), (0.0, 0.0), sign * mag * 10.0)])
            self.generated_supports.append([region(s)])
        return self.generated

    def describe_cases(self) -> list:
        """JSON-friendly list of every case (for metrics.json): its loads, each a region and a
        force. A case with supports of its own lists them on each load as `supports`."""
        def reg(r: Region):
            for k in ("box", "circle", "polygon", "edge"):
                v = getattr(r, k)
                if v is not None:
                    d = {k: [list(map(float, x)) if isinstance(x, (list, tuple)) else float(x) for x in v]}
                    if k == "edge":
                        d["width"] = r.width
                    return d
            return {"side": r.side, "width": r.width}
        out = []
        for case, sups in zip(self.all_cases, self.case_supports):
            extra = {} if sups is self.supports else {"supports": [reg(r) for r in sups]}
            out.append([{"region": reg(ld.region), "force": [float(f) for f in ld.force],
                         **({"moment": float(ld.moment)} if ld.moment else {}), **extra} for ld in case])
        return out

    @staticmethod
    def from_dict(d: dict | None) -> "LoadSpec":
        d = dict(d or {})
        sup = [Region.from_dict(r) for r in d.get("supports", [])]

        def loads(xs):
            if isinstance(xs, dict):     # a case written as a single load
                xs = [xs]
            return [Load2D(Region.from_dict(x["region"]), tuple(x.get("force", (0.0, 0.0))),
                           float(x.get("moment", 0.0))) for x in xs]

        lds = loads(d.get("loads", []))
        pls = [PointLoad(tuple(x["pos"]), tuple(x["force"])) for x in d.get("point_loads", [])]
        cases = [loads(c) for c in d.get("load_cases", [])]
        return LoadSpec(sup, lds, pls, bool(d.get("split_by_height", True)), cases=cases,
                        random_cases=int(d.get("random_cases", 0)), random_moments=int(d.get("random_moments", 0)),
                        random_seed=int(d.get("random_seed", 0)),
                        random_mode=str(d.get("random_mode", "edge")),
                        random_weights=parse_load_weights(d.get("random_weights", LoadSpec.random_weights)))


def parse_load_weights(v) -> tuple | None:
    """random_weights from yaml or the command line: three odds (hole, side, arc) as a list or
    "3,2,1", or "uniform" / None for every edge alike."""
    if v is None or (isinstance(v, str) and v.strip().lower() == "uniform"):
        return None
    xs = [float(x) for x in (v.split(",") if isinstance(v, str) else v)]
    if len(xs) != 3 or min(xs) < 0 or sum(xs) <= 0:
        raise ValueError(f"random_weights: three odds >= 0 (hole, side, arc) or 'uniform'; got {v!r}")
    return tuple(xs)


@dataclass
class ScoreConfig:
    p_norm: float = 8.0
    c_stroke: float = 1.0      # per stroke (one retraction each)
    c_travel: float = 0.05     # per cell pitch of travel
    c_boundary_end: float = 0.5
    c_free_end: float = 1.0    # per path end no other bead touches (rltp.cover.beads.is_anchored)
    sigma_floor: float = 1e-6
    norm: str = "relative"     # term scales: "relative" (to the part's baseline) or "std" (per-edit std)
    stack: str = "auto"        # in-plane stack model: "split", "laminate", or "auto" (laminate when sparse)
    fea_refine: int = 1        # FEA elements per cell side (toolpath cells stay one bead wide)
    fea_coarsen: int = 1       # cells per FEA element side (fine bead grids; exclusive with fea_refine)
    clamp_band: float = 0.0    # mm: per case, cells this close to its own supports and loads leave its FI
    solver: str = "fast"       # FEA linear solver: "fast" (symmetric LU, CG to 1e-7) or "legacy"
    directions: str = "classes"  # bead directions the laminate scores: "classes" (0/45/90/135 from the
                                 # grid steps; a cell takes the worst class it holds) or "path" (each
                                 # bead along its own path, rltp.cover.paths; fine grids, laminate)


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
