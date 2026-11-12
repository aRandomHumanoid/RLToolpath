import numpy as np
import pytest
from shapely.geometry import Point, box

from rltp.cover.baselines import fill_baseline
from rltp.model import Frame, Layer
from rltp.slicing.raster import rasterize


def make_layer(geom, p=1.0, kind="zigzag", H=None, W=None, k=0):
    frame = Frame(0.0, 0.0, 0.0, p)
    x0, y0, x1, y1 = geom.bounds
    H = H or int(np.ceil(y1 / p)) + 1
    W = W or int(np.ceil(x1 / p)) + 1
    mask, _ = rasterize(geom, frame, H, W, min_cells=1)
    L = Layer.empty(k, 0.2 * (k + 1), 0.2, mask, frame, poly=geom)
    return fill_baseline(L, kind)


SHAPES = {
    "square": box(0, 0, 12, 12),
    "plate_hole": box(0, 0, 20, 14).difference(Point(10, 7).buffer(4)),
    "L": box(0, 0, 16, 6).union(box(0, 0, 6, 16)),
    "two_islands": box(0, 0, 6, 6).union(box(9, 1, 15, 8)),
    "rib": box(0, 0, 18, 3).union(box(0, 0, 3, 12)).union(box(8, 0, 9, 12)),
}


@pytest.fixture(params=list(SHAPES))
def shape(request):
    return SHAPES[request.param]
