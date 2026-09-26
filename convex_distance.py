"""Closest points, separation certificates and penetration depth (GJK/EPA).

Public contract
---------------
distance(A, B, *, tol=1e-9, max_iter=128) -> DistanceResult

* ``A`` / ``B``: sequences of 1..64 points; each point is a sequence of
  exactly 3 finite real numbers (``int`` / ``float`` / ``Fraction``;
  ``bool`` is rejected).  Each input is interpreted as the convex hull of
  its points.  Computation is performed in IEEE-754 double precision.
* ``tol``: relative convergence tolerance, a finite real with
  ``0 < tol < 1``.  ``bool`` is rejected.
* ``max_iter``: GJK iteration budget, an ``int`` with
  ``1 <= max_iter <= 128``.  ``bool`` is rejected.

Invalid input raises :class:`InputError` (a ``ValueError``) with a message
that locates the offending element, e.g. ``"A[3][1]: ..."``.  Nothing is
silently corrected.

DistanceResult fields
---------------------
status           "separated" | "intersecting" | "not_converged"
distance         float; for "separated"/"not_converged" the norm of the
                 best closest point of the Minkowski difference to the
                 origin; for "intersecting" a residual <= the zero
                 threshold (NOT a penetration depth).
witness_a        closest point on conv(A) (barycentric combination of A).
witness_b        closest point on conv(B).
separating_axis  unit vector n with min_A a.n >= wa.n and
                 max_B b.n <= wb.n (a verified certificate when
                 status == "separated"); None when "intersecting"; for
                 "not_converged" the best known direction, NOT verified.
iterations       GJK iterations actually used.
residual         final relative gap (||v||^2 - w.v) / ||v||^2; 0.0 for
                 "intersecting".  For "separated" it is <= tol; for
                 "not_converged" it is > tol and bounds the remaining
                 relative uncertainty of ``distance``.

"not_converged" (search budget exhausted) is reported distinctly from
"intersecting" (origin proven inside the Minkowski difference); the two
are never conflated.  Touching hulls (distance exactly zero) are reported
as "intersecting" with distance <= the zero threshold.

Penetration query
-----------------
penetration(A, B, *, tol=1e-9, max_iter=128) -> PenetrationResult

Same input contract as distance().  Phase 1 confirms intersection with
GJK; phase 2 expands a bounding polytope of the Minkowski difference
(EPA) and reports the minimum translation that separates the hulls.

PenetrationResult fields
------------------------
status           "penetrating" | "touching" | "separated" |
                 "not_converged"
depth_lower      verifiable lower bound on the penetration depth
                 (origin-to-boundary distance of the final inner
                 polytope; 0.0 for "touching"/"separated").
depth_upper      verifiable upper bound (support of the Minkowski
                 difference along the final direction).
direction        unit vector n: translating B by depth*n separates the
                 hulls (n points from A towards B, i.e. contact_a -
                 contact_b ~ depth*n).  None for "separated"; for
                 "touching"/"not_converged" the best known direction,
                 NOT a verified unique answer.
contact_a        contact point on conv(A) (barycentric combination of A);
                 None for "separated".
contact_b        contact point on conv(B).
iterations       total iterations used (GJK + EPA share the budget).
residual         relative gap (depth_upper - depth_lower) /
                 max(1, depth_upper); <= tol when "penetrating".

"touching" (coplanar/contact without positive-volume penetration) is an
explicit zero-depth state, distinct from "penetrating".  Degenerate
faces, repeated support points and budget exhaustion never fabricate a
unique normal: bounds are kept and "not_converged" is reported.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "penetration",
    "DistanceResult",
    "PenetrationResult",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
    "STATUS_PENETRATING",
    "STATUS_TOUCHING",
]

__version__ = "1.1.0"

MAX_POINTS = 64
MAX_ITER = 128
DEFAULT_TOL = 1e-9

STATUS_SEPARATED = "separated"
STATUS_INTERSECTING = "intersecting"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_PENETRATING = "penetrating"
STATUS_TOUCHING = "touching"

# Origin-containment threshold, relative to the input scale.  Distinct
# from ``tol``: this decides "intersecting" vs "separated".
_ZERO_TOL_REL = 1e-11
# Feasibility slack for barycentric coordinates (dimensionless).
_BARY_TOL = 1e-9
# Singularity threshold for the KKT solves, relative to the Gram scale.
_KKT_TOL_REL = 1e-12


class InputError(ValueError):
    """Raised for any contract-violating input; message locates the fault."""


@dataclass(frozen=True)
class DistanceResult:
    status: str
    distance: float
    witness_a: tuple
    witness_b: tuple
    separating_axis: tuple | None
    iterations: int
    residual: float

    @property
    def converged(self) -> bool:
        return self.status != STATUS_NOT_CONVERGED


@dataclass(frozen=True)
class PenetrationResult:
    status: str
    depth_lower: float
    depth_upper: float
    direction: tuple | None
    contact_a: tuple | None
    contact_b: tuple | None
    iterations: int
    residual: float

    @property
    def depth(self):
        """Best point estimate of the depth (midpoint of the bounds)."""
        return 0.5 * (self.depth_lower + self.depth_upper)


# ---------------------------------------------------------------------------
# vector helpers (plain 3-tuples of floats)
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _norm2(u):
    return _dot(u, u)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def _validate_points(points, name):
    if isinstance(points, (str, bytes)):
        raise InputError(f"{name}: expected a sequence of 3D points, got text")
    try:
        pts = list(points)
    except TypeError:
        raise InputError(f"{name}: expected a sequence of 3D points, "
                         f"got {type(points).__name__}") from None
    if not 1 <= len(pts) <= MAX_POINTS:
        raise InputError(f"{name}: expected 1..{MAX_POINTS} points, "
                         f"got {len(pts)}")
    out = []
    for i, p in enumerate(pts):
        if isinstance(p, (str, bytes)):
            raise InputError(f"{name}[{i}]: expected a 3-coordinate point, "
                             f"got text")
        try:
            coords = list(p)
        except TypeError:
            raise InputError(f"{name}[{i}]: expected a 3-coordinate point, "
                             f"got {type(p).__name__}") from None
        if len(coords) != 3:
            raise InputError(f"{name}[{i}]: expected exactly 3 coordinates, "
                             f"got {len(coords)}")
        vec = []
        for j, c in enumerate(coords):
            if isinstance(c, bool) or not isinstance(c, numbers.Real):
                raise InputError(f"{name}[{i}][{j}]: expected a real number, "
                                 f"got {type(c).__name__} ({c!r})")
            f = float(c)
            if not math.isfinite(f):
                raise InputError(f"{name}[{i}][{j}]: non-finite coordinate "
                                 f"{c!r} (NaN/Infinity rejected)")
            vec.append(f)
        out.append((vec[0], vec[1], vec[2]))
    return out


def _validate_tol(tol):
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
        raise InputError(f"tol: expected a real number, "
                         f"got {type(tol).__name__} ({tol!r})")
    t = float(tol)
    if not math.isfinite(t) or not 0.0 < t < 1.0:
        raise InputError(f"tol: expected a finite value in (0, 1), got {tol!r}")
    return t


def _validate_max_iter(max_iter):
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise InputError(f"max_iter: expected an int, "
                         f"got {type(max_iter).__name__} ({max_iter!r})")
    if not 1 <= max_iter <= MAX_ITER:
        raise InputError(f"max_iter: expected 1..{MAX_ITER}, got {max_iter}")
    return max_iter


# ---------------------------------------------------------------------------
# support function of the Minkowski difference A - B (no faces are built)
# ---------------------------------------------------------------------------

def _support(A, B, d):
    """Return (w, a, b) with a = argmax_A a.d, b = argmin_B b.d, w = a - b.

    Ties are broken deterministically by lowest index.
    """
    ai = 0
    best = _dot(A[0], d)
    for i in range(1, len(A)):
        s = _dot(A[i], d)
        if s > best:
            best = s
            ai = i
    bi = 0
    best = _dot(B[0], d)
    for i in range(1, len(B)):
        s = _dot(B[i], d)
        if s < best:
            best = s
            bi = i
    a = A[ai]
    b = B[bi]
    return (_sub(a, b), a, b)


# ---------------------------------------------------------------------------
# closest point of the origin to a simplex, by exact subset enumeration
# (Johnson-style).  Affinely dependent subsets are skipped, so degenerate
# segments / triangles / repeated points are handled by dimension
# reduction -- never by random perturbation.
# ---------------------------------------------------------------------------

def _affine_barycentric(ws):
    """Barycentric coords of the origin's projection onto aff(ws).

    Solves the KKT system  [G 1; 1^T 0] [lam; mu] = [0; 1]  with partial
    pivoting.  Returns None when ws is (numerically) affinely dependent;
    the caller then relies on smaller subsets (dimension reduction).
    """
    k = len(ws)
    if k == 1:
        return (1.0,)
    n = k + 1
    m = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(k):
        wi = ws[i]
        for j in range(k):
            m[i][j] = _dot(wi, ws[j])
        m[i][k] = 1.0
        m[k][i] = 1.0
    m[k][n] = 1.0
    gscale = 0.0
    for i in range(k):
        for j in range(k):
            if abs(m[i][j]) > gscale:
                gscale = abs(m[i][j])
    if gscale == 0.0:
        return None
    eps = _KKT_TOL_REL * gscale
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(m[r][col]) > abs(m[piv][col]):
                piv = r
        if abs(m[piv][col]) <= eps:
            return None
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n]
        for c in range(r + 1, n):
            s -= m[r][c] * x[c]
        x[r] = s / m[r][r]
    return tuple(x[:k])


def _closest_on_simplex(simplex):
    """Closest point of conv(simplex) to the origin.

    simplex: list of (w, a, b) support records.  Returns
    (v, lambdas, kept_indices) where v = sum(lambdas[i] * w[kept[i]]).
    Every non-empty subset is examined; the closest point is the
    minimum-norm affine projection whose barycentric coordinates are all
    >= -_BARY_TOL.  Singletons always succeed, so a result always exists.
    """
    n = len(simplex)
    best = None
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        lam = _affine_barycentric([simplex[i][0] for i in idx])
        if lam is None:
            continue
        feasible = True
        for l in lam:
            if l < -_BARY_TOL:
                feasible = False
                break
        if not feasible:
            continue
        vx = vy = vz = 0.0
        for i, l in zip(idx, lam):
            w = simplex[i][0]
            vx += l * w[0]
            vy += l * w[1]
            vz += l * w[2]
        d2 = vx * vx + vy * vy + vz * vz
        if best is None or d2 < best[0]:
            best = (d2, (vx, vy, vz), lam, idx)
    _, v, lam, idx = best
    return v, lam, idx


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------

def _input_scale(pts_a, pts_b):
    scale = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    return math.sqrt(scale)


def _gjk(pts_a, pts_b, tol, max_iter, zero_tol2):
    """GJK on the Minkowski difference A - B.

    Returns (status, simplex, lam, v, v2, iterations, residual) where
    simplex is the terminal list of (w, a, b) support records, lam the
    barycentric weights of v = closest point of conv(simplex) to the
    origin, and v2 = ||v||^2.
    """
    # Initial search direction: centroid difference (deterministic).
    na = len(pts_a)
    ca = (sum(p[0] for p in pts_a) / na,
          sum(p[1] for p in pts_a) / na,
          sum(p[2] for p in pts_a) / na)
    nb = len(pts_b)
    cb = (sum(p[0] for p in pts_b) / nb,
          sum(p[1] for p in pts_b) / nb,
          sum(p[2] for p in pts_b) / nb)
    d = _sub(ca, cb)
    if _norm2(d) == 0.0:
        d = (1.0, 0.0, 0.0)

    simplex = []          # list of (w, a, b)
    v = None              # current closest point of conv(simplex)
    v2 = 0.0
    lam = ()
    residual = math.inf
    status = STATUS_NOT_CONVERGED
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        w, sa, sb = _support(pts_a, pts_b, d)
        if v is not None:
            # w = argmin_{c in C} c.v, so ||c*|| >= w.v / ||v|| and the
            # relative gap (v2 - w.v)/v2 bounds the remaining error.
            gap = v2 - _dot(w, v)
            residual = gap / v2 if v2 > 0.0 else 0.0
            if gap <= tol * v2:
                status = STATUS_SEPARATED
                break
        if any(w == s[0] for s in simplex):
            # Support point already in the simplex: w.v == v2, the gap is
            # exactly zero and no further progress is possible.
            residual = 0.0
            status = STATUS_SEPARATED
            break
        simplex.append((w, sa, sb))
        v, lam, keep = _closest_on_simplex(simplex)
        simplex = [simplex[i] for i in keep]
        v2 = _norm2(v)
        if v2 <= zero_tol2:
            # Origin lies in conv(simplex) subset C: hulls intersect.
            residual = 0.0
            status = STATUS_INTERSECTING
            break
        d = (-v[0], -v[1], -v[2])

    return status, simplex, lam, v, v2, iterations, residual


def _witnesses(simplex, lam):
    wax = way = waz = wbx = wby = wbz = 0.0
    for (w, sa, sb), l in zip(simplex, lam):
        wax += l * sa[0]
        way += l * sa[1]
        waz += l * sa[2]
        wbx += l * sb[0]
        wby += l * sb[1]
        wbz += l * sb[2]
    return (wax, way, waz), (wbx, wby, wbz)


def distance(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Closest distance between conv(A) and conv(B) with witnesses.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    scale = _input_scale(pts_a, pts_b)
    zero_tol2 = (_ZERO_TOL_REL * scale) ** 2

    status, simplex, lam, v, v2, iterations, residual = _gjk(
        pts_a, pts_b, tol, max_iter, zero_tol2)

    # Barycentric witnesses on both bodies.
    witness_a, witness_b = _witnesses(simplex, lam)

    dist = math.sqrt(v2)
    if status == STATUS_INTERSECTING:
        axis = None
    elif v2 > 0.0:
        inv = 1.0 / dist
        axis = (v[0] * inv, v[1] * inv, v[2] * inv)
    else:
        axis = None

    return DistanceResult(
        status=status,
        distance=dist,
        witness_a=witness_a,
        witness_b=witness_b,
        separating_axis=axis,
        iterations=iterations,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# penetration depth: EPA on the Minkowski difference
# ---------------------------------------------------------------------------

def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _det3(a, b, c):
    return _dot(a, _cross(b, c))


def _initial_polytope(pts_a, pts_b, simplex, scale):
    """Tetrahedron inside the Minkowski difference that contains the
    origin, or None when the hulls only touch (no positive-volume
    penetration).

    Candidates are the terminal GJK simplex plus support points along
    the six axis directions; every 4-subset is tested for affine
    independence, positive volume and origin containment.  Deterministic:
    the valid tetrahedron with the largest volume is chosen.
    """
    cands = list(simplex)
    dirs = []
    for sx in (1.0, -1.0):
        dirs.append((sx, 0.0, 0.0))
        dirs.append((0.0, sx, 0.0))
        dirs.append((0.0, 0.0, sx))
        for sy in (1.0, -1.0):
            dirs.append((sx, sy, 0.0))
            dirs.append((sx, 0.0, sy))
            dirs.append((0.0, sx, sy))
            for sz in (1.0, -1.0):
                dirs.append((sx, sy, sz))
    for d in dirs:
        cands.append(_support(pts_a, pts_b, d))
    # Deterministic dedup of repeated support points.
    uniq = []
    for c in cands:
        if not any(c[0] == u[0] for u in uniq):
            uniq.append(c)
    vol_tol = (1e-7 * max(1.0, scale)) ** 3
    best = None
    n = len(uniq)
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            e1 = _sub(uniq[j][0], uniq[i][0])
            for k in range(j + 1, n - 1):
                e2 = _sub(uniq[k][0], uniq[i][0])
                for l in range(k + 1, n):
                    e3 = _sub(uniq[l][0], uniq[i][0])
                    vol = abs(_det3(e1, e2, e3))
                    if vol <= vol_tol:
                        continue
                    ws = [uniq[i][0], uniq[j][0], uniq[k][0], uniq[l][0]]
                    lam = _affine_barycentric(ws)
                    if lam is None or min(lam) < -_BARY_TOL:
                        continue
                    # Origin strictly inside is required for a sane EPA
                    # start; prefer the tetrahedron whose origin is most
                    # deeply interior (max min face distance).
                    md = min(
                        abs(_det3(_sub(ws[a], ws[b]),
                                  _sub(ws[c], ws[b]), ws[b]))
                        / (math.sqrt(_norm2(_cross(_sub(ws[a], ws[b]),
                                                   _sub(ws[c], ws[b]))))
                           or 1.0)
                        for a, b, c in ((1, 2, 3), (0, 3, 1),
                                        (0, 2, 3), (0, 1, 2)))
                    if best is None or md > best[0]:
                        best = (md, (i, j, k, l))
    if best is None:
        # No positive-volume tetrahedron contains the origin: the hulls
        # only touch (coplanar contact, no positive-volume penetration).
        return None
    return [uniq[i] for i in best[1]]


def _face_data(verts, face, escale2, interior):
    """(outward_unit_normal, distance) of a triangular face, or
    (None, inf) for a degenerate face.  Origin is inside the polytope,
    so the outward normal satisfies dot(n, any_vertex) = distance >= 0.
    Orientation uses a known interior point (vertex centroid), which
    stays unambiguous even when the origin lies exactly on a face.
    """
    w0 = verts[face[0]][0]
    e1 = _sub(verts[face[1]][0], w0)
    e2 = _sub(verts[face[2]][0], w0)
    n = _cross(e1, e2)
    nn = math.sqrt(_norm2(n))
    if nn <= 1e-12 * escale2:
        return None, math.inf
    inv = 1.0 / nn
    n = (n[0] * inv, n[1] * inv, n[2] * inv)
    if _dot(n, _sub(w0, interior)) < 0.0:
        n = (-n[0], -n[1], -n[2])
    d = _dot(n, w0)
    return n, d


def _epa(pts_a, pts_b, simplex, tol, budget, scale):
    """Expanding Polytope Algorithm.  Returns
    (converged, direction, depth_lower, depth_upper, face_records,
    face_lambdas, iterations_used).
    """
    verts = _initial_polytope(pts_a, pts_b, simplex, scale)
    if verts is None:
        return None  # touching: no positive-volume penetration
    escale2 = max(1.0, scale) ** 2
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    used = 0
    converged = False
    best = None  # (normal, lower, upper)
    while True:
        m = 1.0 / len(verts)
        interior = (sum(vtx[0][0] for vtx in verts) * m,
                    sum(vtx[0][1] for vtx in verts) * m,
                    sum(vtx[0][2] for vtx in verts) * m)
        data = [_face_data(verts, f, escale2, interior) for f in faces]
        fi = min(range(len(faces)), key=lambda i: data[i][1])
        n, lower = data[fi]
        if n is None:
            break  # all faces degenerate; keep last bounds
        w, sa, sb = _support(pts_a, pts_b, n)
        upper = _dot(w, n)
        best = (n, lower, upper)
        if upper - lower <= tol * max(1.0, lower):
            converged = True
            break
        if any(w == vtx[0] for vtx in verts):
            converged = True  # repeated support point: no progress left
            break
        if used >= budget:
            break
        used += 1
        new_idx = len(verts)
        verts.append((w, sa, sb))
        # Remove faces visible from w; horizon edges border the hole.
        visible = set()
        for i, f in enumerate(faces):
            ni = data[i][0]
            if ni is not None and _dot(ni, _sub(w, verts[f[0]][0])) > 0.0:
                visible.add(i)
        if not visible:
            converged = True  # numerical: w inside current polytope
            verts.pop()
            break
        horizon = {}
        for i in visible:
            f = faces[i]
            for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                if (e[1], e[0]) in horizon:
                    del horizon[(e[1], e[0])]
                else:
                    horizon[e] = True
        faces = [f for i, f in enumerate(faces) if i not in visible]
        for a, b in horizon:
            faces.append((a, b, new_idx))
    if best is None:
        return None
    n, lower, upper = best
    # Contact points: barycentric weights of the origin's projection
    # (lower * n) on the closest face, applied to both bodies' supports.
    data = [_face_data(verts, f, escale2, interior) for f in faces]
    fi = min(range(len(faces)), key=lambda i: data[i][1])
    tri = [verts[j] for j in faces[fi]]
    lam = _affine_barycentric([t[0] for t in tri])
    if lam is None:
        lam = (1.0, 0.0, 0.0)  # degenerate face: fall back to a vertex
    return converged, n, lower, upper, tri, lam, used


def penetration(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Minimum translation that separates conv(A) and conv(B).

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    scale = _input_scale(pts_a, pts_b)
    zero_tol = _ZERO_TOL_REL * scale
    zero_tol2 = zero_tol * zero_tol

    status, simplex, lam, v, v2, it, _res = _gjk(
        pts_a, pts_b, tol, max_iter, zero_tol2)

    if status == STATUS_SEPARATED:
        wa, wb = _witnesses(simplex, lam)
        return PenetrationResult(
            status=STATUS_SEPARATED,
            depth_lower=0.0,
            depth_upper=0.0,
            direction=None,
            contact_a=wa,
            contact_b=wb,
            iterations=it,
            residual=0.0,
        )

    epa = _epa(pts_a, pts_b, simplex, tol, max_iter - it, scale)
    if epa is None or status == STATUS_NOT_CONVERGED and v2 <= zero_tol2:
        # Origin confirmed inside but no positive-volume polytope could
        # be built: coplanar / touching configuration, zero depth.
        wa, wb = _witnesses(simplex, lam)
        return PenetrationResult(
            status=STATUS_TOUCHING,
            depth_lower=0.0,
            depth_upper=0.0,
            direction=None,
            contact_a=wa,
            contact_b=wb,
            iterations=it,
            residual=0.0,
        )

    converged, n, lower, upper, tri, tri_lam, used = epa
    iterations = it + used
    contact_a, contact_b = _witnesses(tri, tri_lam)
    if upper <= zero_tol:
        # Penetration, if any, is below the zero threshold: report the
        # explicit zero-depth touching state rather than a fake normal.
        return PenetrationResult(
            status=STATUS_TOUCHING,
            depth_lower=0.0,
            depth_upper=0.0,
            direction=None,
            contact_a=contact_a,
            contact_b=contact_b,
            iterations=iterations,
            residual=0.0,
        )
    residual = (upper - lower) / max(1.0, upper)
    return PenetrationResult(
        status=STATUS_PENETRATING if converged else STATUS_NOT_CONVERGED,
        depth_lower=lower,
        depth_upper=upper,
        direction=n,
        contact_a=contact_a,
        contact_b=contact_b,
        iterations=iterations,
        residual=residual,
    )
