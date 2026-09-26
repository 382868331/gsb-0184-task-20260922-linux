"""Closest points, separation certificates and penetration depth (GJK + EPA).

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

penetration(A, B, *, tol=1e-9, max_iter=128) -> PenetrationResult

Once GJK has proven the origin lies inside the Minkowski difference
C = A - B, an Expanding Polytope Algorithm (EPA) grows a polytope inside
C around the origin and reports the minimum translation that separates
the hulls:

status           "penetrating" | "touching" | "separated" | "not_converged"
depth            best estimate of the penetration depth (midpoint of the
                 bounds); 0.0 for "touching"/"separated".
depth_lower      verifiable lower bound: distance from the origin to the
                 closest triangle of the inner polytope P subset C, so the
                 true depth is >= depth_lower.
depth_upper      verifiable upper bound: minimum of support_C(n).n over
                 every queried unit normal n, so the true depth is <=
                 depth_upper.  Always finite (the initial tetrahedron
                 already contributes four queries).
direction        unit vector n; translating B by t*n with any
                 t > depth_upper separates the hulls, and depth*n is the
                 (near-)minimal such translation.  Points from B towards
                 A, matching the separating_axis convention of distance().
                 None for "touching"/"separated".
witness_a        contact point on conv(A) (barycentric combination of A):
                 the point where the hulls touch after B is translated by
                 depth_lower * direction.  None for "separated".
witness_b        contact point on conv(B); witness_a - witness_b =
                 depth_lower * direction.
iterations       EPA expansion iterations actually used (the max_iter
                 budget applies to this phase; the seeding GJK phase has
                 its own identical budget).
residual         (depth_upper - depth_lower) / depth_upper; <= ~tol when
                 converged.

"touching" is an explicit zero-depth status: the hulls intersect (GJK
proved origin in C) but no positive-volume penetration could be certified
-- the origin lies on a face/edge/vertex of the final simplex or every
attempt to build a tetrahedron around it collapsed (coplanar contact,
shared corner, degenerate flat hulls).  No normal is invented in that
case: direction is None.

"not_converged" means the EPA iteration budget was exhausted (or a
duplicate support point / degenerate face stopped the expansion early
without meeting tol).  depth_lower/depth_upper remain valid, verifiable
bounds and direction is the best known normal -- it is NOT a certified
unique answer.
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
    depth: float
    depth_lower: float
    depth_upper: float
    direction: tuple | None
    witness_a: tuple | None
    witness_b: tuple | None
    iterations: int
    residual: float

    @property
    def converged(self) -> bool:
        return self.status != STATUS_NOT_CONVERGED


# ---------------------------------------------------------------------------
# vector helpers (plain 3-tuples of floats)
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _norm2(u):
    return _dot(u, u)


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _norm(u):
    return math.sqrt(_norm2(u))


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
# main entry point
# ---------------------------------------------------------------------------

def distance(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Closest distance between conv(A) and conv(B) with witnesses.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    scale = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    scale = math.sqrt(scale)
    zero_tol2 = (_ZERO_TOL_REL * scale) ** 2

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

    # Barycentric witnesses on both bodies.
    wax = way = waz = wbx = wby = wbz = 0.0
    for (w, sa, sb), l in zip(simplex, lam):
        wax += l * sa[0]
        way += l * sa[1]
        waz += l * sa[2]
        wbx += l * sb[0]
        wby += l * sb[1]
        wbz += l * sb[2]
    witness_a = (wax, way, waz)
    witness_b = (wbx, wby, wbz)

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
# penetration depth: GJK seeding + Expanding Polytope Algorithm (EPA)
# ---------------------------------------------------------------------------

def _gjk_intersect(pts_a, pts_b, zero_tol2, max_iter):
    """GJK intersection test; returns (status, simplex, iterations).

    status is STATUS_INTERSECTING when the origin is proven inside the
    Minkowski difference (the returned simplex then contains it in its
    convex hull), STATUS_SEPARATED when a separating witness is found,
    and STATUS_NOT_CONVERGED when the budget ran out first.
    """
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

    simplex = []
    v = None
    v2 = 0.0
    for it in range(1, max_iter + 1):
        w, sa, sb = _support(pts_a, pts_b, d)
        if v is not None and _dot(w, v) >= v2:
            return STATUS_SEPARATED, simplex, it
        if any(w == s[0] for s in simplex):
            return STATUS_SEPARATED, simplex, it
        simplex.append((w, sa, sb))
        v, lam, keep = _closest_on_simplex(simplex)
        simplex = [simplex[i] for i in keep]
        v2 = _norm2(v)
        if v2 <= zero_tol2:
            return STATUS_INTERSECTING, simplex, it
        d = (-v[0], -v[1], -v[2])
    return STATUS_NOT_CONVERGED, simplex, max_iter


def _build_tetra(pts_a, pts_b, simplex, zero_tol):
    """Grow the GJK simplex into a tetrahedron enclosing the origin.

    Returns a list of 4 (w, a, b) support records, or None when no such
    tetrahedron exists (coplanar / edge / point contact, degenerate flat
    hulls with no volume): the caller then reports an explicit zero-depth
    "touching".  The origin may lie on the tetrahedron boundary (the GJK
    simplex is often a degenerate segment/triangle); whether that means
    zero depth is decided later by the EPA lower bound, not here.  Adding
    support points only grows the convex hull, so the origin stays inside
    conv(verts) throughout.
    """
    verts = list(simplex)
    if len(verts) > 4:
        return None
    if len(verts) < 4:
        ws = [v[0] for v in verts]
        dirs = []
        if len(verts) == 3:
            n = _cross(_sub(ws[1], ws[0]), _sub(ws[2], ws[0]))
            dirs = [n, (-n[0], -n[1], -n[2])]
        elif len(verts) == 2:
            u = _sub(ws[1], ws[0])
            e = (1.0, 0.0, 0.0)
            if abs(_dot(u, e)) > 0.9 * _norm(u):
                e = (0.0, 1.0, 0.0)
            n1 = _cross(u, e)
            n2 = _cross(u, n1)
            dirs = [n1, (-n1[0], -n1[1], -n1[2]),
                    n2, (-n2[0], -n2[1], -n2[2])]
        dirs += [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                 (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
        for d in dirs:
            if len(verts) == 4:
                break
            if _norm2(d) == 0.0:
                continue
            w, sa, sb = _support(pts_a, pts_b, d)
            if any(w == v[0] for v in verts):
                continue
            trial = [v[0] for v in verts] + [w]
            if _affine_barycentric(trial) is None:
                continue  # affinely dependent: keeps the hull flat
            verts.append((w, sa, sb))
    if len(verts) < 4:
        return None
    ws = [v[0] for v in verts]
    lam = _affine_barycentric(ws)
    if lam is None or any(l < -_BARY_TOL for l in lam):
        return None
    return verts


def _make_face(verts, i, j, k, area_eps, interior):
    """Outward-oriented triangular face, or None if degenerate (zero area).

    ``interior`` is a point strictly inside the polytope (its centroid);
    the outward normal is the one pointing away from it.  The winding is
    flipped together with the normal to keep the
    counter-clockwise-from-outside convention used by the horizon walk.
    """
    w0 = verts[i][0]
    n = _cross(_sub(verts[j][0], w0), _sub(verts[k][0], w0))
    nn = _norm(n)
    if nn <= area_eps:
        return None
    inv = 1.0 / nn
    n = (n[0] * inv, n[1] * inv, n[2] * inv)
    if _dot(n, _sub(interior, w0)) > 0.0:
        n = (-n[0], -n[1], -n[2])
        j, k = k, j
    return [i, j, k, n[0], n[1], n[2], _dot(n, w0)]


def _epa_add_vertex(faces, verts, new_index, area_eps, interior):
    """Expand the polytope with verts[new_index] via horizon-edge walk.

    Removes every face visible from the new vertex and stitches new
    triangles over the horizon edges.  Returns False when no face is
    visible (numerical dead end: the vertex lies inside the polytope).
    """
    x = verts[new_index][0]
    edges = {}
    kept = []
    for f in faces:
        n = (f[3], f[4], f[5])
        if _dot(n, _sub(x, verts[f[0]][0])) > 0.0:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (a, b) if a < b else (b, a)
                if key in edges:
                    edges[key][1] += 1
                else:
                    edges[key] = [(a, b), 1]
        else:
            kept.append(f)
    if len(kept) == len(faces):
        return False
    faces[:] = kept
    for (a, b), count in edges.values():
        if count != 1:
            continue  # interior edge shared by two visible faces
        f = _make_face(verts, a, b, new_index, area_eps, interior)
        if f is not None:
            faces.append(f)
    return True


def penetration(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Minimum translation that separates conv(A) and conv(B), with bounds.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    scale = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    scale = math.sqrt(scale)
    zero_tol = _ZERO_TOL_REL * scale
    zero_tol2 = zero_tol * zero_tol
    area_eps = _KKT_TOL_REL * scale * scale

    status, simplex, gjk_iters = _gjk_intersect(pts_a, pts_b, zero_tol2,
                                                max_iter)
    if status != STATUS_INTERSECTING:
        if status == STATUS_SEPARATED:
            res = distance(pts_a, pts_b, tol=tol, max_iter=max_iter)
            return PenetrationResult(
                status=STATUS_SEPARATED, depth=0.0, depth_lower=0.0,
                depth_upper=0.0, direction=None,
                witness_a=res.witness_a, witness_b=res.witness_b,
                iterations=gjk_iters, residual=0.0)
        return PenetrationResult(
            status=STATUS_NOT_CONVERGED, depth=0.0, depth_lower=0.0,
            depth_upper=math.inf, direction=None,
            witness_a=None, witness_b=None,
            iterations=gjk_iters, residual=math.inf)

    verts = _build_tetra(pts_a, pts_b, simplex, zero_tol)
    if verts is None:
        # Intersection proven, but no positive-volume penetration could be
        # certified: coplanar / edge / point contact.  Explicit zero depth;
        # no normal is invented.
        res = distance(pts_a, pts_b, tol=tol, max_iter=max_iter)
        return PenetrationResult(
            status=STATUS_TOUCHING, depth=0.0, depth_lower=0.0,
            depth_upper=0.0, direction=None,
            witness_a=res.witness_a, witness_b=res.witness_b,
            iterations=gjk_iters, residual=0.0)

    interior = (sum(v[0][0] for v in verts) / 4.0,
                sum(v[0][1] for v in verts) / 4.0,
                sum(v[0][2] for v in verts) / 4.0)
    faces = []
    for i, j, k in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)):
        f = _make_face(verts, i, j, k, area_eps, interior)
        if f is not None:
            faces.append(f)
    if len(faces) < 4:
        res = distance(pts_a, pts_b, tol=tol, max_iter=max_iter)
        return PenetrationResult(
            status=STATUS_TOUCHING, depth=0.0, depth_lower=0.0,
            depth_upper=0.0, direction=None,
            witness_a=res.witness_a, witness_b=res.witness_b,
            iterations=gjk_iters, residual=0.0)

    # Verifiable upper bound from the start: depth <= support_C(n).n for
    # any unit n, and the four seed faces already give four queries.
    upper = math.inf
    for f in faces:
        w, _, _ = _support(pts_a, pts_b, (f[3], f[4], f[5]))
        s = _dot(w, (f[3], f[4], f[5]))
        if s < upper:
            upper = s

    converged = False
    epa_iters = 0
    for it in range(1, max_iter + 1):
        epa_iters = it
        f = min(faces, key=lambda f: f[6])
        n = (f[3], f[4], f[5])
        w, sa, sb = _support(pts_a, pts_b, n)
        s = _dot(w, n)
        if s < upper:
            upper = s
        if s - f[6] <= tol * max(1.0, s):
            converged = True
            break
        if any(w == v[0] for v in verts):
            converged = True  # duplicate support point: no progress left
            break
        verts.append((w, sa, sb))
        interior = (interior[0] + (w[0] - interior[0]) / len(verts),
                    interior[1] + (w[1] - interior[1]) / len(verts),
                    interior[2] + (w[2] - interior[2]) / len(verts))
        if not _epa_add_vertex(faces, verts, len(verts) - 1, area_eps,
                               interior):
            converged = True  # numerical dead end; bounds remain valid
            break

    # Verifiable lower bound + witnesses: closest point of the origin to
    # the closest triangle of the inner polytope.
    best = None
    for f in faces:
        tri = [verts[f[0]], verts[f[1]], verts[f[2]]]
        v, lam, keep = _closest_on_simplex(tri)
        d2 = _norm2(v)
        if best is None or d2 < best[0]:
            best = (d2, v, lam, keep, tri)
    d2, v, lam, keep, tri = best
    lower = math.sqrt(d2)
    kept_tri = [tri[i] for i in keep]

    if lower <= zero_tol:
        res = distance(pts_a, pts_b, tol=tol, max_iter=max_iter)
        return PenetrationResult(
            status=STATUS_TOUCHING, depth=0.0, depth_lower=0.0,
            depth_upper=0.0, direction=None,
            witness_a=res.witness_a, witness_b=res.witness_b,
            iterations=gjk_iters + epa_iters, residual=0.0)

    wax = way = waz = wbx = wby = wbz = 0.0
    for (w, sa, sb), l in zip(kept_tri, lam):
        wax += l * sa[0]
        way += l * sa[1]
        waz += l * sa[2]
        wbx += l * sb[0]
        wby += l * sb[1]
        wbz += l * sb[2]

    inv = 1.0 / lower
    direction = (v[0] * inv, v[1] * inv, v[2] * inv)
    depth = 0.5 * (lower + upper)
    residual = (upper - lower) / upper if upper > 0.0 else 0.0

    return PenetrationResult(
        status=STATUS_PENETRATING if converged else STATUS_NOT_CONVERGED,
        depth=depth,
        depth_lower=lower,
        depth_upper=upper,
        direction=direction,
        witness_a=(wax, way, waz),
        witness_b=(wbx, wby, wbz),
        iterations=gjk_iters + epa_iters,
        residual=residual,
    )
