"""Closest points and separation certificates for convex polytopes (GJK).

Public contract
---------------
Two entry points share one GJK engine:

* ``distance(A, B, *, tol=1e-9, max_iter=128) -> DistanceResult``
* ``distance_shelled(A, B, LA=None, LB=None, *, tol=1e-9,
  max_iter=128) -> ShelledDistanceResult``

``A`` / ``B``: sequences of 1..64 points; each point is a sequence of
exactly 3 finite real numbers (``int`` / ``float`` / ``Fraction``;
``bool`` is rejected).  Each input is interpreted as the convex hull of
its points.  Computation is performed in IEEE-754 double precision.

``LA`` / ``LB``: ``None`` or an invertible 3x3 matrix of finite reals
(``bool`` rejected).  A non-None matrix ``L`` adds the ellipsoidal
Minkowski shell ``L @ S2`` (``S2`` the closed Euclidean unit ball) to the
body: the queried set is ``conv(points) + L S2``.  The shell is an
ellipsoid whose directional thickness is ``||L^T n||``; it is never
collapsed to a largest-radius sphere.

``tol``: relative convergence tolerance, a finite real with
``0 < tol < 1``.  ``bool`` is rejected.
``max_iter``: GJK iteration budget, an ``int`` with
``1 <= max_iter <= 128``.  ``bool`` is rejected.

Invalid input raises :class:`InputError` (a ``ValueError``) with a message
that locates the offending element, e.g. ``"A[3][1]: ..."`` or
``"LA[1][2]: ..."``.  Nothing is silently corrected.

Shell support
-------------
For a unit direction ``n`` the support offset of ``L S2`` is

    h_off(L, n) = L @ L^T @ n / ||L^T n|| ,

the unique point of the ellipsoid exposed in direction ``n`` (with
``h_off(L, n).n = ||L^T n||``).  The expression is positively homogeneous,
so it is valid for any non-zero direction vector without prior
normalisation; a zero direction is rejected.  The support point of the
shelled body in direction ``d`` is

    argmax_P p.d + h_off(L,  d)   (body A)
    argmin_P p.d + h_off(L, -d)   (body B).

ShelledDistanceResult fields
----------------------------
status           "separated" | "intersecting" | "not_converged"
distance         float; for "separated"/"not_converged" the norm of the
                 best closest point of the Minkowski difference to the
                 origin; for "intersecting" a residual <= the zero
                 threshold (NOT a penetration depth).
witness_a        closest feasible point on the *shelled* body A,
witness_b        i.e. core witness + shell offset.
core_a / core_b  closest points on conv(A) / conv(B) (barycentric
                 combinations of the input vertices).  These alone do
                 NOT represent the inflated surface.
offset_a/offset_b shell offsets with core + offset = witness.  An offset
                 lies on/in L S2: ||L^{-1} offset|| <= 1 (== 1 for the
                 active contact in a separated case, up to roundoff).
separating_axis  unit vector n (B -> A); verified certificate when
                 status == "separated"; None when "intersecting"; for
                 "not_converged" the best known direction, NOT verified.
lower_bound /    checkable bracket on the true shelled distance:
upper_bound      lower_bound <= distance* <= upper_bound.  For
                 "intersecting" both are 0.  lower_bound derives from the
                 support-gap certificate, upper_bound is the feasible
                 witness distance.
iterations       GJK iterations actually used.
residual         final relative gap (||v||^2 - w.v) / ||v||^2; 0.0 for
                 "intersecting".  For "separated" it is <= tol and bounds
                 the remaining relative uncertainty.
shell_a / shell_b whether each body carried an ellipsoidal shell.

"not_converged" (search budget exhausted) is reported distinctly from
"intersecting" (origin proven inside the Minkowski difference); the two
are never conflated, and an un-converged run still returns the bracket.
Touching hulls (distance exactly zero) are reported as "intersecting"
with distance <= the zero threshold.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "distance_shelled",
    "DistanceResult",
    "ShelledDistanceResult",
    "shell_support_offset",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
]

__version__ = "2.0.0"

MAX_POINTS = 64
MAX_ITER = 128
DEFAULT_TOL = 1e-9

STATUS_SEPARATED = "separated"
STATUS_INTERSECTING = "intersecting"
STATUS_NOT_CONVERGED = "not_converged"

# Origin-containment threshold, relative to the input scale.  Distinct
# from ``tol``: this decides "intersecting" vs "separated".
_ZERO_TOL_REL = 1e-11
# Feasibility slack for barycentric coordinates (dimensionless).
_BARY_TOL = 1e-9
# Singularity threshold for the KKT solves, relative to the Gram scale.
_KKT_TOL_REL = 1e-12
# Invertibility threshold for shell matrices, relative to max |entry|.
_SING_TOL_REL = 1e-12

_ZERO = (0.0, 0.0, 0.0)


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
class ShelledDistanceResult:
    status: str
    distance: float
    witness_a: tuple
    witness_b: tuple
    core_a: tuple
    core_b: tuple
    offset_a: tuple
    offset_b: tuple
    separating_axis: tuple | None
    lower_bound: float
    upper_bound: float
    iterations: int
    residual: float
    shell_a: bool
    shell_b: bool

    @property
    def converged(self) -> bool:
        return self.status != STATUS_NOT_CONVERGED


# ---------------------------------------------------------------------------
# vector helpers (plain 3-tuples of floats)
# ---------------------------------------------------------------------------

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _neg(u):
    return (-u[0], -u[1], -u[2])


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


def _validate_matrix(L, name):
    """Validate an invertible 3x3 real matrix; return float tuple-of-tuples."""
    if L is None:
        return None
    if isinstance(L, (str, bytes)):
        raise InputError(f"{name}: expected a 3x3 matrix, got text")
    try:
        rows = list(L)
    except TypeError:
        raise InputError(f"{name}: expected a 3x3 matrix, "
                         f"got {type(L).__name__}") from None
    if len(rows) != 3:
        raise InputError(f"{name}: expected a 3x3 matrix, "
                         f"got {len(rows)} rows")
    mat = []
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise InputError(f"{name}[{i}]: expected a row of 3 reals, "
                             f"got text")
        try:
            entries = list(row)
        except TypeError:
            raise InputError(f"{name}[{i}]: expected a row of 3 reals, "
                             f"got {type(row).__name__}") from None
        if len(entries) != 3:
            raise InputError(f"{name}[{i}]: expected exactly 3 entries, "
                             f"got {len(entries)}")
        out_row = []
        for j, c in enumerate(entries):
            if isinstance(c, bool) or not isinstance(c, numbers.Real):
                raise InputError(f"{name}[{i}][{j}]: expected a real number, "
                                 f"got {type(c).__name__} ({c!r})")
            f = float(c)
            if not math.isfinite(f):
                raise InputError(f"{name}[{i}][{j}]: non-finite matrix entry "
                                 f"{c!r} (NaN/Infinity rejected)")
            out_row.append(f)
        mat.append((out_row[0], out_row[1], out_row[2]))

    # Invertibility via Gaussian elimination with partial pivoting.  A
    # singular (or near-singular) shell would not bound a finite ellipsoid
    # and is rejected up front rather than producing infinities/NaNs.
    scale = max(abs(mat[i][j]) for i in range(3) for j in range(3))
    if scale == 0.0:
        raise InputError(f"{name}: matrix is singular (the zero matrix is "
                         f"not invertible)")
    m = [list(row) for row in mat]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= _SING_TOL_REL * scale:
            raise InputError(f"{name}: matrix is singular (smallest pivot "
                             f"{m[piv][col]:.3e} <= {_SING_TOL_REL:.0e} of "
                             f"the entry scale {scale:.3e}); an invertible "
                             f"matrix is required")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col + 1, 3):
                m[r][c] -= f * m[col][c]
    return tuple(mat)


# ---------------------------------------------------------------------------
# ellipsoidal Minkowski shell
# ---------------------------------------------------------------------------

def shell_support_offset(L, n):
    """Support point of the ellipsoid L S2 in direction n.

    Returns ``L @ L^T @ n / ||L^T n||``.  Valid for any non-zero vector
    ``n`` (the formula is positively homogeneous of degree 0, so
    normalising ``n`` is unnecessary); raises :class:`InputError` for a
    zero direction.
    """
    Lm = _validate_matrix(L, "L")
    if Lm is None:
        raise InputError("L: expected a 3x3 matrix, got None")
    if isinstance(n, (str, bytes)):
        raise InputError("n: expected a 3-vector, got text")
    try:
        nv = list(n)
    except TypeError:
        raise InputError(f"n: expected a 3-vector, got {type(n).__name__}") \
            from None
    if len(nv) != 3:
        raise InputError(f"n: expected exactly 3 components, got {len(nv)}")
    for j, c in enumerate(nv):
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise InputError(f"n[{j}]: expected a real number, "
                             f"got {type(c).__name__} ({c!r})")
        if not math.isfinite(float(c)):
            raise InputError(f"n[{j}]: non-finite direction component {c!r}")
    d = (float(nv[0]), float(nv[1]), float(nv[2]))
    if _norm2(d) == 0.0:
        raise InputError("n: zero support direction (the shell has no "
                         "distinguished support point at n = 0)")
    return _shell_offset(Lm, d)


def _shell_offset(L, d):
    """L @ L^T @ d / ||L^T d|| for validated invertible L, non-zero d."""
    # y = L^T d
    y = (L[0][0] * d[0] + L[1][0] * d[1] + L[2][0] * d[2],
         L[0][1] * d[0] + L[1][1] * d[1] + L[2][1] * d[2],
         L[0][2] * d[0] + L[1][2] * d[1] + L[2][2] * d[2])
    ylen = math.sqrt(_dot(y, y))
    # L is invertible and d != 0, so ylen > 0; clamp only against roundoff
    # underflow on pathological scales.
    if not (ylen > 0.0) or not math.isfinite(ylen):
        raise InputError("L: shell offset undefined for the given scale")
    inv = 1.0 / ylen
    return (L[0][0] * y[0] * inv + L[0][1] * y[1] * inv + L[0][2] * y[2] * inv,
            L[1][0] * y[0] * inv + L[1][1] * y[1] * inv + L[1][2] * y[2] * inv,
            L[2][0] * y[0] * inv + L[2][1] * y[1] * inv + L[2][2] * y[2] * inv)


# ---------------------------------------------------------------------------
# support function of the Minkowski difference A - B (no faces are built)
#
# Each support record is a 7-tuple
#   (w, pa, pb, ca, cb, oa, ob)
# where ca/cb are the core hull vertices selected, oa/ob the shell
# offsets applied, pa = ca + oa, pb = cb + ob, w = pa - pb.
# ---------------------------------------------------------------------------

def _support_record(pts_a, pts_b, La, Lb, d):
    ai = 0
    best = _dot(pts_a[0], d)
    for i in range(1, len(pts_a)):
        s = _dot(pts_a[i], d)
        if s > best:
            best = s
            ai = i
    bi = 0
    best = _dot(pts_b[0], d)
    for i in range(1, len(pts_b)):
        s = _dot(pts_b[i], d)
        if s < best:
            best = s
            bi = i
    ca = pts_a[ai]
    cb = pts_b[bi]
    oa = _shell_offset(La, d) if La is not None else _ZERO
    ob = _shell_offset(Lb, _neg(d)) if Lb is not None else _ZERO
    pa = _add(ca, oa)
    pb = _add(cb, ob)
    return (_sub(pa, pb), pa, pb, ca, cb, oa, ob)


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

    simplex: list of support records.  Returns (v, lambdas, kept_indices)
    where v = sum(lambdas[i] * record[0][kept[i]]).  Every non-empty
    subset is examined; the closest point is the minimum-norm affine
    projection whose barycentric coordinates are all >= -_BARY_TOL.
    Singletons always succeed, so a result always exists.
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
# shared GJK engine
# ---------------------------------------------------------------------------

def _run_gjk(pts_a, pts_b, La, Lb, tol, max_iter):
    """GJK over the (possibly shelled) Minkowski difference.

    Returns (status, v2, simplex, lambdas, iterations, residual, scale).
    """
    scale = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > scale:
            scale = n2
    for Lm in (La, Lb):
        if Lm is not None:
            for i in range(3):
                for j in range(3):
                    a = abs(Lm[i][j])
                    if a > scale:
                        scale = a
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

    simplex = []          # list of support records
    v = None              # current closest point of conv(simplex)
    v2 = 0.0
    lam = ()
    residual = math.inf
    status = STATUS_NOT_CONVERGED
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        rec = _support_record(pts_a, pts_b, La, Lb, d)
        w = rec[0]
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
        simplex.append(rec)
        v, lam, keep = _closest_on_simplex(simplex)
        simplex = [simplex[i] for i in keep]
        v2 = _norm2(v)
        if v2 <= zero_tol2:
            # Origin lies in conv(simplex) subset C: shelled bodies
            # intersect (shells included).
            residual = 0.0
            status = STATUS_INTERSECTING
            break
        d = _neg(v)

    return status, v2, simplex, lam, iterations, residual


def _assemble_witnesses(simplex, lam):
    """Barycentric core witnesses, shell offsets and shelled witnesses."""
    cax = cay = caz = cbx = cby = cbz = 0.0
    oax = oay = oaz = obx = oby = obz = 0.0
    for rec, l in zip(simplex, lam):
        _, _, _, ca, cb, oa, ob = rec
        cax += l * ca[0]
        cay += l * ca[1]
        caz += l * ca[2]
        cbx += l * cb[0]
        cby += l * cb[1]
        cbz += l * cb[2]
        oax += l * oa[0]
        oay += l * oa[1]
        oaz += l * oa[2]
        obx += l * ob[0]
        oby += l * ob[1]
        obz += l * ob[2]
    core_a = (cax, cay, caz)
    core_b = (cbx, cby, cbz)
    offset_a = (oax, oay, oaz)
    offset_b = (obx, oby, obz)
    witness_a = _add(core_a, offset_a)
    witness_b = _add(core_b, offset_b)
    return core_a, core_b, offset_a, offset_b, witness_a, witness_b


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def distance(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Closest distance between conv(A) and conv(B) with witnesses.

    See the module docstring for the full contract.  No Minkowski shells.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    status, v2, simplex, lam, iterations, residual = _run_gjk(
        pts_a, pts_b, None, None, tol, max_iter)

    (_core_a, _core_b, _off_a, _off_b,
     witness_a, witness_b) = _assemble_witnesses(simplex, lam)

    dist = math.sqrt(v2)
    if status == STATUS_INTERSECTING:
        axis = None
    elif v2 > 0.0:
        inv = 1.0 / dist
        axis = tuple(c * inv for c in _sub(witness_a, witness_b))
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


def distance_shelled(A, B, LA=None, LB=None, *,
                     tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Distance between conv(A) + LA S2 and conv(B) + LB S2.

    ``LA`` / ``LB`` are ``None`` (no shell) or invertible 3x3 matrices.
    The shelled witnesses are returned split into core hull witnesses and
    ellipsoid offsets, together with a checkable distance bracket.  See
    the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    La = _validate_matrix(LA, "LA")
    Lb = _validate_matrix(LB, "LB")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    status, v2, simplex, lam, iterations, residual = _run_gjk(
        pts_a, pts_b, La, Lb, tol, max_iter)

    (core_a, core_b, offset_a, offset_b,
     witness_a, witness_b) = _assemble_witnesses(simplex, lam)

    dist = math.sqrt(v2)
    if status == STATUS_INTERSECTING:
        axis = None
        lower = upper = 0.0
    else:
        if v2 > 0.0:
            inv = 1.0 / dist
            axis = tuple(c * inv for c in _sub(witness_a, witness_b))
        else:
            axis = None
        # Support-gap certificate: every c in C = A' - B' obeys
        # c.v >= w.v, hence d* >= w.v/||v|| = ||v|| (1 - residual).
        rel = residual
        if rel < 0.0:
            rel = 0.0
        elif rel > 1.0:
            rel = 1.0
        lower = dist * (1.0 - rel)
        if lower < 0.0:
            lower = 0.0
        upper = dist

    return ShelledDistanceResult(
        status=status,
        distance=dist,
        witness_a=witness_a,
        witness_b=witness_b,
        core_a=core_a,
        core_b=core_b,
        offset_a=offset_a,
        offset_b=offset_b,
        separating_axis=axis,
        lower_bound=lower,
        upper_bound=upper,
        iterations=iterations,
        residual=residual,
        shell_a=La is not None,
        shell_b=Lb is not None,
    )
