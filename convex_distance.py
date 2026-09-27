"""Closest points and separation certificates for convex polytopes (GJK).

Public contract
---------------
distance(A, B, *, tol=1e-9, max_iter=128, warm_start=None) -> DistanceResult

* ``A`` / ``B``: sequences of 1..64 points; each point is a sequence of
  exactly 3 finite real numbers (``int`` / ``float`` / ``Fraction``;
  ``bool`` is rejected).  Each input is interpreted as the convex hull of
  its points.  Computation is performed in IEEE-754 double precision.
* ``tol``: relative convergence tolerance, a finite real with
  ``0 < tol < 1``.  ``bool`` is rejected.
* ``max_iter``: GJK iteration budget, an ``int`` with
  ``1 <= max_iter <= 128``.  ``bool`` is rejected.
* ``warm_start``: optional previous :class:`DistanceResult` from an
  earlier query (e.g. the previous animation frame).  Only the final
  simplex's support *indices* are reused: they are re-evaluated against
  the *new* inputs to seed the search with candidate points of the
  current Minkowski difference.  No cached distance, witness, or
  separation conclusion is ever trusted -- every reported result is
  recomputed and re-verified by the same convergence and containment
  checks as a cold run.  Indices that do not match the current input
  sizes (shape changed) are discarded; if nothing usable remains, the
  solver falls back to the normal cold initialisation.  Any other type
  is rejected with :class:`InputError`.

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

Warm-start reuse only affects the *search path* (and hence the reported
iteration count), never the acceptance criteria: a warm-started result
satisfies exactly the same certificate guarantees as a cold one.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "DistanceResult",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
]

__version__ = "1.0.0"

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
    # Internal warm-start token: support-index pairs of the final
    # simplex.  Not part of the mathematical result; only reused as
    # candidate starting points by a subsequent distance() call.
    _cache: tuple | None = None

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
    """Return (w, a, b, ai, bi): support of the Minkowski difference.

    a = A[ai] = argmax_A a.d, b = B[bi] = argmin_B b.d, w = a - b.
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
    return (_sub(a, b), a, b, ai, bi)


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

def _validate_warm_start(warm_start):
    if warm_start is None:
        return None
    if not isinstance(warm_start, DistanceResult):
        raise InputError(
            f"warm_start: expected a DistanceResult returned by an "
            f"earlier distance() call, got "
            f"{type(warm_start).__name__} ({warm_start!r})")
    return warm_start._cache


def distance(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER, warm_start=None):
    """Closest distance between conv(A) and conv(B) with witnesses.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)
    cache = _validate_warm_start(warm_start)

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
    simplex = []          # list of (w, a, b, ai, bi)
    v = None              # current closest point of conv(simplex)
    v2 = 0.0
    lam = ()
    residual = math.inf
    status = STATUS_NOT_CONVERGED
    iterations = 0

    # Warm start: re-evaluate the cached support indices against the
    # *current* points.  Each seeded w = A[ai] - B[bi] is a genuine
    # point of the current Minkowski difference, so the seed only
    # influences the search path; every conclusion below is still
    # established by the usual fresh convergence/containment checks.
    # Stale indices (shape changed) are dropped; if nothing usable
    # remains we fall through to the cold initialisation.
    seeded = False
    if cache:
        seen = set()
        for ai, bi in cache:
            if not (0 <= ai < na and 0 <= bi < nb):
                continue
            key = (ai, bi)
            if key in seen:
                continue
            seen.add(key)
            a = pts_a[ai]
            b = pts_b[bi]
            w = _sub(a, b)
            if any(w == s[0] for s in simplex):
                continue
            simplex.append((w, a, b, ai, bi))
        if simplex:
            v, lam, keep = _closest_on_simplex(simplex)
            simplex = [simplex[i] for i in keep]
            v2 = _norm2(v)
            seeded = True
            if v2 <= zero_tol2:
                # Seeded candidates already contain the origin: the
                # hulls intersect (proven on current inputs).
                residual = 0.0
                status = STATUS_INTERSECTING
    if not seeded:
        # Initial search direction: centroid difference (deterministic).
        d = _sub(ca, cb)
        if _norm2(d) == 0.0:
            d = (1.0, 0.0, 0.0)
    else:
        d = (-v[0], -v[1], -v[2])

    for it in range(1, max_iter + 1):
        if status == STATUS_INTERSECTING:
            break  # already proven by the warm-started seed simplex
        iterations = it
        w, sa, sb, ai, bi = _support(pts_a, pts_b, d)
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
        simplex.append((w, sa, sb, ai, bi))
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
    for (w, sa, sb, ai, bi), l in zip(simplex, lam):
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
        _cache=tuple((s[3], s[4]) for s in simplex),
    )
