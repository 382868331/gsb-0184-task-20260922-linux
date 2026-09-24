"""Closest points and separation certificates for convex polytopes (GJK).

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
                 "intersecting".  For "separated" it is <= tol, with one
                 exception: when the iteration stops because a support
                 point repeats at the float64 noise floor (the remaining
                 improvement of ||v||^2 is below machine resolution), the
                 best iterate is reported with the honest residual, which
                 may then exceed tol; ``distance`` itself is already at
                 the float64 optimum in that case.  For "not_converged"
                 the residual is > tol and bounds the remaining relative
                 uncertainty of ``distance``.

"not_converged" (search budget exhausted) is reported distinctly from
"intersecting" (origin proven inside the Minkowski difference); the two
are never conflated.  Touching hulls (distance exactly zero) are reported
as "intersecting" with distance <= the zero threshold.

sweep(A, B, velocity, t_window, *, tol=1e-9, max_iter=128) -> SweepResult
-------------------------------------------------------------------------
Translational sweep: ``A`` is static, ``B`` moves with the constant
velocity ``velocity`` (a rigid translation, no rotation; only the
relative motion matters, so this covers any pair of constant-velocity
translations).  ``t_window = (t0, t1)`` with ``t0 < t1`` is the time
window.  The contact set of a line with a convex body is an interval,
so the answer is the first contact time ``t_enter`` and the last
contact time ``t_exit`` inside the window.

SweepResult fields
------------------
status               "miss" | "touching" | "intersecting"
                     ("touching": contact interval degenerates to one
                     instant; "intersecting": contact over an interval
                     of positive length, including initial overlap)
t_enter / t_exit     first / last contact time in the window; None on miss
initially_overlapping  True when the bodies already touch/overlap at t0
stationary           True when velocity is exactly (0, 0, 0)
relative_velocity    the velocity actually used (B relative to A)
window               (t0, t1) as validated
t_closest            time of minimum distance inside the window
min_distance         distance at t_closest (<= zero_tolerance on contact)
witness_a / witness_b  closest points in world coordinates at t_enter
                     (at t_closest on a miss)
separating_axis      unit certificate direction at t_closest on a miss;
                     None on contact
evaluations          number of GJK distance evaluations used
zero_tolerance       distance threshold used to decide contact

Invalid input raises InputError before any computation; nothing is
silently corrected and no partial state is left behind.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "DistanceResult",
    "sweep",
    "SweepResult",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
    "STATUS_MISS",
    "STATUS_TOUCHING",
]

__version__ = "1.1.0"

MAX_POINTS = 64
MAX_ITER = 128
DEFAULT_TOL = 1e-9

STATUS_SEPARATED = "separated"
STATUS_INTERSECTING = "intersecting"
STATUS_NOT_CONVERGED = "not_converged"

# Sweep (translational contact) statuses.
STATUS_MISS = "miss"
STATUS_TOUCHING = "touching"

# Origin-containment threshold, relative to the input scale.  Distinct
# from ``tol``: this decides "intersecting" vs "separated".
_ZERO_TOL_REL = 1e-11
# Feasibility slack for barycentric coordinates (dimensionless).  Must
# stay well below the origin-containment threshold: a lambda of -eps can
# underestimate the true convex distance by ~eps * simplex diameter, so
# 1e-12 keeps that underestimate below the 1e-11 zero threshold while
# remaining ~1000x above the KKT solver's own float64 error.
_BARY_TOL = 1e-12
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
class SweepResult:
    status: str
    t_enter: float | None
    t_exit: float | None
    initially_overlapping: bool
    stationary: bool
    relative_velocity: tuple
    window: tuple
    t_closest: float
    min_distance: float
    witness_a: tuple | None
    witness_b: tuple | None
    separating_axis: tuple | None
    evaluations: int
    zero_tolerance: float

    @property
    def contact(self) -> bool:
        return self.status != STATUS_MISS

    @property
    def contact_interval(self):
        if self.t_enter is None:
            return None
        return (self.t_enter, self.t_exit)


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


def _validate_vector(vec, name):
    """Validate a single 3-vector of finite reals (velocity etc.)."""
    if isinstance(vec, (str, bytes)):
        raise InputError(f"{name}: expected a 3-component vector, got text")
    try:
        comps = list(vec)
    except TypeError:
        raise InputError(f"{name}: expected a 3-component vector, "
                         f"got {type(vec).__name__}") from None
    if len(comps) != 3:
        raise InputError(f"{name}: expected exactly 3 components, "
                         f"got {len(comps)}")
    out = []
    for j, c in enumerate(comps):
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise InputError(f"{name}[{j}]: expected a real number, "
                             f"got {type(c).__name__} ({c!r})")
        f = float(c)
        if not math.isfinite(f):
            raise InputError(f"{name}[{j}]: non-finite component "
                             f"{c!r} (NaN/Infinity rejected)")
        out.append(f)
    return (out[0], out[1], out[2])


def _validate_window(t_window):
    if isinstance(t_window, (str, bytes)):
        raise InputError("t_window: expected a (t0, t1) pair, got text")
    try:
        comps = list(t_window)
    except TypeError:
        raise InputError(f"t_window: expected a (t0, t1) pair, "
                         f"got {type(t_window).__name__}") from None
    if len(comps) != 2:
        raise InputError(f"t_window: expected exactly 2 endpoints, "
                         f"got {len(comps)}")
    out = []
    for j, c in enumerate(comps):
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise InputError(f"t_window[{j}]: expected a real number, "
                             f"got {type(c).__name__} ({c!r})")
        f = float(c)
        if not math.isfinite(f):
            raise InputError(f"t_window[{j}]: non-finite endpoint "
                             f"{c!r} (NaN/Infinity rejected)")
        out.append(f)
    if not out[0] < out[1]:
        raise InputError(f"t_window: expected t0 < t1, "
                         f"got ({out[0]!r}, {out[1]!r})")
    return (out[0], out[1])


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

def _input_scale(pts_a, pts_b):
    """max(1, largest squared coordinate norm) of both point sets."""
    s2 = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > s2:
            s2 = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > s2:
            s2 = n2
    return math.sqrt(s2)


def _gjk_distance(pts_a, pts_b, tol, max_iter, scale):
    """Core GJK closest-distance computation on validated points.

    ``scale`` is the reference length for the origin-containment zero
    threshold; callers sweeping a body over time pass one fixed scale so
    the threshold does not drift between evaluations.
    """
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
    seen = set()          # every support point w ever produced
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
        if w in seen:
            # Support point produced before but since dropped from the
            # simplex: the deterministic iteration is cycling because the
            # possible improvement of v2 is below the float64 resolution.
            # The current iterate is the best float64 can do; report it
            # with the honest remaining residual (which may exceed tol
            # near the zero threshold -- see the module docstring).
            if v2 > 0.0:
                residual = max(0.0, (v2 - _dot(w, v)) / v2)
            else:
                residual = 0.0
            status = STATUS_SEPARATED
            break
        seen.add(w)
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


def distance(A, B, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Closest distance between conv(A) and conv(B) with witnesses.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)
    return _gjk_distance(pts_a, pts_b, tol, max_iter,
                         _input_scale(pts_a, pts_b))


# ---------------------------------------------------------------------------
# translational sweep: first contact and contact interval in a time window
# ---------------------------------------------------------------------------
#
# B(t) = B + t * velocity, A static.  d(t) = dist(A, B(t)) is the distance
# from the point t * velocity to the fixed convex Minkowski difference
# A - B, hence a convex function of t; its zero set {t : d(t) = 0} is the
# intersection of a line with a convex body, i.e. a single interval.  The
# sweep therefore reduces to: minimise d on the window (golden section on
# a convex function), then bisect each side of the minimiser for the
# contact boundary.  For polytopes d is piecewise linear in t, so a
# tangential (zero-length) contact stays sharply distinguishable from a
# positive-length contact interval at the zero threshold.

def sweep(A, B, velocity, t_window, *, tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """First contact and contact interval of B translating past A.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    vel = _validate_vector(velocity, "velocity")
    t0, t1 = _validate_window(t_window)
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    # One fixed reference scale for the whole swept motion (positions at
    # t = 0, t0, t1), so the contact threshold is time-invariant.
    s2 = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > s2:
            s2 = n2
    for b in pts_b:
        for t in (0.0, t0, t1):
            n2 = _norm2((b[0] + t * vel[0],
                         b[1] + t * vel[1],
                         b[2] + t * vel[2]))
            if n2 > s2:
                s2 = n2
    scale = math.sqrt(s2)
    zero_tol = _ZERO_TOL_REL * scale

    evaluations = 0

    def query(t):
        nonlocal evaluations
        evaluations += 1
        shifted = [(b[0] + t * vel[0], b[1] + t * vel[1], b[2] + t * vel[2])
                   for b in pts_b]
        return _gjk_distance(pts_a, shifted, tol, max_iter, scale)

    def touches(t):
        return query(t).status == STATUS_INTERSECTING

    speed2 = _norm2(vel)
    r0 = query(t0)
    initially = r0.status == STATUS_INTERSECTING

    def result(status, t_enter, t_exit, t_closest, representative):
        return SweepResult(
            status=status,
            t_enter=t_enter,
            t_exit=t_exit,
            initially_overlapping=initially,
            stationary=speed2 == 0.0,
            relative_velocity=vel,
            window=(t0, t1),
            t_closest=t_closest,
            min_distance=representative.distance,
            witness_a=representative.witness_a,
            witness_b=representative.witness_b,
            separating_axis=representative.separating_axis,
            evaluations=evaluations,
            zero_tolerance=zero_tol,
        )

    # Relative rest: d(t) is constant, one evaluation decides everything.
    if speed2 == 0.0:
        if initially:
            return result(STATUS_INTERSECTING, t0, t1, t0, r0)
        return result(STATUS_MISS, None, None, t0, r0)

    speed = math.sqrt(speed2)
    # Time resolution: the distance profile crosses the zero threshold
    # with slope at most ``speed``, so resolving t to ~1e-3 *
    # zero_tol / speed places the sampled minimum two orders of
    # magnitude below the contact threshold; plus an absolute floor
    # (~1000 ulps of the window magnitude) to stay float-representable.
    t_tol = max(1e-13 * max(1.0, abs(t0), abs(t1)),
                1e-3 * zero_tol / speed)

    # Golden-section minimisation of the convex function d(t).
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = t0, t1
    c = hi - gr * (hi - lo)
    d_ = lo + gr * (hi - lo)
    fc = query(c).distance
    fd = query(d_).distance
    for _ in range(200):
        if hi - lo <= t_tol:
            break
        if fc < fd:
            hi = d_
            d_ = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = query(c).distance
        else:
            lo = c
            c = d_
            fc = fd
            d_ = lo + gr * (hi - lo)
            fd = query(d_).distance
    t_star = 0.5 * (lo + hi)
    r_star = query(t_star)

    if r_star.status != STATUS_INTERSECTING:
        # Minimum stays above the zero threshold: complete miss.
        return result(STATUS_MISS, None, None, t_star, r_star)

    # Contact exists.  Bracket the entry on [t0, t*] and the exit on
    # [t*, t1]; d is non-increasing / non-decreasing on those ranges.
    if initially:
        t_enter = t0
    else:
        lo_, hi_ = t0, t_star   # touches(lo_) False, touches(hi_) True
        for _ in range(100):
            if hi_ - lo_ <= t_tol:
                break
            mid = 0.5 * (lo_ + hi_)
            if touches(mid):
                hi_ = mid
            else:
                lo_ = mid
        t_enter = hi_

    if touches(t1):
        t_exit = t1
    else:
        lo_, hi_ = t_star, t1   # touches(lo_) True, touches(hi_) False
        for _ in range(100):
            if hi_ - lo_ <= t_tol:
                break
            mid = 0.5 * (lo_ + hi_)
            if touches(mid):
                lo_ = mid
            else:
                hi_ = mid
        t_exit = lo_

    # A tangential graze has a zero-length contact interval; polytope
    # distance profiles are piecewise linear, so the measured interval at
    # the zero threshold stays within O(zero_tol / speed) of the true one
    # and a clear gap separates the two cases.
    touch_tol = max(64.0 * t_tol, 1e-9 * max(1.0, abs(t0), abs(t1)))
    if t_exit - t_enter <= touch_tol:
        status = STATUS_TOUCHING
    else:
        status = STATUS_INTERSECTING

    r_enter = r0 if t_enter == t0 else query(t_enter)
    return result(status, t_enter, t_exit, t_star, r_enter)
