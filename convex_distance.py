"""Closest points, separation certificates and translational sweeps.

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

sweep_contact(A, B, velocity, t_start, t_end, *, tol=1e-9, max_iter=128)
    -> SweepResult

  Body A undergoes the fixed translation ``A + velocity * t``; B is
  stationary (only the *relative* motion matters, so the convention is
  immaterial).  Reports the first contact time, the intersection
  interval within ``[t_start, t_end]`` and reviewable contact witnesses.
  Handles initial overlap, tangential (single-instant) contact, relative
  rest and complete miss.  ``velocity`` is one vector of 3 finite reals;
  ``t_start``/``t_end`` are finite reals with ``t_start < t_end``.
  ``max_iter`` is the per-GJK-solve iteration budget (``1..128``); the
  sweep issues a bounded number of such solves.

Invalid input raises :class:`InputError` (a ``ValueError``) with a message
that locates the offending element, e.g. ``"A[3][1]: ..."``.  Nothing is
silently corrected; both entry points are pure functions, so a rejected
call never leaves partially updated state.

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

SweepResult fields
------------------
status           "separated" (never meet) | "intersecting" (an overlap
                 interval of positive duration exists) | "touching"
                 (contact at exactly one time instant) |
                 "not_converged".
first_contact    ContactInfo at entry, or None when separated; when the
                 bodies already overlap at t_start its time is t_start.
last_contact     ContactInfo at exit, or None when separated; equals
                 first_contact for a touch; time t_end when the bodies
                 are still overlapping at the window end.
interval         (t_enter, t_exit) overlap interval clipped to the
                 window, or None when separated; for "touching" both
                 ends are equal.
initial_state    "separated" | "intersecting" at t_start.
relative_rest    True when the relative velocity is numerically zero.
closest_time / closest_distance
                 time and value of the minimum separation over the
                 whole window (0.0 at/throughout contact).
iterations       total GJK iterations used by all internal solves.
residual         final relative gap of the last GJK solve.

ContactInfo fields: ``time``, ``point_a``, ``point_b`` (world-coordinate
witnesses at that time, coincident within numerical error at contact),
``normal`` (unit axis from B toward A; ``None`` when it is not defined,
e.g. deep overlap at the window boundary), ``kind`` -- one of "impact",
"touch", "separation", "initial_overlap", "window_end", "rest_overlap".

Mathematical basis of the sweep: with C(t) = C0 + velocity * t the
squared separation d(t)^2 = dist(0, C(t))^2 is a *convex* function of t
(projection norm onto an affinely moving convex set), so there is at
most one overlap interval and d'(t) is non-decreasing.  While the GJK
active simplex persists, the closest point moves affinely: its KKT
Hessian (the Gram matrix of the active vertices) is fixed, hence the
barycentric weights vary linearly and the closest point obeys
x(s) = x0 + q s with q = velocity + sum_l h_l w_l.  The march predicts
the stationary point of this quadratic, *validates the prediction with
an independent GJK solve* and halves the step when the active feature
switched first; once a validated solve has d' >= 0 convexity proves
contact can never follow.  The zero crossings themselves are found by
bracketed GJK bisection (robust to the degenerate exact-tangency
instant, where the static solver may cycle, via straddling jittered
probes) and refined linearly from the separating side's distance and
normal rate.  An overlap shorter than the resolvable contact-time
resolution (float ulp or zero-threshold distance over the approach
rate) is reported honestly as "touching".

For the static entry, "not_converged" (search budget exhausted) is
reported distinctly from "intersecting" (origin proven inside the
Minkowski difference); the two are never conflated.  Touching hulls
(distance exactly zero) are reported as "intersecting" with distance
<= the zero threshold.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "sweep_contact",
    "DistanceResult",
    "SweepResult",
    "ContactInfo",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "MAX_SWEEP_EVENTS",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
    "STATUS_TOUCHING",
]

__version__ = "2.0.0"

MAX_POINTS = 64
MAX_ITER = 128
DEFAULT_TOL = 1e-9
# Bound on feature hops in one sweep.  Convexity makes the real number of
# hops tiny (the active-feature minimum marches monotonically to the
# global minimum); the cap only guards against pathological cycling and
# its exhaustion is reported as "not_converged".
MAX_SWEEP_EVENTS = 256

STATUS_SEPARATED = "separated"
STATUS_INTERSECTING = "intersecting"
STATUS_NOT_CONVERGED = "not_converged"
# Sweep-only status: contact at exactly one time instant (grazing).
STATUS_TOUCHING = "touching"

# Origin-containment threshold, relative to the input scale.  Distinct
# from ``tol``: this decides "intersecting" vs "separated".
_ZERO_TOL_REL = 1e-11
# Feasibility slack for barycentric coordinates (dimensionless).
_BARY_TOL = 1e-9
# Singularity threshold for the KKT solves, relative to the Gram scale.
_KKT_TOL_REL = 1e-12
# Bisection budget for locating a root of the distance (each halving is an
# independent GJK solve).  64 halvings drive the bracket far below one
# float ulp of the window, after which midpoints stop moving anyway.
_ROOT_BISECT_STEPS = 64
# A zero velocity is one whose norm is below this fraction of the scale.
_REST_VEL_REL = 1e-14
# Time progress below this fraction of the window counts as "already at
# the stationary point" for the affine march.
_TIME_EPS_REL = 1e-14
# Two bracketed roots closer than this many float ulps of their magnitude
# are a single tangency: a shorter positive interval is not representable
# in double precision anyway.
_TOUCH_ULP = 8.0


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
class ContactInfo:
    """Reviewable record of one contact instant."""

    time: float
    point_a: tuple
    point_b: tuple
    normal: tuple | None
    kind: str


@dataclass(frozen=True)
class SweepResult:
    status: str
    first_contact: ContactInfo | None
    last_contact: ContactInfo | None
    interval: tuple | None
    initial_state: str
    relative_rest: bool
    closest_time: float
    closest_distance: float
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


def _add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _mul(u, s):
    return (u[0] * s, u[1] * s, u[2] * s)


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


def _validate_velocity(velocity):
    if isinstance(velocity, (str, bytes)):
        raise InputError("velocity: expected a 3-component vector, got text")
    try:
        comps = list(velocity)
    except TypeError:
        raise InputError("velocity: expected a 3-component vector, "
                         f"got {type(velocity).__name__}") from None
    if len(comps) != 3:
        raise InputError("velocity: expected exactly 3 components, "
                         f"got {len(comps)}")
    out = []
    for j, c in enumerate(comps):
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise InputError(f"velocity[{j}]: expected a real number, "
                             f"got {type(c).__name__} ({c!r})")
        f = float(c)
        if not math.isfinite(f):
            raise InputError(f"velocity[{j}]: non-finite component {c!r} "
                             "(NaN/Infinity rejected)")
        out.append(f)
    return (out[0], out[1], out[2])


def _validate_time(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InputError(f"{name}: expected a real number, "
                         f"got {type(value).__name__} ({value!r})")
    t = float(value)
    if not math.isfinite(t):
        raise InputError(f"{name}: expected a finite time, got {value!r}")
    return t


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

    The Gram block is divided by its largest entry first.  Without this
    scaling the trailing pivot (a squared affine edge divided by a
    Gram-scale number) is systematically tiny for long support vectors
    and non-degenerate segments were wrongly declared singular, which
    made GJK cycle forever on large coordinates.
    """
    k = len(ws)
    if k == 1:
        return (1.0,)
    gscale = 0.0
    for wi in ws:
        for wj in ws:
            d = abs(_dot(wi, wj))
            if d > gscale:
                gscale = d
    if gscale == 0.0:
        return None
    n = k + 1
    m = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(k):
        wi = ws[i]
        for j in range(k):
            m[i][j] = _dot(wi, ws[j]) / gscale
        m[i][k] = 1.0
        m[k][i] = 1.0
    m[k][n] = 1.0
    eps = _KKT_TOL_REL
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
# GJK solver core.  A solve record carries the active simplex plus the
# sample time, so sweeps can translate the warm-start records to later
# times (each support vertex moves with the relative velocity).
# ---------------------------------------------------------------------------

def _gjk_solve(pts_a, pts_b, tol, max_iter, scale, init=None):
    """Closest-point GJK on conv(A) - conv(B).

    init: optional warm start, a list of (w, sa, sb, t_sample) records in
    world coordinates at ``t_sample``; the caller translates them.

    Returns a dict with status, v, v2, simplex (records), lam (weights
    aligned with the simplex), iterations, residual.
    """
    zero_tol2 = (_ZERO_TOL_REL * scale) ** 2

    if init:
        simplex = [(rec[0], rec[1], rec[2]) for rec in init]
        v, lam, keep = _closest_on_simplex(simplex)
        simplex = [simplex[i] for i in keep]
        v2 = _norm2(v)
        if v2 <= zero_tol2:
            return dict(status=STATUS_INTERSECTING, v=v, v2=v2,
                        simplex=simplex, lam=lam, iterations=0,
                        residual=0.0)
        d = (-v[0], -v[1], -v[2])
        v_init = v
    else:
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

    return dict(status=status, v=v, v2=v2, simplex=simplex, lam=lam,
                iterations=iterations, residual=residual)


# ---------------------------------------------------------------------------
# static entry point
# ---------------------------------------------------------------------------

def _input_scale(pts_a, pts_b):
    scale = 1.0
    for pts in (pts_a, pts_b):
        for p in pts:
            n2 = _norm2(p)
            if n2 > scale:
                scale = n2
    return math.sqrt(scale)


def _witnesses(simplex, lam):
    """Barycentric witness points from active simplex records."""
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
    sol = _gjk_solve(pts_a, pts_b, tol, max_iter, scale)

    v2 = sol["v2"]
    dist = math.sqrt(max(0.0, v2))
    witness_a, witness_b = _witnesses(sol["simplex"], sol["lam"])
    if sol["status"] == STATUS_INTERSECTING or v2 <= 0.0:
        axis = None
    else:
        inv = 1.0 / dist
        v = sol["v"]
        axis = (v[0] * inv, v[1] * inv, v[2] * inv)

    return DistanceResult(
        status=sol["status"],
        distance=dist,
        witness_a=witness_a,
        witness_b=witness_b,
        separating_axis=axis,
        iterations=sol["iterations"],
        residual=sol["residual"],
    )


# ---------------------------------------------------------------------------
# translational sweep (continuous collision detection, linear motion)
#
# A moves as A(t) = A + v t, B is fixed; the Minkowski difference is
# C(t) = C0 + v t.  The squared separation f(t) = dist(0, C(t))**2 is
# convex in t (projection of the origin onto an affinely moving convex
# set), hence there is at most one contact interval and f' never
# decreases: once f' >= 0, future contact is impossible.
#
# The sweep advances in *certified feature pieces*.  While the active GJK
# simplex {w_l} and its barycentric weights stay optimal, the KKT system
# for the weights has a fixed Hessian G0 (w_l . w_m), which makes the
# weights affine in the local time s:
#     G0 h = m*1 - g,  1.h = 0,   g_l = w_l . v,
# and the closest point moves affinely
#     x(s) = x0 + q s,  q = v + sum_l h_l w_l.
# A piece is certified up to the first s at which the active set can
# change: a weight hits zero, or any Minkowski vertex c_ij(s) =
# (a_i + v s) - b_j violates the closest-point inequality
# (c_ij(s) - x(s)) . x(s) >= 0 (a quadratic in s, enumerated over all
# vertex pairs).  Contact inside a certified piece is the closed-form
# quadratic root of ||x0 + q s||^2 = 0, additionally verified by an
# independent GJK solve; at every piece boundary GJK restarts (warm
# started by translating the old simplex records).
# ---------------------------------------------------------------------------

def _translated(points, velocity, t):
    return [_add(p, _mul(velocity, t)) for p in points]


def _carry(records, velocity, dt):
    """Translate warm-start simplex records by velocity * dt."""
    shift = _mul(velocity, dt)
    return [(_add(w, shift), _add(sa, shift), sb)
            for (w, sa, sb) in records]


def _solve_at(pts_a0, pts_b, velocity, t, tol, max_iter, scale,
              warm=None, cold=False):
    """GJK between A + velocity*t and B.

    warm: (records, t_sample) carried over from a previous solve.
    """
    At = _translated(pts_a0, velocity, t)
    init = None
    if warm is not None and not cold:
        records, t_sample = warm
        init = _carry(records, velocity, t - t_sample)
    sol = _gjk_solve(At, pts_b, tol, max_iter, scale, init=init)
    if (sol["status"] == STATUS_NOT_CONVERGED and init is not None
            and not cold):
        # A warm simplex that contains or nearly contains the origin can
        # trap GJK in cycling near a tangency.  One cold restart escapes
        # it (the static solver converges from the centroid direction).
        sol = _gjk_solve(At, pts_b, tol, max_iter, scale, init=None)
    sol["time"] = t
    return sol


def _piece_law(simplex, lam, velocity):
    """Affine law of the closest point on the current feature piece.

    Returns (h, x0, q): weights lambda(s) = lam + h*s, closest point
    x(s) = x0 + q*s, for local time s while the feature persists.
    """
    k = len(simplex)
    ws = [rec[0] for rec in simplex]
    x0 = (0.0, 0.0, 0.0)
    for l, w in zip(lam, ws):
        x0 = _add(x0, _mul(w, l))
    g = tuple(_dot(w, velocity) for w in ws)
    # [G0 1; 1^T 0] [h; m] = [-g; 0], same matrix structure as the
    # barycentric KKT solve.
    n = k + 1
    m = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for i in range(k):
        for j in range(k):
            m[i][j] = _dot(ws[i], ws[j])
        m[i][k] = 1.0
        m[k][i] = 1.0
        rhs[i] = -g[i]
    for col in range(n):
        piv = col
        for rr in range(col + 1, n):
            if abs(m[rr][col]) > abs(m[piv][col]):
                piv = rr
        m[col], m[piv] = m[piv], m[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        inv = 1.0 / m[col][col]
        for rr in range(col + 1, n):
            f = m[rr][col] * inv
            if f == 0.0:
                continue
            for cc in range(col, n):
                m[rr][cc] -= f * m[col][cc]
            rhs[rr] -= f * rhs[col]
    z = [0.0] * n
    for rr in range(n - 1, -1, -1):
        s = rhs[rr] - sum(m[rr][cc] * z[cc] for cc in range(rr + 1, n))
        z[rr] = s / m[rr][rr]
    h = tuple(z[:k])
    q = velocity
    for hl, w in zip(h, ws):
        q = _add(q, _mul(w, hl))
    return h, x0, q


def _sep_contact_ci(t, sol, scale, kind, velocity=None):
    """ContactInfo at bracketed root t using a separated-side solve.

    The solve's A-side witness is recorded at sol["time"]; shift it to t
    along the motion before averaging the two witnesses.  The normal is
    taken from the unshifted witness gap (it is invariant under the time
    shift and would otherwise be destroyed when the refinement lands the
    shifted A witness exactly on the B witness)."""
    wa, wb = _witnesses(sol["simplex"], sol["lam"])
    g = _sub(wa, wb)
    n = None
    if _dot(g, g) > 0.0:
        inv = 1.0 / math.sqrt(_dot(g, g))
        n = (g[0] * inv, g[1] * inv, g[2] * inv)
    if velocity is not None:
        dt = t - sol["time"]
        wa = _add(wa, _mul(velocity, dt))
    mid = tuple(0.5 * (wa[k] + wb[k]) for k in range(3))
    return ContactInfo(time=t, point_a=mid, point_b=mid, normal=n,
                       kind=kind)


def _ulp_span(t):
    return _TOUCH_ULP * (math.ulp(abs(t)) if t != 0.0 else 1e-16)


def _zero_time(scale, rate):
    """Time equivalent of the containment zero threshold at separation
    rate ``rate``: the finest contact-time resolution the static GJK
    classification can certify."""
    zero_d = _ZERO_TOL_REL * scale
    if rate <= 0.0:
        return math.inf
    return zero_d / rate


def _refine_root(t_sep, sol_sep, velocity, lo, hi, scale, side):
    """Linear refinement of a bracketed distance root.

    The separated sample t_sep carries distance d and unit closest
    direction n; the local separation rate is |v . n|.  ``side`` is +1
    when the root lies above t_sep (entry bisection), -1 when it lies
    below (exit bisection).  Pure status bisection brackets the
    zero-THRESHOLD crossing, so both bracket ends can sit on one side of
    the true root -- hence the projection is only clamped to the overall
    search interval [lo, hi], never to the bracket.
    """
    d = math.sqrt(max(0.0, sol_sep["v2"]))
    vv = sol_sep["v"]
    n2 = _dot(vv, vv)
    if n2 <= 0.0:
        return t_sep
    rate = abs(_dot(velocity, vv)) / math.sqrt(n2)
    if rate <= 1e-300:
        return t_sep
    root = t_sep + side * d / rate
    return min(max(root, lo), hi)


def _classify(pts_a0, pts_b, velocity, t, tol, max_iter, scale,
              warm=None):
    """GJK classification at t, robust to cycling at a degenerate root.

    The static GJK solver can exhaust its budget at the exact tangency
    instant (the active simplex is affinely degenerate there).  Probes
    jittered to both sides then decide: opposite statuses mean t is the
    contact boundary itself ("boundary", sep_sol_below, in_sol_above,
    delta); agreeing statuses classify t directly.
    """
    sol = _solve_at(pts_a0, pts_b, velocity, t, tol, max_iter, scale,
                    warm=warm)
    if sol["status"] != STATUS_NOT_CONVERGED:
        return sol["status"], sol, 0.0
    span = scale * 1e-12
    for _ in range(6):
        below = _solve_at(pts_a0, pts_b, velocity, t - span, tol,
                          max_iter, scale)
        above = _solve_at(pts_a0, pts_b, velocity, t + span, tol,
                          max_iter, scale)
        sb = below["status"] != STATUS_NOT_CONVERGED
        sa = above["status"] != STATUS_NOT_CONVERGED
        if sb and sa:
            if below["status"] != above["status"]:
                return ("boundary", below, above, span)
            return below["status"], below, span
        span *= 100.0
    return STATUS_NOT_CONVERGED, sol, 0.0


def _bisect_exit(pts_a0, pts_b, velocity, t_lo, t_hi, sol_hi_sep,
                 tol, max_iter, scale, total_it):
    """Exit root in (t_lo, t_hi): bodies overlap at/near t_lo, sol_hi_sep
    proves separation at t_hi.  Midpoints classify themselves by GJK.

    Returns (root, sep_sol, deepest_in, total_it) where ``deepest_in`` is
    the largest time at which an intersecting sample was found (t_lo when
    the overlap has no classifiable interior, i.e. tangency); or None.
    """
    a, b = t_lo, t_hi
    sb = sol_hi_sep
    sa = None
    deepest = t_lo
    for _ in range(_ROOT_BISECT_STEPS):
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        warm_rec = sa["simplex"] if sa is not None else sb["simplex"]
        warm_t = sa["time"] if sa is not None else sb["time"]
        cls = _classify(pts_a0, pts_b, velocity, mid, tol, max_iter,
                        scale, warm=(warm_rec, warm_t))
        total_it += cls[1]["iterations"]
        status = cls[0]
        if status == "boundary":
            below, above, delta = cls[1], cls[2], cls[3]
            total_it += above["iterations"]
            if below["status"] == STATUS_INTERSECTING:
                a, sa = mid - delta, below
                if a > deepest:
                    deepest = a
                b, sb = mid + delta, above
            else:
                b, sb = mid - delta, below
                a, sa = mid + delta, above
                if a > deepest:
                    deepest = a
            continue
        if status == STATUS_NOT_CONVERGED:
            return None
        if status == STATUS_INTERSECTING:
            a, sa = mid, cls[1]
            if mid > deepest:
                deepest = mid
        else:
            b, sb = mid, cls[1]
    root = _refine_root(b, sb, velocity, t_lo, t_hi, scale, side=-1)
    return root, sb, deepest, total_it


def _march_separated(pts_a0, pts_b, velocity, t0, t_end, sol0,
                     tol, max_iter, scale, total_it):
    """Sample-validated march forward from a separated state.

    The affine feature law predicts the stationary point s* of the
    current quadratic piece; a real GJK solve validates the prediction.
    On mismatch (a feature switch happened first) the step is halved and
    retried.  Sampling the predicted minimum guarantees a contact can
    never be skipped: by convexity of d(t), if the predicted minimum is
    positive, distance is non-decreasing past it -- contact after that
    would contradict convexity -- so f' >= 0 confirmed by the solve ends
    the search.  A non-converged classification at a degenerate instant
    is resolved by straddling probes.

    Returns (tag, payload, total_it):
      "miss"          (best_time, best_distance, residual)
      "touch"         (ContactInfo, residual)
      "impact"        (t_in, ci_in, exit_info, residual), exit_info
                      None (still overlapping at t_end) or (t_out, ci_out)
      "not_converged" residual
    """
    cur_t = t0
    cur_sol = sol0
    best_t = t0
    best_d = math.sqrt(max(0.0, sol0["v2"]))
    residual = sol0["residual"]
    span = t_end - t0
    rebuilds = 0

    for _ in range(MAX_SWEEP_EVENTS * 4):
        h, x0, q = _piece_law(cur_sol["simplex"], cur_sol["lam"],
                              velocity)
        a = _dot(q, q)
        xq = _dot(x0, q)
        remaining = t_end - cur_t

        # Global convexity certificate: non-decreasing distance now.
        if xq >= 0.0 and remaining > 0.0:
            probe_t = min(t_end, cur_t + max(span * 1e-12,
                                             _ulp_span(cur_t)))
            cls = _classify(pts_a0, pts_b, velocity, probe_t, tol,
                            max_iter, scale,
                            warm=(cur_sol["simplex"], cur_t))
            total_it += cls[1]["iterations"]
            if cls[0] == "boundary":
                total_it += cls[2]["iterations"]
                ci = _sep_contact_ci(probe_t, cls[1], scale, "touch",
                                     velocity)
                return ("touch", (ci, residual), total_it)
            if cls[0] != STATUS_NOT_CONVERGED:
                psol = cls[1]
                pd = math.sqrt(max(0.0, psol["v2"]))
                if psol["status"] == STATUS_INTERSECTING or pd < best_d:
                    # Convexity contradicted by the probe => the affine
                    # law is stale; fall through to re-step (handled by
                    # the mismatch path below).
                    if psol["status"] == STATUS_INTERSECTING:
                        return _bisect_entry_and_finish(
                            pts_a0, pts_b, velocity, t_end, cur_t,
                            cur_sol, probe_t, psol, tol, max_iter,
                            scale, total_it, residual)
                else:
                    if pd < best_d:
                        best_t, best_d = probe_t, pd
                    return ("miss", (best_t, best_d, residual),
                            total_it)

        # Predict the quadratic stationary point on this piece.
        if a > 0.0 and xq < 0.0:
            step = min(-xq / a, remaining)
        else:
            step = remaining
        if step <= 0.0:
            step = remaining

        target_t = cur_t + step
        attempt = step
        nxt = None
        # Validate the affine prediction by a real GJK solve; halve on
        # feature mismatch.
        for _halves in range(40):
            cls = _classify(pts_a0, pts_b, velocity, target_t, tol,
                            max_iter, scale,
                            warm=(cur_sol["simplex"], cur_t))
            total_it += cls[1]["iterations"]
            if cls[0] == "boundary":
                total_it += cls[2]["iterations"]
                ci = _sep_contact_ci(target_t, cls[1], scale, "impact",
                                     velocity)
                return _finish_impact(pts_a0, pts_b, velocity, target_t,
                                      ci, t_end, cls[2], tol, max_iter,
                                      scale, total_it, residual)
            if cls[0] == STATUS_NOT_CONVERGED:
                return ("not_converged", cls[1]["residual"], total_it)
            cand = cls[1]
            if cand["status"] == STATUS_INTERSECTING:
                return _bisect_entry_and_finish(
                    pts_a0, pts_b, velocity, t_end, cur_t, cur_sol,
                    target_t, cand, tol, max_iter, scale, total_it,
                    residual)
            xp = (x0[0] + q[0] * attempt,
                  x0[1] + q[1] * attempt,
                  x0[2] + q[2] * attempt)
            vv = cand["v"]
            err = math.sqrt(max(0.0, _dot(
                (vv[0] - xp[0], vv[1] - xp[1], vv[2] - xp[2]),
                (vv[0] - xp[0], vv[1] - xp[1], vv[2] - xp[2]))))
            ref = max(1.0, math.sqrt(_dot(vv, vv)),
                      math.sqrt(_dot(xp, xp)))
            tol_d = max(1e-8 * ref, 1e-10 * scale,
                        _ZERO_TOL_REL * scale)
            if err <= tol_d:
                nxt = cand
                break
            # Feature switched inside the step: halve to stay on the
            # current verified piece and approach the switch.
            attempt *= 0.5
            target_t = cur_t + attempt

        nudged = False
        if nxt is None:
            # The affine law disagrees with GJK even at ulp-sized steps:
            # the active feature switches exactly at cur_t, so a simplex
            # rebuilt at cur_t is still the pre-switch one.  Cross the
            # switch with a cold solve a few window-ulp ahead.  The nudge
            # is far below the zero threshold distance divided by any
            # finite approach rate, so a threshold-detectable contact
            # cannot hide inside it.
            if rebuilds < 6:
                rebuilds += 1
                nudged = True
                nudge = max(8.0 * math.ulp(max(abs(cur_t), 1.0)),
                            8.0 * math.ulp(max(abs(t_end), 1.0)))
                nudge = min(nudge, max(remaining, 0.0))
                if nudge <= 0.0:
                    return ("not_converged", residual, total_it)
                nt = cur_t + nudge
                cls = _classify(pts_a0, pts_b, velocity, nt, tol,
                                max_iter, scale)
                total_it += cls[1]["iterations"]
                if cls[0] == "boundary":
                    total_it += cls[2]["iterations"]
                    ci = _sep_contact_ci(nt, cls[1], scale, "impact",
                                         velocity)
                    return _finish_impact(pts_a0, pts_b, velocity, nt,
                                          ci, t_end, cls[2], tol,
                                          max_iter, scale, total_it,
                                          residual)
                if cls[0] == STATUS_NOT_CONVERGED:
                    return ("not_converged", cls[1]["residual"],
                            total_it)
                nxt = cls[1]
                target_t = nt
                if nxt["status"] == STATUS_INTERSECTING:
                    return _bisect_entry_and_finish(
                        pts_a0, pts_b, velocity, t_end, cur_t, cur_sol,
                        target_t, nxt, tol, max_iter, scale, total_it,
                        residual)
                # Fall through to the shared post-solve checks below.
            else:
                return ("not_converged", residual, total_it)

        d_t = math.sqrt(max(0.0, nxt["v2"]))
        if d_t < best_d:
            best_t, best_d = target_t, d_t

        if target_t >= t_end - _ulp_span(t_end):
            return ("miss", (best_t, best_d, nxt["residual"]),
                    total_it)
        if d_t <= _ZERO_TOL_REL * scale:
            return _bisect_entry_and_finish(
                pts_a0, pts_b, velocity, t_end, cur_t, cur_sol,
                target_t, nxt, tol, max_iter, scale, total_it, residual)
        if not nudged and target_t <= cur_t + _ulp_span(cur_t):
            # No representable progress on the current feature; confirm
            # separation ahead once.
            return ("miss", (best_t, best_d, residual), total_it)
        cur_t, cur_sol = target_t, nxt
        residual = nxt["residual"]
        rebuilds = 0

    return ("not_converged", residual, total_it)


def _bisect_entry_and_finish(pts_a0, pts_b, velocity, win_end, t_sep,
                             sol_sep, t_over, sol_over, tol, max_iter,
                             scale, total_it, residual):
    """Fallback: entry root between a separated and an intersecting
    sample, found by plain bisection and then resolved like an impact."""
    a, b = t_sep, t_over
    sa, sb = sol_sep, sol_over
    for _ in range(_ROOT_BISECT_STEPS):
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        cls = _classify(pts_a0, pts_b, velocity, mid, tol, max_iter,
                        scale, warm=(sa["simplex"], sa["time"]))
        total_it += cls[1]["iterations"]
        if cls[0] == STATUS_NOT_CONVERGED:
            return ("not_converged", cls[1]["residual"], total_it)
        if cls[0] == "boundary":
            total_it += cls[2]["iterations"]
            below, above = cls[1], cls[2]
            if below["status"] == STATUS_SEPARATED:
                a, sa = mid, below
                b, sb = mid, above
            else:
                b, sb = mid, below
                a, sa = mid, above
            continue
        if cls[0] == STATUS_INTERSECTING:
            b, sb = mid, cls[1]
        else:
            a, sa = mid, cls[1]
    root = _refine_root(a, sa, velocity, t_sep, t_over, scale, side=+1)
    hint = sb if sb["status"] == STATUS_INTERSECTING else sol_over
    ci_in = _sep_contact_ci(root, sa, scale, "impact", velocity)
    return _finish_impact(pts_a0, pts_b, velocity, root, ci_in, win_end,
                          hint, tol, max_iter, scale, total_it, residual)


def _contact_rate(velocity, n):
    """Normal separation rate |velocity . n|; 0 without a unit normal."""
    if n is None:
        return 0.0
    return abs(_dot(velocity, n))


def _finish_impact(pts_a0, pts_b, velocity, t_in, ci_in, t_end,
                   interior_sol, tol, max_iter, scale, total_it,
                   residual):
    """Build the impact payload: decide whether/where the exit occurs.

    Tangency (zero-duration contact, no intersecting instant distinct
    from the entry) is downgraded to "touch"; an exit within the
    numerical contact-time resolution is indistinguishable from it and
    reported honestly as touching rather than inflating a phantom
    positive interval.
    """
    cls = _classify(pts_a0, pts_b, velocity, t_end, tol, max_iter,
                    scale, warm=(interior_sol["simplex"],
                                 interior_sol["time"]))
    total_it += cls[1]["iterations"]
    if cls[0] == "boundary":
        total_it += cls[2]["iterations"]
        # Exit exactly at the window end: overlap throughout [t_in, t_end]
        return ("impact", (t_in, ci_in, None, residual), total_it)
    end_sol = cls[1]
    if end_sol["status"] == STATUS_NOT_CONVERGED:
        return ("not_converged", end_sol["residual"], total_it)
    rate_in = _contact_rate(velocity, ci_in.normal)
    touch_gate = max(_ulp_span(t_in), _zero_time(scale, rate_in))
    if end_sol["status"] == STATUS_INTERSECTING:
        if t_end - t_in <= touch_gate:
            ci = ContactInfo(t_in, ci_in.point_a, ci_in.point_b,
                             ci_in.normal, "touch")
            return ("touch", (ci, residual), total_it)
        return ("impact", (t_in, ci_in, None, residual), total_it)
    br = _bisect_exit(pts_a0, pts_b, velocity, t_in, t_end, end_sol,
                      tol, max_iter, scale, total_it)
    if br is None:
        return ("not_converged", residual, total_it)
    root_out, sep_sol, deepest, total_it = br
    rate_out = abs(_dot(velocity, sep_sol["v"])) / math.sqrt(
        max(_dot(sep_sol["v"], sep_sol["v"]), 1e-300))
    gate = max(_ulp_span(t_in), _ulp_span(root_out),
               2.0 * _zero_time(scale, max(rate_in, rate_out)))
    if deepest - t_in <= gate or root_out - t_in <= gate:
        ci = ContactInfo(t_in, ci_in.point_a, ci_in.point_b,
                         ci_in.normal, "touch")
        return ("touch", (ci, residual), total_it)
    ci_out = _sep_contact_ci(root_out, sep_sol, scale, "separation",
                             velocity)
    return ("impact", (t_in, ci_in, (root_out, ci_out), residual),
            total_it)


def sweep_contact(A, B, velocity, t_start, t_end, *,
                  tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """First contact and intersection interval under a fixed translation.

    See the module docstring for the full contract.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    vel = _validate_velocity(velocity)
    t0 = _validate_time(t_start, "t_start")
    t1 = _validate_time(t_end, "t_end")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)
    if not t0 < t1:
        raise InputError(f"t_start < t_end required, got "
                         f"t_start={t0!r}, t_end={t1!r}")

    scale = _input_scale(pts_a, pts_b)
    vnorm = math.sqrt(_dot(vel, vel))
    rest = vnorm <= _REST_VEL_REL * max(1.0, scale)
    total_it = 0

    cls0 = _classify(pts_a, pts_b, vel, t0, tol, max_iter, scale)
    total_it += cls0[1]["iterations"]
    if cls0[0] == "boundary":
        below, above = cls0[1], cls0[2]
        total_it += above["iterations"]
        if below["status"] == STATUS_INTERSECTING:
            # Contact ends exactly at t0 (bodies part inside the window
            # only before it): a single start-instant contact.
            ci = _sep_contact_ci(t0, above, scale, "touch", vel)
            return SweepResult(
                STATUS_TOUCHING, ci, ci, (t0, t0),
                STATUS_INTERSECTING, rest, t0, 0.0,
                total_it, above["residual"])
        sol0 = above  # entering at t0 -> initial overlap
    else:
        sol0 = cls0[1]
    if sol0["status"] == STATUS_NOT_CONVERGED:
        return SweepResult(
            STATUS_NOT_CONVERGED, None, None, None,
            STATUS_SEPARATED, rest, t0, math.nan,
            total_it, sol0["residual"])

    initial_overlap = sol0["status"] == STATUS_INTERSECTING

    # ---- relative rest: separation is constant in time -----------------
    if rest:
        d0 = math.sqrt(max(0.0, sol0["v2"]))
        if not initial_overlap:
            return SweepResult(
                STATUS_SEPARATED, None, None, None,
                STATUS_SEPARATED, True, t0, d0,
                total_it, sol0["residual"])
        wa, wb = _witnesses(sol0["simplex"], sol0["lam"])
        ci0 = ContactInfo(t0, wa, wb, None, "initial_overlap")
        ci1 = ContactInfo(t1, wa, wb, None, "rest_overlap")
        return SweepResult(
            STATUS_INTERSECTING, ci0, ci1, (t0, t1),
            STATUS_INTERSECTING, True, t0, 0.0, total_it, 0.0)

    # ---- initially overlapping: one exit at most (convexity) ------------
    if initial_overlap:
        cls1 = _classify(pts_a, pts_b, vel, t1, tol, max_iter, scale)
        total_it += cls1[1]["iterations"]
        if cls1[0] == STATUS_NOT_CONVERGED:
            return SweepResult(
                STATUS_NOT_CONVERGED, None, None, None,
                STATUS_INTERSECTING, False, t0, math.nan,
                total_it, cls1[1]["residual"])
        if cls1[0] == "boundary":
            total_it += cls1[2]["iterations"]
            # Exit exactly at the window end.
            root, sep_sol = t1, cls1[1]
            deepest = t1
            wa0, wb0 = _witnesses(sol0["simplex"], sol0["lam"])
            ci_start = ContactInfo(t0, wa0, wb0, None,
                                   "initial_overlap")
            ci_exit = _sep_contact_ci(t1, sep_sol, scale, "separation",
                                      vel)
            return SweepResult(
                STATUS_INTERSECTING, ci_start, ci_exit, (t0, t1),
                STATUS_INTERSECTING, False, t0, 0.0,
                total_it, sep_sol["residual"])
        end_sol = cls1[1]
        wa0, wb0 = _witnesses(sol0["simplex"], sol0["lam"])
        ci_start = ContactInfo(t0, wa0, wb0, None, "initial_overlap")
        if end_sol["status"] == STATUS_INTERSECTING:
            wae, wbe = _witnesses(end_sol["simplex"], end_sol["lam"])
            ci_end = ContactInfo(t1, wae, wbe, None, "window_end")
            return SweepResult(
                STATUS_INTERSECTING, ci_start, ci_end, (t0, t1),
                STATUS_INTERSECTING, False, t0, 0.0,
                total_it, end_sol["residual"])
        br = _bisect_exit(pts_a, pts_b, vel, t0, t1, end_sol,
                          tol, max_iter, scale, total_it)
        if br is None:
            return SweepResult(
                STATUS_NOT_CONVERGED, None, None, None,
                STATUS_INTERSECTING, False, t0, math.nan,
                total_it, end_sol["residual"])
        root, sep_sol, deepest, total_it = br
        rate_out = abs(_dot(vel, sep_sol["v"])) / math.sqrt(
            max(_dot(sep_sol["v"], sep_sol["v"]), 1e-300))
        gate = max(_ulp_span(root), 2.0 * _zero_time(scale, rate_out))
        if deepest - t0 <= gate or root - t0 <= gate:
            ci = _sep_contact_ci(t0, sep_sol, scale, "touch", vel)
            return SweepResult(
                STATUS_TOUCHING, ci, ci, (t0, t0),
                STATUS_INTERSECTING, False, t0, 0.0,
                total_it, sep_sol["residual"])
        ci_exit = _sep_contact_ci(root, sep_sol, scale, "separation",
                                  vel)
        return SweepResult(
            STATUS_INTERSECTING, ci_start, ci_exit, (t0, root),
            STATUS_INTERSECTING, False, t0, 0.0,
            total_it, sep_sol["residual"])

    # ---- initially separated: certified march ---------------------------
    tag, payload, total_it = _march_separated(
        pts_a, pts_b, vel, t0, t1, sol0, tol, max_iter, scale, total_it)

    if tag == "not_converged":
        return SweepResult(
            STATUS_NOT_CONVERGED, None, None, None,
            STATUS_SEPARATED, False, t0, math.nan,
            total_it, payload)

    if tag == "miss":
        best_t, best_d, residual = payload
        return SweepResult(
            STATUS_SEPARATED, None, None, None,
            STATUS_SEPARATED, False, best_t, best_d,
            total_it, residual)

    if tag == "touch":
        ci, residual = payload
        return SweepResult(
            STATUS_TOUCHING, ci, ci, (ci.time, ci.time),
            STATUS_SEPARATED, False, ci.time, 0.0,
            total_it, residual)

    t_in, ci_in, exit_info, residual = payload
    if exit_info is None:
        # Still overlapping at t_end: attach a window-end record.
        cls_e = _classify(pts_a, pts_b, vel, t1, tol, max_iter, scale)
        total_it += cls_e[1]["iterations"]
        end_sol2 = (cls_e[2] if cls_e[0] == "boundary"
                    and cls_e[2]["status"] == STATUS_INTERSECTING
                    else cls_e[1])
        wae, wbe = _witnesses(end_sol2["simplex"], end_sol2["lam"])
        ci_end = ContactInfo(t1, wae, wbe, None, "window_end")
        return SweepResult(
            STATUS_INTERSECTING, ci_in, ci_end, (t_in, t1),
            STATUS_SEPARATED, False, t_in, 0.0,
            total_it, residual)
    t_out, ci_out = exit_info
    return SweepResult(
        STATUS_INTERSECTING, ci_in, ci_out, (t_in, t_out),
        STATUS_SEPARATED, False, t_in, 0.0,
        total_it, residual)
