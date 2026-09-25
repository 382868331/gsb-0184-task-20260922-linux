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
                 "intersecting".  For "separated" it is <= tol; for
                 "not_converged" it is > tol and bounds the remaining
                 relative uncertainty of ``distance``.

"not_converged" (search budget exhausted) is reported distinctly from
"intersecting" (origin proven inside the Minkowski difference); the two
are never conflated.  Touching hulls (distance exactly zero) are reported
as "intersecting" with distance <= the zero threshold.

Ellipsoidal Minkowski shells
----------------------------
distance_shell(A, B, *, shell_a=None, shell_b=None, tol=1e-9,
               max_iter=128) -> ShellDistanceResult

Each body is the Minkowski sum  conv(vertices) (+) L * unit_ball  where
``shell_a`` / ``shell_b`` is an invertible 3x3 matrix ``L`` (row-major,
3 rows of 3 finite reals; ``None`` means no shell for that body).  The
shell is a genuine ellipsoid: its support offset in direction ``n`` is

    off(n) = L L^T n / ||L^T n||

which is direction-dependent and is NOT replaced by a maximal-radius
sphere.  A singular ``L`` (or NaN/Infinity/bool entries, wrong shape)
raises :class:`InputError`.  A zero direction yields a zero offset
(reachable only as the intersecting case, where no direction exists).

Because a barycentric combination of the input vertices no longer
represents the dilated surface, the result splits the witness into a
core part and a shell part:

ShellDistanceResult fields
--------------------------
status / distance / separating_axis / iterations / residual
                 same meaning as DistanceResult.
core_witness_a   closest core point, a barycentric combination of A.
core_witness_b   same for B.
shell_offset_a   ellipsoid offset applied at the final direction;
                 witness_a = core_witness_a + shell_offset_a lies on the
                 dilated shell conv(A) (+) L_a * ball.  Zero vector when
                 status == "intersecting" (no direction exists).
shell_offset_b   same for B.
witness_a/b      actual closest points on the two dilated shells.
lower_bound      certified lower bound on the true distance, from the
                 support evaluations (max over iterations of w.v/||v||,
                 clamped to >= 0).  Independently checkable via the
                 support function of the dilated bodies.
upper_bound      ||witness_a - witness_b||, a realisable upper bound on
                 the true distance (the witnesses are feasible points).
lower_bound <= d_true <= upper_bound always; on converged separation
upper_bound - lower_bound is within the convergence tolerance.  For
"not_converged" the interval [lower_bound, upper_bound] is the honest
output -- no exact intersection is claimed.  For "intersecting"
lower_bound is 0 and distance is a residual <= the zero threshold.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

__all__ = [
    "distance",
    "distance_shell",
    "DistanceResult",
    "ShellDistanceResult",
    "InputError",
    "MAX_POINTS",
    "MAX_ITER",
    "DEFAULT_TOL",
    "STATUS_SEPARATED",
    "STATUS_INTERSECTING",
    "STATUS_NOT_CONVERGED",
]

__version__ = "1.1.0"

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

    @property
    def converged(self) -> bool:
        return self.status != STATUS_NOT_CONVERGED


@dataclass(frozen=True)
class ShellDistanceResult:
    status: str
    distance: float
    witness_a: tuple
    witness_b: tuple
    core_witness_a: tuple
    core_witness_b: tuple
    shell_offset_a: tuple
    shell_offset_b: tuple
    separating_axis: tuple | None
    lower_bound: float
    upper_bound: float
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


def _validate_shell(shell, name):
    """Validate an ellipsoid shell matrix L; None means "no shell".

    Returns a row-major 3x3 tuple of floats, or None.  L must be
    invertible: the shell is L * unit_ball, and a singular L would make
    the support offset L L^T n / ||L^T n|| undefined for some directions.
    """
    if shell is None:
        return None
    if isinstance(shell, (str, bytes)):
        raise InputError(f"{name}: expected a 3x3 matrix, got text")
    try:
        rows = list(shell)
    except TypeError:
        raise InputError(f"{name}: expected a 3x3 matrix, "
                         f"got {type(shell).__name__}") from None
    if len(rows) != 3:
        raise InputError(f"{name}: expected exactly 3 rows, got {len(rows)}")
    mat = []
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise InputError(f"{name}[{i}]: expected a 3-coordinate row, "
                             f"got text")
        try:
            coords = list(row)
        except TypeError:
            raise InputError(f"{name}[{i}]: expected a 3-coordinate row, "
                             f"got {type(row).__name__}") from None
        if len(coords) != 3:
            raise InputError(f"{name}[{i}]: expected exactly 3 columns, "
                             f"got {len(coords)}")
        vec = []
        for j, c in enumerate(coords):
            if isinstance(c, bool) or not isinstance(c, numbers.Real):
                raise InputError(f"{name}[{i}][{j}]: expected a real number, "
                                 f"got {type(c).__name__} ({c!r})")
            f = float(c)
            if not math.isfinite(f):
                raise InputError(f"{name}[{i}][{j}]: non-finite entry "
                                 f"{c!r} (NaN/Infinity rejected)")
            vec.append(f)
        mat.append(vec)
    # Invertibility: Gaussian elimination with partial pivoting.
    m = [row[:] for row in mat]
    maxabs = max(abs(x) for row in m for x in row)
    eps = _KKT_TOL_REL * maxabs
    for col in range(3):
        piv = col
        for r in range(col + 1, 3):
            if abs(m[r][col]) > abs(m[piv][col]):
                piv = r
        if abs(m[piv][col]) <= eps:
            raise InputError(f"{name}: singular 3x3 matrix "
                             f"(the shell L must be invertible)")
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, 3):
            f = m[r][col] * inv
            for c in range(col, 3):
                m[r][c] -= f * m[col][c]
    return ((mat[0][0], mat[0][1], mat[0][2]),
            (mat[1][0], mat[1][1], mat[1][2]),
            (mat[2][0], mat[2][1], mat[2][2]))


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
# ellipsoid shell: support offset of L * unit_ball
# ---------------------------------------------------------------------------

def _shell_offset(L, d):
    """Support point of the ellipsoid L * unit_ball in direction d.

    off(d) = L L^T d / ||L^T d||  =  L u  with  u = L^T d / ||L^T d||
    on the unit sphere, so the offset lies exactly on the ellipsoid
    boundary.  The offset is direction-dependent (anisotropic); it is
    NOT a maximal-radius sphere.  A zero direction (only possible when
    the dilated bodies intersect, since L is invertible) yields a zero
    offset by contract.
    """
    # u_raw = L^T d
    ux = L[0][0] * d[0] + L[1][0] * d[1] + L[2][0] * d[2]
    uy = L[0][1] * d[0] + L[1][1] * d[1] + L[2][1] * d[2]
    uz = L[0][2] * d[0] + L[1][2] * d[1] + L[2][2] * d[2]
    n2 = ux * ux + uy * uy + uz * uz
    if n2 == 0.0:
        return (0.0, 0.0, 0.0)
    s = 1.0 / math.sqrt(n2)
    ux *= s
    uy *= s
    uz *= s
    # L u
    return (L[0][0] * ux + L[0][1] * uy + L[0][2] * uz,
            L[1][0] * ux + L[1][1] * uy + L[1][2] * uz,
            L[2][0] * ux + L[2][1] * uy + L[2][2] * uz)


def _support_shell(A, B, La, Lb, d):
    """Support of the dilated Minkowski difference C = A' - B' where
    A' = conv(A) (+) La*ball, B' = conv(B) (+) Lb*ball.

    Returns (w, core_a, core_b): the cores are plain vertices of A / B
    (argmax / argmin of the dot product); the shell offsets are added to
    the difference point only:

        w = (core_a - core_b) + off_a(d) + off_b(d)

    because the B-side support point in -d carries offset off_b(-d) =
    -off_b(d).  Keeping cores and offsets separate is what lets the
    caller split the final witness into core and shell parts.
    """
    w, a, b = _support(A, B, d)
    if La is not None:
        o = _shell_offset(La, d)
        w = (w[0] + o[0], w[1] + o[1], w[2] + o[2])
    if Lb is not None:
        o = _shell_offset(Lb, d)
        w = (w[0] + o[0], w[1] + o[1], w[2] + o[2])
    return (w, a, b)


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
# ellipsoid-shell entry point
# ---------------------------------------------------------------------------

def distance_shell(A, B, *, shell_a=None, shell_b=None,
                   tol=DEFAULT_TOL, max_iter=MAX_ITER):
    """Distance between conv(A) (+) shell_a*ball and conv(B) (+) shell_b*ball.

    See the module docstring for the full contract.  ``shell_a`` /
    ``shell_b`` are invertible 3x3 matrices ``L`` (or None for no shell);
    the shell support offset in direction n is L L^T n / ||L^T n||.
    """
    pts_a = _validate_points(A, "A")
    pts_b = _validate_points(B, "B")
    La = _validate_shell(shell_a, "shell_a")
    Lb = _validate_shell(shell_b, "shell_b")
    tol = _validate_tol(tol)
    max_iter = _validate_max_iter(max_iter)

    # Scale for the origin-containment threshold: core extents plus the
    # shell magnitudes (Frobenius norm bounds every axis of L*ball).
    scale2 = 1.0
    for p in pts_a:
        n2 = _norm2(p)
        if n2 > scale2:
            scale2 = n2
    for p in pts_b:
        n2 = _norm2(p)
        if n2 > scale2:
            scale2 = n2
    for L in (La, Lb):
        if L is not None:
            for row in L:
                for x in row:
                    scale2 += x * x
    scale = math.sqrt(scale2)
    zero_tol2 = (_ZERO_TOL_REL * scale) ** 2

    # Initial search direction: core centroid difference (deterministic).
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

    simplex = []          # list of (w, core_a, core_b)
    v = None              # current closest point of conv(simplex)
    v2 = 0.0
    lam = ()
    residual = math.inf
    status = STATUS_NOT_CONVERGED
    iterations = 0
    best_lower = 0.0      # best certified lower bound on the distance

    for it in range(1, max_iter + 1):
        iterations = it
        w, sa, sb = _support_shell(pts_a, pts_b, La, Lb, d)
        if v is not None:
            # w = argmin_{c in C} c.v, so d_true >= w.v / ||v|| for every
            # iteration; keep the best (largest) such bound.
            dv = math.sqrt(v2)
            lw = _dot(w, v) / dv
            if lw > best_lower:
                best_lower = lw
            gap = v2 - _dot(w, v)
            residual = gap / v2 if v2 > 0.0 else 0.0
            if gap <= tol * v2:
                status = STATUS_SEPARATED
                break
        if any(w == s[0] for s in simplex):
            # Support point already in the simplex: the gap is exactly
            # zero and no further progress is possible.
            residual = 0.0
            status = STATUS_SEPARATED
            break
        simplex.append((w, sa, sb))
        v, lam, keep = _closest_on_simplex(simplex)
        simplex = [simplex[i] for i in keep]
        v2 = _norm2(v)
        if v2 <= zero_tol2:
            # Origin lies in conv(simplex) subset C: shells intersect.
            residual = 0.0
            status = STATUS_INTERSECTING
            break
        d = (-v[0], -v[1], -v[2])

    # Core witnesses: barycentric combinations of the input vertices.
    wax = way = waz = wbx = wby = wbz = 0.0
    for (w, sa, sb), l in zip(simplex, lam):
        wax += l * sa[0]
        way += l * sa[1]
        waz += l * sa[2]
        wbx += l * sb[0]
        wby += l * sb[1]
        wbz += l * sb[2]
    core_witness_a = (wax, way, waz)
    core_witness_b = (wbx, wby, wbz)

    dist = math.sqrt(v2)
    if status == STATUS_INTERSECTING:
        # No direction exists (v ~ 0): zero offsets by contract; the
        # witnesses are the core witnesses and distance is a residual.
        axis = None
        off_a = (0.0, 0.0, 0.0)
        off_b = (0.0, 0.0, 0.0)
        lower = 0.0
    else:
        # n points from B to A.  The witness on A's shell is the core
        # witness plus A's shell offset in direction -n (towards B);
        # symmetrically for B.  Each offset lies exactly on its
        # ellipsoid boundary, so the witnesses are feasible points of
        # the dilated bodies.
        inv = 1.0 / dist
        n = (v[0] * inv, v[1] * inv, v[2] * inv)
        neg_n = (-n[0], -n[1], -n[2])
        off_a = _shell_offset(La, neg_n) if La is not None else (0.0,) * 3
        off_b = _shell_offset(Lb, n) if Lb is not None else (0.0,) * 3
        axis = n
        lower = best_lower if best_lower > 0.0 else 0.0

    witness_a = (core_witness_a[0] + off_a[0],
                 core_witness_a[1] + off_a[1],
                 core_witness_a[2] + off_a[2])
    witness_b = (core_witness_b[0] + off_b[0],
                 core_witness_b[1] + off_b[1],
                 core_witness_b[2] + off_b[2])
    upper = math.sqrt(_norm2(_sub(witness_a, witness_b)))

    return ShellDistanceResult(
        status=status,
        distance=dist,
        witness_a=witness_a,
        witness_b=witness_b,
        core_witness_a=core_witness_a,
        core_witness_b=core_witness_b,
        shell_offset_a=off_a,
        shell_offset_b=off_b,
        separating_axis=axis,
        lower_bound=lower,
        upper_bound=upper,
        iterations=iterations,
        residual=residual,
    )
