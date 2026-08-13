import numpy as np
from solver import (swap_rows, scale_row, add_multiple_of_row,
                    elimination, backsubstitution, inversion, RREF, null_space_special)

TOL = 1e-10

# ---------------------------------------------------------------- harness

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed.append(f"{name}: {detail}" if detail else name)

    def report(self):
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} passed")
        for f in self.failed:
            print(f"  FAIL  {f}")
        return len(self.failed) == 0


R = Results()


def echelon_violations(U, pivot_cols, tol=None):
    """Return a list of reasons U is not in row echelon form."""
    problems = []
    m, n = U.shape
    if tol is None:      # scale with the matrix, or absolute tests break
        tol = max(np.max(np.abs(U)), 1.0) * np.finfo(float).eps * max(m, n) * 10

    if list(pivot_cols) != sorted(set(pivot_cols)):
        problems.append(f"pivot_cols not strictly increasing: {pivot_cols}")

    for c in pivot_cols:
        if not (0 <= c < n):
            problems.append(f"pivot column {c} out of range for {n} columns")

    for r, c in enumerate(pivot_cols):
        if 0 <= c < n:
            below = U[r+1:, c]
            if below.size and np.max(np.abs(below)) > tol:
                problems.append(f"nonzero below pivot at row {r}, col {c}")

    k = len(pivot_cols)
    if k < m and U[k:].size and np.max(np.abs(U[k:])) > tol:
        problems.append(f"rows {k}+ should be zero, got {U[k:]}")

    return problems


def factorisation_violations(A, U, L, P, tol=TOL):
    """Return a list of reasons L, P fail to be a valid PA = LU factorisation."""
    problems = []
    m = A.shape[0]

    if L.shape != (m, m):
        problems.append(f"L should be {m}x{m}, is {L.shape}")
        return problems                      # later checks would crash
    if P.shape != (m, m):
        problems.append(f"P should be {m}x{m}, is {P.shape}")
        return problems

    if not np.allclose(L, np.tril(L), atol=tol):
        problems.append(f"L is not lower triangular:\n{L}")
    if not np.allclose(np.diag(L), 1.0, atol=tol):
        problems.append(f"L's diagonal is not all ones: {np.diag(L)}")

    # P must be a permutation: one 1 per row and per column, and P @ P.T == I
    if not (np.allclose(P.sum(axis=1), 1.0, atol=tol)
            and np.allclose(P.sum(axis=0), 1.0, atol=tol)
            and np.allclose(P @ P.T, np.eye(m), atol=tol)):
        problems.append(f"P is not a permutation matrix:\n{P}")

    if not np.allclose(L @ U, P @ A, atol=1e-8):
        problems.append(f"L @ U != P @ A\nL@U=\n{L @ U}\nP@A=\n{P @ A}")

    return problems


def same_row_space(A, U):
    """A and U span the same row space iff stacking them adds no rank.

    Uses numpy's default tolerance, which is RELATIVE to the largest singular
    value. Passing an absolute tol here silently reports rank 0 for any
    matrix scaled below it.
    """
    def unit(M):
        s = np.max(np.abs(M))
        return M if s == 0 else M / s

    # Normalise first: otherwise the smaller matrix vanishes inside vstack,
    # because numpy's tolerance is relative to the LARGEST singular value.
    A, U = unit(A), unit(U)
    # Generous tolerance: U and R carry accumulated roundoff in their zero
    # rows, and numpy's default would count that noise as rank.
    tol = 1e-7
    rank_A = np.linalg.matrix_rank(A, tol=tol)
    rank_U = np.linalg.matrix_rank(U, tol=tol)
    rank_stacked = np.linalg.matrix_rank(np.vstack([A, U]), tol=tol)
    return rank_A == rank_U == rank_stacked


def run_elimination(A):
    """Call elimination with a throwaway b and copies, so A survives."""
    b = np.zeros(A.shape[0])
    U, b_out, pivot_cols, L, P = elimination(A.copy(), b)
    return U, pivot_cols, L, P


# ---------------------------------------------------- 1. row operations
# Fresh fixtures per check. Helpers mutate, so nothing may be reused.

def fresh_A():
    return np.array([[2., 1., 5.],
                     [-6., 2., 4.],
                     [-2., 5., -1.]])


def primitive_tests():
    # --- hand-computed anchors -------------------------------------------
    got = swap_rows(fresh_A(), 0, 2)
    want = np.array([[-2., 5., -1.],
                     [-6., 2., 4.],
                     [2., 1., 5.]])
    R.check("swap_rows anchor", np.allclose(got, want), f"got\n{got}")

    got = scale_row(fresh_A(), 2, -0.5)
    want = np.array([[2., 1., 5.],
                     [-6., 2., 4.],
                     [1., -2.5, 0.5]])
    R.check("scale_row anchor", np.allclose(got, want), f"got\n{got}")

    got = add_multiple_of_row(fresh_A(), 0, 2, -0.5)
    want = np.array([[2., 1., 5.],
                     [-6., 2., 4.],
                     [-3., 4.5, -3.5]])
    R.check("add_multiple anchor", np.allclose(got, want), f"got\n{got}")

    # --- elementary-matrix identity: op(A) == op(I) @ A -------------------
    for name, op, args in [
        ("swap_rows", swap_rows, (0, 2)),
        ("scale_row", scale_row, (2, -0.5)),
        ("add_multiple_of_row", add_multiple_of_row, (0, 2, -0.5)),
    ]:
        E = op(np.eye(3), *args)
        lhs = op(fresh_A(), *args)
        rhs = E @ fresh_A()
        R.check(f"{name} == E @ A", np.allclose(lhs, rhs),
                f"\nop(A)=\n{lhs}\nE@A=\n{rhs}")

    # --- inverse round trips ----------------------------------------------
    A0 = fresh_A()
    R.check("swap_rows round trip",
            np.allclose(swap_rows(swap_rows(A0.copy(), 0, 2), 0, 2), A0))
    R.check("scale_row round trip",
            np.allclose(scale_row(scale_row(A0.copy(), 2, -0.5), 2, -2.0), A0))
    R.check("add_multiple round trip",
            np.allclose(add_multiple_of_row(
                add_multiple_of_row(A0.copy(), 0, 2, -0.5), 0, 2, 0.5), A0))

    # --- untouched rows stay untouched ------------------------------------
    out = scale_row(fresh_A(), 2, -0.5)
    R.check("scale_row leaves other rows alone", np.allclose(out[:2], A0[:2]))
    out = add_multiple_of_row(fresh_A(), 0, 2, -0.5)
    R.check("add_multiple leaves other rows alone", np.allclose(out[:2], A0[:2]))

    # --- c = 0 is legal and is a no-op ------------------------------------
    try:
        out = add_multiple_of_row(fresh_A(), 0, 2, 0.0)
        R.check("add_multiple with c=0", np.allclose(out, A0), f"got\n{out}")
    except AssertionError:
        R.check("add_multiple with c=0", False, "raised AssertionError")


# ----------------------------------------------- 2. elimination shapes

SHAPES = {
    "square, full rank": (
        np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]]), [0, 1, 2]),
    "square, late swap": (
        np.array([[2., 1., 1.], [4., 2., 3.], [1., 5., 2.]]), [0, 1, 2]),
    "square, zero first pivot": (
        np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]]), [0, 1, 2]),
    "free middle column": (
        np.array([[1., 2., 3.], [2., 4., 7.], [1., 2., 4.]]), [0, 2]),
    "free column needing a swap": (
        np.array([[1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 2., 0.]]), [0, 2, 3]),
    "free last column": (
        np.array([[1., 2., 0.], [0., 0., 0.], [2., 4., 0.]]), [0]),
    "rank deficient square": (
        np.array([[1., 2., 3.], [4., 5., 6.], [5., 7., 9.]]), [0, 1]),
    "wide, leading zero column entries": (
        np.array([[0., 1., 5., 6.], [0., -3., -6., -18.], [-2., 5., 3., 1.]]), [0, 1, 2]),
    "tall, full column rank": (
        np.array([[1., 2.], [2., 5.], [3., 7.], [1., 1.]]), [0, 1]),
    "tall, zero first column": (
        np.array([[0., 1.], [0., 2.], [0., 3.], [0., 4.]]), [1]),
    "tall, rank deficient": (
        np.array([[8., 14., 22.], [2., 4., 6.], [-2., -6., -8.], [2., 4., 6.]]), [0, 1]),
    "all zeros": (np.zeros((3, 3)), []),
    "single element": (np.array([[7.]]), [0]),
    "single zero element": (np.array([[0.]]), []),
}


def elimination_tests():
    for name, (A, want_pivots) in SHAPES.items():
        A_before = A.copy()
        try:
            U, pivot_cols, L, P = run_elimination(A)
        except Exception as ex:
            R.check(f"[{name}] runs", False, f"{type(ex).__name__}: {ex}")
            continue

        R.check(f"[{name}] does not mutate A", np.array_equal(A, A_before))
        R.check(f"[{name}] pivot columns", list(pivot_cols) == want_pivots,
                f"want {want_pivots}, got {list(pivot_cols)}")

        problems = echelon_violations(U, pivot_cols)
        R.check(f"[{name}] echelon form", not problems,
                "; ".join(problems) + f"\nU=\n{U}")

        want_rank = np.linalg.matrix_rank(A_before)
        R.check(f"[{name}] rank == #pivots", len(pivot_cols) == want_rank,
                f"numpy says {want_rank}, got {len(pivot_cols)}")

        R.check(f"[{name}] row space preserved", same_row_space(A_before, U),
                f"U=\n{U}")

        problems = factorisation_violations(A_before, U, L, P)
        R.check(f"[{name}] PA = LU", not problems, "; ".join(problems))


def elimination_b_tests():
    """b rides along through the same row operations as A."""
    A = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]])
    b = np.array([1., 2., 3.])
    U, b_out, pivot_cols, L, P = elimination(A.copy(), b.copy())

    # the same operations hit b, so undoing them with L must give back P @ b
    R.check("b transformed consistently with A",
            np.allclose(L @ b_out, P @ b),
            f"L@b_out={L @ b_out}, P@b={P @ b}")

    b_orig = np.array([1., 2., 3.])
    b_passed = b_orig.copy()
    elimination(A.copy(), b_passed)
    R.check("elimination does not mutate caller's b",
            np.array_equal(b_passed, b_orig), f"b became {b_passed}")


# ------------------------------------------- 3. elimination, randomised

def random_matrix(rng, force_rank_deficient=False):
    m = int(rng.integers(1, 7))
    n = int(rng.integers(1, 7))
    if force_rank_deficient and min(m, n) > 1:
        k = int(rng.integers(1, min(m, n)))
        return (rng.integers(-3, 4, size=(m, k)).astype(float)
                @ rng.integers(-3, 4, size=(k, n)).astype(float))
    return rng.integers(-4, 5, size=(m, n)).astype(float)


def random_elimination_tests(trials=300):
    rng = np.random.default_rng(12345)
    bad = 0
    for t in range(trials):
        A = random_matrix(rng, force_rank_deficient=(t % 2 == 0)) * (10.0 ** rng.integers(-8, 9))
        A_before = A.copy()
        try:
            U, pivot_cols, L, P = run_elimination(A)
        except Exception as ex:
            bad += 1
            print(f"  crash on trial {t}, shape {A.shape}:\n{A_before}\n"
                  f"  {type(ex).__name__}: {ex}")
            continue

        problems = echelon_violations(U, pivot_cols)
        problems += factorisation_violations(A_before, U, L, P)
        if len(pivot_cols) != np.linalg.matrix_rank(A_before):
            problems.append(f"pivots {pivot_cols}, numpy rank "
                            f"{np.linalg.matrix_rank(A_before)}")
        if not same_row_space(A_before, U):
            problems.append("row space changed")

        if problems:
            bad += 1
            if bad <= 3:
                print(f"  trial {t}:\n{A_before}\n  U=\n{U}\n  {problems}")

    R.check(f"randomised elimination + PA=LU ({trials} trials)",
            bad == 0, f"{bad} bad cases")


# ------------------------------------------------- 4. back-substitution

def backsubstitution_tests():
    A = np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]])
    b = np.array([5., 1., -6.])
    want = np.array([346/9, -2/9, -11.])
    got = backsubstitution(A.copy(), b.copy())
    R.check("backsub anchor", got is not None and np.allclose(got, want),
            f"want {want}, got {got}")

    b2 = np.array([3., -1., 7.])
    got = backsubstitution(np.eye(3), b2.copy())
    R.check("backsub with identity", got is not None and np.allclose(got, b2),
            f"got {got}")

    A = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]])
    b = np.array([1., 2., 3.])
    got = backsubstitution(A.copy(), b.copy())
    R.check("backsub with required swap",
            got is not None and np.allclose(A @ got, b), f"got {got}")

    A = np.array([[1., 2., 3.], [4., 5., 6.], [5., 7., 9.]])
    b = np.array([1., 2., 3.])
    try:
        got = backsubstitution(A.copy(), b.copy())
        R.check("backsub: singular system raises", False,
                f"returned {got} instead of raising")
    except ValueError:
        R.check("backsub: singular system raises", True)
    except Exception as ex:
        R.check("backsub: singular system raises", False,
                f"raised {type(ex).__name__}, expected ValueError")


def random_backsubstitution_tests(trials=200):
    rng = np.random.default_rng(7)
    bad = 0
    ran = 0
    for t in range(trials):
        n = int(rng.integers(1, 7))
        A = rng.integers(-9, 10, size=(n, n)).astype(float) * (10.0 ** rng.integers(-8, 9))
        b = rng.integers(-9, 10, size=n).astype(float)
        if np.linalg.matrix_rank(A) < n:
            continue
        ran += 1
        x = backsubstitution(A.copy(), b.copy())
        if x is None:
            bad += 1
            print(f"  returned None, trial {t}:\n{A}")
        elif not np.allclose(A @ x, b):
            bad += 1
            print(f"  residual fails, trial {t}:\n{A}\n  b={b}  x={x}")
        elif not np.allclose(x, np.linalg.solve(A, b)):
            bad += 1
            print(f"  disagrees with numpy, trial {t}:\n{A}\n  b={b}  x={x}")

    R.check(f"randomised backsubstitution ({ran} nonsingular trials)",
            bad == 0, f"{bad} bad cases")


# ------------------------------------------------------- 5. inversion

INVERTIBLE = {
    "general dense": np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]]),
    "upper triangular": np.array([[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]]),
    "lower triangular": np.array([[1., 0., 0.], [-2., 1., 0.], [3., -4., 1.]]),
    "identity 4x4": np.eye(4),
    "zero at (0,0)": np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]]),
    "zero appears mid-elimination": np.array([[1., 1., 1.], [1., 1., 3.], [2., 5., 8.]]),
    "permutation": np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
    "diagonal, mixed signs": np.array([[2., 0.], [0., -3.]]),
    "1x1": np.array([[4.]]),
}

SINGULAR = {
    "duplicate rows": np.array([[1., 2.], [2., 4.]]),
    "zero row": np.array([[1., 2., 3.], [0., 0., 0.], [4., 5., 6.]]),
    "zero column": np.array([[1., 0., 3.], [4., 0., 6.], [7., 0., 9.]]),
    "row 3 = row 1 + row 2": np.array([[1., 2., 3.], [4., 5., 6.], [5., 7., 9.]]),
    "all zeros": np.zeros((3, 3)),
    "1x1 zero": np.array([[0.]]),
}


def inversion_tests():
    for name, M in INVERTIBLE.items():
        M_before = M.copy()
        try:
            inv = inversion(M)
        except Exception as ex:
            R.check(f"[inv: {name}] runs", False, f"{type(ex).__name__}: {ex}")
            continue

        n = len(M_before)
        R.check(f"[inv: {name}] inv @ M == I",
                np.allclose(inv @ M_before, np.eye(n)), f"got\n{inv @ M_before}")
        # a true inverse commutes -- a one-sided check can pass by accident
        R.check(f"[inv: {name}] M @ inv == I",
                np.allclose(M_before @ inv, np.eye(n)), f"got\n{M_before @ inv}")
        R.check(f"[inv: {name}] does not mutate input",
                np.array_equal(M, M_before), f"input became\n{M}")
        R.check(f"[inv: {name}] agrees with numpy",
                np.allclose(inv, np.linalg.inv(M_before)),
                f"got\n{inv}\nnumpy\n{np.linalg.inv(M_before)}")

    for name, M in SINGULAR.items():
        try:
            got = inversion(M.copy())
            R.check(f"[inv: {name}] raises ValueError", False, f"returned\n{got}")
        except ValueError:
            R.check(f"[inv: {name}] raises ValueError", True)
        except Exception as ex:
            R.check(f"[inv: {name}] raises ValueError", False,
                    f"raised {type(ex).__name__}: {ex}")

    # inverting twice returns the original
    M = np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]])
    R.check("inversion is its own inverse",
            np.allclose(inversion(inversion(M.copy())), M))

    # the inverse of a product is the product of inverses, order reversed
    X = np.array([[2., 1.], [1., 1.]])
    Y = np.array([[3., 0.], [4., 2.]])
    R.check("(XY)^-1 == Y^-1 X^-1",
            np.allclose(inversion(X @ Y), inversion(Y.copy()) @ inversion(X.copy())))

    # non-square input must be rejected
    try:
        inversion(np.array([[1., 2., 3.], [4., 5., 6.]]))
        R.check("inversion rejects non-square", False, "accepted a 2x3")
    except (AssertionError, ValueError):
        R.check("inversion rejects non-square", True)


def random_inversion_tests(trials=400):
    rng = np.random.default_rng(99)
    bad = 0
    ran = 0
    for t in range(trials):
        n = int(rng.integers(1, 8))
        M = rng.integers(-9, 10, size=(n, n)).astype(float) * (10.0 ** rng.integers(-8, 9))
        # det scales like s**n, so it is useless here: use rank instead
        singular = np.linalg.matrix_rank(M) < n
        try:
            inv = inversion(M.copy())
            if singular:
                bad += 1
                print(f"  singular matrix not caught, trial {t}:\n{M}")
            elif not (np.allclose(inv @ M, np.eye(n))
                      and np.allclose(M @ inv, np.eye(n))):
                bad += 1
                print(f"  wrong inverse, trial {t}:\n{M}")
            else:
                ran += 1
        except ValueError:
            if not singular:
                bad += 1
                print(f"  falsely rejected an invertible matrix, trial {t}:\n{M}")
        except Exception as ex:
            bad += 1
            print(f"  crash on trial {t}: {type(ex).__name__}: {ex}\n{M}")

    R.check(f"randomised inversion ({ran} invertible matrices)",
            bad == 0, f"{bad} bad cases")


def inversion_conditioning_report():
    """Hilbert matrices are invertible but badly conditioned.

    Informational, not pass/fail: the drift is a property of the pivoting
    strategy (first nonzero vs. largest magnitude), not a bug.
    """
    for n in (4, 6, 8):
        H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        err = np.max(np.abs(inversion(H.copy()) @ H - np.eye(n)))
        err_np = np.max(np.abs(np.linalg.inv(H) @ H - np.eye(n)))
        print(f"  {n}x{n} Hilbert: yours {err:.2e}, numpy {err_np:.2e}")


# ------------------------------------------------------------- 6. RREF

def reference_rref(M, tol=None):
    """An independent RREF, written with a different structure on purpose.

    The RREF of a matrix is UNIQUE -- unlike U, which depends on pivoting
    choices. So any correct implementation must agree with this one exactly,
    which makes it a real oracle rather than just a second opinion.
    """
    R = M.astype(float).copy()
    m, n = R.shape
    if tol is None:
        # Same relative rule the solver uses, with headroom for the roundoff
        # this reference accumulates -- it pivots on the first nonzero rather
        # than the largest, so it is less numerically stable than the solver.
        tol = np.max(np.abs(R)) * np.finfo(float).eps * max(m, n) * 1e3
    row = 0
    for col in range(n):
        if row >= m:
            break
        p = None
        for k in range(row, m):
            if abs(R[k, col]) > tol:
                p = k
                break
        if p is None:
            continue                      # free column: no row consumed
        R[[row, p]] = R[[p, row]]
        R[row] = R[row] / R[row, col]
        for k in range(m):
            if k != row and abs(R[k, col]) > tol:
                R[k] = R[k] - R[k, col] * R[row]
        row += 1
    R[np.abs(R) < tol] = 0.0
    return R


def rref_violations(R, pivot_cols, tol=None):
    """Return a list of reasons R is not in reduced row echelon form."""
    problems = []
    m, n = R.shape
    if tol is None:
        # R is normalised, so its entries are O(1) regardless of input scale.
        # But the leftover roundoff in R is driven by the CONDITIONING of the
        # input, not its scale, so this has to be generous. A checker that is
        # too strict reports bugs that aren't there.
        tol = max(np.max(np.abs(R)) if R.size else 1.0, 1.0) * 1e-7

    # locate the pivots directly from R, without trusting pivot_cols
    found = []
    for r in range(m):
        nz = np.where(np.abs(R[r]) > tol)[0]
        if nz.size:
            found.append((r, int(nz[0])))

    # rows holding pivots must be the top rows -- zero rows sink to the bottom
    if [r for r, _ in found] != list(range(len(found))):
        problems.append(f"zero rows not at the bottom: nonzero rows "
                        f"{[r for r, _ in found]}")

    for idx, (r, c) in enumerate(found):
        if abs(R[r, c] - 1.0) > tol:
            problems.append(f"pivot at ({r},{c}) is {R[r, c]}, not 1")

        column = R[:, c].copy()
        column[r] = 0.0
        if column.size and np.max(np.abs(column)) > tol:
            problems.append(f"pivot column {c} not cleared above/below: {R[:, c]}")

        if idx > 0 and c <= found[idx - 1][1]:
            problems.append(f"pivot columns not increasing: "
                            f"{[p[1] for p in found]}")

    # the reported pivot_cols must match what's actually in R
    if list(pivot_cols) != [c for _, c in found]:
        problems.append(f"pivot_cols says {list(pivot_cols)}, R shows "
                        f"{[c for _, c in found]}")

    return problems


RREF_SHAPES = {
    "square, full rank": np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]]),
    "tall 4x3": np.array([[5., 2., 3.], [9., 5., 6.], [9., -6., 4.], [1., 2., 6.]]),
    "free middle column": np.array([[1., 2., 3.], [2., 4., 7.], [1., 2., 4.]]),
    "rank deficient": np.array([[1., 2., 3.], [4., 5., 6.], [5., 7., 9.]]),
    "wide 3x5": np.array([[1., 2., 0., 3., 1.],
                          [2., 4., 1., 7., 3.],
                          [0., 0., 1., 1., 1.]]),
    "tall, full column rank": np.array([[1., 2.], [2., 5.], [3., 7.], [1., 1.]]),
    "zero first column": np.array([[0., 1., 2.], [0., 3., 4.], [0., 5., 7.]]),
    "needs a swap": np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]]),
    "all zeros": np.zeros((3, 3)),
    "identity": np.eye(3),
    "single element": np.array([[7.]]),
    "single zero element": np.array([[0.]]),
}


def rref_tests():
    for name, A in RREF_SHAPES.items():
        A_before = A.copy()
        try:
            Rmat, pivot_cols, free_cols, rank = RREF(A)
        except Exception as ex:
            R.check(f"[rref: {name}] runs", False, f"{type(ex).__name__}: {ex}")
            continue

        shown = np.round(Rmat, 8)

        problems = rref_violations(Rmat, pivot_cols)
        R.check(f"[rref: {name}] is in RREF", not problems,
                "; ".join(problems) + f"\nR=\n{shown}")

        want = reference_rref(A_before)
        R.check(f"[rref: {name}] matches independent implementation",
                np.allclose(Rmat, want, atol=1e-8),
                f"\nyours=\n{shown}\nreference=\n{np.round(want, 8)}")

        R.check(f"[rref: {name}] does not mutate A",
                np.array_equal(A, A_before), f"A became\n{A}")

        R.check(f"[rref: {name}] rank == #pivots",
                rank == np.linalg.matrix_rank(A_before),
                f"numpy says {np.linalg.matrix_rank(A_before)}, got {rank}")

        R.check(f"[rref: {name}] row space preserved",
                same_row_space(A_before, Rmat), f"R=\n{shown}")

        R.check(f"[rref: {name}] pivot + free cols partition all columns",
                np.array_equal(np.sort(pivot_cols + free_cols),
                               np.arange(len(Rmat[0]))),
                f"pivots {pivot_cols}, free {free_cols}, "
                f"{len(Rmat[0])} columns")

        R.check(f"[rref: {name}] rank <= min(m, n)",
                rank <= min(A_before.shape),
                f"rank {rank}, shape {A_before.shape}")

        # RREF is idempotent: reducing an already-reduced matrix changes nothing
        R2, pivot_cols2, free_cols2, rank2 = RREF(Rmat.copy())
        R.check(f"[rref: {name}] idempotent",
                np.allclose(R2, Rmat, atol=1e-8)
                and list(pivot_cols2) == list(pivot_cols),
                f"second pass gave\n{np.round(R2, 8)}")


def rref_identity_test():
    """For an invertible A, RREF(A) is I, and RREF([A | I]) is [I | A^-1]."""
    A = np.array([[1., 2., 3.], [2., -5., 7.], [1., 2., 4.]])
    Rmat, pivot_cols, _, _ = RREF(A.copy())
    R.check("rref of an invertible matrix is I",
            np.allclose(Rmat, np.eye(3), atol=1e-8)
            and list(pivot_cols) == [0, 1, 2], f"got\n{np.round(Rmat, 8)}")

    aug = np.hstack([A, np.eye(3)])
    Rmat, pivot_cols, _, _ = RREF(aug)
    R.check("rref of [A | I] gives [I | A^-1]",
            np.allclose(Rmat[:, :3], np.eye(3), atol=1e-8)
            and np.allclose(Rmat[:, 3:], inversion(A.copy()), atol=1e-8),
            f"got\n{np.round(Rmat, 8)}")


def random_rref_tests(trials=300):
    rng = np.random.default_rng(2024)
    bad = 0
    for t in range(trials):
        A = random_matrix(rng, force_rank_deficient=(t % 2 == 0)) * (10.0 ** rng.integers(-8, 9))
        A_before = A.copy()
        try:
            Rmat, pivot_cols, free_cols, rank = RREF(A)
        except Exception as ex:
            bad += 1
            if bad <= 3:
                print(f"  crash on trial {t}, shape {A.shape}:\n{A_before}\n"
                      f"  {type(ex).__name__}: {ex}")
            continue

        problems = rref_violations(Rmat, pivot_cols)
        # atol must allow for accumulated roundoff in the ORIGINAL matrix,
        # which the normalised R no longer shows the scale of
        ref = reference_rref(A_before)
        scale = max(np.max(np.abs(A_before)), 1.0)
        if not np.allclose(Rmat, ref, atol=scale * np.finfo(float).eps * 1e4):
            problems.append(f"disagrees with the independent implementation "
                            f"(max diff {np.max(np.abs(Rmat - ref)):.2e})")
        if rank != np.linalg.matrix_rank(A_before):
            problems.append(f"rank {rank}, numpy rank "
                            f"{np.linalg.matrix_rank(A_before)}")
        if not np.array_equal(np.sort(pivot_cols + free_cols),
                              np.arange(A_before.shape[1])):
            problems.append(f"pivots {pivot_cols} + free {free_cols} "
                            f"don't partition {A_before.shape[1]} columns")
        if not same_row_space(A_before, Rmat):
            problems.append("row space changed")

        if problems:
            bad += 1
            if bad <= 3:
                print(f"  trial {t}:\n{A_before}\n  yours=\n"
                      f"{np.round(Rmat, 6)}\n  ref=\n"
                      f"{np.round(reference_rref(A_before), 6)}\n  {problems}")

    R.check(f"randomised RREF ({trials} trials)", bad == 0, f"{bad} bad cases")

def null_space_test(trials=300):
    rng = np.random.default_rng(2024)
    bad = 0
    for t in range(trials):
        A = random_matrix(rng, force_rank_deficient=(t % 2 == 0)) * (10.0 ** rng.integers(-8, 9))
        A_before = A.copy()
        try:
            N = null_space_special(A)        
        except Exception as ex:
            bad += 1
            if bad <= 3:
                print(f"  crash on trial {t}, shape {A.shape}:\n{A_before}\n"
                      f"  {type(ex).__name__}: {ex}")
            continue
        problems = []
        
        if len(N) != A.shape[1] - np.linalg.matrix_rank(A):
            problems.append("n - rank != number of basis null vectors") 
            
        if np.linalg.matrix_rank(np.array(N)) != len(N):
            problems.append("non independent basis null vectors")
            
        scale = max(np.max(np.abs(A_before)), 1.0)
        for n in N:
            if not np.allclose(A @ n, 0, atol=scale * np.finfo(float).eps * 1e4):
                problems.append("vector doesnt belong in nullspace")
                
        if problems:
            bad += 1
            if bad <= 3:
                print(f"trial {t}: \n non-null-space vectors have been returned")
                
    R.check(f"null space specials ({trials} trials)", bad == 0, f"{bad} bad cases")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    primitive_tests()
    elimination_tests()
    elimination_b_tests()
    random_elimination_tests()
    backsubstitution_tests()
    random_backsubstitution_tests()
    inversion_tests()
    random_inversion_tests()
    rref_tests()
    rref_identity_test()
    random_rref_tests()
    null_space_test()

    print("\nconditioning (informational, not pass/fail):")
    inversion_conditioning_report()

    ok = R.report()
    raise SystemExit(0 if ok else 1)