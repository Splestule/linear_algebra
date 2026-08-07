import numpy as np

A = np.array([[1., 2., 3.],
              [2., -5., 7.],
              [1., 2., 4.]])

b = np.array([5., 1., -6.])

def swap_rows(A, i, j):
    assert 0 <= i < len(A) and 0 <= j < len(A)
    temp = A.copy()
    A[i], A[j] = temp[j], temp[i]
    return A

def scale_row(A, i, c):
    assert c != 0 and 0 <= i < len(A)
    A[i] = A[i] * c
    return A

def add_multiple_of_row(A, source, target, c):
    assert source != target and 0 <= source < len(A) and 0 <= target < len(A)
    A[target] = A[target] + c * A[source]
    return A

def inversion(E):
    assert E.shape[0] == E.shape[1]
    tol = 1e-10
    A = E.copy()
    I = np.eye(len(A))
    for i in range(len(A)):
        for j in range(i, len(A)):
            if abs(A[j][i]) > tol:
                swap_rows(A, i, j)
                swap_rows(I, i, j)
                break
        else:
            raise ValueError("Matrix is singular")
        for j in range(i+1, len(A)):
            ratio = A[j][i]/A[i][i]
            add_multiple_of_row(A, i, j, -1*ratio)
            add_multiple_of_row(I, i, j, -1*ratio)
    for i in range(len(A)):
        for j in range(0,i):
            ratio = A[j][i]/A[i][i]
            add_multiple_of_row(A, i, j, -1*ratio)
            add_multiple_of_row(I, i, j, -1*ratio)
    for i in range(len(A)):
            ratio = 1/A[i][i]
            scale_row(A, i, ratio)
            scale_row(I, i, ratio)
    return I


def elimination(A, b):
    tol = 1e-10
    E = A.copy()
    D = b.copy()
    pivot_cols = list()
    row = 0
    L = np.eye(len(A))
    P = np.eye(len(A))
    for i in range(max(len(E[0]), len(E))):
        for j in range(row,len(E)):
            if abs(E[j][min(i, len(E[0])-1)]) > tol:
                swap_rows(E, row, j)
                swap_rows(D, row, j)
                swap_rows(P, row, j)
                L[row][:row], L[j][:row] = L[j][:row].copy(), L[row][:row].copy()
                pivot_cols.append(i)
                break
        if min(i, len(E[0])-1) in pivot_cols:
            for k in range(row+1, len(E)):
                ratio = E[k][min(i, len(E[0])-1)]/E[row][min(i, len(E[0])-1)]
                add_multiple_of_row(E, row, k, -1*ratio)
                add_multiple_of_row(D, row, k, -1*ratio)
                L[k][row] = ratio
            if pivot_cols[-1] == len(E[0])-1:
                return E, D, pivot_cols, L, P
            row += 1
    return E, D, pivot_cols, L, P



def backsubstitution(A, b):
    U, b, pivot_cols, L, P = elimination(A, b)
    X = np.zeros(len(U[0]))
    if len(pivot_cols) != len(U[0]):
        raise ValueError(f"singular system: {len(pivot_cols)} pivots for {len(U[0])} columns")
    for i in range(len(b)):
        r = 0
        for j in range(i):
            r += U[len(b)-i-1][len(b)-j-1]*X[len(b)-j-1]
        X[len(b)-i-1] = (b[len(b)-i-1]-r)/U[len(b)-i-1][len(b)-i-1]

    return X

rng = np.random.default_rng(0)
fails = 0

for trial in range(50):
    A = rng.integers(-9, 10, size=(5, 5)).astype(float)
    b = rng.integers(-9, 10, size=5).astype(float)

    if abs(np.linalg.det(A)) < 1e-8:
        continue
    x = backsubstitution(A, b.copy())

    matches_numpy = np.allclose(x, np.linalg.solve(A, b.copy()))
    residual_ok = np.allclose(A @ x, b.copy())

    if not (matches_numpy and residual_ok):
        fails += 1
        print(f"FAIL on trial {trial}\nA =\n{A}\nb = {b}\ngot {x}")

print("all passed" if fails == 0 else f"{fails} failures")