import numpy as np

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
    tol = np.max(np.abs(E)) * np.finfo(float).eps * max(len(E), len(E[0]))
    A = E.copy()
    I = np.eye(len(A))
    for i in range(len(A)):
        maximum, pos = 0, None
        for j in range(i, len(A)):
            if abs(A[j][i]) - maximum > tol:
                maximum = abs(A[j][i])
                pos = j
        if pos != None:
            swap_rows(A, i, pos)
            swap_rows(I, i, pos)
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
    tol = np.max(np.abs(A)) * np.finfo(float).eps * max(len(A), len(A[0]))
    E = A.copy()
    D = b.copy()
    pivot_cols = list()
    row = 0
    L = np.eye(len(A))
    P = np.eye(len(A))
    for i in range(max(len(E[0]), len(E))):
        maximum, pos = 0, None
        for j in range(row,len(E)):
            if abs(E[j][min(i, len(E[0])-1)]) - maximum > tol:
                maximum = abs(E[j][min(i, len(E[0])-1)])
                pos = j
        if pos != None:
            swap_rows(E, row, pos)
            swap_rows(D, row, pos)
            swap_rows(P, row, pos)
            L[row][:row], L[pos][:row] = L[pos][:row].copy(), L[row][:row].copy()
            pivot_cols.append(i)
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

def RREF(A, b=None):
    if b is None:
        b = np.zeros(len(A))
    R, d, pivot_cols, _, _ = elimination(A, b)
    for r, c in enumerate(pivot_cols):
        for i in range(r):
            ratio = R[i][c]/R[r][c]
            add_multiple_of_row(R, r, i, -1*ratio)
            add_multiple_of_row(d, r, i, -1*ratio)
        ratio = 1/R[r][c]
        scale_row(R, r, ratio)
        scale_row(d, r, ratio)
    free_cols = []
    for c in range(len(R[0])):
        if c not in pivot_cols: free_cols.append(c)
    rank = len(pivot_cols)
    return R, pivot_cols, free_cols, rank, d

def elimination_old(A, b):
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

def null_space_special(A, pivot_cols=None, free_cols=None):
    if pivot_cols and free_cols and len(pivot_cols) + len(free_cols) == len(A[0]):
        R = A.copy()
    else:
        R, pivot_cols, free_cols, rank, b = RREF(A.copy())
    S = []
    for i, c in enumerate(free_cols):
        X = np.zeros(len(R[0]))
        for j in range(len(pivot_cols)):
            X[pivot_cols[j]] = -1 * R[j][c]
        X[c] = 1
        S.append(X)
    return S

def solve(A, b):
    R, pivot_cols, free_cols, rank, b = RREF(A.copy(), b.copy())
    if np.allclose(np.zeros(len(R) - rank), b[rank:len(b)]):
        X = np.zeros(len(R[0]))
        for i, c in enumerate(pivot_cols):
            X[c] = b[i]
        null_space = null_space_special(R, pivot_cols, free_cols)
    else:
        return None, None

    return X, null_space

def four_subspaces(A):
    R, pivot_cols, free_cols, rank, b = RREF(A.copy())
    N = null_space_special(R, pivot_cols, free_cols)
    C = []
    for c in range(rank):
        C.append(A[:, pivot_cols[c]])
    At = A.T.copy()
    Rt, pivot_colst, free_colst, rankt, bt = RREF(At.copy())
    Nt = null_space_special(Rt, pivot_colst, free_colst)
    Ct = []
    for c in range(rankt):
        Ct.append(At[:, pivot_colst[c]])
    return C, N, Ct, Nt

def determinant(A):
    if len(A)==1: return A[0][0]
    else:
        det = 0
        for i in range(len(A)):
            if A[0][i] == 0: continue
            else:
                cofactor_matrix_pre = np.delete(A, i, axis=1)
                det += (-1)**i*A[0][i]*determinant(cofactor_matrix_pre[1:])
    return det

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = np.array([[1., 2., 3.],
                [2., -5., 7.],
                [1., 2., 4.]])

    b = np.array([5., 1., -6.])

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

    G = np.array([[0.000000001, 1.],
                [1., 1.]])
    print(elimination(G, np.zeros(len(G)))[0])
    print(elimination_old(G, np.zeros(len(G)))[0])

    F = np.array([[5., 10., 3.],
                [9., 18., 6.],
                [9., 18., 4.],
                [1., 2., 6.]])
    C, N, Ct, Nt = four_subspaces(F)
    print("C: ", C, "N: ", N, "Ct: ", Ct, "Nt: ", Nt)
    
    H = rng.integers(-9, 10, size=(5, 5)).astype(float)
    print(np.allclose(determinant(H),np.linalg.det(H)))
    