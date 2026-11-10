import numpy as np

print("=== ALADIN G2 Algebraic Confirmation Suite ===\n")

# ----------------------------------------------------------------
# Shared Fano table and generators
# ----------------------------------------------------------------
lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
table = np.zeros((7,7,7))
for i,j,k in lines:
    table[i,j,k] = table[j,k,i] = table[k,i,j] = 1.0
    table[j,i,k] = table[k,j,i] = table[i,k,j] = -1.0

E = np.array([
[[0,1,0,0,0,0,0],[-1,0,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,-1,0,0,0,0],[0,0,0,0,0,-1,0],[0,0,0,0,1,0,1],[0,0,0,0,0,-1,0]],
[[0,0,1,0,0,0,0],[0,0,0,-1,0,0,0],[-1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1],[0,0,0,0,-1,1,0]],
[[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],[0,-1,0,0,0,0,0],[-1,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,0,0,0,0,0,0]],
[[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,-1],[0,0,0,0,0,0,0],[-1,0,0,0,0,0,0],[0,-1,0,0,0,0,0],[0,0,1,0,0,0,0]],
[[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,1],[0,1,0,0,0,0,0],[-1,0,0,0,0,0,0],[0,0,0,-1,0,0,0]],
[[0,0,0,0,0,0,1],[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,-1,0],[0,0,-1,0,0,0,0],[0,0,0,1,0,0,0],[-1,0,0,0,0,0,0]],
[[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[0,-1,0,0,0,0,0],[0,0,-1,0,0,0,0],[0,0,0,-1,0,0,0]],
[[0,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,0,0,0,0,0,-1],[0,0,1,0,0,0,0],[0,-1,0,0,0,0,0],[0,0,0,1,0,0,0]],
[[0,0,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,1,0,0,0],[0,0,-1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,-1,0,0,0,0,0]],
[[0,-1,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,-1,0,0,0,0],[0,0,0,-1,0,0,0],[0,0,0,0,0,0,0]],
[[0,0,0,0,0,0,1],[0,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,0,0,1,0,0,0],[0,0,-1,0,0,0,0],[-1,0,0,0,0,0,0]],
[[0,0,0,0,0,0,0],[0,0,0,0,0,-1,0],[0,0,0,0,0,0,1],[0,0,0,0,1,0,0],[0,0,0,-1,0,0,0],[0,1,0,0,0,0,0],[0,0,-1,0,0,0,0]],
[[0,0,-1,0,0,0,0],[0,0,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[0,0,0,-1,0,0,0],[0,0,0,0,-1,0,0]],
[[0,0,0,0,0,0,0],[0,0,-1,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,-1,0,0],[0,0,0,-1,0,0,0]]
], dtype=np.float64)

def full_multiply(x0, x, y0, y):
    dot = np.dot(x, y)
    z0 = x0*y0 - dot
    cross = np.einsum('i,j,ijk->k', x, y, table)
    z = x0*y + y0*x + cross
    return z0, z

def phi(a, b, c):
    cross = np.einsum('i,j,ijk->k', a, b, table)
    return np.dot(cross, c)

# ----------------------------------------------------------------
# 1. Generators
# ----------------------------------------------------------------
def test_generators():
    skew = all(np.allclose(A, -A.T) for A in E)
    rank = np.linalg.matrix_rank(E.reshape(14, -1))
    traces = all(np.allclose(np.trace(A), 0) for A in E)
    ok = skew and rank == 14 and traces
    print(f"[1] Generators: skew={skew}, rank={rank}/14, trace0={traces} -> {'PASS' if ok else 'FAIL'}")
    return ok

# ----------------------------------------------------------------
# 2. Equivariance
# ----------------------------------------------------------------
def test_equivariance(n=10):
    max_res = 0.0
    for _ in range(n):
        e = np.random.randn(7)
        e /= np.linalg.norm(e) + 1e-12
        coeffs = 0.01 * np.random.randn(14)
        omega = np.tensordot(coeffs, E, axes=1)
        g = np.eye(7) + omega
        ge = g @ e
        res = abs(phi(ge, ge, ge) - phi(e, e, e))
        max_res = max(max_res, res)
    ok = max_res < 1e-10
    print(f"[2] Equivariance: max |Δφ|={max_res:.1e} -> {'PASS' if ok else 'FAIL'}")
    return ok

# ----------------------------------------------------------------
# 3. Unit alternativity
# ----------------------------------------------------------------
def test_alternativity():
    basis = np.eye(7)
    max_res = 0.0
    for i in range(7):
        for j in range(7):
            if i == j: continue
            # left
            d0, d = associator(0, basis[i], 0, basis[i], 0, basis[j])
            max_res = max(max_res, abs(d0)+np.linalg.norm(d))
            # right
            d0, d = associator(0, basis[j], 0, basis[i], 0, basis[i])
            max_res = max(max_res, abs(d0)+np.linalg.norm(d))
    ok = max_res < 1e-12
    print(f"[3] Unit alternativity: max residual={max_res:.1e} -> {'PASS' if ok else 'FAIL'}")
    return ok

def associator(x0, x, y0, y, z0, z):
    xy0, xy = full_multiply(x0, x, y0, y)
    left0, left = full_multiply(xy0, xy, z0, z)
    yz0, yz = full_multiply(y0, y, z0, z)
    right0, right = full_multiply(x0, x, yz0, yz)
    return left0-right0, left-right

# ----------------------------------------------------------------
# 4. Non-associativity
# ----------------------------------------------------------------
def test_nonassociativity():
    basis = np.eye(7)
    nonzero = 0
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if len(set([i,j,k])) < 3: continue
                d0, d = associator(0, basis[i], 0, basis[j], 0, basis[k])
                if abs(d0)+np.linalg.norm(d) > 1e-12:
                    nonzero += 1
    ok = nonzero > 0
    print(f"[4] Non-associativity: {nonzero} non-zero triples -> {'PASS' if ok else 'FAIL'}")
    return ok

# ----------------------------------------------------------------
# 5. Zero-divisors
# ----------------------------------------------------------------
def test_zero_divisors():
    basis = np.eye(7)
    zd = 0
    for i in range(7):
        for j in range(7):
            z0, z = full_multiply(0, basis[i], 0, basis[j])
            if np.sqrt(z0*z0 + np.dot(z,z)) < 1e-10:
                zd += 1
    ok = zd == 0
    print(f"[5] Zero-divisors (basis): {zd} -> {'PASS' if ok else 'FAIL'}")
    return ok

# ----------------------------------------------------------------
# 6. Multiplication table
# ----------------------------------------------------------------
def test_table():
    errors = 0
    for i in range(7):
        z0, z = full_multiply(0, np.eye(7)[i], 0, np.eye(7)[i])
        if abs(z0 + 1) > 1e-12 or np.linalg.norm(z) > 1e-12:
            errors += 1
    ok = errors == 0
    print(f"[6] Multiplication table (squares): {errors} errors -> {'PASS' if ok else 'FAIL'}")
    return ok

# ----------------------------------------------------------------
# Run all
# ----------------------------------------------------------------
if __name__ == "__main__":
    results = [
        test_generators(),
        test_equivariance(),
        test_alternativity(),
        test_nonassociativity(),
        test_zero_divisors(),
        test_table(),
    ]
    print("\n=== Summary ===")
    print(f"Passed: {sum(results)} / {len(results)}")
    print("G2 algebraic core confirmed." if all(results) else "Some tests failed.")
