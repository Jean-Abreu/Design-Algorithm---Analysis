from __future__ import annotations
from typing import Tuple
import numpy as np
import math
import time

DEBUG = False # set True while debugging small cases

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def is_square_same_shape(A: np.ndarray, B: np.ndarray) -> bool:
    return (
        isinstance(A, np.ndarray) and isinstance(B, np.ndarray) and
        A.ndim == 2 and B.ndim == 2 and 
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    )

def next_power_of_two(n: int) -> int:
    """ Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def pad_to_power_of_two(A: np.ndarray, target:int) -> np.ndarray:
    """ Pad a square matrix A (n x n) with zeros to size (target x target)."""
    n = A.shape[0]
    if n == target:
        return A
    P = np.zeros((target, target), dtype=A.dtype)
    P[:n, :n] = A
    return P

def unpad(C: np.ndarray, n: int) -> np.ndarray:
    """Crop C back to (n x n)."""
    return C[:n, :n]

def split_quadrants(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split A into 4 submatrices A11, A12, A21, A22 of size n/2 x n/2."""
    n = A.shape[0]
    mid = n //2
    return A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]

def join_quadrants(C11: np.ndarray, C12: np.ndarray, C21: np.ndarray, C22: np.ndarray) -> np.ndarray:
    """Join 4 blocks into one square matrix."""
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

# Step 2: naive multiply and validator
def naive_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A straightforward O(n^3) triple-loop multiply.
    Assumes A and B are square and same shape.
    """
    assert is_square_same_shape(A, B), "A and B must be same square shape."
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            a_ik = Ai[k]
            Ci += a_ik * B[k]
    return C

def validate_multiply(A: np.ndarray, B: np.ndarray, fn, name: str = "candidate") -> None:
    """
    Compare the output of fn(A, B) with NumPy's matmul.
    Raises AssertionError if they differ.
    """
    assert is_square_same_shape(A, B), "A and B must be same square shape."
    C_ref = A @ B
    C_out = fn(A, B)
    if not np.array_equal(C_ref, C_out):
        diff = np.abs(C_ref - C_out)
        max_err = diff.max()
        where = np.argwhere(diff != 0)
        debug_print(f"[validate] Mismatch. max_err={max_err}, first diffs idx={where[:5]}")
        raise AssertionError(f"{name} multiply produced incorrect result.")
    debug_print(f"[validate] {name} multiply matches NumPy for shape {A.shape}.")

# Step 3: Strassen with padding and base-threshold
def _strassen_recursive(A: np.ndarray, B: np.ndarray, base_threshold: int = 64) -> np.ndarray:
    """
    Strassen's recursive multiply on power-of-two shapes.
    Assumes A and B are square, same shape, and n is a power of two.
    """
    n = A.shape[0]
    if n <= base_threshold:
        return naive_multiply(A, B)
    
    A11, A12, A21, A22 = split_quadrants(A)
    B11, B12, B21, B22 = split_quadrants(B)

    # Strassen's 7 products
    M1 = _strassen_recursive(A11 + A22, B11 + B22, base_threshold)
    M2 = _strassen_recursive(A21 + A22, B11, base_threshold)
    M3 = _strassen_recursive(A11, B12 - B22, base_threshold)
    M4 = _strassen_recursive(A22, B21 - B11, base_threshold)
    M5 = _strassen_recursive(A11 + A12, B22, base_threshold)
    M6 = _strassen_recursive(A21 - A11, B11 + B12, base_threshold)
    M7 = _strassen_recursive(A12 - A22, B21 + B22, base_threshold)

    # Combine into C blocks
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    return join_quadrants(C11, C12, C21, C22)

def strassen_multiply(A: np.ndarray, B: np.ndarray, base_threshold: int = 64) -> np.ndarray:
    """
    Public entry: Strassen multiply with automatic padding for arbitary n.
    Works for any square n x n matrices A and B of the same shape.
    """
    assert is_square_same_shape(A, B), "A and B must be same square shape."
    n = A.shape[0]
    # need to be generous with dtype to avoid overflow on small random ints
    dtype = np.result_type(A.dtype, B.dtype, np.int64)
    A = A.astype(dtype, copy=False)
    B = B.astype(dtype, copy=False)

    target = next_power_of_two(n)
    Ap = pad_to_power_of_two(A, target)
    Bp = pad_to_power_of_two(B, target)

    Cp = _strassen_recursive(Ap, Bp, base_threshold=base_threshold)
    C = unpad(Cp, n)
    return C

# Step 4: timing helpers and smoke tests
def time_once(fn, *args, **kwargs) -> float:
    """Time a single call to fn(*args, **kwargs) and return elapsed seconds."""
    t0 = time.perf_counter()
    _ = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0

def time_trials(fn, A: np.ndarray, B: np.ndarray, trials: int = 3, **kwargs) -> float:
    """
    Run fn(A,B,**kwargs) several times and return the median elapsed seconds.
    Also checks correctness vs NumPy on the first run.
    """
    # correctness check (first run)
    validate_multiply(A, B, lambda X, Y: fn(X, Y, **kwargs), name=getattr(fn, "__name__", "candidate"))

    times = []
    _ = fn(A, B, **kwargs)  # warm-up (not timed)

    for _i in range(trials):
        elapsed = time_once(fn, A, B, **kwargs)
        times.append(elapsed)
    return float(np.median(times))

# Tiny self-test for helpers
if __name__ == "__main__":
    DEBUG = True
    
    rng = np.random.default_rng(123)

    def rand_int_matrix(n: int, low: int = -5, high: int = 6, dtype=np.int64): 
        return rng.integers(low, high, size=(n, n), dtype=dtype)

    sizes = [16, 32, 45, 50]
    base_threshold = 64

    print("\n=== Strassen Multiply: Smoke Timing ===")
    for n in sizes:
        A = rand_int_matrix(n)
        B = rand_int_matrix(n)
        sec = time_trials(strassen_multiply, A, B, trials=3, base_threshold=base_threshold)
        print(f"n={n:4d}  median_time={sec*1000:9.3f} ms  (trials=3, base_threshold={base_threshold})")

    print("\nNotes:")
    print(" - Each run is validated against NumPy; any mismatch raises AssertionError.")
    print(" - For assignment tables, test around n=50, 500, 1000 (±10%).")
    print(" - Consider increasing trials (5–7) and adjusting base_threshold for large n.")
