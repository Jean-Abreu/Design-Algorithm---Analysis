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

def pad_to_power_of_two(A: np.ndarray, target: int) -> np.ndarray:
    """ Pad a square matrix A (n x n) with zeros to size target x target."""
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
    mid = n // 2
    return A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]

def join_quadrants(C11: np.ndarray, C12: np.ndarray, C21: np.ndarray, C22: np.ndarray) -> np.ndarray:
    """Join 4 blocks into one single matrix."""
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
    # Use local variables for speed in python loop
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            a_ik = Ai[k]
            # fused inner loop: Ci[j] += a_ik * B[k][j]
            Ci += a_ik * B[k]
    return C

def validate_multiply(A: np.ndarray, B: np.ndarray, fn, name: str = "candidate") -> None:
    """
    Compare the output of fn(A, B) to with Numpy's matmul.
    Raises AssertionError if mismatch is detected.
    """
    assert is_square_same_shape(A, B), "A and B must be same square shape."
    C_ref = A @ B
    C_out = fn(A, B)
    if not np.array_equal(C_ref, C_out):
        # Give a small diff summary to help debug
        diff = np.abs(C_ref - C_out)
        max_err = diff.max()
        where = np.argwhere(diff != 0)
        debug_print(f"[validate] Mismatch detected. max_err={max_err},first few diffs indices={where[:5]}")
        raise AssertionError(f"{name} produced incorrect result.")
    debug_print(f"[validate] {name} multiply NumPy for shape {A.shape}.")

# Step 3: standard divide-and-conquer with padding
def _dc_recursive(A: np.ndarray, B: np.ndarray, base_threshold: int = 64) -> np.ndarray:
    """
    Standard divide-and conquer matrix multiplication on power-of-two shapes.
    Assumes A and B are square and same shape with n a power of two.
    """
    n = A.shape[0]
    if n <= base_threshold:
        return A @ B
    
    A11, A12, A21, A22 = split_quadrants(A)
    B11, B12, B21, B22 = split_quadrants(B)

    # 8 sub-multiplies
    P1 = _dc_recursive(A11, B11, base_threshold)
    P2 = _dc_recursive(A12, B21, base_threshold)
    P3 = _dc_recursive(A11, B12, base_threshold)
    P4 = _dc_recursive(A12, B22, base_threshold)
    P5 = _dc_recursive(A21, B11, base_threshold)
    P6 = _dc_recursive(A22, B21, base_threshold)
    P7 = _dc_recursive(A21, B12, base_threshold)
    P8 = _dc_recursive(A22, B22, base_threshold)

    C11 = P1 + P2
    C12 = P3 + P4
    C21 = P5 + P6
    C22 = P7 + P8

    return join_quadrants(C11, C12, C21, C22)

def standard_multiply(A: np.ndarray, B: np.ndarray, base_threshold: int = 64) -> np.ndarray:
    """
    Public entr point: Standard divide-and-conquer multiply with automatic padding.
    Works for any square n x n matrices A and B of the same shape.
    """
    assert is_square_same_shape(A, B), "A and B must be the same square shape."
    n = A.shape[0]
    # Choose dtype that avoids overflow on small random ints
    dtype = np.result_type(A.dtype, B.dtype, np.int32)
    A = A.astype(dtype, copy=False)
    B = B.astype(dtype, copy=False)

    # Pad to power-of-two
    target = next_power_of_two(n)
    Ap = pad_to_power_of_two(A, target)
    Bp = pad_to_power_of_two(B, target)

    Cp = _dc_recursive(Ap, Bp, base_threshold=base_threshold)
    C = unpad(Cp, n)
    return C

# Step 4: timing helpers and smokes tests
def time_once(fn, *args, **kwargs) -> float:
    """
    Time a single execution of fn(*args, **kwargs) and return elapsed seconds.
    """
    t0 = time.perf_counter()
    _ = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0

def time_trials(fn, A: np.ndarray, B: np.ndarray, trials: int = 3, **kwargs) -> float:
    """
    Run fn(A, B, **kwargs) several times and return the median elapsed seconds.
    Also check correctness vs NumPy on the first run.
    """
    # correctness check (first run)
    validate_multiply(A, B, lambda X, Y: fn(X, Y, **kwargs), name = getattr(fn, "__name__", "candidate"))

    times = []
    # one warm-up (not timed) to charge caches/JIT/etc. (helps with variability)
    _ = fn(A, B, **kwargs)

    for _i in range(trials):
        elapsed = time_once(fn, A, B, **kwargs)
        times.append(elapsed)
    return float(np.median(times))
    
    


# Main loop for quick tests

if __name__ == "__main__":
    DEBUG = True

    rng = np.random.default_rng(42)
    def ran_int_matrix(n: int, low: int = -5, high: int = 6, dtype=np.int64):
        return rng.integers(low, high, size=(n, n), dtype=dtype)
    
    #Choosing modest sizes so the smoke test finishes quickly
    sizes = [16, 32, 45, 50] # includes a non-power-of-two size (45, 50)
    base_threshold = 64

    print("\n==== Standard D&C Multiply: Smoke Timing ====")
    for n in sizes:
        A = ran_int_matrix(n)
        B = ran_int_matrix(n)
        sec = time_trials(standard_multiply, A, B, trials=3, base_threshold=base_threshold)
        print(f"n={n:4d} median_time={sec*1000:9.3f} ms (trials=3, base_threshold={base_threshold})")

    print("\nNote:")
    print(" - Validation runs compare results with NumPy each time; failures will raise AssertionError.")
    print(" - For your assignment tables, switch sizes to around 50, 500, and 1000 (±10%).")
    print(" - Use more trials (e.g., 5–7) when collecting final numbers for the report.")