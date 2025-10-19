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

# Tiny self-test for helpers
if __name__ == "__main__":
    DEBUG = True
    n = 5
    A = np.arange(n*n).reshape(n, n)
    debug_print("A=\n", A)

    # Pad test
    target = next_power_of_two(n)
    Ap = pad_to_power_of_two(A, target)
    debug_print(f"\nPadded to {target}:\n", Ap)

    # Split-join test
    B = np.arange(16).reshape(4,4)
    B11, B12, B21, B22 = split_quadrants(B)
    B_back = join_quadrants(B11, B12, B21, B22)
    assert np.array_equal(B, B_back), "Split-join failed"
    debug_print("\nSplit-join successful on B=\n", B)  