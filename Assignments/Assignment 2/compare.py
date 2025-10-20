# compare.py
from __future__ import annotations
import argparse
import csv
import time
import numpy as np
from typing import Callable, List, Tuple

# Local imports
from standard import standard_multiply
from strassen import strassen_multiply, next_power_of_two

# -----------------------
# Utilities
# -----------------------

def rand_int_matrix(n: int, rng: np.random.Generator, low: int = -5, high: int = 6, dtype=np.int64):
    return rng.integers(low, high, size=(n, n), dtype=dtype)

def time_once(fn: Callable, A: np.ndarray, B: np.ndarray, **kwargs) -> float:
    t0 = time.perf_counter()
    _ = fn(A, B, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0

def time_trials(fn: Callable, A: np.ndarray, B: np.ndarray, trials: int = 3, **kwargs) -> float:
    # One warm-up to stabilize caches
    _ = fn(A, B, **kwargs)
    times = []
    for _ in range(trials):
        times.append(time_once(fn, A, B, **kwargs))
    return float(np.median(times))

def warn_if_excessive_padding(n: int, method: str):
    # Strassen & our standard D&C both pad to next power-of-two.
    padded = next_power_of_two(n)
    ratio = padded / n
    if ratio >= 1.6:  # heuristic threshold to warn
        print(f"[warn] {method}: n={n} pads to {padded} (x{ratio:.2f}). "
              f"This may be very slow and memory-heavy. Consider using sizes nearer {padded}.")

# -----------------------
# Benchmark runner
# -----------------------

def run_benchmarks(
    sizes_by_case: List[Tuple[str, List[int]]],
    trials: int,
    base_threshold_std: int,
    base_threshold_strassen: int,
    seed: int,
    csv_path: str
):
    rng = np.random.default_rng(seed)

    rows = []  # for CSV: [case, n, method, median_ms]
    for case_name, sizes in sizes_by_case:
        print(f"\n=== {case_name} ===")
        print(f"{'n':>6}  {'Std(ms)':>12}  {'Strassen(ms)':>14}")
        print("-" * 36)
        for n in sizes:
            # (Optional) warnings about severe padding
            warn_if_excessive_padding(n, "Standard D&C")
            warn_if_excessive_padding(n, "Strassen")

            A = rand_int_matrix(n, rng)
            B = rand_int_matrix(n, rng)

            # Standard D&C
            t_std = time_trials(
                standard_multiply, A, B,
                trials=trials, base_threshold=base_threshold_std
            ) * 1000.0

            # Strassen
            t_strassen = time_trials(
                strassen_multiply, A, B,
                trials=trials, base_threshold=base_threshold_strassen
            ) * 1000.0

            print(f"{n:6d}  {t_std:12.3f}  {t_strassen:14.3f}")

            rows.append([case_name, n, "Standard", f"{t_std:.3f}"])
            rows.append([case_name, n, "Strassen", f"{t_strassen:.3f}"])

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "n", "Method", "MedianTime_ms"])
        writer.writerows(rows)

    print(f"\nSaved results to {csv_path}")
    print("Tip: paste rows per case into tables in your report template. "
          "Add brief observations under each, as requested in the assignment. "
          "(Case 1 ~ n≈50, Case 2 ~ n≈500, Case 3 ~ n≈1000.)")

# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Standard D&C vs Strassen for assignment tables "
                    "(3 sizes near ~50, ~500, ~1000)."
    )
    parser.add_argument("--trials", type=int, default=3, help="Timing trials per (n, method).")
    parser.add_argument("--std-threshold", type=int, default=64,
                        help="Base threshold for standard D&C fallback to naive.")
    parser.add_argument("--strassen-threshold", type=int, default=64,
                        help="Base threshold for Strassen fallback to naive.")
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed for reproducibility.")
    parser.add_argument("--csv", type=str, default="results.csv", help="Output CSV path.")
    parser.add_argument("--strict-sizes", action="store_true",
                        help="Use the report’s exact table sizes (45/50/55, 450/500/550, 900/1000/1100). "
                             "Warning: 1100 pads to 2048 and can be very heavy for Strassen.")
    args = parser.parse_args()

    if args.strict_sizes:
        sizes_by_case = [
            ("Case 1 (≈50)",  [45, 50, 55]),     # matches the small-n table in the prompt
            ("Case 2 (≈500)", [450, 500, 550]),  # matches the medium-n table
            ("Case 3 (≈1000)", [900, 1000, 1100])  # matches the large-n table (watch padding!)
        ]
    else:
        # Performance-friendly alternatives to avoid extreme padding:
        # keep “≈50, ≈500, ≈1000” spirit but choose sizes that pad less.
        sizes_by_case = [
            ("Case 1 (≈50)",   [48, 50, 56]),     # pads to 64 at worst
            ("Case 2 (≈500)",  [480, 512, 544]),  # 512 is exact power-of-two
            ("Case 3 (≈1000)", [960, 1000, 1024]) # 1024 is power-of-two; avoids 2048 padding
        ]

    run_benchmarks(
        sizes_by_case=sizes_by_case,
        trials=args.trials,
        base_threshold_std=args.std_threshold,
        base_threshold_strassen=args.strassen_threshold,
        seed=args.seed,
        csv_path=args.csv
    )

if __name__ == "__main__":
    main()
