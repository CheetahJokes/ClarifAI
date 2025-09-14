# filename: pi_ray_benchmark.py
import time
import random
import math
from typing import List

# ----- Config -----
TOTAL_SAMPLES = 20_000_000     # total dart throws
NUM_TASKS     = None           # None => auto (CPU count). Or set e.g. 8
N_TRIALS      = 3              # run multiple trials and take best
WARMUP        = True           # one warmup per mode to stabilize JIT/memory

# ----- Pure Python baseline (sequential) -----
def pi_seq(sample_count: int) -> float:
    inc = 0
    # Use local variables for tiny speed win in Python
    rand = random.random
    for _ in range(sample_count):
        x = rand(); y = rand()
        if x*x + y*y <= 1.0:
            inc += 1
    return 4.0 * (inc / sample_count)

# ----- Ray parallel version -----
import multiprocessing
import ray

@ray.remote
def _pi_chunk(sample_count: int) -> int:
    inc = 0
    rand = random.random
    for _ in range(sample_count):
        x = rand(); y = rand()
        if x*x + y*y <= 1.0:
            inc += 1
    return inc

def pi_ray(total_samples: int, num_tasks: int) -> float:
    # Split work across tasks (spread remainder)
    base = total_samples // num_tasks
    rem  = total_samples % num_tasks
    splits: List[int] = [base + (1 if i < rem else 0) for i in range(num_tasks)]

    futures = [_pi_chunk.remote(n) for n in splits]
    inc_total = sum(ray.get(futures))
    return 4.0 * (inc_total / total_samples)

def best_time(fn, *args):
    # optional warmup
    if WARMUP:
        fn(*args)
    best = math.inf
    last_val = None
    for _ in range(N_TRIALS):
        t0 = time.time()
        val = fn(*args)
        dt = time.time() - t0
        if dt < best:
            best = dt
            last_val = val
    return best, last_val

def main():
    # Initialize Ray (local)
    ray.init(ignore_reinit_error=True)

    # Choose number of tasks
    if NUM_TASKS is None:
        # Prefer Ray’s view of CPUs if available; else fall back to mp
        try:
            cpus = int(ray.available_resources().get("CPU", 0))
        except Exception:
            cpus = 0
        if cpus <= 0:
            cpus = multiprocessing.cpu_count()
        num_tasks = max(1, cpus)
    else:
        num_tasks = int(NUM_TASKS)

    print(f"\n--- Monte Carlo π Benchmark ---")
    print(f"Total samples: {TOTAL_SAMPLES:,}")
    print(f"Ray tasks    : {num_tasks}")
    print(f"Trials/mode  : {N_TRIALS} (best time reported)")
    print(f"Warmup       : {WARMUP}\n")

    # Sequential
    t_seq, pi_seq_val = best_time(pi_seq, TOTAL_SAMPLES)
    print(f"[Sequential]  π ≈ {pi_seq_val:.8f} | time = {t_seq:.3f}s")

    # Ray parallel
    t_ray, pi_ray_val = best_time(pi_ray, TOTAL_SAMPLES, num_tasks)
    print(f"[Ray]         π ≈ {pi_ray_val:.8f} | time = {t_ray:.3f}s")

    # Speedup
    if t_ray > 0:
        speedup = t_seq / t_ray
    else:
        speedup = float('inf')
    print(f"\nSpeedup (Seq / Ray): {speedup:.2f}×")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
