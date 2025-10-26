import math
import sys
import time
from multiprocessing import Pool, cpu_count
from itertools import count

CHUNK = 1_000_000        # numbers per part
INITIAL_BASE = 100_000   # starting base primes for worker cache
WORKERS = cpu_count()    # use all CPU cores
DURATION_SECS = 120      # run time in seconds
PRINT_PROGRESS = False

_base_primes = []
_base_max = 0

def sieve_simple(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    limit = int(n**0.5)
    for p in range(2, limit + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = b'\x00' * ((n - p*p)//p + 1)
    return [i for i, isprime in enumerate(sieve) if isprime]

def init_worker(initial_base):
    global _base_primes, _base_max
    _base_primes = sieve_simple(initial_base)
    _base_max = initial_base

def ensure_base_primes(limit):
    global _base_primes, _base_max
    if limit > _base_max:
        _base_primes = sieve_simple(limit)
        _base_max = limit

def segmented_sieve_segment(args):
    low, high, seg_idx = args
    if low < 2:
        low = 2
    limit = int(math.isqrt(high)) + 1
    ensure_base_primes(limit)
    base = [p for p in _base_primes if p <= limit]
    size = high - low
    sieve = bytearray(b'\x01') * size
    for p in base:
        start = (-(low % p) + p) % p
        first = low + start
        if first == p:
            start += p
        sieve[start:size:p] = b'\x00' * (((size - 1 - start) // p) + 1)
    return sum(sieve)  # count of primes in this part

def segment_generator(start=2, chunk=CHUNK):
    idx = 0
    for k in count(0):
        low = start + k * chunk
        high = low + chunk
        yield (low, high, idx)
        idx += 1

def main():
    try:
        print(f"# Prime generator speed test — {WORKERS} workers for {DURATION_SECS}s", file=sys.stderr)
        start_time = time.time()
        total_primes = 0
        total_segments = 0

        with Pool(processes=WORKERS, initializer=init_worker, initargs=(INITIAL_BASE,)) as pool:
            seg_iter = segment_generator()
            for count_primes in pool.imap_unordered(segmented_sieve_segment, seg_iter, chunksize=1):
                total_primes += count_primes
                total_segments += 1
                elapsed = time.time() - start_time
                if elapsed >= DURATION_SECS:
                    break
                if PRINT_PROGRESS and total_segments % 5 == 0:
                    print(f"{elapsed:6.1f}s  segments={total_segments}  primes={total_primes}", file=sys.stderr)

        elapsed = time.time() - start_time
        rate = total_primes / elapsed
        rate_per_core = rate / WORKERS

        print("\n--- Performance summary ---")
        print(f"Elapsed time:     {elapsed:,.1f} s")
        print(f"Total primes:     {total_primes:,}")
        print(f"Primes / second:  {rate:,.1f}")
        print(f"Per core rate:    {rate_per_core:,.1f}")
        print(f"Cores used:       {WORKERS}")
        print(f"Chunk size:       {CHUNK:,}")
        print("---------------------------")

    except KeyboardInterrupt:
        print("\n# Interrupted by user.", file=sys.stderr)

if __name__ == "__main__":
    main()
