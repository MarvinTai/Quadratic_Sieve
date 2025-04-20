# Quadratic Sieve – adaptive & multi‑dependency edition
# ----------------------------------------------------
#  • Pure‑Python (no external libs).
#  • Comfortable up to 9‑digit semiprimes on a laptop.
#  • Gathers relations once, then explores **all** null‑space
#    dependencies before enlarging the factor base – this
#    fixes the “trivial dependency” loop the user hit.
# ----------------------------------------------------

from __future__ import annotations
import math
from collections import defaultdict
from math import gcd

# --------------------------------------------------
# 1.  Prime generation – sieve of Eratosthenes
# --------------------------------------------------

def primes_up_to(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = b"\x00" * (((n - p * p) // p) + 1)
    return [i for i, f in enumerate(sieve) if f]

# --------------------------------------------------
# 2.  Legendre symbol
# --------------------------------------------------

def legendre(a: int, p: int) -> int:
    return pow(a, (p - 1) >> 1, p)

# --------------------------------------------------
# 3.  Build factor base (−1 & 2 always present)
# --------------------------------------------------

def build_factor_base(n: int, B: int) -> list[int]:
    base = [-1]
    for p in primes_up_to(B):
        if p == 2:
            base.append(2)
        elif legendre(n % p, p) == 1:
            base.append(p)
    return base

# --------------------------------------------------
# 4.  Trial division of Q(x) over the factor base
# --------------------------------------------------

def trial_divide(q: int, base: list[int]):
    fac = defaultdict(int)
    if q < 0:
        fac[-1] = 1
        q = -q
    for p in base:
        if p < 2:
            continue
        while q % p == 0:
            fac[p] += 1
            q //= p
    return fac if q == 1 else None

# --------------------------------------------------
# 5.  Factorisation → exponent vector mod 2 (GF(2))
# --------------------------------------------------

def fac_to_vec(fac: dict[int, int], base: list[int]):
    return [(fac.get(p, 0) & 1) for p in base]

# --------------------------------------------------
# 6.  Null‑space of a binary matrix (bitset Gauss)
# --------------------------------------------------

def nullspace_masks(rows: list[list[int]]) -> list[int]:
    """Return list of bit‑masks – each mask is a dependency."""
    m, n = len(rows), len(rows[0])
    bit_rows = [int("".join(map(str, r[::-1])), 2) for r in rows]
    combos   = [1 << i for i in range(m)]

    col_pivot = {}
    for col in range(n):
        mask = 1 << col
        pivot = next((r for r in range(m) if (bit_rows[r] & mask) and col not in col_pivot.values()), None)
        if pivot is None:
            continue
        col_pivot[col] = pivot
        for r in range(m):
            if r != pivot and (bit_rows[r] & mask):
                bit_rows[r] ^= bit_rows[pivot]
                combos[r]   ^= combos[pivot]

    # zero rows → dependencies
    deps = [combos[r] for r in range(m) if bit_rows[r] == 0]
    return deps

# --------------------------------------------------
# 7.  Quadratic Sieve with multi‑dependency attempt
# --------------------------------------------------

def quadratic_sieve(n: int, B0: int | None = None):
    if n % 2 == 0:
        return 2
    r = math.isqrt(n)
    if r * r == n:
        return r

    # small trial division first
    for p in primes_up_to(100):
        if n % p == 0:
            return p

    # heuristic starting bound
    if B0 is None:
        B0 = int(max(50, 1.2 * math.exp(0.5 * math.sqrt(math.log(n) * math.log(math.log(n))))))
    B = B0

    while True:
        base = build_factor_base(n, B)
        need = len(base) + 10
        print(f"B = {B}  |base| = {len(base)}  collecting ≥{need} relations…")

        relations: list[tuple[int, dict[int, int], list[int]]] = []
        x = math.isqrt(n) + 1
        limit = x + 400_000  # adaptive window; increase if needed
        while x < limit and len(relations) < need:
            q = x * x - n
            fac = trial_divide(q, base)
            if fac:
                relations.append((x, fac, fac_to_vec(fac, base)))
            x += 1

        if len(relations) < need:
            B = int(B * 1.6)
            continue  # try with larger factor base

        # compute all dependencies
        deps = nullspace_masks([vec for *_, vec in relations])
        tried_trivial = False
        for mask in deps:
            idx = [i for i in range(len(relations)) if (mask >> i) & 1]
            if not idx:
                continue
            a, exps = 1, defaultdict(int)
            for i in idx:
                x_i, fac, _ = relations[i]
                a = (a * x_i) % n
                for p, e in fac.items():
                    exps[p] += e
            b = 1
            for p, e in exps.items():
                if p == -1:
                    continue
                b = (b * pow(p, e // 2, n)) % n
            g = gcd((a - b) % n, n)
            if g not in (1, n):
                return g
            tried_trivial = True

        # if we’re here, every dependency was trivial → enlarge search
        if tried_trivial:
            # try gathering more relations with same base first
            print("All dependencies trivial; gathering extra relations…")
            extra_target = len(base) + 30
            while len(relations) < extra_target and x < limit * 2:
                q = x * x - n
                fac = trial_divide(q, base)
                if fac:
                    relations.append((x, fac, fac_to_vec(fac, base)))
                x += 1
            if len(relations) > need:  # recompute dependencies
                deps = nullspace_masks([vec for *_, vec in relations])
                for mask in deps:
                    idx = [i for i in range(len(relations)) if (mask >> i) & 1]
                    a, exps = 1, defaultdict(int)
                    for i in idx:
                        x_i, fac, _ = relations[i]
                        a = (a * x_i) % n
                        for p, e in fac.items():
                            exps[p] += e
                    b = 1
                    for p, e in exps.items():
                        if p == -1:
                            continue
                        b = (b * pow(p, e // 2, n)) % n
                    g = gcd((a - b) % n, n)
                    if g not in (1, n):
                        return g
            # still nothing: enlarge factor base and retry
        B = int(B * 1.6)

# --------------------------------------------------
# Demo block
# --------------------------------------------------
if __name__ == "__main__":
    n = 10000019*98765431# problematic 7‑digit example
    print(f"Factoring {n}…")
    f = quadratic_sieve(n)
    print(f"→ {n} = {f} × {n//f}")
