# Pure‑Python Quadratic Sieve + Pollard‑Brent prefilter
# =====================================================

from __future__ import annotations
import math, random
from collections import defaultdict
from math import gcd, isqrt, log


# ----------------------------------------------------
# 0.  Pollard–Brent ρ  (cheap small‑factor filter)
# ----------------------------------------------------
def pollard_brent_rho(n: int) -> int | None:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y, c, m = random.randrange(1, n), random.randrange(1, n), 1
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r <<= 1
        if g == n:                             # cycle back‑tracking
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


# ----------------------------------------------------
# 1.  Sieve of Eratosthenes
# ----------------------------------------------------
def primes_up_to(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = b"\x00" * (((n - p * p) // p) + 1)
    return [i for i, f in enumerate(sieve) if f]


# ----------------------------------------------------
# 2.  Tonelli–Shanks square root  (mod p)
# ----------------------------------------------------
def mod_sqrt(n: int, p: int) -> tuple[int, int] | None:
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:                     # fast path
        r = pow(n, (p + 1) // 4, p)
        return r, p - r
    q, s = p - 1, 0
    while not q & 1:
        q >>= 1
        s += 1
    z = next(z for z in range(2, p) if pow(z, (p - 1) // 2, p) == p - 1)
    c, r = pow(z, q, p), pow(n, (q + 1) // 2, p)
    t, m = pow(n, q, p), s
    while t != 1:
        i, tmp = 1, pow(t, 2, p)
        while tmp != 1:
            tmp = pow(tmp, 2, p)
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        r, c = (r * b) % p, pow(b, 2, p)
        t, m = (t * c) % p, i
    return r, p - r


# ----------------------------------------------------
# 3.  Factor base with roots  (p, r1, r2)
# ----------------------------------------------------
def build_factor_base(N: int, B: int):
    fb = []
    for p in primes_up_to(B):
        if p == 2:
            if N & 1:                       #  N odd ⇒ root = 1 mod 2
                fb.append((2, 1, 1))
            continue
        roots = mod_sqrt(N % p, p)
        if roots:
            fb.append((p, *roots))
    return fb


# ----------------------------------------------------
# 4.  Block‑sieve  → relations list
# ----------------------------------------------------
def sieve_block(N: int, fb, x0: int, length: int, slack_bits=1.5):
    import math
    logs = [0.0] * length
    for p, r1, r2 in fb:
        logp = math.log(p, 2)
        for r in (r1, r2):
            off = (r - x0) % p
            for pos in range(off, length, p):
                logs[pos] += logp

    rels = []
    for i, s in enumerate(logs):
        x = x0 + i
        Q = x * x - N
        absQ = abs(Q)
        if absQ == 0:
            continue
        if s >= math.log(absQ, 2) - slack_bits:          # likely smooth
            fac = defaultdict(int)
            if Q < 0:
                fac[-1] = 1
            q = absQ
            for p, *_ in fb:
                while q % p == 0:
                    fac[p] += 1
                    q //= p
            if q == 1:
                rels.append((x, fac))
    return rels


# ----------------------------------------------------
# 5.  Factorisation dict → parity vector
# ----------------------------------------------------
def fac_to_vec(fac, primes):
    return [(fac.get(p, 0) & 1) for p in primes]


# ----------------------------------------------------
# 6.  Binary Gaussian elimination  (GF(2) null‑space)
# ----------------------------------------------------
def nullspace_masks(rows):
    m, n = len(rows), len(rows[0])
    bits = [int("".join(map(str, r[::-1])), 2) for r in rows]
    combos = [1 << i for i in range(m)]
    piv = {}
    for col in range(n):
        mask = 1 << col
        piv_row = next((r for r in range(m)
                        if (bits[r] & mask) and col not in piv.values()), None)
        if piv_row is None:
            continue
        piv[col] = piv_row
        for r in range(m):
            if r != piv_row and bits[r] & mask:
                bits[r] ^= bits[piv_row]
                combos[r] ^= combos[piv_row]
    return [combos[r] for r in range(m) if bits[r] == 0]


# ----------------------------------------------------
# 7.  Quadratic Sieve
# ----------------------------------------------------
def quadratic_sieve(N: int, B0=None):
    if N % 2 == 0:
        return 2
    r = isqrt(N)
    if r * r == N:
        return r

    # Pollard–Brent first
    g = pollard_brent_rho(N)
    if g and g not in (1, N):
        return g

    if B0 is None:                                  # Pomerance heuristic
        B0 = int(2.0 * math.exp(0.55 * math.sqrt(log(N) * log(log(N)))))
    B = B0

    while True:
        fb = build_factor_base(N, B)
        primes = [p for p, *_ in fb]
        need = len(primes) + 15
        window = int(8_000 + 160 * len(primes))
        print(f"B={B}  |FB|={len(primes)}  need≈{need}  window={window}")

        relations = []
        x0 = isqrt(N) + 1
        offset = 0
        while len(relations) < need:
            relations += sieve_block(N, fb, x0 + offset, window)
            offset += window
            if offset > 30 * window and len(relations) < need // 4:
                break                              # base too small

        if len(relations) < need:                 # enlarge B and retry
            B = int(B * 1.5)
            continue

        vecs = [fac_to_vec(fac, primes) for _, fac in relations]
        for mask in nullspace_masks(vecs):
            idx = [i for i in range(len(relations)) if mask >> i & 1]
            if not idx:
                continue
            a, expo = 1, defaultdict(int)
            for i in idx:
                x, fac = relations[i]
                a = (a * x) % N
                for p, e in fac.items():
                    expo[p] += e
            b = 1
            for p, e in expo.items():
                if p == -1:
                    continue
                b = (b * pow(p, e // 2, N)) % N
            g = gcd((a - b) % N, N)
            if g not in (1, N):
                return g

        B = int(B * 1.5)                          # still unlucky → grow FB


# ----------------------------------------------------
# 8.  Demo
# ----------------------------------------------------
if __name__ == "__main__":
    N = 46839566299936919234246726809       # 9.88×10^15
    print(f"Factoring {N} …")
    f = quadratic_sieve(N)
    print(f"\n→ {N} = {f} × {N // f}")
