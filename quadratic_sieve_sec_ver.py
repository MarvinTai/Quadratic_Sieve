import math
from math import gcd, isqrt
from collections import defaultdict
import sys

###############################################################################
# 1) PRIME SIEVE: Sieve of Eratosthenes
###############################################################################
def sieve_of_eratosthenes(limit):
    """Return a list of all primes up to 'limit'."""
    sieve = [True]*(limit+1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5)+1):
        if sieve[i]:
            for multiple in range(i*i, limit+1, i):
                sieve[multiple] = False
    return [p for p, is_prime in enumerate(sieve) if is_prime]

###############################################################################
# 2) LEGENDRE SYMBOL
###############################################################################
def legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p) using Euler's criterion.
    For an odd prime p:
        (a/p) =  1  if a is a quadratic residue mod p,
               -1  if a is a non-residue,
                0  if p divides a.
    """
    ls = pow(a, (p - 1)//2, p)
    if ls == 0:
        return 0
    elif ls == p - 1:
        return -1
    else:
        return ls  # 1

###############################################################################
# 3) GENERATE FACTOR BASE
###############################################################################
def generate_factor_base(n, B):
    """
    Build a factor base consisting of:
      - The prime -1 to handle signs.
      - All primes p up to B such that:
           • p == 2 is always included
           • for odd p, n is a quadratic residue modulo p (i.e. legendre_symbol(n, p) == 1)
    """
    base_primes = sieve_of_eratosthenes(B)
    factor_base = [-1]  # begin with -1 to handle sign of Q(x)
    for p in base_primes:
        if p == 2:
            factor_base.append(p)
        else:
            if legendre_symbol(n, p) == 1:
                factor_base.append(p)
    return factor_base

###############################################################################
# 4) TRIAL DIVISION FOR SMOOTHNESS
###############################################################################
def trial_division(num, factor_base):
    """
    Try to fully factor 'num' (which will be Q(x) = x^2 - n) over the factor base.
    Returns a dictionary {prime: exponent} if the complete factorization over the factor
    base is found (i.e. num becomes 1), otherwise returns None.
    """
    factors = defaultdict(int)
    if num < 0:
        factors[-1] = 1
        num = -num
    for p in factor_base:
        if p == -1:
            continue  # already handled
        while num % p == 0:
            factors[p] += 1
            num //= p
    return dict(factors) if num == 1 else None

###############################################################################
# 5) FACTORIZATION TO EXPONENT VECTOR (MOD 2)
###############################################################################
def factorization_to_vector(factors, factor_base):
    """
    Given a factorization (dictionary: prime -> exponent) and the factor base (list),
    produce an exponent vector (list of 0s and 1s) modulo 2 in the order of the factor base.
    """
    return [factors.get(p, 0) % 2 for p in factor_base]

###############################################################################
# 6) GAUSSIAN ELIMINATION OVER GF(2) TO FIND A DEPENDENCY
#
# This routine works on the matrix of smooth relation exponent vectors (each row is a
# relation, with entries in {0,1}). We wish to find a nonzero vector v (over GF(2)) such that:
#
#     v[0]*row0 + v[1]*row1 + ... + v[R-1]*row_{R-1} = 0    (all arithmetic mod 2)
#
# If we set up the augmented matrix [A | I] with A the R x C matrix and I the identity of
# size R, then any row of zeros in the A–part will have the corresponding row of I giving a
# dependency vector on the original rows.
###############################################################################
def find_dependency_GF2(matrix):
    """
    Given a list 'matrix' of R rows (each row is a list of 0/1 of length C),
    perform in-place Gaussian elimination over GF(2) (using an augmented identity matrix)
    and return a dependency vector v (of length R, with entries in {0,1}) such that
       sum_{i=0}^{R-1} v[i]*matrix[i] == 0 (mod 2).
    
    Returns None if no dependency is found.
    """
    R = len(matrix)
    if R == 0:
        return None
    C = len(matrix[0])
    
    # Build augmented matrix: each row becomes row || identity_row.
    # The augmented matrix will have C+R columns.
    A = [row[:] + [1 if i == j else 0 for j in range(R)] for i, row in enumerate(matrix)]
    
    pivot_row = 0
    for col in range(C):
        # Find a pivot row with a 1 in current column, starting at pivot_row.
        pivot = None
        for r in range(pivot_row, R):
            if A[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            # No pivot in this column means that column is "free."
            # (We continue since this can later produce a dependency.)
            continue
        # Swap pivot row into position if necessary.
        if pivot != pivot_row:
            A[pivot_row], A[pivot] = A[pivot], A[pivot_row]
        # Eliminate 1's in the current column in all rows except pivot_row.
        for r in range(R):
            if r != pivot_row and A[r][col] == 1:
                # Add (XOR) pivot row to row r.
                A[r] = [ (a ^ b) for a, b in zip(A[r], A[pivot_row]) ]
        pivot_row += 1
        if pivot_row == R:
            break

    # Look for a row that became zero in the left part (first C columns).
    for r in range(R):
        if all(A[r][j] == 0 for j in range(C)):
            # Return the augmented part of that row (the dependency vector).
            dependency = A[r][C:]
            # Verify that it is nontrivial.
            if any(dependency):
                return dependency
    return None

###############################################################################
# 7) QUADRATIC SIEVE: MAIN FUNCTION
###############################################################################
def quadratic_sieve(n, 
                    factor_base_bound=300000,   # Increased bound for larger numbers
                    max_sieve_attempts=3000000,  # Increase search range for smooth relations
                    extra_relations=200         # Increase extra relations to improve chance for dependency
                   ):
    """
    Attempt to factor a composite number n using a single–polynomial Quadratic Sieve.
    
    Parameters:
      n                 : Composite number to factor.
      factor_base_bound : The maximum prime (by value) to include in the factor base.
                          (A higher bound increases smoothness probability.)
      max_sieve_attempts: Maximum number of x values to try (starting near sqrt(n)).
      extra_relations   : Number of extra smooth relations (above factor base size) to gather.
    
    Returns:
      A nontrivial factor of n on success, or None if the QS did not produce a factor.
    """
    # --- Preliminary Checks ---
    r = isqrt(n)
    if r*r == n:
        return r  # n is a perfect square

    # Quick trial division using very small primes.
    small_primes = sieve_of_eratosthenes(2000)
    for p in small_primes:
        if n % p == 0 and p > 1:
            return p

    # --- Build Factor Base ---
    # If the bound is not given, one could use a heuristic. Here, we use the parameter.
    factor_base = generate_factor_base(n, factor_base_bound)
    base_size = len(factor_base)
    required_relations = base_size + extra_relations

    print(f"[+] n = {n}")
    print(f"[+] Using factor base bound = {factor_base_bound}, with factor base size = {base_size}")
    print(f"[+] Need at least {required_relations} smooth relations.")

    # --- Collect Smooth Relations ---
    relations = []  # Each relation: (x, factorization dict, exponent vector mod 2)
    start_x = r + 1
    x = start_x
    attempts = 0
    while len(relations) < required_relations and attempts < max_sieve_attempts:
        Qx = x*x - n
        factors = trial_division(Qx, factor_base)
        if factors is not None:
            vec = factorization_to_vector(factors, factor_base)
            relations.append((x, factors, vec))
        x += 1
        attempts += 1

    print(f"[+] Collected {len(relations)} smooth relations after {attempts} attempts.")
    if len(relations) < required_relations:
        print("[-] Not enough smooth relations were found. Increase factor_base_bound or max_sieve_attempts.")
        return None

    # --- Build the Exponent Matrix over GF(2) ---
    matrix = [rel[2] for rel in relations]  # Each row is a binary vector

    # --- Find a Dependency (Linear Combination that Sums to Zero mod 2) ---
    dep = find_dependency_GF2(matrix)
    if not dep:
        print("[-] No dependency found or the dependency was invalid.")
        return None

    # --- Combine the Selected Relations ---
    subset_indices = [i for i, bit in enumerate(dep) if bit == 1]
    if not subset_indices:
        print("[-] The dependency vector is trivial (all zeros). No factor found.")
        return None

    print(f"[+] Using {len(subset_indices)} relations from the dependency combination.")
    # Compute 'a' = product of x values (mod n) and combine factorizations.
    a = 1
    combined_factors = defaultdict(int)
    for i in subset_indices:
        x_i, fac_i, _ = relations[i]
        a = (a * x_i) % n
        for p, e in fac_i.items():
            combined_factors[p] += e

    # Form 'b' from the square root of the combined factorization
    b = 1
    for p, total_exp in combined_factors.items():
        if total_exp % 2 != 0:
            print("[!] Odd exponent encountered in combined factorization. Dependency invalid.")
            return None
        b = (b * pow(p, total_exp//2, n)) % n

    # --- Compute the Greatest Common Divisor ---
    candidate = gcd(abs(a - b), n)
    if candidate in (1, n):
        print("[-] Obtained only trivial factors. Dependency did not yield a nontrivial factor.")
        return None

    return candidate

###############################################################################
# DEMONSTRATION / DRIVER CODE
###############################################################################
if __name__ == "__main__":
    # The following candidates are the ones you mentioned.
    candidates = [
        16921456439215439701,
        46839566299936193234246726809,
        617283508641975203683304919691358469663,
        374483408052961599091981510330554205500926021947
    ]

    for num in candidates:
        print("="*80)
        print(f"Trying to factor: {num}")
        factor = quadratic_sieve(
            num,
            factor_base_bound=300000,    # Increased bound (tweak as needed)
            max_sieve_attempts=3000000,   # More attempts for smooth relations
            extra_relations=200           # More extra relations increases dependency chances
        )
        if factor:
            other_factor = num // factor
            print(f"[SUCCESS] Factor found: {factor}")
            print(f"          Other factor: {other_factor}")
            print(f"          Verification: {factor * other_factor == num}")
        else:
            print("[FAILURE] Could not factor with current parameters.\n")
