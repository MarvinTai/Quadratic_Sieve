import math
import sys
from collections import defaultdict
from functools import reduce
from operator import mul
from math import gcd

##############################################################
# Helper Function: Sieve of Eratosthenes
#
# Generates all prime numbers up to a specified limit.
# This is used to build our “factor base” – the set of primes
# we will use to test smoothness of the numbers in our sequence.
##############################################################
def sieve_of_eratosthenes(limit):
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]
    for num in range(2, int(limit ** 0.5) + 1):
        if sieve[num]:
            for multiple in range(num*num, limit+1, num):
                sieve[multiple] = False
    return [p for p, is_prime in enumerate(sieve) if is_prime]


##############################################################
# Helper Function: Legendre Symbol
#
# Computes the Legendre symbol (a/p) which indicates whether
# a is a quadratic residue modulo the prime p.
# (a/p) is 1 if a is a nonzero quadratic residue modulo p,
# -1 if it is a non-residue, and 0 if p divides a.
#
# This is used to decide whether a prime p should be added to
# the factor base based on whether n is a quadratic residue modulo p.
##############################################################
def legendre_symbol(a, p):
    # Using Euler’s criterion: a^((p-1)//2) mod p is 1 if residue, p-1 if non-residue.
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls  # return -1 if ls == p-1 else ls (which is 0 or 1)


##############################################################
# Helper Function: Modular Square Root (Tonelli-Shanks)
#
# Finds a solution to x^2 ≡ a (mod p) where p is an odd prime and
# a is known to be a quadratic residue modulo p.
#
# This function is necessary to compute the roots of the quadratic
# congruence x^2 - n ≡ 0 (mod p), which in turn tells us the positions
# in our sieving interval that might be divisible by p.
##############################################################
def mod_sqrt(a, p):
    # Simple cases
    if legendre_symbol(a, p) != 1:
        return None
    if a == 0:
        return 0
    if p == 2:
        return a  # only possible value here is a mod 2

    # Write p-1 as Q*2^S with Q odd.
    S = 0
    Q = p - 1
    while Q % 2 == 0:
        Q //= 2
        S += 1

    # Find a quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1

    # Set initial values
    M = S
    c = pow(z, Q, p)
    t = pow(a, Q, p)
    R = pow(a, (Q + 1) // 2, p)

    # Loop until t becomes 1 (i.e. R is the square root)
    while t != 1:
        # Find the smallest i (0 < i < M) such that t^(2^i)=1
        i = 1
        temp = pow(t, 2, p)
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
            if i == M:
                return None

        # Update values
        b = pow(c, 2 ** (M - i - 1), p)
        M = i
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R * b) % p

    return R


##############################################################
# Function: Generate Factor Base
#
# Builds the factor base for the quadratic sieve.
# We include small primes up to the bound B for which n is a quadratic
# residue (i.e. Legendre symbol (n/p) == 1). We also include -1 so that
# we can handle negative Q(x) values.
##############################################################
def generate_factor_base(n, B):
    primes = sieve_of_eratosthenes(B)
    factor_base = []
    for p in primes:
        # We include p if n is a quadratic residue mod p.
        # Also, ignore p=2 here or handle it separately.
        if p == 2:
            # Always include 2 in the factor base.
            factor_base.append(p)
        elif legendre_symbol(n, p) == 1:
            factor_base.append(p)
    # We also add -1 to the factor base to factor the sign.
    factor_base.insert(0, -1)
    return factor_base


##############################################################
# Function: Trial Division for Smoothness
#
# Given a number Q (which is x^2 - n), attempt to fully factor it
# over the factor base.
#
# Returns a dictionary with keys being primes and values being their
# exponents in the factorization if the number is B-smooth (up to sign).
# Returns None if the remaining unfactored part (after dividing by the
# primes in the factor base) is not ±1.
##############################################################
def trial_division(Q, factor_base):
    original_Q = Q  # Keep a copy for debugging/annotation
    factors = defaultdict(int)
    # Handle negative numbers: factor out the sign using -1 in the factor base.
    if Q < 0:
        factors[-1] = 1
        Q = -Q
    
    # Try dividing by each prime in the factor base (ignore -1 as it is already handled)
    for p in factor_base:
        if p == -1:
            continue
        while Q % p == 0:
            factors[p] += 1
            Q //= p
    # If after division the remainder is 1, then the number is B-smooth.
    if Q == 1:
        return dict(factors)
    else:
        return None


##############################################################
# Function: Compute Exponent Vector Mod 2
#
# From a complete factorization (dictionary: prime -> exponent), create
# the exponent vector over GF(2) corresponding to the factor base ordering.
#
# This vector (with entries 0 or 1) is used later in the linear algebra step
# to find a dependency that reveals a square.
##############################################################
def factorization_to_vector(factorization, factor_base):
    vector = []
    for p in factor_base:
        exp = factorization.get(p, 0)
        # Taking exponent mod 2.
        vector.append(exp % 2)
    return vector


##############################################################
# Function: Gaussian Elimination over GF(2)
#
# Given a binary matrix (list of lists of 0's and 1's) representing the
# exponent vectors from our smooth numbers, perform Gaussian elimination
# to find a nontrivial linear dependency.
#
# This implementation uses a simple elimination process and keeps track of
# the operations on the rows so that we can later reconstruct a dependency.
#
# Returns a list 'dep' of indices (a bitmask) for each row that sum to the
# zero vector mod 2. If no dependency is found (which is unlikely when
# there are more rows than columns), returns None.
##############################################################
def find_dependency(matrix):
    # Number of rows and columns.
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    
    # Create a copy of the matrix for elimination.
    A = [row[:] for row in matrix]
    # Create an identity matrix that will track the row operations.
    # Each row in 'ops' corresponds to the combination (in GF(2)) of original rows that yield that row.
    ops = [[1 if i == j else 0 for j in range(n_rows)] for i in range(n_rows)]
    
    # This list will mark which column pivot is used.
    pivot_rows = [-1] * n_cols
    
    row = 0
    for col in range(n_cols):
        # Find pivot: a row from 'row' onward with a 1 in column 'col'.
        pivot = None
        for r in range(row, n_rows):
            if A[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            # No pivot in this column; dependency exists.
            # Construct a dependency vector with 1 in the position of a row that was not used as pivot.
            dep = [0] * n_rows
            dep[col] = 1  # Mark the free variable in this column.
            return dep  # Simple return; in a full implementation, one would continue to find a full nontrivial vector.
        # Swap current row with pivot row if necessary.
        if pivot != row:
            A[row], A[pivot] = A[pivot], A[row]
            ops[row], ops[pivot] = ops[pivot], ops[row]
        pivot_rows[col] = row
        # Eliminate ones in this column for all other rows.
        for r in range(n_rows):
            if r != row and A[r][col] == 1:
                # Add (XOR) the pivot row to this row.
                A[r] = [(a ^ b) for a, b in zip(A[r], A[row])]
                ops[r] = [(a ^ b) for a, b in zip(ops[r], ops[row])]
        row += 1
        if row == n_rows:
            break

    # At this point, if there are more rows than pivots, the extra rows represent dependencies.
    for r in range(n_rows):
        if all(x == 0 for x in A[r]):
            # The r-th row of A is zero, so the corresponding combination (ops[r]) is a dependency.
            return ops[r]
    # In case no dependency is found (very unlikely), return None.
    return None


##############################################################
# Main Function: Quadratic Sieve Factorization
#
# Given a composite number n, the quadratic sieve attempts to find a
# nontrivial factor of n.
#
# The process is:
#  1. Remove small factors by trial division.
#  2. Choose a factor base (primes up to a bound B for which n is a quadratic residue).
#  3. For x starting near ceil(sqrt(n)), compute Q(x) = x^2 - n.
#     Use a sieving/trial division method to detect when Q(x) is B-smooth.
#  4. Store the smooth relations and their exponent vectors mod 2.
#  5. When enough relations are gathered, set up a matrix over GF(2) and
#     find a linear dependency.
#  6. Combine the corresponding x and factorization data to produce a
#     congruence of squares modulo n.
#  7. Compute gcd(|a - b|, n) to obtain a nontrivial factor.
#
# This implementation follows the “proto-algorithm” described in the paper :contentReference[oaicite:1]{index=1}.
##############################################################
def quadratic_sieve(n, B_bound=100, max_relations_multiplier=2):
    # Step 0. Basic checks: if n is even or a perfect square.
    if n % 2 == 0:
        return 2
    # Check if n is a perfect square.
    root_n = math.isqrt(n)
    if root_n * root_n == n:
        return root_n

    # Step 1. Remove small prime factors by trial division.
    for p in sieve_of_eratosthenes(50):
        if n % p == 0:
            return p

    # Step 2. Select parameter B for the factor base.
    # In practice, B is chosen according to heuristics related to n.
    B = B_bound  # For demonstration, we use the given bound.
    factor_base = generate_factor_base(n, B)
    print(f"Factor base (primes for which n is quadratic residue) = {factor_base}")
    
    # We need more relations than the size of the factor base.
    required_relations = len(factor_base) + 5  # A few extra to ensure a dependency.
    
    # Step 3. Sieving: search for x such that Q(x) = x^2 - n factors entirely over the factor base.
    relations = []   # Each element will be a tuple (x, factorization, exponent_vector)
    x_start = math.isqrt(n) + 1
    x = x_start
    while len(relations) < required_relations:
        Qx = x*x - n
        # Attempt to factor Q(x) over the factor base.
        factorization = trial_division(Qx, factor_base)
        if factorization is not None:
            # Successfully expressed Q(x) as a product of factor base primes.
            exp_vector = factorization_to_vector(factorization, factor_base)
            relations.append((x, factorization, exp_vector))
            print(f"Found smooth relation: x = {x}, Q(x) = {Qx}, factors: {factorization}, exp_vector: {exp_vector}")
        x += 1
        # A safe exit in case of too many iterations (for demonstration only).
        if x - x_start > 10000:
            print("Exceeded search range for smooth numbers. Increase B_bound or search range.")
            break

    if len(relations) < required_relations:
        print("Not enough smooth relations found. Try increasing B_bound.")
        return None

    # Step 4. Build the binary matrix (rows: smooth relations, columns: factor base).
    matrix = [exp_vector for (_, _, exp_vector) in relations]

    # Step 5. Perform Gaussian elimination over GF(2) to find a dependency.
    dependency = find_dependency(matrix)
    if dependency is None:
        print("No dependency found among relations. Try collecting more smooth relations.")
        return None

    # 'dependency' is a binary vector indicating which relations to use.
    # Extract the indices of the relations used (those positions where dependency is 1).
    indices = [i for i, bit in enumerate(dependency) if bit == 1]
    if not indices:
        print("No nontrivial dependency found.")
        return None
    print(f"Using dependency from rows: {indices}")

    # Step 6. Combine the chosen relations.
    # Compute the product of x values (this will be 'a') and the combined factorization.
    a = 1
    combined_factors = defaultdict(int)
    for i in indices:
        xi, factorization, _ = relations[i]
        a *= xi
        for p, exp in factorization.items():
            combined_factors[p] += exp

    # Now, since the exponents add up, they should all be even (in a true dependency).
    # Compute b as the square root of the product corresponding to the combined factorization.
    b = 1
    for p, exp in combined_factors.items():
        if exp % 2 != 0:
            # This should not happen in an ideal dependency.
            print("Error: combined exponents are not all even.")
            return None
        b *= pow(p, exp // 2)
    
    # Reduce a and b modulo n.
    a = a % n
    b = b % n
    print(f"Combined product: a = {a} and b = {b} satisfy a^2 ≡ b^2 (mod n)")

    # Step 7. The factor is found by computing gcd(|a - b|, n).
    factor = gcd(a - b, n)
    if factor == 1 or factor == n:
        print("Trivial factor found; dependency did not yield a nontrivial factor.")
        return None
    else:
        print(f"Nontrivial factor found: {factor}")
        return factor


##############################################################
# Main Execution: Test the Quadratic Sieve
#
# For demonstration, we factor a modest composite number.
# You can replace 'n' with any composite number you wish to factor.
##############################################################
if __name__ == "__main__":
    # Example composite: 1649 from the paper. Its factors are 17 and 97.
    n = 16921456439215439701
    print(f"Attempting to factor n = {n} using the Quadratic Sieve")
    factor = quadratic_sieve(n, B_bound=30)
    if factor is not None:
        print(f"Factorization result: {n} = {factor} * {n // factor}")
    else:
        print("Failed to factor n using the current parameters.")
