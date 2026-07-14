#!/usr/bin/env python
"""Depth-robust complexity estimators. Dependency-free so it is unit-testable with plain python3.

THE PROBLEM THIS SOLVES
    Sequencing depth varies ~6x across the samples in this project (94k .. 551k payload reads). Any
    RICHNESS-style count -- distinct UMIs, distinct fragment vectors, shared species between two
    samples -- GROWS WITH DEPTH, because the harder you look the more distinct things you find. So a
    raw species count is a property of the RUN, not of the LIBRARY, and comparing it across samples of
    different depth is invalid. (Concretely: A549_R1 looks like it has 8.3x more vectors than A549_R5,
    but rarefied to R5's depth the honest ratio is 2.4x.)

    Rule of thumb: if a number gets bigger when you sequence more, it is not a property of your library.

TWO DEPTH-ROBUST ALTERNATIVES

  c_eff(counts)      Effective complexity = 1 / P(two random distinct reads came from the same species)
                     = N(N-1) / sum_i n_i(n_i - 1)   [the inverse-Simpson / Hill-2 number]
                     Estimates a property of the POOL (its collision probability), so it does NOT
                     inflate with depth the way a species count does.

  rarefy(counts, m)  Expected number of distinct species if only m reads had been sequenced:
                     E[distinct] ~= sum_i [ 1 - (1 - m/N)^n_i ]
                     Analytic (no random subsampling) and deterministic, but an APPROXIMATION -- it is the
                     binomial / with-replacement form, not the exact hypergeometric one. See rarefy().
                     Use it to put every sample on a COMMON depth before comparing richness.

CAVEAT on c_eff -- IT IS NOT A HEADCOUNT. This is a theorem, not a rule of thumb:

    By Cauchy-Schwarz on (p_1..p_R) and (1..1):   1 = (sum_i p_i)^2 <= R * sum_i p_i^2 = R * P

        =>   c_eff = 1/P  <=  R          with equality IFF every species is equally abundant.

    So c_eff is a LOWER BOUND on the true number of species, always, and meets it only for a uniform pool.
    A skewed pool has c_eff far below its richness, because abundant species dominate the collision
    probability. That is exactly the quantity governing duplication -- but a FALL in c_eff means the pool
    CONCENTRATED, which can happen either because species were lost (R fell) or because the survivors
    became uneven (R held, skew grew). c_eff alone CANNOT distinguish the two. To tell them apart, read it
    alongside dup_pairs, the maximum n_i, and the share of sum_i n_i(n_i-1) carried by the top few species:
    lost species do not create high-count UMIs; PCR skew and clonal expansion do.
"""
from __future__ import annotations


def collision_prob(counts):
    """THE PRIMITIVE QUANTITY: sum_i p_i^2 -- the probability that two randomly chosen distinct reads
    came from the same species.

        estimator:  sum_i n_i(n_i - 1) / (N(N - 1))

    This is UNBIASED AT ANY DEPTH (by linearity of expectation over the N(N-1) ordered read-pairs), which
    is the whole reason the method is depth-robust. Prefer this over c_eff when you care about the
    statistics: c_eff is its RECIPROCAL, and 1/x is convex, so by Jensen's inequality the reciprocal of an
    unbiased estimate is NOT unbiased -- c_eff is biased UPWARD, worst when duplicates are scarce.

    Returns 0.0 when no duplicates were observed (then only a bound on c_eff is available).
    """
    n = list(counts)
    N = sum(n)
    if N < 2:
        return 0.0
    return sum(x * (x - 1) for x in n) / (N * (N - 1))


def c_eff(counts):
    """Effective complexity = 1 / collision_prob. A RECIPROCAL, for readability.

    Returns (N, n_distinct, dup_pairs, c_eff, is_lower_bound).

    CAVEAT: c_eff is *consistent* but NOT unbiased -- see collision_prob() above. It overshoots when the
    duplicate-pair count is small, so always read it alongside `dup_pairs`.

    EDGE CASE -- no duplicates at all (every species seen exactly once), which is exactly what an
    un-bottlenecked empty/PCR-control round looks like: the collision probability estimate is 0, so c_eff
    = 1/0 is unbounded. This is not a special case needing a hack -- it is the reciprocal blowing up. The
    only honest answer is a LOWER BOUND.

    WHY THE BOUND IS NOT THE "RULE OF THREE". The rule of three (0 events in n INDEPENDENT Bernoulli
    trials => p < 3/n at 95%) does not apply here: the M = N(N-1)/2 read-pairs are NOT independent -- if
    a~b and b~c then a~c necessarily. Linearity of expectation survives that dependence (which is why
    collision_prob is unbiased); a binomial tail bound does NOT.

    The bound below is instead justified by a POISSON approximation (Chen-Stein): collisions are rare and
    only weakly dependent, so their count is ~ Poisson(lambda) with lambda = M * P. Observing zero has
    probability exp(-lambda), which stays consistent at the 5% level while exp(-lambda) >= 0.05, i.e.
    lambda <= 3 (since -ln 0.05 = 2.996). Hence P <= 3/M and c_eff >= M/3 = N(N-1)/6.

    Same constant as the rule of three, sound justification. `is_lower_bound` is True in that case --
    report it as "> x", never as "= x".
    """
    n = list(counts)
    N = sum(n)
    D = len(n)
    if N < 2:
        return N, D, 0, float("inf"), True
    S = sum(x * (x - 1) for x in n)          # ordered pairs of distinct reads sharing a species
    if S == 0:
        return N, D, 0, N * (N - 1) / 6.0, True     # rule of three (95% lower bound)
    return N, D, S // 2, N * (N - 1) / S, False


def rarefy(counts, m):
    """Expected number of distinct species when subsampling to m reads (analytic rarefaction).

    THIS IS AN APPROXIMATION, and it is worth being precise about which one.

        exact (sampling WITHOUT replacement):  E[distinct] = sum_i [ 1 - C(N - n_i, m) / C(N, m) ]
        used here (binomial / WITH replacement): E[distinct] = sum_i [ 1 - (1 - m/N)^n_i ]

    The binomial form treats each read as retained independently with probability m/N. It agrees closely
    with the exact hypergeometric form when N is large and n_i << N, which is the regime here -- but they
    are not the same formula, and this is the approximate one.

    Rarefy every sample to the SHALLOWEST sample's depth before comparing richness across samples.
    Returns float('nan') if m > N (you cannot rarefy up).

    BLIND SPOT: if the pool is much larger than the depth, nearly every read lands on an unseen species and
    E[distinct] -> m for ANY sufficiently large pool. Rarefied richness then SATURATES at the read count and
    cannot distinguish a 1e6 pool from a 1e8 one. In that regime "richness barely changed" says nothing
    about the pool -- it says this instrument is blind. Use collision_prob(), which stays informative.
    """
    n = list(counts)
    N = sum(n)
    if N == 0:
        return 0.0
    if m > N:
        return float("nan")
    if m == N:
        return float(len(n))
    f = 1.0 - (m / N)
    return sum(1.0 - f ** x for x in n)


def summarize(counts, rarefy_to=None):
    """Convenience: the depth-robust complexity fields for one sample's count vector."""
    N, D, pairs, C, lower = c_eff(counts)
    out = {
        "n_reads": N,
        "n_distinct": D,                       # DEPTH-DEPENDENT -- do not compare across samples raw
        "dup_pairs": pairs,                    # read c_eff alongside this: few pairs => c_eff unreliable
        "collision_prob": collision_prob(counts),   # sum p_i^2 -- THE unbiased, depth-robust quantity
        "c_eff": None if C == float("inf") else round(C, 1),   # its reciprocal (biased up when pairs few)
        "c_eff_is_lower_bound": lower,
        # SIGNIFICANT FIGURES, not fixed decimals. round(x, 4) fabricates a ZERO for exactly the sample
        # this field matters most on: an un-bottlenecked empty/PCR-only round has no duplicates, so its
        # c_eff is enormous, so N/c_eff is ~4e-5 -- and round(4.071e-05, 4) is 0.0. That zero then
        # travels into the TSV and out into the report as "0.000", which reads as "you sampled none of
        # the pool" instead of "you sampled a ten-thousandth of it". A destroyed value cannot be
        # recovered downstream, so it must not be destroyed here.
        # (When c_eff is only a LOWER bound, this ratio is correspondingly an UPPER bound -- read it
        # next to c_eff_is_lower_bound, which the reports render as a `>` on c_eff itself.)
        "n_over_c_eff": None if C in (0, float("inf")) else float(f"{N / C:.4g}"),
    }
    if rarefy_to is not None:
        out["rarefied_to"] = rarefy_to
        r = rarefy(counts, rarefy_to)
        out["n_distinct_rarefied"] = None if r != r else round(r, 1)   # NaN-safe
    return out


if __name__ == "__main__":  # tiny self-test: python3 complexity_lib.py
    # all singletons -> no duplicates -> lower bound only
    N, D, p, C, lo = c_eff([1] * 1000)
    assert p == 0 and lo and abs(C - 1000 * 999 / 6) < 1e-6, "rule-of-three branch broken"
    # one species seen twice among 1000 reads
    N, D, p, C, lo = c_eff([2] + [1] * 998)
    assert p == 1 and not lo and abs(C - (1000 * 999) / 2) < 1e-6
    # rarefying to full depth returns the observed richness
    assert abs(rarefy([5, 3, 2], 10) - 3.0) < 1e-9
    # rarefying to a shallower depth loses species
    assert rarefy([1] * 100, 50) < 100
    print("complexity_lib self-test OK")
