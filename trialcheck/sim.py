"""How many real-robot trials does it take to say policy B beats policy A, and which test design lies?

World with known truth: each trial has an initial condition (object pose, lighting...) with a latent difficulty z;
success prob = sigmoid(logit(p_policy) + beta*z - drift*t), where t is the trial's position in the session
(batteries, calibration, lighting drift). Every method claims "B wins" at one-sided alpha = 0.025.
"""
import numpy as np
from scipy.stats import binom, norm

Z_CRIT = norm.isf(0.025)  # 1.96


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw(rng, R, N, pA, pB, beta, drift, bad_reset=0.0, blocked=False):
    """Returns success arrays for A and B plus the pairing structure. Shapes (R, N)."""
    zA = rng.standard_normal((R, N))
    zB = zA if not blocked else rng.standard_normal((R, N))  # interleaved paired design reuses the initial condition
    i = np.arange(N)
    if blocked:  # all A trials first, then all B
        tA, tB = i, N + i
    else:        # A B A B ...
        tA, tB = 2 * i, 2 * i + 1
    sA = rng.random((R, N)) < sigmoid(logit(pA) + beta * zA - drift * tA)
    sB = rng.random((R, N)) < sigmoid(logit(pB) + beta * zB - drift * tB)
    if bad_reset:  # a failed reset ruins a trial regardless of policy
        sA &= rng.random((R, N)) >= bad_reset
        sB &= rng.random((R, N)) >= bad_reset
    return sA, sB


def z_unpaired(sA, sB):
    n = sA.shape[1]
    a, b = sA.mean(1), sB.mean(1)
    p = (a + b) / 2
    se = np.sqrt(np.maximum(p * (1 - p) * 2 / n, 1e-12))
    return (b - a) / se


def z_paired(sA, sB):
    """McNemar-style z on discordant pairs: (c - b) / sqrt(b + c)."""
    b = (sA & ~sB).sum(1)
    c = (~sA & sB).sum(1)
    return (c - b) / np.sqrt(np.maximum(b + c, 1))


def exact_paired_p(sA, sB):
    b = (sA & ~sB).sum(1)
    c = (~sA & sB).sum(1)
    return binom.sf(c - 1, b + c, 0.5)  # P(X >= c), one-sided


def methods(sA_blk, sB_blk, sA_int, sB_int, N):
    """Each method returns (claimed_win bool array, trials_per_policy used array)."""
    R = sA_blk.shape[0]
    out = {}
    out["A-then-B, unpaired z"] = (z_unpaired(sA_blk, sB_blk) >= Z_CRIT, np.full(R, N))
    out["interleaved, unpaired z"] = (z_unpaired(*(x for x in _shuffle(sA_int, sB_int))) >= Z_CRIT, np.full(R, N))
    out["interleaved + same initial condition, exact"] = (exact_paired_p(sA_int, sB_int) <= 0.025, np.full(R, N))
    # peeking every 10 trials at the same 1.96 threshold, stop at the first crossing
    looks = list(range(10, N + 1, 10)) or [N]
    out["paired, peek every 10 trials (z>=1.96)"] = _seq(sA_int, sB_int, looks, lambda k: Z_CRIT)
    # five equally spaced looks with O'Brien-Fleming-type boundary
    idx = [max(1, round(N * k / 5)) for k in range(1, 6)]
    out["paired, 5 looks, O'Brien-Fleming"] = _seq(sA_int, sB_int, idx, lambda k: 2.04 * np.sqrt(5 / k), k_of=idx)
    return out


def _shuffle(sA, sB):
    """Unpaired interleaved design: the two policies do not share initial conditions."""
    return sA, np.roll(sB, 1, axis=1)  # decouple B's trial i from A's initial condition i without redrawing


def _seq(sA, sB, looks, bound, k_of=None):
    R = sA.shape[0]
    won = np.zeros(R, bool)
    used = np.full(R, looks[-1])
    for j, n in enumerate(looks):
        z = z_paired(sA[:, :n], sB[:, :n])
        crit = bound(j + 1)
        hit = (z >= crit) & ~won
        used = np.where(hit, n, used)
        won |= hit
    return won, used
