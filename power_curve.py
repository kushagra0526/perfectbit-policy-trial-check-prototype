"""What-if added after the held-out run: trials per policy needed for the paired exact test to detect a 10-point gain."""
from trialcheck import run

print("N per policy | power % (paired exact, 10-point gain) at pA = 0.5, beta 2.0, a fixed total warm-up of 0.7 logit over the session, 5% bad resets")
for N in (60, 120, 250, 400, 600, 900):
    r = run.one(dict(N=N, pA=0.5, beta=2.0, drift=-0.7 / (2 * N), bad_reset=0.05), run.DELTA, 7, R=2000)
    print(f"{N:12} | {100 * r['interleaved + same initial condition, exact'][0]:.0f}")
