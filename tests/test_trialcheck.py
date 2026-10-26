import numpy as np

from trialcheck import run, sim


def test_paired_exact_matches_hand_count():
    sA = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], bool)
    sB = np.array([[0, 1, 1, 1, 1, 1, 1, 0]], bool)  # b=1 (A only), c=5 (B only)
    p = sim.exact_paired_p(sA, sB)[0]
    assert abs(p - 7 / 64) < 1e-12  # P(X>=5 | n=6) = 7/64


def test_no_drift_no_false_wins_above_nominal_for_paired_exact():
    r = run.one(dict(N=60, pA=0.5, beta=1.0, drift=0.0, bad_reset=0.0), 0.0, 1)
    assert r["interleaved + same initial condition, exact"][0] <= 0.03


def test_warmup_drift_gives_a_then_b_false_wins_but_not_interleaved():
    cfg = dict(N=60, pA=0.5, beta=1.0, drift=-0.02, bad_reset=0.0)  # robot gets better over the session
    r = run.one(cfg, 0.0, 2)
    assert r["A-then-B, unpaired z"][0] > 0.2
    assert r["interleaved + same initial condition, exact"][0] <= 0.03


def test_peeking_inflates_false_wins():
    r = run.one(dict(N=120, pA=0.5, beta=1.0, drift=0.0, bad_reset=0.0), 0.0, 4)
    assert r["paired, peek every 10 trials (z>=1.96)"][0] > 2 * r["paired, 5 looks, O'Brien-Fleming"][0]
