import json

import numpy as np

from . import sim

# (N trials per policy, pA, pB, beta, drift per trial, bad_reset). dev uses mild heterogeneity; held-out is different (the robot improves over the session instead of degrading)
DEV = [dict(N=N, pA=0.6, beta=1.0, drift=0.004, bad_reset=0.0) for N in (30, 60, 120)]
HELD = [dict(N=N, pA=pa, beta=2.0, drift=-0.006, bad_reset=0.05) for N in (30, 60, 120) for pa in (0.3, 0.5, 0.85)]
DELTA = 0.10  # true improvement when B really is better


def one(cfg, delta, seed, R=4000):
    rng = np.random.default_rng(seed)
    N, pA = cfg["N"], cfg["pA"]
    pB = min(pA + delta, 0.97)
    blk = sim.draw(rng, R, N, pA, pB, cfg["beta"], cfg["drift"], cfg["bad_reset"], blocked=True)
    inter = sim.draw(rng, R, N, pA, pB, cfg["beta"], cfg["drift"], cfg["bad_reset"], blocked=False)
    return {k: (float(w.mean()), float(u.mean())) for k, (w, u) in sim.methods(*blk, *inter, N).items()}


def table(cfgs, seed0):
    rows = {}
    for i, cfg in enumerate(cfgs):
        null = one(cfg, 0.0, seed0 + i)       # B is exactly as good as A: a "win" is a false win
        real = one(cfg, DELTA, seed0 + 100 + i)
        for m in null:
            r = rows.setdefault(m, {"false_win": [], "power": [], "trials": []})
            r["false_win"].append(null[m][0])
            r["power"].append(real[m][0])
            r["trials"].append(real[m][1] / cfg["N"])
    return {m: {"false_win_pct_mean": float(round(100 * np.mean(v["false_win"]), 1)), "false_win_pct_worst": float(round(100 * max(v["false_win"]), 1)),
                "power_pct_mean": float(round(100 * np.mean(v["power"]), 1)), "power_pct_at_N120": [float(round(100 * p, 1)) for p, c in zip(v["power"], cfgs) if c["N"] == 120],
                "trials_used_share": float(round(np.mean(v["trials"]), 2))}
            for m, v in rows.items()}


def main():
    res = {"dev": table(DEV, 0), "held_out": table(HELD, 1000)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print(f"  {m:46} {d}")


if __name__ == "__main__":
    main()
