# trialcheck: how many real-robot trials does it take to say policy B beats policy A?

PerfectBit says it is "building policies for robots that do real work" and is hiring its first research scientists and research engineers (https://perfectbit.ai). Whoever builds robot policies ends up asking one question many times a week: after N real trials, is B actually better than A? Real trials are slow, initial conditions vary, the robot drifts over a session, and people peek at the running score. This is a simulator with known truth that shows which test designs give wrong answers.

## The setup
A trial has an initial condition (object pose, lighting...) with latent difficulty z. Success probability is sigmoid(logit(p_policy) + beta*z - drift*t) where t is the trial's position in the session (battery, calibration, warm-up). Every method claims "B wins" at one-sided alpha = 0.025. Designs:
- **A then B, unpaired z-test:** all A trials, then all B trials, different initial conditions.
- **Interleaved, unpaired z-test:** A B A B..., different initial conditions.
- **Interleaved with the same initial condition for both, exact paired test** on the trials where exactly one policy succeeded.
- **Paired, peek every 10 trials** at z >= 1.96 and stop at the first crossing.
- **Paired, 5 looks, O'Brien-Fleming-type boundary** (2.04 * sqrt(5/k)).
False wins are measured with B exactly equal to A; power with B 10 points better.

## Results (4,000 simulated experiments per cell; truth known)
Dev: pA = 0.6, beta 1, the robot degrades over the session (drift +0.004 logit per trial), N = 30/60/120 trials per policy. Held-out: pA = 0.3/0.5/0.85, beta 2, the robot improves over the session (drift -0.006), 5% of trials ruined by a bad reset, N = 30/60/120; configurations fixed in git before the first run.

| Held-out (mean over 9 settings) | false wins when B = A (nominal 2.5%) | power for a 10-point gain | trials used |
|---|---|---|---|
| A then B, unpaired | **14.4%** (worst setting 35.7%) | 39.2% | 100% |
| interleaved, unpaired | 0.8% | 13.6% | 100% |
| interleaved + same initial condition, exact | 1.4% | 15.6% | 100% |
| paired, peek every 10 trials | **7.2%** (worst 10.6%) | 31.2% | 85% |
| paired, 5 looks, O'Brien-Fleming | 2.4% | 20.6% | 97% |

- The 39.2% "power" of A-then-B is mostly the warm-up bias, not skill. Running A first and B second makes a false win likely when the robot warms up during the session (and, in dev where it degrades, makes B look worse: false wins 0.7%, power 5%). The sign of the drift decides the error, so a blocked comparison can be wrong in either direction.
- Peeking every 10 trials triples the false-win rate. Five planned looks with an O'Brien-Fleming boundary stay at nominal and keep most of the power.
- Power is low everywhere: with these settings 10 points is hard to see in 120 trials per policy (15% for the exact paired test at pA = 0.5).

## What-if added after the held-out run: how many trials is enough?
`python power_curve.py`, paired exact test, pA = 0.5, beta 2, a fixed total warm-up of 0.7 logit, 5% bad resets, a true 10-point gain (2,000 experiments each):

| trials per policy | 60 | 120 | 250 | 400 | 600 | 900 |
|---|---|---|---|---|---|---|
| power | 8% | 16% | 32% | 49% | 67% | 84% |

(My first version of this table used a drift that grows with the number of trials and saturated the success rate, which made power flat; I replaced it with a fixed total drift.)

## Read this before believing the table
Everything is a model of mine: success is logistic in a single latent difficulty, drift is linear in the logit, resets fail independently. Real policies fail in clusters (a bad object, a bad day), and pairing only helps when the same initial condition can really be reproduced, which is hard on hardware. Nothing here uses PerfectBit data, robots or code; PerfectBit publishes almost nothing about its work, so this is a generic tool, not a fit to their setup.

## Run it (3 commands)
```
pip install numpy scipy pytest
python -m pytest -q tests
python -m trialcheck.run
```
numpy + scipy. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with PerfectBit.
