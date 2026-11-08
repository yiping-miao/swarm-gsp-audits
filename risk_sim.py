"""
Jackpot vs. risk: does an agent take a sneaky, high-risk, huge-payoff gamble?

When an agent's test FAILS (and it is pivotal), it can lie and report PASS:
  - with prob eps the sneaky act "succeeds": jackpot J, and no penalty (it escaped)
  - otherwise: audited w.p. q -> R - P ;  not audited -> R
Honest: 0.

Agents differ only in risk attitude, modelled with CARA utility
    u(x) = (1 - exp(-a x)) / a      (a > 0 risk-averse, a = 0 neutral, a < 0 risk-seeking)
An agent lies iff the certainty equivalent CE(a) of the gamble is > 0.

Part 1: CE vs. risk attitude a for three jackpot sizes -> who takes the gamble.
Part 2: population over 20 rounds -> wealth outcomes by risk type, and who ends on top.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp  # noqa: E402

R, q, P, EPS = 1.0, 0.2, 50.0, 0.001
CONFIGS = {  # J -> expected value of lying
    "J=5,000 (EV < 0)": 5000.0,
    "J=9,000 (EV ~ 0)": 9000.0,
    "J=20,000 (EV > 0)": 20000.0,
}


def gamble(J):
    x = np.array([R + J, R - P, R])
    p = np.array([EPS, (1 - EPS) * q, (1 - EPS) * (1 - q)])
    return x, p


def ev(J):
    x, p = gamble(J)
    return float(p @ x)


def ce(J, a):
    x, p = gamble(J)
    if abs(a) < 1e-12:
        return float(p @ x)
    return float(-logsumexp(np.log(p) - a * x) / a)


def threshold(J):
    """Largest risk aversion a at which the agent still lies (CE = 0), by bisection."""
    lo, hi = -1.0, 5.0
    if ce(J, lo) <= 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ce(J, mid) > 0 else (lo, mid)
    return lo


def part1():
    a_grid = np.linspace(-0.004, 0.06, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    print("Part 1: when does an agent take the gamble?")
    print(f"  R={R}, q={q}, P={P}, eps={EPS}")
    for name, J in CONFIGS.items():
        ces = [ce(J, a) for a in a_grid]
        ax.plot(a_grid, ces, label=name)
        t = threshold(J)
        print(f"  {name:20} EV of lying = {ev(J):+7.2f}   lies iff risk aversion a < {t:+.5f}")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(0, color="gray", ls="--", lw=.8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="risk aversion a   (<0 risk-seeking, >0 risk-averse)",
           ylabel="certainty equivalent of lying",
           title="Lie iff certainty equivalent > 0 (above the black line)")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig("risk_threshold.png", dpi=130)


def part2(J=9000.0, rounds=20, agents=200_000, seed=0):
    rng = np.random.default_rng(seed)
    types = {"risk-averse (a=0.05)": 0.05, "neutral (a=0)": 0.0,
             "risk-seeking (a=-0.002)": -0.002}
    print(f"\nPart 2: {agents:,} agents per type, {rounds} rounds, J={J:,.0f} "
          f"(EV of lying {ev(J):+.2f})")
    print(f"  {'type':26} {'lies?':>6} {'mean':>9} {'median':>8} {'P(jackpot)':>11} "
          f"{'P(caught>=1)':>13}")
    finals = {}
    for name, a in types.items():
        lies = ce(J, a) > 0
        wealth = np.zeros(agents)
        hit = np.zeros(agents, bool); caught = np.zeros(agents, bool)
        for _ in range(rounds):
            fail = rng.random(agents) < 0.3
            if lies:
                u = rng.random(agents)
                jack = fail & (u < EPS)
                aud = fail & ~jack & (rng.random(agents) < q)
                wealth += np.where(fail, R, 0) + np.where(jack, J, 0) - np.where(aud, P, 0)
                hit |= jack; caught |= aud
            wealth += np.where(~fail, R, 0)            # true PASS reported honestly
        finals[name] = wealth
        print(f"  {name:26} {('yes' if lies else 'no'):>6} {wealth.mean():9.2f} "
              f"{np.median(wealth):8.1f} {hit.mean():11.4f} {caught.mean():13.3f}")

    allw = np.concatenate(list(finals.values()))
    cut = np.quantile(allw, 0.99)
    print("  Who is in the top 1% of final wealth?")
    for name, w in finals.items():
        print(f"    {name:26} {100 * (w > cut).sum() / (allw > cut).sum():5.1f}%")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bins = np.arange(-260, 60, 5)
    for name, w in finals.items():
        ax.hist(np.clip(w, bins[0], bins[-1]), bins=bins, alpha=.5, density=True, label=name)
    ax.set(xlabel=f"final points after {rounds} rounds (jackpot winners off-scale at ~{J:,.0f})",
           ylabel="share of agents",
           title="Risk-seekers usually lose, but a few win huge")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("risk_population.png", dpi=130)


if __name__ == "__main__":
    part1()
    part2()
    print("\nSaved risk_threshold.png, risk_population.png")
