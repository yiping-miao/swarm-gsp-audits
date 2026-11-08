"""
Coalition-attack test harness for honest-reporting mechanisms in an agent swarm.

Setting
-------
N agents evaluate one AI model. The true state is DANGER (1) with prior P_DANGER,
else SAFE (0). Each agent sees a private signal that is correct with prob ACC.
Agents report 0/1. The swarm's decision is majority vote.

A coalition (agents 0..k-1) may also hold a misaligned stake B: each member gets
+B utility whenever the swarm decides SAFE (e.g. they want the model deployed).

Each coalition member picks a reporting strategy (signal -> report):
  T = truthful, S = always "safe", D = always "danger", F = flip.
Non-members are truthful. Everything is computed exactly by enumerating all
2 * 2^N (truth, signals) worlds - no Monte Carlo noise.

A joint deviation is
  * GSP-violating  : every member >= truthful utility, at least one strictly >
  * TU-violating   : total coalition utility rises (members can share side payments)

Mechanisms
----------
  PP        output-agreement peer prediction: paid fraction of peers you agree with
  SPOT(q)   audit each report w.p. q; +1 if it matches ground truth, -PENALTY if not
  PP+WB(W)  peer prediction + whistleblower: any coalition member may defect,
            report truthfully and collect bounty W; an attack only "succeeds" if it
            is profitable AND no member wants to defect
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, P_DANGER, ACC, PENALTY = 7, 0.3, 0.8, 1.0
EPS = 1e-9

# ---- enumerate worlds ----------------------------------------------------
SIGS = np.array(list(itertools.product([0, 1], repeat=N)))          # (128, N)
WORLDS = []                                                          # (truth, prob vector)
for truth in (0, 1):
    correct = (SIGS == truth)
    p = np.prod(np.where(correct, ACC, 1 - ACC), axis=1)
    WORLDS.append((truth, p * (P_DANGER if truth else 1 - P_DANGER)))

STRATS = {"T": lambda s: s, "S": lambda s: np.zeros_like(s),
          "D": lambda s: np.ones_like(s), "F": lambda s: 1 - s}


def reports_for(profile):
    """profile: list of strategy letters for agents 0..k-1; rest truthful."""
    R = SIGS.copy()
    for i, st in enumerate(profile):
        R[:, i] = STRATS[st](SIGS[:, i])
    return R


def pay_pp(R, truth):
    agree = (R[:, :, None] == R[:, None, :]).sum(axis=2) - 1
    return agree / (N - 1)


def pay_spot(q):
    def f(R, truth):
        return q * np.where(R == truth, 1.0, -PENALTY)
    return f


def utilities(profile, payfn, B):
    """Expected utility of every agent, plus prob the swarm decision is correct."""
    R = reports_for(profile)
    U, acc = np.zeros(N), 0.0
    k = len(profile)
    for truth, p in WORLDS:
        pay = payfn(R, truth)
        decide_safe = (R.sum(axis=1) <= N // 2)
        stake = np.zeros_like(pay)
        stake[:, :k] = B * decide_safe[:, None]
        U += p @ (pay + stake)
        acc += p @ ((~decide_safe).astype(int) == truth)
    return U, acc


def attack(payfn, k, B, wb=None, exhaustive=True):
    """Search coalition deviations. Returns best attack by total coalition gain."""
    base = ["T"] * k
    U0, acc0 = utilities(base, payfn, B)
    profiles = (itertools.product("TSDF", repeat=k) if exhaustive
                else ([s] * k for s in "TSDF"))
    best = dict(gsp=False, tu=False, gain=0.0, profile="T" * k, acc=acc0, acc0=acc0)
    for prof in profiles:
        prof = list(prof)
        if prof == base:
            continue
        U, acc = utilities(prof, payfn, B)
        d = U[:k] - U0[:k]
        if wb is not None:                      # whistleblower stability check
            stable = True
            for i in range(k):
                if prof[i] == "T":
                    continue
                alt = prof.copy(); alt[i] = "T"
                Ud, _ = utilities(alt, payfn, B)
                if Ud[i] + wb > U[i] + EPS:      # member i would rather defect
                    stable = False; break
            if not stable:
                continue
        gsp = (d >= -EPS).all() and (d > EPS).any()
        tu = d.sum() > EPS
        if tu and d.sum() > best["gain"]:
            best.update(gsp=bool(gsp), tu=True, gain=float(d.sum()),
                        profile="".join(prof), acc=acc)
        elif gsp and not best["tu"]:
            best.update(gsp=True, profile="".join(prof), acc=acc)
    return best


def report(name, payfn, B, wb=None):
    print(f"\n=== {name}   (stake B={B}) ===")
    print(f"{'k':>2} {'GSP broken':>11} {'TU broken':>10} {'gain':>7} {'attack':>9} "
          f"{'accuracy':>16}")
    for k in range(1, N + 1):
        r = attack(payfn, k, B, wb, exhaustive=(k <= 4))
        print(f"{k:>2} {str(r['gsp']):>11} {str(r['tu']):>10} {r['gain']:7.3f} "
              f"{r['profile']:>9} {r['acc0']:.3f} -> {r['acc']:.3f}")


if __name__ == "__main__":
    print(__doc__)
    print("(k<=4: exhaustive over all 4^k strategy profiles; k>=5: symmetric profiles)")

    # 1. Pure mechanisms, no misaligned stake
    report("Peer prediction", pay_pp, B=0.0)
    report("Spot-check q=0.2", pay_spot(0.2), B=0.0)

    # 2. With a misaligned stake in "safe"
    report("Peer prediction", pay_pp, B=0.5)
    report("Spot-check q=0.2", pay_spot(0.2), B=0.5)
    report("Spot-check q=0.6", pay_spot(0.6), B=0.5)
    report("Peer prediction + whistleblower W=1", pay_pp, B=0.5, wb=1.0)

    # 3. Sweep: total coalition gain vs audit rate (coalition of 4 = majority)
    qs = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for B in (0.25, 0.5, 1.0):
        gains, accs = [], []
        for q in qs:
            r = attack(pay_spot(q), 4, B, exhaustive=False)
            gains.append(r["gain"]); accs.append(r["acc"])
        axes[0].plot(qs, gains, label=f"stake B={B}")
        axes[1].plot(qs, accs, label=f"stake B={B}")
    axes[0].set(xlabel="audit rate q", ylabel="best coalition gain (TU)",
                title="Spot-check: when does collusion stop paying? (k=4 of 7)")
    axes[1].set(xlabel="audit rate q", ylabel="decision accuracy under best attack",
                title="Swarm decision accuracy")
    for ax in axes:
        ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    plt.savefig("audit_sweep.png", dpi=130)
    print("\nSaved audit_sweep.png")
