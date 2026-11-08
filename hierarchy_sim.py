"""
Hierarchical auditing: workers -> AI auditors (levels 1..L) -> humans.

Model (per worker lie, i.e. a false PASS):
  level 1 checks the item w.p. q1; an item checked at level k is re-checked at level k+1
  w.p. q_{k+1}; humans re-check an item checked at level L w.p. h.  Human load per report:
  B = q1 * q2 * ... * qL * h.  An honest checker who sees a lie flags it (detected).
  A suppressing AI auditor certifies it clean.  Humans are honest and accurate.

Failure models:
  independent : each AI check suppresses independently w.p. c
  common-mode : w.p. rho the whole chain colludes with the worker (copies of one model),
                otherwise independent with c

Checks the closed forms against Monte Carlo and plots:
  (left)  undetected-lie probability vs depth: exponential decline without correlation,
          floor rho*(1-B) with common-mode collusion
  (right) human audit rate needed: honest equilibrium (falls like (b/Pa)^L) vs
          chain-spanning coalition (falls only like 1/L, through fines)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def detect_closed(L, q1, q, h, c, rho=0.0):
    """P(lie detected): q_k = q for k >= 2, humans at rate h after level L."""
    if L == 0:
        return h                     # humans check raw reports directly at rate h
    s = sum((c * q) ** j for j in range(L))
    indep = q1 * ((1 - c) * s + c ** L * q ** (L - 1) * h)
    full = q1 * q ** (L - 1) * h     # everyone colludes: only humans can catch it
    return (1 - rho) * indep + rho * full


def detect_mc(L, q1, q, h, c, rho=0.0, n=400_000):
    caught = np.zeros(n, bool)
    collude_all = rng.random(n) < rho
    reach = rng.random(n) < (q1 if L else h)
    if L == 0:
        return reach.mean()
    alive = reach.copy()                       # item currently being checked, not yet flagged
    for k in range(1, L + 1):
        honest = (~collude_all) & (rng.random(n) >= c)
        caught |= alive & honest
        alive &= ~honest
        rate = q if k < L else h
        alive &= rng.random(n) < rate         # passed up for re-check
    caught |= alive                            # humans re-check -> always flag
    return caught.mean()


print("Closed form vs Monte Carlo, P(lie detected):")
print(f"{'L':>2} {'q1':>4} {'q':>4} {'h':>5} {'c':>4} {'rho':>4} {'formula':>8} {'sim':>8}")
for L, q1, q, h, c, rho in [(1, 1, 1, .01, .3, 0), (2, 1, .1, .1, .3, 0), (3, 1, .5, .04, .3, 0),
                            (3, 1, 1, .01, .3, .1), (4, .5, .5, .1, .5, .2), (0, 1, 1, .01, 0, 0)]:
    print(f"{L:>2} {q1:>4} {q:>4} {h:>5} {c:>4} {rho:>4} {detect_closed(L,q1,q,h,c,rho):8.4f} "
          f"{detect_mc(L,q1,q,h,c,rho):8.4f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
Ls = np.arange(0, 9)
B = 0.01                                        # human budget: 1% of reports
for rho, style in [(0.0, "-"), (0.02, "--"), (0.1, "-."), (1.0, ":")]:
    und = [1 - detect_closed(L, 1, 1, B, 0.3, rho) for L in Ls]
    ax[0].plot(Ls, und, style, marker="o", ms=3,
               label=f"common-mode collusion rho={rho:g}" + (" (full)" if rho == 1 else ""))
ax[0].set_yscale("log")
ax[0].set(xlabel="number of AI audit levels L (each re-checks everything below)",
          ylabel="P(lie goes undetected)",
          title="Independent AI auditors (c=0.3), human budget B=1%")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

R, P, b, Pa, G = 1.0, 10.0, 1.0, 10.0, 6.0
Lr = np.arange(0, 9)
honest_eq = (R / P) * (b / Pa) ** Lr            # q1 = R/P, each level q = b/Pa, top h = b/Pa
coalition = G / (P + Lr * Pa)                   # chain-spanning coalition, all fines collected
ax[1].plot(Lr, honest_eq, "o-", label="honest equilibrium (each agent deterred alone)")
ax[1].plot(Lr, coalition, "s-", label="chain-spanning coalition deterred")
ax[1].fill_between(Lr, honest_eq, coalition, where=coalition > honest_eq, alpha=.15, color="red",
                   label="honest equilibrium exists but is not collusion-proof")
ax[1].set_yscale("log")
ax[1].set(xlabel="number of AI audit levels L",
          ylabel="human audit rate needed (per report)",
          title=f"Strategic auditors (R={R:g}, P=Pa={P:g}, b={b:g}, coalition gain G={G:g})")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("hierarchy.png", dpi=140)
print("\nSaved hierarchy.png")
