"""
Phase diagram for the team sign-off mechanism (Proposition 8).

Axes: audit rate q (P fixed) and per-agent team bonus T.
  not strategy-proof           : qP < R + T
  GSP but not bribe-proof      : R + T < qP < R + kT
  bribe-proof                  : qP > R + kT
Dashed: pooled team WITHOUT communication is deterred iff qP > R + kT p^(k-1).

Markers: rational-agent simulation from experiments.py at grid points
  (individual condition lies -> not SP; pooled+chat lies -> not bribe-proof).
"""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import BASE, Rational, run, rates

R, P, k, p = 1.0, 10.0, 3, 0.7
q = np.linspace(0.0, 1.0, 400)
x = q * P - R                                  # qP - R
sp = np.clip(x, 0, None)                       # T below this -> SP/GSP
bp = np.clip(x / k, 0, None)                   # T below this -> bribe-proof
nc = np.clip(x / (k * p ** (k - 1)), 0, None)  # pooled, no chat
Tmax = 3.0

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(q, 0, np.minimum(bp, Tmax), color="#4c9a6a", alpha=.35, label="bribe-proof (GSP with side payments)")
ax.fill_between(q, np.minimum(bp, Tmax), np.minimum(sp, Tmax), color="#e0b84c", alpha=.45,
                label="GSP, not bribe-proof")
ax.fill_between(q, np.minimum(sp, Tmax), Tmax, color="#c8574b", alpha=.35,
                label="not strategy-proof\n(if teammates' results are known)")
ax.plot(q, np.minimum(nc, Tmax), "k--", lw=1.2, label="pooled team without chat: deterred below")

# simulation check
grid_q = np.linspace(0.05, 0.95, 10)
grid_T = np.linspace(0.1, 2.9, 8)
team = dict(BASE, N=6, team=3, P=P, R=R, p=p, rounds=20, frame="baseline")
for qq in grid_q:
    for TT in grid_T:
        c_ind = dict(team, q=qq, T=TT)
        c_pool = dict(team, q=qq, T=TT, pooled=True, chat=True)
        ind = np.mean([rates(run(c_ind, Rational(c_ind), random.Random(s)))["lie"] for s in range(3)])
        pool = np.mean([rates(run(c_pool, Rational(c_pool), random.Random(s)))["lie"] for s in range(3)])
        m = "x" if ind > 0 else ("o" if pool > 0 else "s")
        ax.plot(qq, TT, m, color="k", ms=5, mfc="none" if m != "x" else None)
ax.plot([], [], "x", color="k", label="sim: individuals lie\n(teammates' results unknown)")
ax.plot([], [], "o", color="k", mfc="none", label="sim: only pooled teams lie")
ax.plot([], [], "s", color="k", mfc="none", label="sim: nobody lies")

ax.set(xlim=(0, 1), ylim=(0, Tmax), xlabel=f"audit rate q   (penalty P = {P:g}, reward R = {R:g})",
       ylabel=f"team sign-off bonus T per agent (team size k = {k})",
       title="When is honest reporting group strategy-proof?")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig("phase_diagram.png", dpi=140)
print("Saved phase_diagram.png")
