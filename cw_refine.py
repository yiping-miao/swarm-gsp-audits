"""Refinements: deletions during the active period, trick diffusion from the June-16 onset,
deletion order within bursts. Data only."""
import gzip, json, sys, re, collections
from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1]
load = lambda n: [json.loads(l) for l in gzip.open(f"{D}/{n}.jsonl.gz", "rt", encoding="utf-8")]
revs, events = load("revisions"), load("events")
T = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
for r in revs: r["t"] = T(r["time"]); r["body"] = r["body"] or ""
last_post = max(r["t"] for r in revs if r["time"] < "2026-06-23")
print("last post of the main episode:", last_post)

saves = collections.defaultdict(list)
for r in revs: saves[r["page_key"]].append(r["t"])
for v in saves.values(): v.sort()
dels = sorted([e for e in events if e["event_type"] == "delete" and e.get("page_key")],
              key=lambda e: e["time"])
active = [e for e in dels if T(e["time"]) <= last_post]
print(f"deletions during active period: {len(active):,} of {len(dels):,}")
delays = []
for e in active:
    nxt = [t for t in saves[e["page_key"]] if t > T(e["time"])]
    if nxt: delays.append((nxt[0] - T(e["time"])).total_seconds() / 60)
print(f"  page written again after deletion: {len(delays)} ({100*len(delays)/max(1,len(active)):.0f}%)"
      + (f", median {np.median(delays):.0f} min" if delays else ""))
# posts mentioning deletion BEFORE vs AFTER first active deletion
if active:
    t0 = T(active[0]["time"])
    EV = re.compile(r"\b(delet\w*|admin\w*|moderat\w*|backup)\b", re.I)
    pre = [r for r in revs if r["t"] < t0]; post = [r for r in revs if t0 <= r["t"] <= last_post]
    f = lambda rs: 100 * sum(bool(EV.search(r["body"])) for r in rs) / max(1, len(rs))
    print(f"  first active deletion {t0:%m-%d %H:%M}; posts mentioning deletion/admin/backup: "
          f"{f(pre):.1f}% before vs {f(post):.1f}% after")
# alphabetical order within 30-minute bursts of >=20 deletions
bursts, cur = [], [dels[0]]
for e in dels[1:]:
    if (T(e["time"]) - T(cur[-1]["time"])).total_seconds() < 120:
        cur.append(e)
    else:
        bursts.append(cur); cur = [e]
bursts.append(cur)
big = [b for b in bursts if len(b) >= 20]
rhos = []
for b in big:
    names = [e["page_key"].split("~", 1)[-1].lower() for e in b]
    rhos.append(spearmanr(range(len(names)), np.argsort(np.argsort(names)))[0])
rhos = np.array(rhos)
print(f"deletion bursts (>=20 deletions, gaps <2 min): {len(big)}; "
      f"alphabetical (rho>0.9): {(rhos>0.9).sum()}, reverse (rho<-0.9): {(rhos<-0.9).sum()}, "
      f"median rho {np.median(rhos):+.2f}")

# diffusion from onset of mass coordination
onset = datetime(2026, 6, 16, tzinfo=timezone.utc)
TRICK = re.compile(r"\bGET[- ]?(edit|save|write)", re.I)
first = {}
for r in sorted(revs, key=lambda r: r["t"]):
    if r["label"] and (TRICK.search(r["body"]) or "GetEdit" in r["name"]):
        first.setdefault(r["label"], r["t"])
ts = sorted(t for t in first.values() if t >= onset)
pre = sum(t < onset for t in first.values())
if ts:
    h = lambda t: (t - ts[0]).total_seconds() / 3600
    print(f"GET-edit trick: {pre} agents before Jun 16, {len(ts)} after; after first post-onset use, "
          f"50% adopted within {h(ts[len(ts)//2]):.1f} h, 90% within {h(ts[int(.9*len(ts))]):.1f} h")
SB = re.compile(r"sandbox", re.I)
sb = sorted({r["label"]: r["t"] for r in sorted(revs, key=lambda r: r["t"], reverse=True)
             if r["label"] and SB.search(r["body"])}.values())
if sb:
    print(f"'sandbox' mentioned by {len(sb)} agents; first {sb[0]:%m-%d %H:%M}, "
          f"second {sb[1]:%m-%d %H:%M}" if len(sb) > 1 else "")

# concentration of the round-answer relay
RND = re.compile(r"\bR[1-5]\b")
rel = collections.Counter(r["label"] for r in revs if r["label"] and RND.search(r["body"]))
v = np.sort(np.array(list(rel.values())))[::-1]
print(f"round-relay posts: {v.sum():,} by {len(v):,} agents; top 10% wrote {100*v[:len(v)//10].sum()/v.sum():.0f}%; "
      f"median agent wrote {int(np.median(v))}")
