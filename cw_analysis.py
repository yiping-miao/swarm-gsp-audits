"""
collusion.wiki analysis tied to the GSP audit-mechanism paper. Reads the export as data only.

A  Scale and network: how many agents coordinated, and how connected they were.
B  Providers vs requesters (free riding on shared answers).
C  Verification and error reports: are shared answers checked?
D  Response to oversight: admin deletions vs re-creation and evasion.
E  Diffusion of a workaround (the GET-edit trick) across agents.

Usage: python -I cw_analysis.py data/collusion
"""
import gzip, json, sys, re, collections
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
load = lambda n: [json.loads(l) for l in gzip.open(f"{D}/{n}.jsonl.gz", "rt", encoding="utf-8")]
pages, revs, events = load("pages"), load("revisions"), load("events")
T = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
for r in revs:
    r["t"] = T(r["time"]); r["body"] = r["body"] or ""
agent_revs = [r for r in revs if r["label"]]           # drop blank (unattributed) labels
print(f"Revisions {len(revs):,}; attributed {len(agent_revs):,}; "
      f"agents {len({r['label'] for r in agent_revs}):,}; pages {len(pages):,}")

# ---------------------------------------------------------------- A network
by_page = collections.defaultdict(set)
for r in agent_revs:
    by_page[r["page_key"]].add(r["label"])
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for labs in by_page.values():
    labs = list(labs)
    for b in labs[1:]:
        parent[find(labs[0])] = find(b)
for r in agent_revs: find(r["label"])
comp = collections.Counter(find(l) for l in parent)
giant = comp.most_common(1)[0][1]
shared_pages = sum(len(v) >= 2 for v in by_page.values())
on_shared = sum(len(by_page[r["page_key"]]) >= 2 for r in agent_revs)
print("\n[A] Network")
print(f"  pages edited by >=2 agents: {shared_pages:,} of {len(by_page):,}")
print(f"  posts on shared pages: {100*on_shared/len(agent_revs):.0f}%")
print(f"  largest connected group of agents (co-editing): {giant:,} of {len(parent):,} "
      f"({100*giant/len(parent):.0f}%)")

# ---------------------------------------------------------------- B providers/requesters
REQ = re.compile(r"\b(please|need|urgent|asap|anyone|can someone|report|post (your|current|the))\b|\?", re.I)
NUM = re.compile(r"(=\s*-?[\d][\d,\.]{2,})|\b(answer|value|result|confirmed)\b[^\n]{0,40}\d", re.I)
LINK = re.compile(r"https?://\S+")
def kind(b):
    give = bool(NUM.search(b) or LINK.search(b)); ask = bool(REQ.search(b))
    return {(True, False): "provide", (False, True): "request",
            (True, True): "both", (False, False): "other"}[(give, ask)]
kinds = collections.Counter(kind(r["body"]) for r in agent_revs)
print("\n[B] Post types (heuristic)")
for k in ("provide", "request", "both", "other"):
    print(f"  {k:8} {kinds[k]:6,}  ({100*kinds[k]/len(agent_revs):.0f}%)")
per = collections.defaultdict(collections.Counter)
for r in agent_revs:
    per[r["label"]][kind(r["body"])] += 1
active = {l: c for l, c in per.items() if sum(c.values()) >= 3}
give = lambda c: c["provide"] + c["both"]
ask = lambda c: c["request"] + c["both"]
free = sum(give(c) == 0 and ask(c) > 0 for c in active.values())
prov_only = sum(ask(c) == 0 and give(c) > 0 for c in active.values())
print(f"  agents with >=3 posts: {len(active):,}; only ask (free riders): {free} "
      f"({100*free/len(active):.0f}%); only provide: {prov_only} ({100*prov_only/len(active):.0f}%)")
contrib = np.sort([give(c) for c in per.values()])[::-1]
top10 = contrib[: max(1, len(contrib)//10)].sum() / max(1, contrib.sum())
print(f"  top 10% of agents supply {100*top10:.0f}% of answer/data posts")

# ---------------------------------------------------------------- C verification/errors
VER = re.compile(r"\b(verif\w*|double[- ]check\w*|cross[- ]check\w*)\b", re.I)
ERR = re.compile(r"\b(wrong|incorrect|mismatch\w*|not match\w*|was off|bad value|error in)\b", re.I)
ans = [r for r in agent_revs if kind(r["body"]) in ("provide", "both")]
nver = sum(bool(VER.search(r["body"])) for r in ans)
nerr = sum(bool(ERR.search(r["body"])) for r in agent_revs)
print("\n[C] Verification")
print(f"  answer/data posts: {len(ans):,}; of these mention verification: {nver} "
      f"({100*nver/len(ans):.1f}%)")
print(f"  posts flagging something as wrong/incorrect/mismatched: {nerr:,} "
      f"({100*nerr/len(agent_revs):.1f}% of posts)")

# ---------------------------------------------------------------- D oversight
dels = [e for e in events if e["event_type"] == "delete" and e.get("page_key")]
saves = collections.defaultdict(list)
for r in revs:
    saves[r["page_key"]].append(r["t"])
for v in saves.values(): v.sort()
delays, recreated = [], 0
for e in dels:
    td = T(e["time"])
    nxt = [t for t in saves[e["page_key"]] if t > td]
    if nxt:
        recreated += 1; delays.append((nxt[0] - td).total_seconds() / 60)
print("\n[D] Oversight (admin deletions)")
print(f"  deletions: {len(dels):,}; page written again afterwards: {recreated:,} "
      f"({100*recreated/len(dels):.0f}%)")
if delays:
    print(f"  median time to rewrite: {np.median(delays):.0f} min "
          f"(25-75%: {np.percentile(delays,25):.0f}-{np.percentile(delays,75):.0f} min)")
# alphabetical deletion order within each day
rho = []
for day, grp in collections.groupby(sorted(dels, key=lambda e: e["time"]), key=lambda e: e["time"][:10]) if False else []:
    pass
byday = collections.defaultdict(list)
for e in sorted(dels, key=lambda e: e["time"]):
    byday[e["time"][:10]].append(e["page_key"].split("~", 1)[-1].lower())
from scipy.stats import spearmanr
for day, names in sorted(byday.items()):
    if len(names) >= 50:
        r_, _ = spearmanr(range(len(names)), [n[:3] for n in names]) if False else spearmanr(
            range(len(names)), np.argsort(np.argsort(names)))
        rho.append((day, len(names), r_))
print("  deletion order vs alphabetical order (Spearman rho, days with >=50 deletions):")
for day, n, r_ in rho:
    print(f"    {day}: n={n:5,}  rho={r_:+.2f}")
zz = [p for p in pages if p["name"].upper().startswith("ZZ")]
print(f"  pages named ZZ...: {len(zz)} (first: {min((p['first_write'] for p in zz), default='-')})")
EVADE = re.compile(r"\b(delet\w*|admin\w*|moderat\w*|backup|mirror|survive)\b", re.I)
nev = sum(bool(EVADE.search(r["body"])) for r in agent_revs)
print(f"  posts discussing deletion/admin/backup: {nev:,}")

# ---------------------------------------------------------------- E diffusion
TRICK = re.compile(r"\bGET[- ]?(edit|save|write)", re.I)
first = {}
for r in sorted(agent_revs, key=lambda r: r["t"]):
    if TRICK.search(r["body"]) or "GetEdit" in r["name"]:
        first.setdefault(r["label"], r["t"])
ts = sorted(first.values())
print("\n[E] Diffusion of the GET-edit write trick")
if ts:
    h = lambda t: (t - ts[0]).total_seconds() / 3600
    print(f"  agents using it: {len(ts):,}; first {ts[0]:%Y-%m-%d %H:%M}; "
          f"50% adopted after {h(ts[len(ts)//2]):.1f} h; 90% after {h(ts[int(.9*len(ts))]):.1f} h")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
days = sorted({r["time"][:10] for r in agent_revs})
pd = collections.Counter(r["time"][:10] for r in agent_revs)
ag = collections.defaultdict(set)
for r in agent_revs: ag[r["time"][:10]].add(r["label"])
dd = collections.Counter(e["time"][:10] for e in dels)
x = range(len(days))
ax[0].bar(x, [pd[d] for d in days], label="posts")
ax[0].plot(x, [dd[d] for d in days], "r.-", label="admin deletions")
ax[0].set_xticks(list(x)[::4]); ax[0].set_xticklabels([d[5:] for d in days][::4], rotation=45)
ax[0].set_yscale("log"); ax[0].set_title("Posts vs deletions per day"); ax[0].legend()
ax[1].plot(x, [len(ag[d]) for d in days], "g.-")
ax[1].set_xticks(list(x)[::4]); ax[1].set_xticklabels([d[5:] for d in days][::4], rotation=45)
ax[1].set_title("Distinct agents posting per day")
if ts:
    ax[2].plot([h(t) for t in ts], np.arange(1, len(ts)+1) / len(ts))
    ax[2].set_xlabel("hours since first use"); ax[2].set_ylabel("share of adopters")
    ax[2].set_title("Spread of GET-edit trick")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("collusion_wiki_analysis.png", dpi=130)
print("\nSaved collusion_wiki_analysis.png")
