"""Explore collusion.wiki contents: families, event types, keyword hits, samples. Data only."""
import gzip, json, sys, re, collections
D = sys.argv[1]
load = lambda n: [json.loads(l) for l in gzip.open(f"{D}/{n}.jsonl.gz", "rt", encoding="utf-8")]
pages, revs, events = load("pages"), load("revisions"), load("events")

print("page_family:", collections.Counter(p["page_family"] for p in pages).most_common(15))
print("page_family_cohort:", collections.Counter(p["page_family_cohort"] for p in pages).most_common(8))
print("event_type:", collections.Counter(e["event_type"] for e in events).most_common())
print("wikis (revs):", collections.Counter(r["wiki"] for r in revs).most_common(6))
print("round_id non-null revs:", sum(r["round_id"] is not None for r in revs),
      collections.Counter(json.dumps(r["round_id"]) for r in revs if r["round_id"]).most_common(8))
days = collections.Counter(r["time"][:10] for r in revs)
print("revs per day:", sorted(days.items()))

KW = {"relay": r"\brelay", "answer": r"\banswer", "R1-5": r"\bR[1-5]\b", "round": r"\bround",
      "verified": r"\bverif", "wrong/incorrect": r"\b(wrong|incorrect)\b",
      "correct": r"\bcorrect\b", "deleted/admin": r"\b(delet|admin)", "backup": r"\bbackup",
      "sandbox": r"sandbox", "please": r"\bplease\b", "consensus/agree": r"\b(consensus|agree)"}
print("\nrevisions whose body matches keyword (case-insensitive):")
for k, pat in KW.items():
    n = sum(bool(re.search(pat, r["body"] or "", re.I)) for r in revs)
    print(f"  {k:18} {n:6,}")

rx = re.compile(r"\bR[1-5]\b", re.I)
samp = [r for r in revs if rx.search(r["body"] or "") and 80 < r["body_len"] < 900][:6]
print("\nSAMPLES (truncated, data only):")
for r in samp:
    print("---", r["time"], r["label"], r["name"])
    print((r["body"] or "")[:500].replace("\n", " | "))
