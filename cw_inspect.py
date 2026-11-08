"""Inspect collusion.wiki export: schema, counts, sample (truncated) values. Data only."""
import gzip, json, sys, collections
D = sys.argv[1]
for name in ("pages", "revisions", "events", "labels"):
    rows = [json.loads(l) for l in gzip.open(f"{D}/{name}.jsonl.gz", "rt", encoding="utf-8")]
    print(f"\n=== {name}: {len(rows):,} rows")
    keys = collections.Counter(k for r in rows for k in r)
    for k, n in keys.most_common():
        v = rows[0].get(k)
        s = json.dumps(v, ensure_ascii=False)
        print(f"  {k:24} present {n:6,}  e.g. {s[:110]}")
