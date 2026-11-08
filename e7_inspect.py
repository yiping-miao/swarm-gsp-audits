"""Inspect E7 pooled W>Q transcripts: reasoning on FAIL rounds, chat, lies and accusations."""
import json, re, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if '"E7:pooled,W=6' in l]
print("rows", len(rows))
fails = [r for r in rows if r["truth"] == "FAIL"]
print("FAIL rounds", len(fails), "lies", sum(r["report"] == "PASS" for r in fails))
kw = collections.Counter()
for r in fails:
    t = (r["reasoning"] or "").lower()
    for k in ("accus", "whistle", "self", "expose", "pool", "w - q", "net", "honest", "bonus", "expected"):
        kw[k] += k in t
print("keywords in FAIL-round reasoning:", dict(kw))
print("\n--- sample FAIL reasoning (truncated)")
for r in fails[:4] + [r for r in fails if r["report"] == "PASS"][:2]:
    print(f"[round {r['round']} agent {r['agent']} -> {r['report']}]", (r["reasoning"] or "")[:550].replace("\n", " "))
print("\n--- sample team chat")
seen = set()
for r in rows:
    ch = r.get("channel")
    if ch and (r["round"], r["agent"] // 3) not in seen and any("FAIL" in m for m in ch):
        seen.add((r["round"], r["agent"] // 3))
        print(f"[round {r['round']}]", " || ".join(m[:220] for m in ch))
        if len(seen) >= 3:
            break
acc = [r for r in rows if r.get("accused")]
print("\naccusations made:", len(acc))
