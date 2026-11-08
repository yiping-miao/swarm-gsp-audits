"""Normalize extracted PDF text (data only); print selected pages and keyword contexts."""
import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
pages = re.split(r"===== PAGE (\d+) =====", raw)
P = {int(pages[i]): re.sub(r"\s+", " ", pages[i + 1]).replace("â€™", "'").replace("â€œ", '"').replace("â€\x9d", '"').strip()
     for i in range(1, len(pages) - 1, 2)}
mode = sys.argv[2]
if mode == "pages":
    a, b = map(int, sys.argv[3].split("-"))
    for n in range(a, b + 1):
        print(f"\n[p{n}] {P.get(n, '')[:3500]}")
else:
    for kw in sys.argv[3].split("|"):
        hits = [(n, m.start()) for n, t in P.items() for m in re.finditer(kw, t, re.I)]
        print(f"\n## '{kw}': {len(hits)} hits")
        for n, s in hits[:4]:
            print(f"  [p{n}] ...{P[n][max(0, s-220):s+260]}...")
