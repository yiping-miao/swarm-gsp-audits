"""Extract text from a PDF (data only): page count, then text of selected pages to a .txt file."""
import sys
from pypdf import PdfReader
src, out = sys.argv[1], sys.argv[2]
r = PdfReader(src)
with open(out, "w", encoding="utf-8") as f:
    for i, pg in enumerate(r.pages, 1):
        f.write(f"\n\n===== PAGE {i} =====\n")
        f.write(pg.extract_text() or "")
print("pages:", len(r.pages))
