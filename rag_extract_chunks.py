# rag_extract_chunks.py
# Extract markdown from otel-docs into chunks.jsonl for indexing.
#
# Run from rag/:
#   python rag_extract_chunks.py
#
# Config (optional):
#   CORPUS_DIR=../otel-docs
#   CHUNKS_OUT=chunks.jsonl
#   CHUNK_CHARS=1400
#   CHUNK_OVERLAP=200

import os
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "../otel-docs")).resolve()
CHUNKS_OUT = Path(os.getenv("CHUNKS_OUT", "chunks.jsonl"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Look here for docs content
CANDIDATE_DIRS = [
    CORPUS_DIR / "content",
    CORPUS_DIR / "content-modules",
]


def iter_markdown_files() -> Iterator[Path]:
    for d in CANDIDATE_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.md"):
            yield p


_frontmatter_re = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    return re.sub(_frontmatter_re, "", text, count=1)


def md_to_plain_ish(text: str) -> str:
    # Keep headings as plain text, remove code fences (keep content), remove links formatting
    text = text.replace("\r\n", "\n")
    text = strip_frontmatter(text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Convert markdown links [text](url) -> text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if chunk_chars <= 200:
        chunk_chars = 200
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_chars:
        overlap = chunk_chars // 4

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_chars, n)

        # Try to break on paragraph boundary
        window = text[start:end]
        cut = window.rfind("\n\n")
        if cut > 200:
            end = start + cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def infer_title_and_section(md_text: str) -> Tuple[str, str]:
    # Very rough: first H1 as title, first H2 as section
    title = ""
    section = ""

    for line in md_text.splitlines():
        if not title and line.startswith("# "):
            title = line[2:].strip()
            continue
        if title and not section and line.startswith("## "):
            section = line[3:].strip()
            break

    return title, section


def rel_url_for(path: Path) -> str:
    # Create a stable "doc path" even without the site generator
    # Example: content/docs/collector.md -> /content/docs/collector.md
    try:
        rel = path.relative_to(CORPUS_DIR)
    except ValueError:
        rel = path.name
    return "/" + str(rel).replace("\\", "/")


def main() -> None:
    files = list(iter_markdown_files())
    if not files:
        raise SystemExit(f"No markdown found under: {', '.join(str(d) for d in CANDIDATE_DIRS)}")

    out_path = CHUNKS_OUT
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for md_file in files:
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
            plain = md_to_plain_ish(raw)
            if len(plain) < 200:
                continue

            title, section = infer_title_and_section(raw)
            url = rel_url_for(md_file)

            chunks = chunk_text(plain, CHUNK_CHARS, CHUNK_OVERLAP)
            for i, ch in enumerate(chunks):
                rec = {
                    "id": f"{url}::chunk{i}",
                    "text": ch,
                    "url": url,
                    "doc_title": title or md_file.stem,
                    "section": section or "",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"✅ Wrote {written} chunks to {out_path.resolve()}")


if __name__ == "__main__":
    main()

