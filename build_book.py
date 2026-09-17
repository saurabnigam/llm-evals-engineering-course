#!/usr/bin/env python3
"""Combine the eval-engineering course's README + module READMEs into one
markdown file and one self-contained, print-friendly HTML file (rendered
client-side via marked.js, matching the existing books/*.html convention
in this repo).

Usage:
    python3 build_book.py <course_root> <out_stem> [--source-prefix PREFIX]

    course_root      Directory containing README.md and NN-slug/ module dirs.
    out_stem         Output path without extension, e.g. books/eval-engineering
                      -> writes <out_stem>.md and <out_stem>.html
    --source-prefix  Prefix used in the "<!-- SOURCE: ... -->" markers before
                      each file's relative path (default: "").

Re-run this after any course revision so the single-file book stays in sync
with the per-module READMEs, which remain the source of truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HTML_HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
/* CSS for print/book layout */
@page {{
    margin: 1.5cm;
}}
body {{
    font-family: 'Georgia', serif;
    font-size: 11pt;
    line-height: 1.4;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    color: #000;
}}
@media print {{
    body {{
        font-size: 9pt; /* Smaller font for less pages */
        line-height: 1.2;
    }}
    h1, h2, h3, h4 {{
        page-break-after: avoid;
    }}
    p, pre, blockquote, ul, ol, li {{
        page-break-inside: avoid;
    }}
    hr {{
        page-break-before: always;
        border: 0;
    }}
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    margin-top: 1em;
    margin-bottom: 0.5em;
}}
h1 {{ font-size: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
h2 {{ font-size: 14pt; }}
h3 {{ font-size: 12pt; }}
pre, code {{
    font-family: 'Consolas', 'Courier New', Courier, monospace;
    font-size: 8pt;
}}
code {{
    background-color: #f4f4f4;
    padding: 2px 4px;
    border-radius: 3px;
}}
pre {{
    background-color: #f4f4f4;
    padding: 10px;
    overflow-x: auto;
    white-space: pre-wrap; /* critical for printing */
    word-wrap: break-word;
    border: 1px solid #ddd;
}}
blockquote {{
    border-left: 4px solid #ccc;
    margin-left: 0;
    padding-left: 10px;
    color: #555;
    font-style: italic;
}}
img {{
    max-width: 100%;
    height: auto;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 1em;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 6px;
    text-align: left;
}}
th {{
    background-color: #f4f4f4;
}}
</style>
</head>
<body>
<div id="content"></div>
<script>
// We pass the markdown content via a JSON string to avoid closing script tag issues
const mdString = {md_json};
document.getElementById('content').innerHTML = marked.parse(mdString);
</script>
</body>
</html>
"""


def module_dirs(course_root: Path) -> list[Path]:
    dirs = [
        d
        for d in course_root.iterdir()
        if d.is_dir() and re.match(r"^\d{2}-", d.name) and (d / "README.md").exists()
    ]
    return sorted(dirs, key=lambda d: d.name)


def build_combined_markdown(course_root: Path, source_prefix: str) -> str:
    files = [course_root / "README.md"] + [d / "README.md" for d in module_dirs(course_root)]
    blocks = []
    for f in files:
        rel = f.relative_to(course_root).as_posix()
        text = f.read_text(encoding="utf-8").strip()
        blocks.append(f"<!-- SOURCE: {source_prefix}{rel} -->\n\n{text}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("course_root", type=Path)
    ap.add_argument("out_stem", type=Path)
    ap.add_argument("--source-prefix", default="")
    ap.add_argument("--title", default="Eval Engineering Book")
    args = ap.parse_args()

    combined_md = build_combined_markdown(args.course_root, args.source_prefix)

    md_path = args.out_stem.with_suffix(".md")
    html_path = args.out_stem.with_suffix(".html")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(combined_md, encoding="utf-8")
    html_path.write_text(
        HTML_HEAD.format(title=args.title, md_json=json.dumps(combined_md)),
        encoding="utf-8",
    )
    print(f"wrote {md_path} ({len(combined_md):,} chars)")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
