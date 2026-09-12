#!/usr/bin/env python3
"""
Strip # comments from a notebook's own code cells, to produce a comment-free copy.

    python pipeline/strip_comments.py IN.ipynb OUT.ipynb

build_colab.py is the annotated version and stays that way; this makes the copy
that goes out, where the glue cells carry no running commentary.

Every code cell is stripped, including the %%writefile cells carrying the
pipeline scripts: the Drive folder ships the notebook alone, not the
repository, so there is nothing for those cells to drift from. The repository
copy keeps its comments either way.

Left alone:

  * markdown cells, which carry the explanation of each stage;
  * docstrings, which are strings rather than comments;
  * outputs, execution counts and cell metadata, so an already-executed notebook
    stays executed and its recorded run is untouched.

Comments are found with tokenize, not a regex, so a # inside a string literal is
never mistaken for one.
"""

from __future__ import annotations

import io
import json
import sys
import tokenize
from pathlib import Path

def _strip_python(src: str) -> str:
    """Remove comment tokens from Python source, and the blank lines they leave."""
    cuts: dict[int, int] = {}
    protected: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                cuts.setdefault(tok.start[0], tok.start[1])
            elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                # Every line of a multi-line string literal, docstrings included.
                # Blank lines in there are part of the value, not layout.
                protected.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                    # not parseable on its own: leave it exactly as is
    if not cuts:
        return src

    lines: list[tuple[int, str]] = []
    for i, line in enumerate(src.splitlines(), 1):
        if i in cuts and i not in protected:
            head = line[: cuts[i]].rstrip()
            if not head:
                continue              # the whole line was a comment
            line = head
        lines.append((i, line))

    def blank(entry: tuple[int, str]) -> bool:
        return not entry[1].strip() and entry[0] not in protected

    kept: list[tuple[int, str]] = []
    for entry in lines:               # collapse blank runs left by a removed block
        if blank(entry) and kept and blank(kept[-1]):
            continue
        kept.append(entry)
    return "\n".join(text for _, text in kept).strip("\n")


def strip_source(src: str) -> str:
    first, sep, rest = src.partition("\n")
    if first.startswith("%%"):        # a cell magic hides the rest from tokenize
        return first + sep + _strip_python(rest)
    return _strip_python(src)


def strip_notebook(src_path: Path, dst_path: Path) -> None:
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new = strip_source(src)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            changed += 1
    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst_path}  ({dst_path.stat().st_size / 1024:.0f} KB, "
          f"{changed} of {len(nb['cells'])} cells stripped)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    strip_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
