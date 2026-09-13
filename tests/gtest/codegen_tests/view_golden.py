#!/usr/bin/env python3
"""Pretty-print compact codegen golden .out files for review. CI diffs the compact form."""

import argparse
import sys
from pathlib import Path


def pretty_msl(src: str) -> str:
    buf = []
    indent = 0
    current = []
    i = 0

    def flush():
        text = "".join(current).strip()
        current.clear()
        if text:
            buf.append(("    " * indent) + text)

    while i < len(src):
        if src.startswith("[[", i):
            end = src.find("]]", i)
            if end < 0:
                raise ValueError("unclosed [[ in shader")
            current.append(src[i : end + 2])
            i = end + 2
            continue
        c = src[i]
        if c == "{":
            current.append("{")
            flush()
            indent += 1
        elif c == "}":
            flush()
            indent = max(0, indent - 1)
            current.append("}")
            flush()
        elif c == ";":
            current.append(";")
            flush()
        elif c == "\n":
            flush()
        else:
            current.append(c)
        i += 1
    flush()
    return "\n".join(buf) + ("\n" if buf else "")


def view_golden(text: str) -> str:
    if "---\n" in text:
        header, shader = text.split("---\n", 1)
    elif text.endswith("---"):
        header, shader = text[:-3], ""
    else:
        header, shader = text, ""
    parts = [header.rstrip("\n"), "---"]
    pretty = pretty_msl(shader)
    if pretty:
        parts.append(pretty.rstrip("\n"))
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="golden .out files (default: all *.out in this directory)",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    paths = args.paths or sorted(here.glob("*.out"))
    if not paths:
        print("no golden .out files", file=sys.stderr)
        sys.exit(1)
    for i, path in enumerate(paths):
        if i:
            print()
        print(f"=== {path} ===")
        print(view_golden(path.read_text()), end="")


if __name__ == "__main__":
    main()
