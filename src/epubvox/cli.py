"""The `epubvox` command.

    epubvox [options] BOOK.epub     convert a book (epubvox --help)
    epubvox web [options]           open the local web page (epubvox web --help)
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["web"]:  # a book literally named "web" can still be given as ./web
        from epubvox import web

        return web.main(argv[1:])
    from epubvox import convert

    return convert.main(argv)
