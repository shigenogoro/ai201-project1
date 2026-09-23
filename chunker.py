"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or 800
    overlap = overlap or 120

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def _title_and_body(text: str) -> tuple[str, str]:
    """The document's `# ...` line, and everything after it."""
    first, _, rest = text.partition("\n")
    if first.startswith("# "):
        return first.strip(), rest.lstrip("\n")
    return "", text


def _sections(body: str) -> list[str]:
    """
    Split on `## ` headings, keeping each heading attached to its own text.

    The capturing group is what keeps the headings: re.split returns
    [preamble, heading, text, heading, text, ...], so the pairs come out of
    zip(parts[1::2], parts[2::2]).
    """
    parts = re.split(r"(?m)^(##\s.*)$", body)

    sections = []
    preamble = parts[0].strip()
    if preamble:                       # five of the fourteen guides have none
        sections.append(preamble)
    for heading, text in zip(parts[1::2], parts[2::2]):
        sections.append(f"{heading.strip()}\n\n{text.strip()}")
    return sections


def _windows(text: str, ceiling: int) -> list[str]:
    """Last resort, for a single paragraph longer than the ceiling."""
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + ceiling].strip()
        if piece:
            pieces.append(piece)
        start += ceiling - config.CHUNK_OVERLAP
    return pieces


def _fit(section: str, ceiling: int) -> list[str]:
    """
    Cut a section down to the ceiling, on paragraph breaks where it can.

    Nothing in city_guides reaches the ceiling; to exercise this, drop
    config.CHUNK_SIZE to 400 and re-run. The heading is repeated on every
    piece, so a section that does come apart still says what it is about.
    """
    if len(section) <= ceiling:
        return [section]

    heading, _, rest = section.partition("\n\n")
    if not heading.startswith("##"):   # the preamble has no heading
        heading, rest = "", section

    room = ceiling - (len(heading) + 2 if heading else 0)

    pieces: list[str] = []
    current = ""
    for paragraph in rest.split("\n\n"):
        if len(paragraph) > room:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_windows(paragraph, room))
        elif current and len(current) + 2 + len(paragraph) > room:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)

    if not heading:
        return pieces
    return [f"{heading}\n\n{piece}" for piece in pieces]


def _merge_short(pieces: list[str], floor: int, ceiling: int) -> list[str]:
    """
    Fold anything under the floor into a neighbour, where that still fits.

    The title prefix already lifts my shortest piece from 123 to 173, so this
    does not fire either. It is here so criterion 4's floor is enforced rather
    than satisfied by accident; raise config.CHUNK_MIN to 250 to watch it work.
    """
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < floor and len(merged[-1]) + 2 + len(piece) <= ceiling:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)

    # The first piece has nothing behind it to fold into, so fold it forward.
    if (
        len(merged) > 1
        and len(merged[0]) < floor
        and len(merged[0]) + 2 + len(merged[1]) <= ceiling
    ):
        merged[1] = f"{merged[0]}\n\n{merged[1]}"
        merged.pop(0)

    return merged


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split each document at its `## ` headings — one section, one chunk.

    Every chunk keeps the document's `# ` title line on the front. All fourteen
    guides share their section headings (ten have a `## Eat and drink`) and a
    section's own text never repeats its town name, so without the title a
    chunk cannot say which town it belongs to. The prefix also lifts the
    shortest piece over criterion 4's floor.

    `## Straightforward` in guide_accessibility.md stays whole at 763
    characters even though it covers three towns: the section exists to compare
    them, and splitting it would answer "is Brightwater step-free?" better at
    the cost of "which towns are easiest?".

    Lengths are measured including the prefix, so the ceiling and floor passed
    down to the helpers have the prefix subtracted from them.
    """
    chunks: list[Chunk] = []

    for doc in documents:
        title, body = _title_and_body(doc.text)
        prefix = f"{title}\n" if title else ""

        pieces: list[str] = []
        for section in _sections(body):
            pieces.extend(_fit(section, config.CHUNK_SIZE - len(prefix)))

        pieces = _merge_short(
            pieces,
            floor=config.CHUNK_MIN - len(prefix),
            ceiling=config.CHUNK_SIZE - len(prefix),
        )

        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=f"{prefix}{piece}",
                    source=doc.source,
                    index=index,
                    produced_by="chunker.py::split_documents",
                )
            )

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
