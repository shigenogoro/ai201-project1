# Milestone 3 — baseline (starter chunker, before my change)

Corpus: city_guides. Command: `python app.py --variant fallback index`

    loaded   14 documents, 28,958 characters, ~2,068 characters per document
    chunked  51 chunks, 650 characters on average (shortest 24, longest 800),
             produced by chunker.py::fallback_split

Kept as Chroma collection `city_guides__fallback`, so I can re-query it after
the change without re-indexing.

## Retrieval baseline (TOP_K = 5, no API calls — embeddings run locally)

| Q | best distance | top-1 source | answer anywhere in top 5? |
|---|---|---|---|
| Q1 Brightwater Tuesday market | 0.363 | guide_eating.md#1 (wrong town) | YES (rank 5) |
| Q2 winter walking +4 min | 0.439 | guide_walking.md#2 | YES |
| Q3 Halden Bay harbour inn | 0.359 | guide_halden_bay.md#0 | **NO** |
| Q4 Marchwood canal walk | 0.483 | guide_accessibility.md#0 (wrong doc) | YES |
| Q5 Corry Vale best season | 0.393 | guide_brightwater.md#2 (wrong town) | **NO** |

Criterion 1 today: **3 of 5 — MISSED** (target 4 of 5).
Criterion 4 today: **FAIL** — shortest chunk 24 chars, floor is 150.

OUT_OF_SCOPE best distances: 0.887, 0.897, 0.903, 0.829, 0.853.
In-corpus group 0.359-0.483, out-of-scope group 0.829-0.903. Wide gap.

## What I read off this

Three of five top-1 hits are the right *section* from the wrong *document*.
All 14 guides use the same `## Getting there / ## Eat and drink / ## When to
go` headings, and the body of a section never repeats the town name — the
Brightwater "Eat and drink" section does not contain the word "Brightwater".
So the embedding has nothing to tell the fourteen towns apart.

Two separate problems, then:
  1. boundaries cut mid-word (chunk #1 starts `'on.\n\n**Brightwater** is...'`)
  2. chunks carry no document identity
