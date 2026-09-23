# The Unofficial Guide

<!-- Replace this line with your name and which corpus you picked. -->

> **This file is your submission.** Fill it in as you go — most sections get
> written during the milestone that produces them, not at the end.
>
> How the starter works, and every command you'll need, is in `RUNNING.md`.
> Leave that file alone.
>
> **Paste everything as text.** No screenshots, no video. A typed table gets
> full credit; a picture of the same table gets none.
>
> Delete these instruction blocks as you replace them. The `<!-- -->` comments
> are notes to you and don't show up when the page renders — you can leave them
> or remove them.

---

# Unit 1

## What This Does

<!-- Three or four sentences. Which corpus you picked, and the kinds of
     questions your system answers. Write it for someone who has never seen
     this repo.

     Milestone 5. -->

## Chunking Strategy

**Chunk size:** 900 characters — a ceiling, not a target.
**Overlap:** 0 between chunks.

My corpus is `city_guides`: 14 documents of roughly 2,000 characters each. Every
one of them is a `# Title` line followed by `## Section` headings. Before
choosing any numbers I measured the structure: 84 sections and 10 preambles, the
sections running 178 to 711 characters and the preambles 123 to 250. A section
is one topic and usually answers one question on its own, so I split on `## `
headings — one section, one chunk — rather than on a character count.

### What the starter's numbers did to these documents

`fallback_split` cut every 800 characters with no regard for structure. On my
corpus that gave 51 chunks, 650 characters on average, shortest 24 and longest
800. Reading them showed three separate problems:

- **20 of the 51 chunks start in the middle of a word.**
  `guide_kestrelford.md#3` opens with `irts.` — the back half of "outskirts".
  `guide_accessibility.md#1` opens with `on.`, the end of "consideration".
- **Every chunk held about three topics**: the tail of one section plus two
  whole sections. `guide_brightwater.md#1` was the end of `## Getting around`,
  all of `## Eat and drink`, and the start of `## What to see` — buses, a
  market and a museum sharing one embedding.
- **6 chunks were under 150 characters**, so criterion 4 in `criteria.md`
  failed before I had run a single question.

### Every chunk carries its document's `#` title line

This is the part I got wrong at first. My plan was to put the section text in
the chunk and nothing else. Then I counted how often the word "Brightwater"
appears in the chunks of `guide_brightwater.md`: twice in chunk 0, once in
chunk 3, and **not at all in chunks 1 and 2** — and chunk 1 is where the answer
to my first test question lives.

The reason is that all fourteen guides use the same headings. Ten of them have a
`## Eat and drink`, and a section's body never repeats its own town name, so
nothing in the Brightwater "Eat and drink" section says "Brightwater".

The one mention in chunk 3 turned out to be worse than no mention. It comes from
this sentence:

> The nearest full hospital is in Brightwater; there is a minor injuries unit
> locally with limited hours.

That sentence appears in **9 of my 14 documents**. It is regional boilerplate, so
it is not evidence that the chunk is about Brightwater — it pulls the other eight
towns' guides toward any question that names Brightwater.

Prefixing the `# ` title line fixes both things at once, and it also fixes the
floor: the shortest piece a pure section split produces is the 123-character
preamble of `guide_accessibility.md`, which is under criterion 4's floor of 150.
With the title line in front it is 173, and criterion 4 passes. Chunk length is
therefore measured including that prefix, which is why the ceiling has to leave
room for it — the longest title line in the corpus is 49 characters.

### Where 900 comes from

The longest chunk this rule can produce is 761 characters: the 711-character
`## Straightforward` section of `guide_accessibility.md` plus its title line. So
the ceiling is squeezed from both sides:

- It must be **above 761**, or that section gets split (see below for why I want
  it whole).
- It must be **at or below 1000**, because criterion 4 in `criteria.md` names
  1000 as the ceiling.

I checked the embedding limit rather than assuming it. `all-MiniLM-L6-v2` takes
256 word-piece tokens and silently drops the rest; Chroma sets that limit at load
time (`onnx_mini_lm_l6_v2.py`) even though the bundled `tokenizer.json` says 128.
Running my own text through that tokenizer, my densest chunk is 3.98 characters
per token and the corpus averages 4.57, so 256 tokens is about **1,020
characters** at worst for this corpus. None of my 94 chunks comes close: the
longest, at 761 characters, is 162 tokens, and **0 of 94** exceed 256.

So the binding constraint is my own criterion 4 at 1000, not the model at ~1020.

That leaves a band of 764 to 1000, and three candidates in it:

| Ceiling | Margin above 761 | Margin below 1000 |
|---|---|---|
| 800 | 39 | 200 |
| **900** | **139** | **100** |
| 1000 | 239 | **0** |

**800 leaves 39 characters** — about half a sentence. I decided below that
`## Straightforward` should stay in one piece; at 800 that decision breaks the
first time someone adds a sentence to that section, and it breaks silently.

**1000 puts the guard exactly where the failure is.** If the ceiling and
criterion 4's limit are the same number, a chunk that reaches the ceiling is
already at the criterion's boundary, with no room for an off-by-one in how I
count length.

900 leaves margin on both sides. I should be straight about what this choice is
worth today: nothing reaches 761, so 800, 900 and 1000 all produce identical
output on the corpus as it stands. The ceiling is a guard, not a driver — the
"section too long, split further" path in my code never runs here. I am choosing
for robustness, not for current behaviour.

### The longest section stays in one piece

`## Straightforward` in `guide_accessibility.md` is 761 characters with the title
line and covers three towns at once:

```
## Straightforward
**Thornby Wells** is the easiest town in the region...
**Marchwood** has a modern tram network with level boarding...
**Brightwater** is level along the river and through the centre...
```

I could have split it again on the `**Bold**` entries to get one town per chunk.
I chose not to. The section exists to compare the towns, and a question like
"which towns are easiest to get around with limited mobility?" is answered by
the whole section and not by any one third of it. The cost is that a question
about a single town's accessibility gets an embedding diluted by two other
towns, and I would rather carry that cost than lose the comparison.

### Why overlap is 0

Overlap exists to repair the damage of cutting at an arbitrary offset: a
sentence that straddles the boundary ends up complete in neither chunk. Section
headings are already meaning boundaries, so nothing straddles them and there is
no damage to repair. The starter's 120 characters would only duplicate text.

Overlap could not have solved the problem it looks like it should solve, either.
Borrowing 120 characters from the end of the previous section does not bring the
town name with it — that is what the title prefix is for, and the two are not
interchangeable.

`config.CHUNK_OVERLAP` is kept only for the case where one section overruns the
900 ceiling and has to be split mid-topic. Nothing in this corpus triggers it.

### Known problem I am not fixing in this unit

The `## Practical notes` section is identical, word for word, in 9 of my 14
documents — the same 277 characters. Splitting on sections therefore produces
nine near-identical chunks that differ only in their title line, all competing
for the same query. I noticed it while tracing the "Brightwater" mentions above
and left it alone, because fixing it means deciding whether shared regional text
should be indexed once or fourteen times, and that is a bigger change than
Milestone 3 asks for.

### The baseline is still in the file

`fallback_split` stays in `chunker.py` with its defaults pinned at the starter's
own 800 and 120 rather than reading `config.py`, so that changing the chunking
numbers above does not quietly move the baseline I am comparing against.

## Sample Chunks

Printed by `python app.py chunks -n 5`, which samples across the corpus rather
than taking five in a row.

**Chunk 1** — source: `guide_accessibility.md#0` — produced by: `chunker.py::split_documents`

```
# Getting around the region with limited mobility
An honest assessment rather than a promotional one. Some of these places are
difficult and it is better to know in advance.
```

**Chunk 2** — source: `guide_corry_vale.md#5` — produced by: `chunker.py::split_documents`

```
# Corry Vale
## Where to stay

Perhaps thirty beds in the entire valley, spread across two pubs and a handful of farmhouse rooms. In summer these are booked months ahead. Camping is permitted on two marked fields and nowhere else.
```

**Chunk 3** — source: `guide_givens_mill.md#2` — produced by: `chunker.py::split_documents`

```
# Givens Mill
## Getting around

Everything is on one street along the river. The mill is at one end and the church at the other, eight minutes apart. The riverside path continues in both directions for as far as you want to walk.
```

**Chunk 4** — source: `guide_kestrelford.md#4` — produced by: `chunker.py::split_documents`

```
# Kestrelford
## What to see

The market square on a Saturday morning is the main event and has run continuously since the 1400s. The parish church has a 13th-century tower you can climb for £2. The old trackbed walk runs six miles to the next village along an easy gradient and is the best half-day here.
```

**Chunk 5** — source: `guide_pellew_sands.md#6` — produced by: `chunker.py::split_documents`

```
# Pellew Sands
## When to go

June and September for the beach without the crowds. July and August are busy and the town is at its most itself, for better and worse. Winter is bleak, largely closed, and has a following among people who like that sort of thing.
```

Each of these answers a question on its own, and each one names its town on the
first line. Chunk 5 is the clearest test of the title prefix: "June and September
for the beach without the crowds" is true of Pellew Sands and false of
Brightwater, and nothing in the section's own text says which.

For comparison, the same command on the starter's chunker returned chunks like
`guide_kestrelford.md#3`, which began `irts. The nearest full hospital is in
Brightwater` — 110 characters, opening on the back half of "outskirts", and
naming a town it is not about.

## Sample Answer

<!-- One complete question and answer, pasted as text, with the source line
     visible. Milestone 4. -->

**Question:**

**Answer:**

```
```

**My relevance cutoff:**

<!-- The number you set in config.py, and how you got there.

     You ran five questions your corpus covers and the five in OUT_OF_SCOPE
     that it clearly doesn't, and wrote down the best distance for each. What
     did those two groups look like? Where was the gap? Put the actual numbers
     here — the table below wants all ten rows.

     Milestone 4. -->

| Question | In corpus? | Best distance |
|---|---|---|
|  |  |  |

## How I Used AI

<!-- Two specific moments. For each: what you asked for, what came back, and
     what you changed about it.

     "I asked Claude to write the chunking function from my notes. It ignored
     the overlap, so I added that myself" is the level of detail we're after.
     "I used AI to help me code" is not.

     Milestone 5. -->

**1.**

**2.**

<!-- ── Stretch features ─────────────────────────────────────────────────────
     Doing one? Say so here BEFORE you start. A feature this README never
     claims earns nothing.
     ───────────────────────────────────────────────────────────────────────── -->

---

# Unit 2

<!-- These sections get ADDED to what's already above. Don't delete or rewrite
     unit 1 — the point is that someone can see what you said before you knew
     how it went. -->

## Run Log — Before

<!-- Your five criteria, three runs each. `python run_eval.py --label before`
     runs the questions, puts the OUT_OF_SCOPE ones through the gate, and
     writes it all into results/ for you. Targets come from criteria.md; the
     verdict column is your call.

     Criterion 3 is measured in one deterministic pass rather than three, so
     the same number goes in all three run columns. That's correct, not lazy.

     Milestone 1. -->

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Retrieved chunk contains the answer | 4 of 5 |  |  |  |  |
| 2. Every answer names a source | 5 of 5 |  |  |  |  |
| 3. Gate stops out-of-corpus questions | 4 of 5 |  |  |  |  |
| 4. | | | | | |
| 5. | | | | | |

<!-- Underneath, paste the REAL output for each criterion from one of your
     runs — the actual text your system produced, not a description of it.
     Name the file and function that produced it. -->

## Verdicts

<!-- MET or MISSED for each of the five, against the target you wrote last
     unit — not a new one. Plus a sentence on how you decided. That sentence
     matters most where it was close.

     If your target said 4 of 5 and your runs came out 4, 3, 4, that's a MISS.
     The target has to hold, not show up occasionally.

     Milestone 2. -->

| # | Criterion | Verdict | How I decided |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |

## Diagnoses

<!-- For each miss: which stage caused it, and how. The stage alone isn't
     enough — you need the mechanism.

     Not a diagnosis: "Question 3 didn't work."
     A diagnosis:     "Question 3 asks about laundry costs. The answer is in
                       one sentence that got split across two chunks, so
                       neither chunk on its own contains it."

     The five stages: loading → chunking → embedding → retrieval → generation.

     Look for a pattern. If three misses all ask about numbers, that's one
     problem, not three.

     Missed nothing? Say so, then say honestly whether your targets were set
     low, and which one you'd tighten and to what.

     Milestone 3. -->

## The Improvement

**What I changed:**

**Why I picked it:**

<!-- Connect it to a specific diagnosis above in one sentence. If you can't,
     you picked a fix because it sounded impressive. -->

### Run Log — After

<!-- Same format, same five criteria, three runs each.
     `python run_eval.py --label after` -->

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Retrieved chunk contains the answer | 4 of 5 |  |  |  |  |
| 2. Every answer names a source | 5 of 5 |  |  |  |  |
| 3. Gate stops out-of-corpus questions | 4 of 5 |  |  |  |  |
| 4. | | | | | |
| 5. | | | | | |

**Did it help?**

<!-- Say plainly whether it did, and how you know. If it made things worse,
     say that — a change that backfired, honestly reported, earns full credit
     and is more interesting than one that worked. What matters is that you can
     tell.

     Milestone 4. -->

## What's Still Broken

<!-- For each criterion still missed after your fix: what you'd do about it,
     and why you stopped where you did.

     "I ran out of time" is fine if it's true. Pretending nothing is left is
     not.

     Milestone 5. -->

## What I'd Do Differently

<!-- Knowing what you know now — which of your five criteria would you write
     differently, and why?

     Milestone 5. -->
