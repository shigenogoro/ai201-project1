# How to Write Acceptance Criteria for a RAG System

My reference for writing criteria in future projects. Covers **what** to test,
**how** to test it, **why** it's worth testing, and **how many** criteria to
write.

---

## 1. What makes a criterion a criterion

A criterion is a claim that can come out **false**. If there's no way to fail
it, it isn't one.

Three parts, all required:

| Part | Example | Fails without it |
|---|---|---|
| **Observable** | "the retrieved chunks include one containing the answer" | "retrieval works" — opinion, not observation |
| **Threshold** | "for at least 4 of my 5 test questions" | "usually works" — can't be scored |
| **Procedure** | "run the 5 questions in `questions.py`, check top-5 by hand" | Two people get two different scores |

The test: **could someone else run this and get my number?** If the answer
depends on my mood that day, rewrite it.

### The "why this target" is half the work

A number with no reasoning behind it is a guess dressed up as rigor. The
reasoning has to point at something real:

- Bad — "80% seemed reasonable."
- Good — "One of my five questions is about laundry costs, and only two
  documents mention it. I expect that one to be the hard one, so I set the bar
  at 4 of 5 rather than 5 of 5."

The target should be **reachable but not guaranteed**. If I can't imagine
missing it, it's measuring nothing. A criterion I set at 4 of 5 and hit 3 of 5
is more useful than one I set at 1 of 5 and hit — the miss is where the
diagnosis lives.

---

## 2. What to test, stage by stage

A RAG pipeline has five stages: **loading → chunking → embedding → retrieval →
generation**. Every failure traces back to exactly one of them, so organizing
criteria by stage means a missed criterion points straight at the culprit.

### Loading

**What to test:** every document made it in, and its text survived intact.

**How:** print the document count and compare against the file count in the
corpus folder. Spot-check one document with awkward formatting (tables, code
blocks, unusual characters) and read its loaded text.

**Why:** this is the cheapest failure to miss and the most expensive to
misdiagnose. If a document silently failed to load, every downstream criterion
about it will fail, and I'll spend hours blaming retrieval.

Usually **not** worth a formal criterion — it's a precondition I check once. It
becomes a criterion if my corpus has files in mixed formats.

### Chunking

**What to test:**

| Property | Criterion shape | How to measure |
|---|---|---|
| **Readability / boundary integrity** | "At least 4 of 5 sampled chunks read as a complete thought, with no sentence cut in half at either end." | `python app.py chunks -n 5`, read them, count |
| **Size distribution** | "No chunk is shorter than 200 characters." | `chunker.describe()` prints shortest/longest/average |
| **Semantic coherence** | "At least 4 of 5 sampled chunks stay within a single section, rather than splicing two unrelated sections together." | Sample chunks, compare against source headings |
| **Retrieval-relevant density** | "In at least 4 of 5 test questions, the answer sits whole inside one retrieved chunk rather than split across two adjacent ones." | Run the questions, look at where the answer text falls relative to chunk boundaries |

**Why this stage matters most:** chunking is the only stage where information
can be **destroyed**. Embedding and retrieval can rank badly and be fixed by
tuning; a sentence cut in half by the chunker is gone, and no amount of
retrieval tuning brings it back. That's why chunk criteria earn their place
even though they feel indirect.

Readability and density are the two I'd pick if I only get two — they catch
different failures. Readability catches *mangling*; density catches
*fragmentation*.

### Embedding

**What to test:** whether semantically similar things land near each other.

**How:** take a question and its known-correct chunk, record the distance.
Take the same question and an unrelated chunk, record that distance. The gap
between them is what the whole system runs on.

**Why:** if there's no gap, nothing downstream can work — retrieval is ranking
noise and the relevance gate has nothing to cut on. Worth measuring once
before writing any retrieval criteria, because it tells me whether a
disappointing retrieval score is a retrieval problem or an embedding problem.

Rarely its own criterion — it shows up as the evidence behind the gate's
threshold.

### Retrieval

**What to test:**

| Property | Criterion shape |
|---|---|
| **Hit rate** | "For at least 4 of 5 questions, the retrieved chunks include one containing the answer." |
| **Rank quality** | "For at least 3 of 5 questions, the *top* result contains the answer." |
| **The relevance gate** | "When I ask a question my documents clearly don't cover, the gate refuses in at least 4 of 5 tries." |

**How:** a fixed question set, written down before running anything, scored the
same way every time. Both halves matter — questions the corpus **covers** and
questions it **clearly doesn't**. The second set is the one people forget.

**Why the gate needs its own criterion:** a system that answers everything
confidently is worse than one that refuses sometimes, because a confident
wrong answer is indistinguishable from a right one until someone gets hurt by
it. Hit rate alone can't catch this — it only looks at questions that *should*
succeed.

**Setting the gate threshold:** run both groups of questions, write down the
best distance for each, and look for the gap. If in-corpus questions cluster
around 0.35 and out-of-corpus ones around 0.75, the cutoff goes in the gap.
If the two groups overlap, that's a real finding and worth writing down — it
means no threshold will cleanly separate them, and the criterion target should
reflect that.

### Generation

**What to test:**

| Property | Criterion shape | Why |
|---|---|---|
| **Attribution present** | "Every answer names at least one source document." | Unverifiable answers are unusable |
| **Attribution *correct*** | "For at least 4 of 5 answers, the named source actually contains the claim." | Stricter and much more interesting — a system can cite reliably and cite *wrongly* |
| **Grounding / no fabrication** | "No answer contains a specific fact (number, name, date) absent from the retrieved chunks." | This is the failure that destroys trust |
| **Refusal honesty** | "When retrieval returns nothing relevant, the answer says so rather than answering from model knowledge." | Catches the model bypassing my corpus entirely |

**How:** read the answer next to the chunks that produced it. Check each
specific claim against the chunk text. Tedious, unavoidable, and where the real
bugs surface.

**Why "names a source" is a weak criterion on its own:** it's nearly free to
pass — the prompt can force a citation whether or not it's the right one. If I
want the criterion to mean something, test that the citation is *correct*, not
merely present.

### Cross-cutting (not tied to one stage)

- **Determinism** — "The same question asked three times gives the same
  verdict." *Why:* if runs disagree, every other number I report is noise, and
  that's worth knowing before I build a table of them.
- **Latency** — "Median answer takes under 5 seconds." *Why:* only if I
  actually care; don't pad the list with it.
- **Cost** — tokens or API calls per question. Same caveat.

---

## 3. How to actually run the test

**Fix the question set before running anything.** Write the questions down
first. Questions invented after seeing results are questions chosen to pass.

**Cover the shapes that break things.** A good five-question set isn't five
easy ones:
- one answered by a single sentence in one document
- one whose answer spans a paragraph
- one whose answer appears in two documents (do they agree?)
- one about something mentioned in only one place — the hard one
- one asking for a specific number or name — fabrication bait

**Run it more than once.** Three runs is the usual minimum. The target has to
**hold**, not show up occasionally: 4 of 5 targeted with runs of 4, 3, 4 is a
**MISS**, not a pass with noise.

Deterministic checks (like a gate threshold comparison) only need one pass —
the same number goes in all three columns, and that's correct, not lazy.

**Record the raw output, not a description of it.** "Retrieval worked well"
is unusable next month. The actual chunk text is evidence.

**Score against the original target, never a revised one.** Missing a target
is data. Lowering the target after missing it deletes the data.

---

## 4. When a criterion can be revised

There's one legitimate reason: the criterion **couldn't be measured**.

> I couldn't judge "the chunks include one that contains the answer" the same
> way twice — I scored two questions differently on Monday than on Wednesday.
> Revised to "the top three results contain the answer," which I can actually
> check.

That's a measurement problem, and fixing it is real work.

This is **not** a revision:

> I said 4 of 5 but got 2 of 5, so 2 of 5 is more realistic.

A number I missed stays where it is, gets diagnosed, and gets a fix attempted.
Keep the original visible and add the revision underneath — the point is that
someone can see what I claimed before I knew the answer.

### Diagnosing a miss

Name the stage **and the mechanism**. The stage alone isn't a diagnosis.

- Not a diagnosis: "Question 3 didn't work."
- A diagnosis: "Question 3 asks about laundry costs. The answer is one
  sentence that got split across two chunks, so neither chunk on its own
  contains it." → chunking stage, fragmentation mechanism.

Look for patterns before fixing anything. Three misses that all ask about
numbers is **one** problem, not three.

---

## 5. How many criteria do people use

**Short answer: 3–7. Five is the common landing spot.**

| Context | Count | What they use |
|---|---|---|
| Course / milestone projects | 5 | Usually retrieval, attribution, refusal, plus two chosen |
| RAGAS (standard RAG eval library) | 4 core | Faithfulness, answer relevancy, context precision, context recall |
| Production RAG teams | 5–8 | The core four plus latency, cost, refusal rate, safety |
| Research papers | 2–4 headline | Plus ablations; headline metrics stay few on purpose |

**Why the number lands there:**

- **Below 3** leaves blind spots. Retrieval and generation fail in unrelated
  ways; one number can't see both.
- **Above 7** and they stop being decision-makers. If 8 criteria mean some
  always fail, "did it work?" has no answer anymore, and I start quietly
  ignoring the inconvenient ones.
- Criteria are meant to be **checked repeatedly**, every run, every change.
  The count is limited by what I'll actually keep scoring by hand — not by
  what's theoretically worth measuring.

**A good spread of five:**

1. Retrieval finds the right material
2. Generation is grounded in it
3. The system refuses when it should
4. Something about the pipeline stage I'm least sure of
5. Something I personally care about getting right

Criteria 1–3 are the skeleton — nearly every RAG evaluation has something in
each of those slots under some name. Slots 4 and 5 are where the project's
specific risks go.

**Don't add a criterion just for coverage.** A fourth chunking criterion that
restates the other three isn't thoroughness, it's padding. Two sharp
criteria that can genuinely fail beat five vague ones that can't.

---

## 6. Diagnostics vs. criteria

Not everything worth measuring should be a criterion. Some things are better
kept as **diagnostics** — tools I reach for when a criterion fails, to explain
*why*:

- Overlap effectiveness (does a sentence cut at the end of chunk N appear
  whole at the start of N+1?)
- Redundancy in the top-k set (are two retrieved chunks 80% the same text?)
- Chunk-size ablation (re-run with half/double the chunk size and compare hit
  rate — the strongest evidence that a size is tuned, and too heavy to run
  every time)
- Distance distributions for in- vs. out-of-corpus questions

**The distinction:** a criterion is something I commit to a number on and check
every run. A diagnostic is something I run once, or when something breaks.
Mixing them up bloats the criteria list and makes every run expensive.

---

## 7. Checklist before committing to a set of criteria

- [ ] Each one names a number, a count, a rate, or something plainly observable
- [ ] Each one could come out false — none are guaranteed passes
- [ ] Someone else could run my procedure and get my number
- [ ] Each "why this target" points at something in **my** corpus or pipeline,
      not at a round number
- [ ] Retrieval, generation, and refusal are each covered by at least one
- [ ] The question set is written down and fixed before the first run
- [ ] The set includes questions the corpus does **not** cover
- [ ] No two criteria are measuring the same failure
- [ ] Between 3 and 7 total
- [ ] I'd be willing to score all of them by hand, three times, more than once
