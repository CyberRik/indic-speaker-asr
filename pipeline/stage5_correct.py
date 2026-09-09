#!/usr/bin/env python3
"""
Stage 5 -- LLM speaker-label correction.

Takes one Stage 4b condition and rewrites speaker labels using an open-weights
LLM reading the transcript as discourse. Writes a new condition that Stage 4c
scores exactly like any other, so baseline and improved rows sit side by side.

    python stage5_correct.py --cond indicconformer__pyannote31 --method rule
    python stage5_correct.py --cond indicconformer__pyannote31 --method llm
    python stage5_correct.py --cond indicconformer__pyannote31 --audit

Output goes to data/attrib/<asr>__<diar>+<method>/. Stage 4c splits a condition
name on the first `__`, so the suffix rides along on the diarizer and every
variant sorts directly under its baseline: `pyannote31`, `pyannote31+rule`,
`pyannote31+llm` read down the results table in that order.

Methods live in the METHODS registry and share one signature,
`(units, speakers, lang, llm, min_conf) -> ([(unit, speaker, confidence)],
rejections)`. Adding the acoustic variant later is one function and one registry
entry; run(), the manifest and Stage 4c need no change. That is deliberate --
Rule, LLM and LLM+acoustic have to be scorable side by side, and they can only
stay comparable if they differ in the edit proposal and in nothing else.


WHAT IT TARGETS, AND WHY NOT cpWER
----------------------------------
Measured attribution_cost (cpWER - DI-cpWER) is 0.01-0.33 across all twelve
Stage 4 conditions. There is no cpWER headroom, and an "improvement" of that
size would be noise. Stage 5 targets WDER, where the spread is real: 20.33 for
pyannote31 against 39.53 for sortformer_stream on the same IndicConformer words.

This also matches the goal. False splits, false merges and speaker swaps are
all speaker-structure errors; none of them is a text error. Stage 5 never edits
a word.


THE EDIT SPACE IS RELABEL, AND ONLY RELABEL
-------------------------------------------
The model may say "unit 7 belongs to Speaker_A". It may not add a speaker,
move a boundary, or touch text.

That is not a limitation, it is the whole design:

  * cpWER, DI-cpWER and WDER are functions of the word -> speaker map alone, so
    relabel is a COMPLETE edit space for every metric this project reports.
  * WER ignores speakers entirely, so WER CANNOT MOVE. A Stage 5 run that
    changes WER by 0.01 is broken. That is a free tripwire and it is asserted
    here rather than left for someone to notice in the results table.

Units are contiguous same-speaker runs, split further at pauses longer than
GAP_SEC. The split matters: with maximal runs, relabel would fix false splits
and swaps but never a false merge, because two speakers inside one run cannot be
separated by relabelling the run. A real speaker change usually has a pause at
it, so splitting on pauses turns many false merges into two units that relabel
can then fix -- without asking the model to point at a word index inside 79%-WER
text, which it would do badly.

"Usually" is doing real work in that sentence, so `--audit` measures it rather
than asserting it. It reports how many units still span two reference speakers
after splitting -- the false merges relabel-only correction can never reach --
against how many splitting rescued. That number is the ceiling on this whole
stage, and it belongs in the writeup next to any improvement claimed.


ABSTENTION
----------
Each edit carries a confidence and anything below --min-conf is discarded. A
wrong relabel costs WDER twice, removing a correct word from one speaker and
adding a wrong one to another, while an abstention costs only the improvement
forgone -- so at 79-87% WER the bias should be conservative and the default
threshold is high. A missing confidence is treated as a rejection, not as
certainty: a model that ignores the schema is exactly the one whose edits should
not bypass the threshold. Both kinds of abstention are counted and reported.


THE RISK, STATED UP FRONT
-------------------------
These transcripts are 79-87% WER. Whether an LLM can recover discourse structure
from text that damaged is genuinely unknown until it is measured. The design
fails safe: no parse, no edits, and the condition scores identically to its
baseline. Every way the model can be ignored is counted in the manifest, so
"the LLM was overruled on N clips" is itself a reportable result.

Generation is greedy (do_sample=False). Stage 4 was bitten once by an unseeded
sampler making a benchmark irreproducible; the same mistake is not available
here.


THE RULE BASELINE
-----------------
`--method rule` relabels any unit shorter than MIN_UNIT_SEC to the neighbour it
is closer to. It costs no GPU and captures the single most common diarization
artefact -- a sliver of a turn dropped inside someone else's speech. Without it
"the LLM improved WDER" is unfalsifiable: the comparison that matters is against
a dumb method, not against doing nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from stage4_attribute import Manifest, write_json

# A unit break needs a pause this long. Below it, a label change is far more
# likely to be diarizer jitter than a real speaker change.
GAP_SEC = 0.5

# Units shorter than this are what --method rule rewrites.
MIN_UNIT_SEC = 1.0

# Above this fraction of units edited, the model is not correcting a transcript,
# it is rewriting one. Drop the whole clip's edits and record it.
#
# The floor matters as much as the fraction: on a 3-unit clip a single correct
# edit is 33% and would trip a bare percentage, so a clip has to exceed BOTH to
# count as rogue. Without it the guard fires hardest on the short clips where
# there is least to get wrong.
MAX_EDIT_FRAC = 0.30
MIN_ROGUE_EDITS = 3

# Units per prompt, and how many trailing units of the previous window are
# reshown as context. Context units are visible but not editable, so no unit is
# ever decided twice.
WINDOW = 25
CONTEXT = 4

# Per-unit character cap in the prompt. Long units are informative for a few
# words and then just spend budget.
UNIT_CHARS = 220

# An edit below this confidence is discarded. At 79-87% WER the model is reading
# heavily corrupted text, so the default is deliberately high: a wrong relabel
# costs WDER twice over -- it removes a correct word from one speaker and adds a
# wrong one to another -- while an abstention costs nothing beyond the
# improvement forgone. Conservative is the right bias here.
MIN_CONF = 0.7

MODEL = "Qwen/Qwen2.5-7B-Instruct"

LANG_NAME = {
    "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "pa": "Punjabi", "or": "Odia", "ur": "Urdu", "ne": "Nepali", "en": "English",
}


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

def build_units(words: list[dict], gap: float = GAP_SEC) -> list[dict]:
    """Attributed words -> editable units.

    A unit breaks on a speaker change, or on a pause longer than `gap` within
    one speaker. See the module docstring for why the pause break is load
    bearing rather than cosmetic.
    """
    units: list[dict] = []
    for i, w in enumerate(words):
        new = (
            not units
            or w["spk"] != units[-1]["spk"]
            or w["start"] - units[-1]["end"] > gap
        )
        if new:
            units.append({"spk": w["spk"], "start": w["start"], "end": w["end"],
                          "idx": [i]})
        else:
            units[-1]["end"] = w["end"]
            units[-1]["idx"].append(i)
    for u in units:
        u["text"] = " ".join(words[i]["w"] for i in u["idx"])
    return units


def render(units: list[dict], lo: int, hi: int, ctx: int) -> str:
    """Units [lo, hi) as prompt text, the first `ctx` of them marked context."""
    lines = []
    for i in range(lo, hi):
        u = units[i]
        text = u["text"][:UNIT_CHARS]
        mark = "  [context, do not edit]" if i < lo + ctx else ""
        lines.append(f'[{i}] {u["spk"]} ({u["start"]:.1f}-{u["end"]:.1f}s)'
                     f'{mark}: {text}')
    return "\n".join(lines)


PROMPT = """You are correcting speaker labels on an automatic transcript of a \
conversation in {lang}. The transcript came from a speech recogniser with a high \
error rate, so the words are unreliable -- judge by conversational structure, \
not by whether the text reads correctly.

The speakers present are: {speakers}

Each line is one unit: [index] speaker (start-end): text

{units}

Some units carry the wrong speaker. The three failures to look for:
- a false split: one person's continuous speech broken across two labels
- a false merge: one unit's label covering what are really two people
- a swap: two speakers' labels exchanged across a boundary

Signals that survive a bad transcript: a question and its answer are different \
speakers; a sentence continuing mid-clause across a label change is one speaker; \
a very short unit inside a long stretch of one speaker is usually that speaker.

Reply with JSON only, no other text:
{{"edits": [{{"unit": <index>, "speaker": "<one of the speakers above>", \
"confidence": <0.0 to 1.0>, "why": "<a few words>"}}]}}

Be conservative. Propose an edit only where the conversational structure makes \
the current label clearly wrong, and set confidence honestly -- below {minconf} \
if you are unsure, and the edit will be discarded rather than applied. Leaving a \
label alone costs nothing; a wrong change makes the transcript worse. If the \
labels look right, reply {{"edits": []}}.
"""


def build_prompt(units, lo, hi, ctx, speakers, lang, min_conf=MIN_CONF) -> str:
    return PROMPT.format(
        lang=LANG_NAME.get(lang or "", "an Indic language"),
        speakers=", ".join(speakers),
        units=render(units, lo, hi, ctx),
        minconf=f"{min_conf:.2f}",
    )


# --------------------------------------------------------------------------
# parsing -- every failure mode here is a silent no-op, never a crash
# --------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """First JSON object in a model reply, fenced or bare."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None

    # Unbalanced: the reply hit max_new_tokens mid-array. Every edit before the
    # cut is still well-formed and independently valid, so salvage them rather
    # than discarding the whole window -- otherwise the model's willingness to
    # explain itself is what loses its edits, which is a silly failure mode.
    tail = text.rfind("}")
    if tail > start:
        try:
            obj = json.loads(text[start:tail + 1] + "]}")
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


REJECTIONS = ("parse_fail", "bad_unit", "bad_speaker", "dup",
              "low_conf", "no_conf", "oom")


def parse_edits(reply: str, editable: range, speakers: set[str],
                min_conf: float = MIN_CONF) -> tuple[list, dict]:
    """Model reply -> [(unit, speaker, conf)], plus every way it was rejected.

    Confidence is required, not optional. A missing or unparseable score is
    counted as `no_conf` and the edit is dropped: treating it as high confidence
    would let a model that ignores the schema bypass the threshold entirely,
    which is precisely the model whose edits are least worth trusting. If
    `no_conf` dominates a run, the model is not following the format and the
    summary says so out loud rather than reporting a quiet zero.
    """
    bad = {k: 0 for k in REJECTIONS}
    obj = extract_json(reply)
    if obj is None or not isinstance(obj.get("edits"), list):
        bad["parse_fail"] = 1
        return [], bad

    seen: set[int] = set()
    out: list[tuple[int, str, float]] = []
    for e in obj["edits"]:
        if not isinstance(e, dict):
            bad["bad_unit"] += 1
            continue
        u, s = e.get("unit"), e.get("speaker")
        if not isinstance(u, int) or u not in editable:
            bad["bad_unit"] += 1
            continue
        if s not in speakers:
            # The model inventing a speaker is the failure that would quietly
            # wreck cpWER, so it is counted separately from a bad index.
            bad["bad_speaker"] += 1
            continue
        try:
            conf = float(e["confidence"])
        except (KeyError, TypeError, ValueError):
            bad["no_conf"] += 1
            continue
        if conf < min_conf:
            bad["low_conf"] += 1
            continue
        if u in seen:
            bad["dup"] += 1
            continue
        seen.add(u)
        out.append((u, s, conf))
    return out, bad


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------

def method_rule(units, speakers, lang, llm, min_conf):
    """Short units go to the neighbour they are closer to in time.

    No model, no GPU, no confidence -- a rule that is certain by construction
    reports 1.0 so the downstream shape matches the LLM's.
    """
    edits = []
    for i, u in enumerate(units):
        if u["end"] - u["start"] >= MIN_UNIT_SEC:
            continue
        prev_u, next_u = (units[i - 1] if i else None,
                          units[i + 1] if i + 1 < len(units) else None)
        cands = []
        if prev_u:
            cands.append((u["start"] - prev_u["end"], prev_u["spk"]))
        if next_u:
            cands.append((next_u["start"] - u["end"], next_u["spk"]))
        cands = [c for c in cands if c[1] != u["spk"]]
        if cands:
            edits.append((i, min(cands)[1], 1.0))
    return edits, {k: 0 for k in REJECTIONS}


class LLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        import os

        # Long Indic prompts produce large, variable attention buffers, and a
        # fragmented allocator fails to serve them even when the free total is
        # sufficient. Must be set before the first CUDA allocation.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")

        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"[env ] {model_name} (4-bit nf4) on {device}")

    def free(self) -> None:
        """Drop cached blocks after an OOM, keeping the model loaded."""
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def close(self) -> None:
        """Release the weights.

        A notebook cell that loads the model holds ~8.8 GiB for the life of the
        kernel, and the next `!python stage5_correct.py` is a SEPARATE process
        that then OOMs on a 14.6 GiB T4. Any in-process use must call this.
        """
        import gc

        import torch

        self.model = None
        self.tok = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, prompt: str, max_new_tokens: int = 900) -> str:
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy: a benchmark must reproduce
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def method_llm(units, speakers, lang, llm, min_conf):
    """Windowed pass over one clip's units.

    A window that will not fit in VRAM is halved and retried rather than
    allowed to kill the clip. Indic scripts tokenize far worse than Latin --
    often near one token per character -- so WINDOW units of text is a much
    bigger prompt here than the unit count suggests, and the long clips are
    exactly the ones worth correcting. Each OOM is counted, so a run that only
    survived by shrinking says so instead of looking clean.
    """
    edits: list[tuple[int, str, float]] = []
    bad = {k: 0 for k in REJECTIONS}
    lo = 0
    while lo < len(units):
        ctx = CONTEXT if lo else 0
        span = min(WINDOW, len(units) - lo)
        reply = None
        while True:
            try:
                reply = llm(build_prompt(units, lo, lo + span, ctx,
                                         speakers, lang, min_conf))
                break
            except Exception as exc:                       # noqa: BLE001
                if not _is_oom(exc):
                    raise
                bad["oom"] += 1
                llm.free()
                if span <= ctx + 2:
                    # Cannot shrink further. Skip the window rather than the
                    # clip: the units in it keep their baseline labels.
                    break
                span = max(ctx + 2, span // 2)

        hi = lo + span
        if reply is not None:
            got, b = parse_edits(reply, range(lo + ctx, hi), set(speakers),
                                 min_conf)
            edits += got
            for k in bad:
                bad[k] += b[k]
        if hi >= len(units):
            break
        lo = hi - CONTEXT
    return edits, bad


# Every method takes (units, speakers, lang, llm, min_conf) and returns
# ([(unit, speaker, confidence)], rejection counters). Adding the acoustic
# variant later means adding one function and one entry here -- nothing in
# run(), the manifest, or Stage 4c needs to know about it.
METHODS = {
    "rule": method_rule,
    "llm": method_llm,
}
NEEDS_GPU = {"llm"}


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def apply_edits(words: list[dict], units: list[dict],
                edits: list[tuple[int, str, float]]) -> list[dict]:
    """New word list with relabelled units. Text is untouched, by construction."""
    out = [dict(w) for w in words]
    for u_i, spk, conf in edits:
        for w_i in units[u_i]["idx"]:
            out[w_i]["spk"] = spk
            out[w_i]["relabelled"] = True
            out[w_i]["conf"] = conf
    return out


def check_text_unchanged(before: list[dict], after: list[dict]) -> None:
    """The invariant that makes WER a tripwire rather than a hope."""
    if len(before) != len(after):
        raise AssertionError(f"word count changed: {len(before)} -> {len(after)}")
    for a, b in zip(before, after):
        if a["w"] != b["w"] or a["start"] != b["start"] or a["end"] != b["end"]:
            raise AssertionError(f"word altered: {a!r} -> {b!r}")


# --------------------------------------------------------------------------
# false-merge audit -- DIAGNOSTIC ONLY, never touches the correction path
# --------------------------------------------------------------------------

def audit_clip(words, ref_turns, gap: float):
    """How much of this clip's false merging is reachable by relabel at all?

    A unit whose words belong to two different REFERENCE speakers cannot be
    fixed by relabelling it -- whichever label it gets, half its words are
    wrong. Splitting units at pauses is what rescues some of them, and this
    measures exactly how many:

      merged_words_maximal  words in a unit spanning >1 true speaker, with NO
                            pause splitting
      merged_words_split    the same after splitting at `gap` -- the residue
                            relabel-only correction can never reach
      freed                 the difference: words that splitting released into
                            separately-labellable units

    Measured in WORDS, and that is not a detail. Splitting a multi-speaker unit
    often yields two units that are each still multi-speaker, so the unit count
    can RISE while the situation improves: the two segmentations have different
    denominators and their unit counts are not comparable. Words are conserved
    under splitting, so they are.

    This reads the reference RTTM and is therefore an ORACLE measurement. It
    exists so the writeup can state the ceiling on relabel-only correction
    instead of implying there is none. Nothing it computes is fed to any model
    or used to choose an edit.
    """
    from stage4_attribute import attribute_word

    starts = [t[0] for t in ref_turns]
    max_dur = max((e - s for s, e, _ in ref_turns), default=0.0)
    truth = []
    for w in words:
        spk, _ov, _spans = attribute_word(w["start"], w["end"], ref_turns,
                                          starts, max_dur)
        truth.append(spk)

    out = {}
    for name, g in (("maximal", float("inf")), ("split", gap)):
        units = build_units(words, g)
        n_multi = n_words = 0
        for u in units:
            spks = {truth[i] for i in u["idx"] if truth[i] is not None}
            if len(spks) > 1:
                n_multi += 1
                n_words += len(u["idx"])
        out[f"merged_{name}"] = n_multi
        out[f"merged_words_{name}"] = n_words
        out[f"units_{name}"] = len(units)
    out["n_words"] = len(words)
    return out


def run_audit(cond: str, data: Path, gap: float, limit: int | None) -> bool:
    from stage4_attribute import load_turns

    src_root = data / "attrib" / cond
    ref_dir = data / "ref" / "rttm"
    if not src_root.is_dir():
        print(f"[skip] no condition at {src_root}")
        return False
    if not ref_dir.is_dir():
        print(f"[skip] no reference RTTMs at {ref_dir} -- the audit is scored "
              f"against ground truth and cannot run without it")
        return False

    clips = sorted(p.stem for p in src_root.glob("*.json"))
    clips = clips[:limit] if limit else clips
    tot = {}
    n = 0
    for clip_id in clips:
        turns = load_turns(ref_dir / f"{clip_id}.rttm")
        if not turns:
            continue
        src = json.loads((src_root / f"{clip_id}.json").read_text(encoding="utf-8"))
        a = audit_clip(src["words"], turns, gap)
        for k, v in a.items():
            tot[k] = tot.get(k, 0) + v
        n += 1

    if not n:
        print(f"[audit] {cond}: nothing to audit")
        return False

    wm, ws = tot["merged_words_maximal"], tot["merged_words_split"]
    freed = wm - ws
    pct = 100.0 * freed / max(wm, 1)
    nw = max(tot["n_words"], 1)
    print(f"\n[audit] {cond}  ({n} clips)  ORACLE DIAGNOSTIC, not a system result")
    print(f"  units    : {tot['units_maximal']:,} unsplit -> "
          f"{tot['units_split']:,} split at {gap:.1f}s")
    print(f"  words trapped in a unit spanning >1 true speaker:")
    print(f"    unsplit: {wm:>7,}  ({100.0 * wm / nw:5.2f}% of all words)")
    print(f"    split  : {ws:>7,}  ({100.0 * ws / nw:5.2f}% of all words)")
    print(f"    freed  : {freed:>7,}  ({pct:.1f}% of the unsplit total)")
    print(f"  CEILING  : {100.0 * ws / nw:.2f}% of words sit in a unit that no "
          f"relabel can fix")
    print(f"  (measured in words: unit counts are NOT comparable across the two "
          f"segmentations, since splitting a multi-speaker unit can yield two "
          f"of them)")
    return True


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run(cond: str, method: str, data: Path, model_name: str,
        limit: int | None, min_conf: float = MIN_CONF,
        shard: tuple[int, int] | None = None) -> bool:
    src_root = data / "attrib" / cond
    if not src_root.is_dir():
        print(f"[skip] no condition at {src_root}")
        return False
    if "+" in cond:
        print(f"[skip] {cond} is already a Stage 5 output; correcting a "
              f"correction is not a condition anyone can interpret")
        return False

    out_root = data / "attrib" / f"{cond}+{method}"
    manifest = Manifest(out_root / "manifest.jsonl")
    clips = sorted(p.stem for p in src_root.glob("*.json"))
    if shard:
        # Sharded BEFORE the done-filter, so each worker owns a fixed set of
        # clips no matter when it starts or how far the others have got. Two
        # workers must never be handed the same clip: they would race on the
        # same output path and double-count in the manifest.
        i, n_sh = shard
        clips = [c for k, c in enumerate(clips) if k % n_sh == i]
    pending = [c for c in clips if not manifest.done(c)]
    todo = pending[:limit] if limit else pending

    tag = f"{cond}+{method}" + (f"  [shard {shard[0]}/{shard[1]}]" if shard else "")
    print(f"\n[cond] {cond} -> {tag}")
    print(f"[plan] {len(clips)} clips: {len(clips) - len(pending)} done, "
          f"{len(pending)} pending, running {len(todo)} now")
    if not todo:
        return True

    llm = LLM(model_name) if method in NEEDS_GPU else None
    fn = METHODS[method]

    n_ok = n_fail = 0
    tot = {"units": 0, "proposed": 0, "applied": 0, "rogue": 0,
           **{k: 0 for k in REJECTIONS}}
    for n, clip_id in enumerate(todo, 1):
        try:
            src = json.loads((src_root / f"{clip_id}.json").read_text(encoding="utf-8"))
            words = src["words"]
            units = build_units(words)
            speakers = sorted({u["spk"] for u in units})

            edits, bad = fn(units, speakers, src.get("lang"), llm, min_conf)

            proposed = len(edits)
            # The guard exists to catch a MODEL that has stopped following
            # instructions. A deterministic rule cannot do that, and applying
            # it there silently cripples the baseline on exactly the noisiest
            # conditions -- 30 of 99 sortformer_stream clips were being dropped,
            # which is the comparison the rule is supposed to provide.
            rogue = (method in NEEDS_GPU
                     and proposed > MIN_ROGUE_EDITS
                     and proposed > MAX_EDIT_FRAC * len(units))
            if rogue:
                # Not a correction pass any more. Keep the baseline labels and
                # say so; a clip silently rewritten would be indistinguishable
                # from one the model genuinely improved.
                edits = []

            new_words = apply_edits(words, units, edits)
            check_text_unchanged(words, new_words)

            rec = dict(src)
            rec["words"] = new_words
            rec["stage5"] = {
                "method": method,
                "model": model_name if method in NEEDS_GPU else None,
                "min_conf": min_conf,
                "n_units": len(units),
                "n_proposed": proposed,
                "n_applied": len(edits),
                "n_words_relabelled": sum(1 for w in new_words if w.get("relabelled")),
                "rogue": rogue,
                **bad,
            }
            write_json(out_root / f"{clip_id}.json", rec)

            tot["units"] += len(units)
            tot["proposed"] += proposed
            tot["applied"] += len(edits)
            tot["rogue"] += int(rogue)
            for k in REJECTIONS:
                tot[k] += bad[k]

            manifest.append({"clip_id": clip_id, "status": "ok",
                             **rec["stage5"]})
            n_ok += 1
            print(f"[{n:3d}/{len(todo)}] {clip_id:44s} ok  "
                  f"{len(units):4d} units  {len(edits):3d} edits"
                  f"{'  ROGUE' if rogue else ''}")
        except Exception as exc:                       # noqa: BLE001
            n_fail += 1
            manifest.append({"clip_id": clip_id, "status": "fail",
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{n:3d}/{len(todo)}] {clip_id:44s} FAIL  "
                  f"{type(exc).__name__}: {exc}")

    pct = 100.0 * tot["applied"] / max(tot["units"], 1)
    print(f"\n[done] ok={n_ok} fail={n_fail}  {tot['units']:,} units, "
          f"{tot['applied']:,} relabelled ({pct:.2f}%), "
          f"{tot['rogue']} clips rogue")
    if method in NEEDS_GPU:
        print(f"[done] abstained: {tot['low_conf']} below conf {min_conf}, "
              f"{tot['no_conf']} with no confidence given")
        if tot["oom"]:
            print(f"[done] {tot['oom']} windows hit OOM and were retried at half "
                  f"width -- results are still valid, but the GPU is the "
                  f"binding constraint on the long clips")
        print(f"[done] rejected: {tot['parse_fail']} unparseable replies, "
              f"{tot['bad_unit']} bad index, {tot['bad_speaker']} invented "
              f"speaker, {tot['dup']} duplicate")
        offered = tot["applied"] + tot["low_conf"] + tot["no_conf"]
        if tot["no_conf"] > max(tot["applied"], 5):
            # Not a conservative model -- a model ignoring the schema. Said
            # loudly, because the run otherwise looks like a clean abstention.
            print("[WARN] most edits arrived without a confidence score, so "
                  "they were dropped. The model is not following the reply "
                  "format; fix the prompt before reading anything into this "
                  "run's WDER.")
        elif offered and tot["applied"] == 0:
            print("[WARN] every proposed edit was abstained away. This scores "
                  "identically to the baseline by construction -- lower "
                  "--min-conf or accept that the model has no usable signal.")
    return n_fail == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cond", nargs="+", required=True,
                    help="Stage 4b condition(s), e.g. indicconformer__pyannote31")
    ap.add_argument("--method", choices=tuple(METHODS), default="llm")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", metavar="I/N",
                    help="process only clips where index %% N == I. Run one "
                         "worker per GPU with CUDA_VISIBLE_DEVICES to halve "
                         "wall-clock: the shards are disjoint, so the two "
                         "workers never touch the same output file.")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF,
                    help=f"discard edits below this confidence "
                         f"(default {MIN_CONF})")
    ap.add_argument("--audit", action="store_true",
                    help="ORACLE DIAGNOSTIC: report how much false merging "
                         "pause splitting recovers, and how much no relabel "
                         "can reach. Reads the reference RTTM, writes no "
                         "condition, and never influences an edit.")
    args = ap.parse_args()

    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        shard = (i, n)

    clean = True
    for c in args.cond:
        if c.endswith("__ref"):
            # Correcting oracle labels would improve a diagnostic that is
            # perfect by construction. It cannot mean anything.
            print(f"[skip] {c} is the oracle condition")
            continue
        if args.audit:
            clean &= run_audit(c, args.data, GAP_SEC, args.limit)
        else:
            clean &= run(c, args.method, args.data, args.model, args.limit,
                         args.min_conf, shard)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
