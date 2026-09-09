#!/usr/bin/env python3
"""
Stage 4c -- ASR + attribution scoring (CPU-only).

Scores every (asr, diar) condition from Stage 4b against the Stage 2 reference
transcripts and writes:

    data/results/asr_per_clip.csv
    data/results/asr_summary.csv
    data/results/asr_summary.md

Four metrics, chosen so that the errors decompose rather than pile into one
number:

  WER       speaker-agnostic, reference text vs hypothesis text. Pure ASR
            quality; identical for every diar condition of the same ASR, which
            is a useful invariant to eyeball -- if it moves, something leaked.
  cpWER     permutation-optimal per-speaker WER. The headline number.
  DI-cpWER  diarization-invariant cpWER. meeteval 0.4.3 ships only the greedy
            approximation, whose error count is an upper bound on the optimal,
            so `cpWER - DI-cpWER` is a LOWER bound on the attribution cost.
  WDER      word diarization error rate, hand-rolled: meeteval has no WDER.

  cpWER - DI-cpWER  isolates what wrong speaker attribution cost, which is
                    exactly the quantity Stage 5 is trying to reduce.

CORPUS NUMBERS ARE ERROR-WEIGHTED: sum(errors) / sum(reference words), not the
mean of per-clip rates. The unweighted mean is reported beside it because it is
the one people publish by accident -- it lets a 50 s clip outweigh a 30 min one.

TWO SUBSETS, ALWAYS BOTH. `sortformer` has 74 of 99 clips and the 25 it is
missing are all long ones, with more speakers and more turn changes than
average. A corpus number over 74 clips is not comparable to one over 99, so
every table reports `all` (each condition over what it has) and `common` (the
clips every condition produced), and only `common` is a fair cross-system
comparison.

    python stage4_score.py --data data

Normalisation is defined in one place, `normalise()`, and applied identically to
reference and hypothesis. WER is extremely sensitive to it, so it is a stated
policy rather than an accident: Unicode NFC, bracketed annotations removed,
punctuation stripped (including the Devanagari danda), case folded, whitespace
collapsed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # checked in main(), so the message arrives in second one
    Levenshtein = None

ORACLE = "ref"

# Bracketed annotations: "(inaudible)", "[laughs]", "<laughter>". Reference-only
# conventions that the ASR cannot produce, so scoring them would be scoring the
# annotator. The angle form matters: PUNCT strips "<" and ">" as punctuation, so
# without this the tag survives as a bare word. 666 of them across the corpus --
# <unintelligible> 263, <noise> 163, <laughter> 146, <vocalization> 81,
# <background_speech> 12, <uhhh> 1 -- each an unmatchable reference token.
BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|<[^>]*>")

# Punctuation across the scripts in this corpus, plus the Devanagari danda and
# its double form, which are sentence terminators rather than words.
PUNCT = re.compile(r"[!-/:-@\[-`{-~।॥‐-‧‰-⁞¡-¿]")


def normalise(text: str) -> list[str]:
    """Text -> comparable word list. The single definition of 'a word' here."""
    text = unicodedata.normalize("NFC", text)
    text = BRACKETS.sub(" ", text)
    text = PUNCT.sub(" ", text)
    return text.casefold().split()


# --------------------------------------------------------------------------
# meeteval, resolved defensively
# --------------------------------------------------------------------------

def resolve_meeteval():
    """Return (cpwer_fn, di_cpwer_fn, siso_fn).

    meeteval has moved these between `meeteval.wer` and `meeteval.wer.wer.*`
    across releases, and DI-cpWER is recent. Rather than fail with an
    AttributeError three hours into a session, try the known spellings and, if
    none match, say exactly what the installed version does expose.
    """
    try:
        import meeteval.wer as mw
    except ImportError:
        raise SystemExit("meeteval is not installed:  pip install meeteval")

    def pick(*names):
        for n in names:
            fn = getattr(mw, n, None)
            if fn is not None:
                return fn
        return None

    cp = pick("cp_word_error_rate", "cpwer")
    siso = pick("siso_word_error_rate", "wer", "word_error_rate")
    # 0.4.3 ships DI-cpWER only in the greedy form. Greedy assigns words without
    # searching every permutation, so its error count is an upper bound on the
    # optimal DI-cpWER -- which makes `cpWER - DI-cpWER` a LOWER bound on the
    # cost of wrong attribution: the real cost is at least this, never less.
    # Stated because Stage 5's entire claim is a reduction in that quantity.
    di = pick("di_cp_word_error_rate", "dicpwer", "di_cpwer",
              "greedy_di_cp_word_error_rate", "greedy_dicpwer")

    missing = [n for n, f in (("cpWER", cp), ("DI-cpWER", di), ("WER", siso)) if f is None]
    if missing:
        avail = sorted(n for n in dir(mw) if "error_rate" in n or n.endswith("wer"))
        raise SystemExit(f"meeteval is missing {missing}. Installed version exposes: {avail}")

    di_name = getattr(di, "__name__", "?")
    print(f"[env ] meeteval bindings: cpWER={getattr(cp, '__name__', '?')}, "
          f"DI-cpWER={di_name}, WER={getattr(siso, '__name__', '?')}")
    if "greedy" in di_name:
        print("[env ] DI-cpWER is the greedy approximation, so attribution_cost "
              "(cpWER - DI-cpWER) is a LOWER bound on the true attribution cost.")
    return cp, di, siso


# meeteval's entry points do not all take the same input shape: cpWER accepts a
# {speaker: text} mapping, while greedy DI-cpWER wants SegLST (a list of segment
# dicts) and raises TypeError on a mapping. Rather than hard-code which is which
# -- it has changed between releases -- try the mapping, fall back to SegLST, and
# remember the answer per function so the probe costs one call, not 99.
_CALL_STYLE: dict[str, str] = {}


def _seglst(mapping: dict[str, str], clip_id: str) -> list[dict]:
    return [{"session_id": clip_id, "speaker": spk, "words": text}
            for spk, text in mapping.items()]


def call_wer(fn, ref: dict[str, str], hyp: dict[str, str], clip_id: str):
    name = getattr(fn, "__name__", repr(fn))
    style = _CALL_STYLE.get(name)

    if style in (None, "mapping"):
        try:
            out = fn(ref, hyp)
            _CALL_STYLE[name] = "mapping"
            return out
        except (TypeError, AttributeError, KeyError):
            if style == "mapping":
                raise

    out = fn(_seglst(ref, clip_id), _seglst(hyp, clip_id))
    if _CALL_STYLE.get(name) != "seglst":
        _CALL_STYLE[name] = "seglst"
        print(f"[env ] {name} takes SegLST, not a speaker mapping")
    return out


def rate(err) -> tuple[int, int]:
    """meeteval ErrorRate -> (errors, reference length), version-tolerantly."""
    errors = getattr(err, "errors", None)
    length = getattr(err, "length", None)
    if errors is None:  # older releases expose only the ratio
        errors, length = round(err.error_rate * err.length), err.length
    return int(errors), int(length)


# --------------------------------------------------------------------------
# WDER -- hand-rolled, because meeteval has no WDER
# --------------------------------------------------------------------------

def align_pairs(ref_words: list[str], hyp_words: list[str]):
    """Yield (ref_index, hyp_index) for every correct or substituted word.

    Insertions and deletions are skipped: WDER is defined over words that exist
    on both sides, since a word the ASR never produced has no speaker to be
    wrong about. Uses rapidfuzz's Levenshtein opcodes (C++); a pure-Python DP
    would be minutes per clip at 3000 words a side.
    """
    for op in Levenshtein.opcodes(ref_words, hyp_words):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                yield op.src_start + k, op.dest_start + k
        elif op.tag == "replace":
            # Pair positionally; the ragged tail is insertions or deletions.
            for k in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                yield op.src_start + k, op.dest_start + k


def speaker_mapping(ref_by_spk: dict[str, list[str]],
                    hyp_by_spk: dict[str, list[str]],
                    cp_result) -> dict[str, str]:
    """hyp speaker -> ref speaker, under cpWER's optimal permutation.

    Prefers the assignment meeteval already computed, so WDER and cpWER agree on
    who is who. Falls back to a Hungarian match on shared word counts if the
    installed version does not expose it -- reported when it happens, because a
    different permutation makes the two metrics tell slightly different stories.
    """
    assignment = getattr(cp_result, "assignment", None)
    if assignment:
        out = {}
        for pair in assignment:
            # meeteval orders these (reference, hypothesis).
            r, h = pair[0], pair[1]
            if h is not None and r is not None:
                out[str(h)] = str(r)
        if out:
            return out

    from collections import Counter

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    refs, hyps = sorted(ref_by_spk), sorted(hyp_by_spk)
    gain = np.zeros((len(refs), len(hyps)))
    for i, r in enumerate(refs):
        rc = Counter(ref_by_spk[r])
        for j, h in enumerate(hyps):
            hc = Counter(hyp_by_spk[h])
            gain[i, j] = sum((rc & hc).values())
    ri, hj = linear_sum_assignment(-gain)
    return {hyps[j]: refs[i] for i, j in zip(ri, hj)}


def wder(ref_seq, hyp_seq, mapping: dict[str, str]) -> tuple[int, int]:
    """(mis-attributed words, scorable words).

    WDER = (S_IS + C_IS) / (S + C): among words that align -- correct or
    substituted -- the fraction whose speaker is wrong. A hypothesis speaker
    with no counterpart in the mapping counts as wrong, which is the honest
    reading: the word was attributed to somebody who does not exist.
    """
    ref_words = [w for w, _ in ref_seq]
    hyp_words = [w for w, _ in hyp_seq]
    wrong = total = 0
    for i, j in align_pairs(ref_words, hyp_words):
        total += 1
        if mapping.get(hyp_seq[j][1]) != ref_seq[i][1]:
            wrong += 1
    return wrong, total


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_reference(path: Path):
    """ref/segments/<clip>.json -> (by_speaker words, time-ordered (word, spk))."""
    d = json.loads(path.read_text(encoding="utf-8"))
    by_spk: dict[str, list[str]] = {}
    seq: list[tuple[str, str]] = []
    for seg in sorted(d["segments"], key=lambda s: (s["start"], s["index"])):
        spk = str(seg["speaker"])
        words = normalise(seg["text"])
        by_spk.setdefault(spk, []).extend(words)
        seq.extend((w, spk) for w in words)
    return by_spk, seq


def load_hypothesis(path: Path):
    """attrib/<cond>/<clip>.json -> (by_speaker words, time-ordered (word, spk))."""
    d = json.loads(path.read_text(encoding="utf-8"))
    by_spk: dict[str, list[str]] = {}
    seq: list[tuple[str, str]] = []
    for w in d["words"]:
        spk = str(w["spk"])
        for token in normalise(w["w"]):  # a "word" may normalise to 0 or 2
            by_spk.setdefault(spk, []).append(token)
            seq.append((token, spk))
    return by_spk, seq, d


# --------------------------------------------------------------------------

def score_clip(ref_by_spk, ref_seq, hyp_by_spk, hyp_seq, fns, clip_id="clip") -> dict:
    cp_fn, di_fn, siso_fn = fns

    ref_txt = {k: " ".join(v) for k, v in ref_by_spk.items()}
    hyp_txt = {k: " ".join(v) for k, v in hyp_by_spk.items()} or {"spk0": ""}

    cp = call_wer(cp_fn, ref_txt, hyp_txt, clip_id)
    di = call_wer(di_fn, ref_txt, hyp_txt, clip_id)
    siso = siso_fn(" ".join(w for w, _ in ref_seq), " ".join(w for w, _ in hyp_seq))

    cp_e, cp_n = rate(cp)
    di_e, di_n = rate(di)
    si_e, si_n = rate(siso)

    mapping = speaker_mapping(ref_by_spk, hyp_by_spk, cp)
    wd_e, wd_n = wder(ref_seq, hyp_seq, mapping)

    return {"cp_errors": cp_e, "cp_len": cp_n,
            "di_errors": di_e, "di_len": di_n,
            "wer_errors": si_e, "wer_len": si_n,
            "wder_errors": wd_e, "wder_len": wd_n,
            "ref_words": len(ref_seq), "hyp_words": len(hyp_seq),
            "ref_spk": len(ref_by_spk), "hyp_spk": len(hyp_by_spk)}


def aggregate(df: pd.DataFrame, label: str) -> dict:
    def ew(e, n):  # error-weighted
        return round(100 * df[e].sum() / max(df[n].sum(), 1), 2)

    def mean(e, n):  # unweighted mean of per-clip rates
        return round(100 * (df[e] / df[n].clip(lower=1)).mean(), 2)

    cp, di = ew("cp_errors", "cp_len"), ew("di_errors", "di_len")
    return {"subset": label, "clips": len(df),
            "WER": ew("wer_errors", "wer_len"),
            "cpWER": cp, "DI_cpWER": di,
            "attribution_cost": round(cp - di, 2),
            "WDER": ew("wder_errors", "wder_len"),
            "cpWER_unweighted_mean": mean("cp_errors", "cp_len"),
            "ref_words": int(df["ref_words"].sum())}


def _as_markdown(df: pd.DataFrame) -> str:
    """to_markdown needs `tabulate`, which is not always installed. A missing
    optional dependency must not discard a completed scoring run."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        fence = "```"
        return fence + "\n" + df.to_string(index=False) + "\n" + fence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N clips")
    args = ap.parse_args()

    data = Path(args.data)
    # Both dependencies are checked before a single clip is scored: discovering
    # a missing one after cpWER has run on 99 clips wastes the whole pass.
    if Levenshtein is None:
        raise SystemExit("WDER needs rapidfuzz for the word alignment:  "
                         "pip install rapidfuzz")
    fns = resolve_meeteval()
    ref_dir = data / "ref" / "segments"
    conds = sorted(p for p in (data / "attrib").glob("*") if p.is_dir())
    if not conds:
        raise SystemExit(f"no conditions under {data / 'attrib'} -- run stage4_attribute.py")

    # Stage 4b needs only ref/rttm, so a dataset can carry the RTTMs and not the
    # transcripts and everything upstream still looks healthy. Scoring needs the
    # text. Say so here rather than skipping all 99 clips and reporting the
    # uninformative "nothing scored".
    if not ref_dir.is_dir():
        raise SystemExit(
            f"no reference transcripts at {ref_dir}. Stage 4b only needed "
            f"ref/rttm/, so this is easy to miss: attach the Stage 2 dataset "
            f"that carries ref/segments/ (one JSON per clip, with the text)."
        )
    n_ref = len(list(ref_dir.glob("*.json")))
    print(f"[env ] {n_ref} reference transcripts, {len(conds)} conditions")
    if n_ref == 0:
        raise SystemExit(f"{ref_dir} exists but holds no *.json")

    rows, skipped = [], []
    for cond in conds:
        asr, diar = cond.name.split("__", 1)
        clips = sorted(p.stem for p in cond.glob("*.json"))
        clips = clips[:args.limit] if args.limit else clips
        print(f"[cond] {cond.name}: {len(clips)} clips")

        for clip_id in clips:
            ref_path = ref_dir / f"{clip_id}.json"
            if not ref_path.exists():
                skipped.append((cond.name, clip_id, "no reference"))
                continue
            ref_by_spk, ref_seq = load_reference(ref_path)
            if not ref_seq:
                # Nothing to score against: an empty reference makes WER either
                # 0/0 or infinite depending on convention, and neither is a
                # result. Excluded and listed rather than silently counted.
                skipped.append((cond.name, clip_id, "empty reference"))
                continue
            hyp_by_spk, hyp_seq, _ = load_hypothesis(cond / f"{clip_id}.json")
            r = score_clip(ref_by_spk, ref_seq, hyp_by_spk, hyp_seq, fns, clip_id)
            rows.append({"asr": asr, "diar": diar, "oracle": diar == ORACLE,
                         "clip_id": clip_id, **r})

    per_clip = pd.DataFrame(rows)
    if per_clip.empty:
        why = {}
        for _, _, reason in skipped:
            why[reason] = why.get(reason, 0) + 1
        raise SystemExit(
            f"nothing scored. Skipped {len(skipped)} (condition, clip) pairs: {why}. "
            f"'no reference' in bulk means the clip ids in data/attrib do not match "
            f"the filenames in {ref_dir}."
        )

    # The common subset: clips every condition produced. This is the only fair
    # cross-system comparison, since sortformer is missing the long clips.
    per_cond = per_clip.groupby(["asr", "diar"])["clip_id"].apply(set)
    common = set.intersection(*per_cond) if len(per_cond) else set()

    summary = []
    for (asr, diar), g in per_clip.groupby(["asr", "diar"]):
        for label, sub in (("all", g), ("common", g[g.clip_id.isin(common)])):
            if sub.empty:
                continue
            summary.append({"asr": asr,
                            "diar": diar + (" (oracle)" if diar == ORACLE else ""),
                            **aggregate(sub, label)})
    summary = pd.DataFrame(summary).sort_values(["subset", "asr", "diar"])

    # DI-cpWER relaxes cpWER's speaker constraint, so it can never be larger.
    # If it is, the resolver bound the wrong meeteval function or the inputs are
    # shaped wrongly -- either way the attribution-cost column is meaningless
    # and must not be quietly written to a results table.
    bad = summary[summary["attribution_cost"] < -0.01]
    if not bad.empty:
        print("\n[WARN] DI-cpWER exceeds cpWER. With the optimal DI-cpWER that is "
              "impossible and means the binding or the input shape is wrong; with "
              "the greedy approximation a small excess is possible on hard clips. "
              "Either way attribution_cost is not trustworthy here:",
              file=sys.stderr)
        print(bad.to_string(index=False), file=sys.stderr)

    out = data / "results"
    out.mkdir(parents=True, exist_ok=True)
    per_clip.to_csv(out / "asr_per_clip.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "asr_summary.csv", index=False, encoding="utf-8")

    md = ["# Stage 4 -- ASR and attribution results", "",
          f"Common subset: {len(common)} of "
          f"{per_clip.clip_id.nunique()} clips scored by every condition.", "",
          "All rates are percentages, error-weighted "
          "(sum of errors / sum of reference words).", "",
          "`ref` is an ORACLE condition using the reference diarization. It is a "
          "diagnostic floor, never a system result.", "",
          _as_markdown(summary), ""]
    if skipped:
        md += ["## Excluded", ""] + [f"- `{c}` / `{k}`: {why}" for c, k, why in skipped] + [""]
    (out / "asr_summary.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print(summary.to_string(index=False))
    if skipped:
        print(f"\nexcluded {len(skipped)} (condition, clip) pairs; see asr_summary.md")
    print(f"\nwrote {out / 'asr_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
