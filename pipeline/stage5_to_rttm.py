#!/usr/bin/env python3
"""
Stage 5d -- corrected word labels back to RTTM, so DER/JER can be re-scored.

    python stage5_to_rttm.py --cond indicconformer__pyannote31+rule --data data
    python stage3_score.py --data data --systems pyannote31 pyannote31+rule

Stage 5 relabels WORDS. DER and JER are defined over time, not words, so the
corrected transcript cannot be scored against them directly -- which leaves the
"baseline vs improved DER/JER" cell of the results table empty. This closes it.


WHY RELABEL TURNS INSTEAD OF BUILDING THEM FROM WORDS
-----------------------------------------------------
The obvious approach -- emit one turn per unit, using the first and last word
times -- is wrong here, and quietly so.

A diarizer's turn spans continuous speech including the pauses inside it. Word
spans do not: they stop at the last word and resume at the next, and they omit
leading/trailing silence within a turn entirely. Turns rebuilt from words are
therefore systematically SHORTER than the turns they replace, so hypothesis
speech time drops, missed speech rises, and DER moves for a reason that has
nothing to do with Stage 5. The improvement would be measuring the conversion.

So the baseline turn boundaries are kept EXACTLY as the diarizer produced them,
and only the label changes: each turn takes the majority label of the corrected
words inside it, weighted by word duration so a long word counts for more than
a filler. Same segmentation, different names. The DER delta then isolates
speaker confusion, which is the only thing a relabel-only stage can affect --
and missed speech, false alarm and total speech time are unchanged by
construction, which the summary asserts.

A turn containing no words keeps its original label: there is no evidence to
revise it, and inventing one would be noise.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

from stage4_attribute import load_turns, rttm_dir


def relabel_turns(turns, words):
    """[(start, end, spk)] + corrected words -> [(start, end, new_spk)].

    Assignment is by duration-weighted majority of the words overlapping each
    turn. Overlap, not containment: a word straddling a boundary should vote in
    both turns it touches rather than in neither.
    """
    starts = [t[0] for t in turns]
    max_dur = max((e - s for s, e, _ in turns), default=0.0)

    votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for w in words:
        # Every turn that could overlap this word: bisect to the first turn
        # starting at or after (word start - longest turn), then walk forward
        # until turns start after the word ends.
        i = bisect.bisect_left(starts, w["start"] - max_dur)
        while i < len(turns) and turns[i][0] < w["end"]:
            s, e, _ = turns[i]
            ov = min(w["end"], e) - max(w["start"], s)
            if ov > 0:
                votes[i][w["spk"]] += ov
            i += 1

    out = []
    for i, (s, e, spk) in enumerate(turns):
        v = votes.get(i)
        if v:
            # max() on (weight, speaker) so ties break on the label, not on
            # dict order -- the output has to be reproducible.
            spk = max(sorted(v.items()), key=lambda kv: kv[1])[0]
        out.append((s, e, spk))
    return out


def write_rttm(path: Path, clip_id: str, turns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"SPEAKER {clip_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>"
        for s, e, spk in turns if e > s
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run(cond: str, data: Path, limit: int | None) -> bool:
    src_root = data / "attrib" / cond
    if not src_root.is_dir():
        print(f"[skip] no condition at {src_root}")
        return False

    # `asr__diar+method` -> the diarizer whose RTTMs supplied the boundaries.
    _asr, rest = cond.split("__", 1)
    diar = rest.split("+")[0]
    # Layout is data/hyp/<system>/rttm/<clip>.rttm -- that extra `rttm` level is
    # what Stage 3 writes and what stage3_score.py reads, so borrow Stage 4b's
    # resolver rather than rebuilding the path and getting it subtly wrong.
    base_dir = rttm_dir(data, diar)
    if not base_dir.is_dir():
        print(f"[skip] no baseline RTTMs at {base_dir}")
        return False

    # Named for the FULL condition, not just the diarizer. Three ASR systems
    # share each diarizer, and each produces a different corrected RTTM from
    # the same baseline turns -- naming these `pyannote31+rule` would have all
    # three overwrite one another, and the survivor would depend on argument
    # order. That the same diarizer repairs differently under different ASR is
    # itself a result; it needs three rows, not one.
    out_name = cond
    out_dir = data / "hyp" / out_name / "rttm"

    clips = sorted(p.stem for p in src_root.glob("*.json"))
    clips = clips[:limit] if limit else clips

    n_ok = n_skip = 0
    n_turns = n_changed = 0
    dur_total = dur_changed = 0.0
    for clip_id in clips:
        turns = load_turns(base_dir / f"{clip_id}.rttm")
        if not turns:
            n_skip += 1
            continue
        rec = json.loads((src_root / f"{clip_id}.json").read_text(encoding="utf-8"))
        new = relabel_turns(turns, rec["words"])
        write_rttm(out_dir / f"{clip_id}.rttm", clip_id, new)
        for (s, e, a), (_s, _e, b) in zip(turns, new):
            n_turns += 1
            dur_total += e - s
            if a != b:
                n_changed += 1
                dur_changed += e - s
        n_ok += 1

    print(f"\n[rttm] {cond} -> hyp/{out_name}")
    print(f"  clips     : {n_ok} written, {n_skip} skipped (no baseline RTTM)")
    if n_ok == 0:
        # A directory that exists but yields nothing is a path bug, not a
        # result. Fail loudly rather than reporting a tidy row of zeroes.
        print(f"  [!] nothing written -- {base_dir} exists but held no *.rttm "
              f"matching these clip ids. Check the path, not the data.")
        return False
    print(f"  turns     : {n_turns:,}, {n_changed:,} relabelled "
          f"({100.0 * n_changed / max(n_turns, 1):.2f}%)")
    print(f"  speech    : {dur_total / 3600:.2f} h, {dur_changed / 3600:.2f} h "
          f"relabelled ({100.0 * dur_changed / max(dur_total, 1e-9):.2f}%)")
    print(f"  boundaries are unchanged, so missed speech, false alarm and total "
          f"speech time are identical to `{diar}` -- any DER/JER delta is "
          f"speaker confusion alone")
    return n_ok > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cond", nargs="+", required=True,
                    help="Stage 5 condition(s), e.g. indicconformer__pyannote31+rule")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    clean = True
    for c in args.cond:
        clean &= run(c, args.data, args.limit)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
