#!/usr/bin/env python3
"""
Stage 4b -- Speaker attribution (CPU-only, seconds per condition).

Takes the words produced by Stage 4a, which never saw a speaker, and labels each
one using ONE diarization hypothesis. Every (asr, diar) pair is a separate
condition written to its own directory, so a cpWER difference between two
conditions is attributable to the thing that differs -- the labelling -- and
never to the ASR having been handed a different slice of audio.

    python stage4_attribute.py --asr whisper --diar pyannote31 sortformer ref

Writes data/attrib/<asr>__<diar>/<clip_id>.json and a manifest per condition.

Attribution rules, all four of which move the score and so are stated here
rather than buried:

  1. MAXIMUM OVERLAP. A word goes to the turn sharing the most time with its
     [start, end] interval. The cheaper rule -- assign by midpoint -- discards
     information exactly where it is most needed, on the long words that
     straddle a turn boundary. Equal overlap breaks toward the earlier turn, so
     the output is deterministic rather than dict-order dependent.

  2. ORPHANS ARE KEPT, NOT DROPPED. A word can land where the diarizer heard
     nothing. Dropping it deletes it from the hypothesis, which shows up in
     cpWER as a deletion and quietly *rewards* a system for missing speech.
     Instead the word is given the nearest turn and flagged `orphan`, so the
     error becomes a substitution when the guess is wrong, and Stage 6 can
     report how much of each system's cpWER came from this rule rather than
     from genuine labelling mistakes.

  3. OVERLAPPED SPEECH needs no special case: rule 1 hands the word to whichever
     simultaneous speaker covers more of it. Contested words are counted so the
     cost of that choice stays visible, and the two ways a word can be contested
     are counted apart -- `n_overlap_words` for genuinely simultaneous speakers,
     `n_boundary_words` for a word crossing between two disjoint turns. Merging
     them would drown the first in the second: the corpus is 7.60% overlapped,
     while every turn change produces boundary words regardless.

  4. `--diar ref` is an ORACLE condition. It attributes with the Stage 2
     reference RTTM, giving a cpWER floor where labelling is perfect by
     construction -- so every other condition reads as "ASR error + what this
     diarizer cost on top". It is a diagnostic only: nothing produced from it is
     fed back into any model, and it must be labelled `oracle` wherever it
     appears in a results table.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path

ORACLE = "ref"


# --------------------------------------------------------------------------
# manifest -- append-only and fsync'd, matching Stage 4a
# --------------------------------------------------------------------------

class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from a hard kill
                self.records[rec["clip_id"]] = rec

    def done(self, clip_id: str) -> bool:
        return self.records.get(clip_id, {}).get("status") == "ok"

    def append(self, rec: dict) -> None:
        self.records[rec["clip_id"]] = rec
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a reader never sees a half-written file


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------

def load_turns(path: Path) -> list[tuple[float, float, str]]:
    """RTTM -> [(start, end, speaker)] sorted by start.

    Parsed by hand for the same reason Stage 3b does it: the loader moved
    between pyannote releases, and this stage should not need pyannote at all.
    Zero- and negative-duration turns are dropped -- they can never win a
    maximum-overlap comparison, but they would pollute the speaker inventory.
    """
    turns: list[tuple[float, float, str]] = []
    if not path.exists():
        return turns
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        if dur <= 0:
            continue
        turns.append((start, start + dur, spk))
    turns.sort(key=lambda t: (t[0], t[1]))
    return turns


def attribute_word(w_start: float, w_end: float, turns, starts, max_dur: float):
    """One word -> (speaker, overlap_seconds, touched_spans).

    `touched_spans` is each intersecting turn clipped to the word, which is what
    lets the caller tell the two ways a word can touch two turns apart:
    simultaneous speech, or a boundary crossing between disjoint turns.

    Only turns that begin within `max_dur` before the word can reach it, so the
    scan starts there instead of at turn zero. Turns may overlap each other, so
    a plain bisect window is not enough on its own -- the longest turn in the
    clip sets how far back to look.
    """
    best_spk, best_ov, spans = None, 0.0, []
    i = bisect.bisect_left(starts, w_start - max_dur)
    for t_start, t_end, spk in turns[i:]:
        if t_start >= w_end:
            break  # sorted by start: nothing later can overlap
        lo, hi = max(w_start, t_start), min(w_end, t_end)
        if hi > lo:
            spans.append((lo, hi))
            # Strict >: ties keep the earlier turn, which the sort fixed.
            if hi - lo > best_ov:
                best_spk, best_ov = spk, hi - lo
    return best_spk, best_ov, spans


def is_simultaneous(spans: list[tuple[float, float]]) -> bool:
    """True if two speakers are active at the same instant inside the word.

    A word crossing the boundary between two disjoint turns also touches two
    turns, and counting that as overlapped speech would make the overlap
    diagnostic mostly boundary noise -- the corpus is only 7.60% overlapped, so
    the distinction is the whole measurement.
    """
    spans = sorted(spans)
    return any(b[0] < a[1] for a, b in zip(spans, spans[1:]))


def nearest_turn(w_start: float, w_end: float, turns) -> str | None:
    """Speaker of the turn with the smallest gap to this word. Ties -> earlier."""
    best_spk, best_gap = None, None
    for t_start, t_end, spk in turns:
        gap = t_start - w_end if t_start > w_end else w_start - t_end
        gap = max(gap, 0.0)
        if best_gap is None or gap < best_gap:
            best_spk, best_gap = spk, gap
    return best_spk


# --------------------------------------------------------------------------

def attribute_clip(words: list[dict], turns) -> dict:
    starts = [t[0] for t in turns]
    max_dur = max((e - s for s, e, _ in turns), default=0.0)

    out, n_orphan, n_overlap, n_boundary = [], 0, 0, 0
    for w in words:
        spk, ov, spans = attribute_word(w["start"], w["end"], turns, starts, max_dur)
        rec = {"w": w["w"], "start": w["start"], "end": w["end"]}
        if "lang" in w:
            rec["lang"] = w["lang"]
        if spk is None:
            # Rule 2: keep it, attributed to the nearest turn, and say so.
            spk = nearest_turn(w["start"], w["end"], turns)
            rec["orphan"] = True
            n_orphan += 1
        elif len(spans) > 1:
            # Both flags mark a contested word, and attribution errors
            # concentrate in them -- but they are contested for different
            # reasons and Stage 6 should be able to separate the two.
            if is_simultaneous(spans):
                rec["overlap"] = len(spans)
                n_overlap += 1
            else:
                rec["boundary"] = True
                n_boundary += 1
        rec["spk"] = spk
        out.append(rec)

    by_speaker: dict[str, list[str]] = {}
    for rec in out:
        by_speaker.setdefault(rec["spk"], []).append(rec["w"])

    return {
        "words": out,
        "by_speaker": {k: " ".join(v) for k, v in sorted(by_speaker.items())},
        "n_words": len(out),
        "n_orphan": n_orphan,
        "n_overlap_words": n_overlap,
        "n_boundary_words": n_boundary,
        "n_turns": len(turns),
        "speakers": sorted({t[2] for t in turns}),
    }


def rttm_dir(data: Path, diar: str) -> Path:
    return data / "ref" / "rttm" if diar == ORACLE else data / "hyp" / diar / "rttm"


def run_condition(asr: str, diar: str, data: Path, limit: int | None) -> bool:
    words_dir = data / "asr" / asr / "words"
    turns_dir = rttm_dir(data, diar)
    if not words_dir.is_dir():
        print(f"[skip] {asr}__{diar}: no words at {words_dir}")
        return False
    if not turns_dir.is_dir():
        print(f"[skip] {asr}__{diar}: no RTTMs at {turns_dir}")
        return False

    out_root = data / "attrib" / f"{asr}__{diar}"
    manifest = Manifest(out_root / "manifest.jsonl")
    clips = sorted(p.stem for p in words_dir.glob("*.json"))

    pending = [c for c in clips if not manifest.done(c)]
    todo = pending[:limit] if limit else pending
    tag = f"{asr}__{diar}" + ("  [ORACLE -- diagnostic only]" if diar == ORACLE else "")
    print(f"\n[cond] {tag}")
    print(f"[plan] {len(clips)} clips: {len(clips) - len(pending)} done, "
          f"{len(pending)} pending, running {len(todo)} now")

    n_ok = n_fail = 0
    tot_words = tot_orphan = tot_overlap = tot_boundary = 0
    for clip_id in todo:
        try:
            src = json.loads((words_dir / f"{clip_id}.json").read_text(encoding="utf-8"))
            turns = load_turns(turns_dir / f"{clip_id}.rttm")
            if not turns:
                # No hypothesis for this clip is a real result, not a crash:
                # Stage 3 left sortformer short on the long clips. Record it as
                # a failure so it is retried if the RTTM appears, and so it can
                # never be mistaken for a clip with zero words.
                raise ValueError("no turns in RTTM")

            res = attribute_clip(src["words"], turns)
            write_json(out_root / f"{clip_id}.json", {
                "clip_id": clip_id,
                "asr": asr,
                "diar": diar,
                "oracle": diar == ORACLE,
                "duration": src.get("duration"),
                "lang": src.get("lang"),
                **res,
            })
            manifest.append({"clip_id": clip_id, "status": "ok",
                             "n_words": res["n_words"],
                             "n_orphan": res["n_orphan"],
                             "n_overlap_words": res["n_overlap_words"],
                             "n_boundary_words": res["n_boundary_words"],
                             "n_turns": res["n_turns"],
                             "n_speakers": len(res["speakers"])})
            n_ok += 1
            tot_words += res["n_words"]
            tot_orphan += res["n_orphan"]
            tot_overlap += res["n_overlap_words"]
            tot_boundary += res["n_boundary_words"]
        except Exception as exc:  # noqa: BLE001 -- one bad clip must not stop 98 good ones
            manifest.append({"clip_id": clip_id, "status": "fail",
                             "error": f"{type(exc).__name__}: {exc}"})
            n_fail += 1
            print(f"  fail: {clip_id[:40]}  {type(exc).__name__}: {exc}")

    if tot_words:
        print(f"[done] ok={n_ok} fail={n_fail}  {tot_words:,} words, "
              f"{tot_orphan:,} orphaned ({100 * tot_orphan / tot_words:.2f}%), "
              f"{tot_overlap:,} in overlapped speech "
              f"({100 * tot_overlap / tot_words:.2f}%), "
              f"{tot_boundary:,} on a turn boundary "
              f"({100 * tot_boundary / tot_words:.2f}%)")
    else:
        print(f"[done] ok={n_ok} fail={n_fail}")
    return n_fail == 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", nargs="+", required=True,
                    help="ASR systems under data/asr/ (e.g. whisper indicconformer)")
    ap.add_argument("--diar", nargs="+", required=True,
                    help=f"diarization systems under data/hyp/, plus '{ORACLE}' "
                         f"for the oracle condition")
    ap.add_argument("--data", default="data", help="pipeline root")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N clips")
    args = ap.parse_args()

    data = Path(args.data)
    clean = True
    for asr in args.asr:
        for diar in args.diar:
            clean &= run_condition(asr, diar, data, args.limit)

    if ORACLE in args.diar:
        print(f"\n[note] the '{ORACLE}' condition used the reference RTTM. It is a "
              f"diagnostic upper bound -- label it 'oracle' in every table.")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
