#!/usr/bin/env python3
"""
Stage 2 -- Parse ground-truth labels into scoring-ready references.

Reads the segment table and turns the two label columns into:

    ref/rttm/<clip_id>.rttm    reference diarization, one file per clip
    ref/all.rttm               all clips concatenated (some scorers want one file)
    ref/segments/<clip_id>.json  speaker <-> text join, one record per turn
    ref/clip_meta.csv          per-clip conditions for the Stage 6 breakdown
    ref/stage2_report.json     anomaly counts for the writeup

Design notes:
  * `diarization_segments` and `asr_segments` are promised to share boundaries in
    the same order. That is ASSERTED here, not assumed -- if the promise ever
    breaks we fail loudly rather than silently misattributing text to speakers.
  * Reference turns are clipped to [0, end_sec - start_sec]. The task is to
    use only that window, so labels running past it are truncated rather than
    padding the audio. Truncation hits reference and hypothesis identically and
    introduces no differential bias.
  * Overlap is computed SPEAKER-AWARE: only regions with >=2 distinct speakers
    count. Counting all interval pairs (including adjacent same-speaker turns)
    overstates it.
  * Text is stored RAW. The reference uses a gloss convention -- native script
    followed by the English/numeric source, e.g. "कॉफी(coffee)", "एक(1)" -- which
    touches ~16% of tokens. Normalising that is a scoring decision and belongs in
    Stage 4, where it can be applied to reference and hypothesis alike.
  * Idempotent rather than checkpointed: it rewrites its output tree each run and
    completes in seconds, so a killed run leaves no half-state.

Usage:
    python stage2_parse_refs.py --input youtube_segments_final.xlsx --out data
    python stage2_parse_refs.py --input ... --out data --manifest data/manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "video_id", "youtube_link", "start_sec", "end_sec",
    "diarization_segments", "asr_segments",
]

# "Speaker A [12.34-56.78]"
DIAR_RE = re.compile(r"^\s*(?P<spk>.+?)\s*\[\s*(?P<a>-?\d+(?:\.\d+)?)\s*-\s*(?P<b>-?\d+(?:\.\d+)?)\s*\]\s*$")
# "[12.34-56.78] some text"
ASR_RE = re.compile(r"^\s*\[\s*(?P<a>-?\d+(?:\.\d+)?)\s*-\s*(?P<b>-?\d+(?:\.\d+)?)\s*\]\s*(?P<text>.*)$", re.S)

MIN_SEG_DUR = 0.01          # segments shorter than this after clipping are noise
TIME_TOL = 1e-6             # tolerance when asserting diar/asr timestamps agree

# Unicode script -> language label. Devanagari covers both Hindi and Marathi and
# cannot be split by script alone; kept as one bucket and flagged as such.
SCRIPT_TO_LANG = {
    "DEVANAGARI": "Devanagari (hi/mr)",
    "BENGALI": "Bengali",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Punjabi",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "ORIYA": "Odia",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
}


@dataclass
class Anomalies:
    inverted: list = field(default_factory=list)          # end < start
    clipped_tail: list = field(default_factory=list)      # truncated at window end
    dropped_outside: list = field(default_factory=list)   # entirely past window
    dropped_tiny: list = field(default_factory=list)      # < MIN_SEG_DUR after clip
    empty_text: int = 0
    empty_clips: list = field(default_factory=list)       # no usable turns left


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def split_turns(blob: str) -> list[str]:
    """Split the pipe-delimited turn list. Asserts no stray '|' inside text."""
    return [p for p in str(blob).split("|")]


def parse_diarization(blob: str, clip_id: str) -> list[tuple[str, float, float]]:
    out = []
    for i, part in enumerate(split_turns(blob)):
        if not part.strip():
            continue
        m = DIAR_RE.match(part)
        if not m:
            raise ValueError(f"{clip_id}: unparseable diarization turn {i}: {part!r}")
        out.append((m.group("spk").strip(), float(m.group("a")), float(m.group("b"))))
    return out


def parse_asr(blob: str, clip_id: str) -> list[tuple[float, float, str]]:
    out = []
    for i, part in enumerate(split_turns(blob)):
        if not part.strip():
            continue
        m = ASR_RE.match(part)
        if not m:
            raise ValueError(f"{clip_id}: unparseable asr turn {i}: {part!r}")
        out.append((float(m.group("a")), float(m.group("b")), m.group("text").strip()))
    return out


def normalise_speaker(label: str) -> str:
    """RTTM is whitespace-delimited, so a space in the speaker name corrupts it."""
    return re.sub(r"\s+", "_", label.strip())


# --------------------------------------------------------------------------
# Clipping + geometry
# --------------------------------------------------------------------------

def clip_turns(turns, duration: float, clip_id: str, anom: Anomalies):
    """Clip reference turns to [0, duration], recording every adjustment."""
    kept = []
    for spk, a, b, text in turns:
        if b < a:
            anom.inverted.append({"clip_id": clip_id, "speaker": spk, "start": a, "end": b})
            continue
        if a >= duration:
            anom.dropped_outside.append({"clip_id": clip_id, "speaker": spk,
                                         "start": a, "end": b, "duration": duration})
            continue
        na, nb = max(0.0, a), min(duration, b)
        if nb - a != b - a or na != a:
            anom.clipped_tail.append({"clip_id": clip_id, "speaker": spk,
                                      "orig": [a, b], "clipped": [na, nb],
                                      "lost_sec": round((b - nb) + (na - a), 3)})
        if nb - na < MIN_SEG_DUR:
            anom.dropped_tiny.append({"clip_id": clip_id, "speaker": spk,
                                      "start": na, "end": nb})
            continue
        kept.append((spk, na, nb, text))
    return kept


def speaker_aware_overlap(turns) -> float:
    """Seconds where >=2 DISTINCT speakers are simultaneously active.

    Builds a sweep over boundary points and counts distinct speakers in each
    elementary interval. Same-speaker adjacent or nested turns do not count.
    """
    if not turns:
        return 0.0
    points = sorted({t for _, a, b, _ in turns for t in (a, b)})
    total = 0.0
    for lo, hi in zip(points, points[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        active = {spk for spk, a, b, _ in turns if a <= mid < b}
        if len(active) >= 2:
            total += hi - lo
    return total


def union_speech(turns) -> float:
    """Seconds with at least one speaker active (overlap counted once)."""
    iv = sorted((a, b) for _, a, b, _ in turns)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in iv:
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    if cur_a is not None:
        total += cur_b - cur_a
    return total


def detect_language(text: str) -> str:
    """Majority Unicode script of the reference text, as a language proxy.

    Bracketed timestamps and parenthetical English glosses are stripped first --
    otherwise the Latin glosses skew the vote on every clip.
    """
    body = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    counts = Counter()
    for ch in body:
        if not ch.isalpha():
            continue
        try:
            script = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[script] += 1
    if not counts:
        return "unknown"
    top = counts.most_common(1)[0][0]
    return SCRIPT_TO_LANG.get(top, top.title())


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def rttm_lines(clip_id: str, turns) -> list[str]:
    """SPEAKER <file> <chan> <onset> <dur> <NA> <NA> <spk> <NA> <NA>"""
    lines = []
    for spk, a, b, _ in turns:
        lines.append(
            f"SPEAKER {clip_id} 1 {a:.3f} {b - a:.3f} "
            f"<NA> <NA> {normalise_speaker(spk)} <NA> <NA>"
        )
    return lines


# --------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    recs = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["clip_id"]] = r
    return recs


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def clip_id_for(video_id: str, start: float, end: float) -> str:
    """Must match Stage 1 exactly, or the audio and references will not join."""
    return f"{video_id}__{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: parse ground truth into RTTM + aligned segments.")
    ap.add_argument("--input", "--csv", dest="input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="root containing ref/ (usually Stage 1's --out)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Stage 1 manifest.jsonl, to mark which clips have audio")
    args = ap.parse_args()

    df = read_table(args.input)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    ref_root = args.out / "ref"
    if ref_root.exists():
        shutil.rmtree(ref_root)          # idempotent: no half-state from a killed run
    (ref_root / "rttm").mkdir(parents=True)
    (ref_root / "segments").mkdir(parents=True)

    manifest = load_manifest(args.manifest) if args.manifest else {}
    anom = Anomalies()
    all_rttm: list[str] = []
    meta_rows: list[dict] = []
    pipe_violations = 0

    for _, row in df.iterrows():
        video_id = str(row.video_id)
        start, end = float(row.start_sec), float(row.end_sec)
        duration = end - start
        clip_id = clip_id_for(video_id, start, end)

        diar = parse_diarization(row.diarization_segments, clip_id)
        asr = parse_asr(row.asr_segments, clip_id)

        # The index-join is only safe if the two columns really do correspond.
        # Assert it; a mismatch means the file changed shape under us.
        if len(diar) != len(asr):
            raise SystemExit(
                f"{clip_id}: diarization has {len(diar)} turns but asr has {len(asr)}. "
                "The index-join assumption is broken -- stop and re-audit the input."
            )
        for i, ((_, da, db), (aa, ab, _)) in enumerate(zip(diar, asr)):
            if abs(da - aa) > TIME_TOL or abs(db - ab) > TIME_TOL:
                raise SystemExit(
                    f"{clip_id}: turn {i} timestamps disagree between columns: "
                    f"diar [{da}-{db}] vs asr [{aa}-{ab}]."
                )

        turns = [(spk, a, b, text) for (spk, a, b), (_, _, text) in zip(diar, asr)]
        for _, _, _, text in turns:
            if "|" in text:
                pipe_violations += 1
            if not text.strip():
                anom.empty_text += 1

        turns = clip_turns(turns, duration, clip_id, anom)
        if not turns:
            anom.empty_clips.append(clip_id)

        # --- artifacts -----------------------------------------------------
        lines = rttm_lines(clip_id, turns)
        (ref_root / "rttm" / f"{clip_id}.rttm").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        all_rttm.extend(lines)

        seg_records = [
            {"index": i, "speaker": normalise_speaker(spk),
             "start": round(a, 3), "end": round(b, 3), "text": text}
            for i, (spk, a, b, text) in enumerate(turns)
        ]
        (ref_root / "segments" / f"{clip_id}.json").write_text(
            json.dumps({"clip_id": clip_id, "video_id": video_id,
                        "start_sec": start, "end_sec": end, "duration": duration,
                        "segments": seg_records}, ensure_ascii=False, indent=1),
            encoding="utf-8")

        # --- per-clip conditions for Stage 6 -------------------------------
        speakers = sorted({normalise_speaker(s) for s, _, _, _ in turns})
        overlap = speaker_aware_overlap(turns)
        speech = union_speech(turns)
        all_text = " ".join(t for _, _, _, t in turns)
        n_words = len(re.findall(r"\S+", re.sub(r"\([^)]*\)", "", all_text)))
        rec = manifest.get(clip_id)

        meta_rows.append({
            "clip_id": clip_id,
            "video_id": video_id,
            "start_sec": start,
            "end_sec": end,
            "duration": round(duration, 3),
            "n_speakers": len(speakers),
            "n_segments": len(turns),
            "speech_sec": round(speech, 3),
            "speech_frac": round(speech / duration, 4) if duration else 0.0,
            "overlap_sec": round(overlap, 3),
            "overlap_frac_of_speech": round(overlap / speech, 4) if speech else 0.0,
            "overlap_frac_of_clip": round(overlap / duration, 4) if duration else 0.0,
            "language": detect_language(all_text),
            "n_ref_words": n_words,
            "has_audio": bool(rec and rec.get("status") in ("ok", "short_source")),
        })

    (ref_root / "all.rttm").write_text("\n".join(all_rttm) + "\n", encoding="utf-8")

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(ref_root / "clip_meta.csv", index=False, encoding="utf-8")

    report = {
        "n_clips": len(meta),
        "n_turns_kept": int(meta.n_segments.sum()),
        "inverted_dropped": len(anom.inverted),
        "dropped_outside_window": len(anom.dropped_outside),
        "dropped_tiny_after_clip": len(anom.dropped_tiny),
        "clipped_at_window_end": len(anom.clipped_tail),
        "total_sec_lost_to_clipping": round(sum(a["lost_sec"] for a in anom.clipped_tail), 3),
        "empty_text_turns": anom.empty_text,
        "clips_with_no_turns": anom.empty_clips,
        "pipe_in_text_violations": pipe_violations,
        "details": {
            "inverted": anom.inverted,
            "dropped_outside": anom.dropped_outside[:50],
            "dropped_tiny": anom.dropped_tiny[:50],
        },
    }
    (ref_root / "stage2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- summary -----------------------------------------------------------
    print("=" * 70)
    print("STAGE 2 SUMMARY")
    print("=" * 70)
    print(f"  clips parsed          : {len(meta)}")
    print(f"  reference turns kept  : {int(meta.n_segments.sum())}")
    print(f"  turns clipped at end  : {len(anom.clipped_tail)} "
          f"({report['total_sec_lost_to_clipping']:.1f}s lost total)")
    print(f"  turns dropped         : {len(anom.inverted)} inverted, "
          f"{len(anom.dropped_outside)} outside window, "
          f"{len(anom.dropped_tiny)} sub-{MIN_SEG_DUR}s")
    if pipe_violations:
        print(f"  [!] '|' found inside {pipe_violations} transcript(s) -- split may be wrong")
    if anom.empty_clips:
        print(f"  [!] clips left with NO turns: {anom.empty_clips}")

    print(f"\n  total speech (union)  : {meta.speech_sec.sum() / 3600:.2f}h "
          f"of {meta.duration.sum() / 3600:.2f}h audio "
          f"({meta.speech_sec.sum() / meta.duration.sum() * 100:.1f}%)")
    print(f"  OVERLAP (speaker-aware): {meta.overlap_sec.sum() / 60:.1f} min = "
          f"{meta.overlap_sec.sum() / meta.speech_sec.sum() * 100:.2f}% of speech")
    print(f"  clips containing overlap: {int((meta.overlap_sec > 0).sum())}/{len(meta)}")

    print("\n  speaker-count distribution:")
    for k, v in sorted(meta.n_speakers.value_counts().items()):
        print(f"    {k} speakers : {v:>3} clips")
    print("\n  language distribution:")
    for k, v in meta.language.value_counts().items():
        print(f"    {k:<22}: {v:>3} clips")
    if "has_audio" in meta:
        print(f"\n  clips with audio ready : {int(meta.has_audio.sum())}/{len(meta)}")
    print(f"\n  wrote -> {ref_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
