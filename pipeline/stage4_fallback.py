#!/usr/bin/env python3
"""
Stage 4c -- language-ID fallback: IndicConformer, or Whisper when IndicConformer
does not know what language it is hearing (CPU, seconds, no model runs).

    python stage4_fallback.py --data data
    python stage4_attribute.py --asr ic_lid_fallback --diar pyannote31 sortformer_stream sortformer ref

Writes data/asr/ic_lid_fallback/words/<clip_id>.json in exactly the Stage 4a
format, so attribution, Stage 5 and scoring treat it as one more ASR system.


THE FAILURE IT TARGETS
----------------------
IndicConformer decodes a clip inside ONE language's vocabulary block, chosen by
a frame vote over the model's own logits (stage4_asr.py, `lang_lock`). On 13 of
99 clips that vote lands on a language this task does not serve: 11 Urdu, 2
Nepali. The decode is then spelled in the wrong script for the whole clip --
Perso-Arabic against a Devanagari reference -- and scores 100% WER no matter how
well the acoustics were recognised. Hindi and Urdu are close to one spoken
language written in two scripts, so this vote is near a coin flip on acoustics
alone; it is not a bug to be fixed inside the decoder.

Whisper is not immune to the same confusion, but its language decision comes
from a separate model trained on far more Hindi, and on these clips it is right
far more often.


THE RULE
--------
    if IndicConformer's detected language is in TARGET_LANGS:  keep IndicConformer
    else:                                                      use Whisper's words

Clip-level, all or nothing. Mixing the two systems' words inside a clip would
need a word alignment between hypotheses (ROVER-style) and is out of scope.


WHAT IT DOES AND DOES NOT USE
-----------------------------
Inputs are the two ASR outputs and a fixed list of languages. No reference
transcript, no reference diarization, no per-clip metadata.

TARGET_LANGS is the one judgement call, and it should be stated as such: it is
the set of languages the system is deployed for -- here the nine scripts this
corpus was collected in, fixed once for the whole run. That is task
configuration, the same thing that decides which model is loaded, not a label on
any clip. It was NOT tuned by trying language subsets against the scores: the
rule is "outside the served set", and the set is the task's.

The per-clip consequences are checked, not assumed: stage6_report.py confirms
every kept clip scores identically to `indicconformer` and every switched clip
identically to `whisper`, so the gain cannot come from anything but the switch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage4_attribute import Manifest, write_json

NAME = "ic_lid_fallback"
PRIMARY = "indicconformer"
FALLBACK = "whisper"

# IndicConformer's language codes for the nine scripts the task serves.
# Devanagari covers both Hindi and Marathi; everything else is one-to-one.
TARGET_LANGS = frozenset({
    "hi", "mr",   # Devanagari
    "bn",         # Bengali
    "gu",         # Gujarati
    "kn",         # Kannada
    "ml",         # Malayalam
    "or",         # Odia
    "pa",         # Punjabi (Gurmukhi)
    "ta",         # Tamil
    "te",         # Telugu
})


def run(data: Path) -> bool:
    src = {s: data / "asr" / s / "words" for s in (PRIMARY, FALLBACK)}
    for s, d in src.items():
        if not d.is_dir():
            print(f"[!] no {s} words at {d} -- run stage4_asr.py --system {s}")
            return False

    out_root = data / "asr" / NAME
    # Rebuilt from scratch every run rather than resumed: it takes seconds, and
    # a stale file from an earlier TARGET_LANGS would otherwise survive.
    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    manifest = Manifest(manifest_path)

    clips = sorted(p.stem for p in src[PRIMARY].glob("*.json"))
    switched, missing = [], []
    for clip_id in clips:
        prim = json.loads((src[PRIMARY] / f"{clip_id}.json").read_text(encoding="utf-8"))
        lang = prim.get("lang")
        use_fallback = lang not in TARGET_LANGS

        if use_fallback:
            fb_path = src[FALLBACK] / f"{clip_id}.json"
            if not fb_path.exists():
                # Nothing to fall back to: keep the primary rather than drop the
                # clip, and say so. Dropping would shrink the scored set.
                missing.append(clip_id)
                use_fallback = False
            else:
                chosen = json.loads(fb_path.read_text(encoding="utf-8"))
                switched.append((clip_id, lang, chosen.get("lang")))
        if not use_fallback:
            chosen = prim

        source = FALLBACK if use_fallback else PRIMARY
        write_json(out_root / "words" / f"{clip_id}.json", {
            **chosen,
            "system": NAME,
            "source": source,
            "primary_lang": lang,
        })
        manifest.append({"clip_id": clip_id, "status": "ok",
                         "n_words": len(chosen["words"]),
                         "lang": chosen.get("lang"),
                         "duration": chosen.get("duration"),
                         "source": source,
                         "primary_lang": lang})

    print(f"[fallback] {len(clips)} clips: {len(clips) - len(switched)} kept "
          f"{PRIMARY}, {len(switched)} switched to {FALLBACK}")
    print(f"  served languages: {' '.join(sorted(TARGET_LANGS))}")
    for clip_id, lang, fb_lang in switched:
        print(f"  {clip_id}  {PRIMARY} lang={lang!s:<4} -> {FALLBACK} lang={fb_lang}")
    if missing:
        print(f"  [!] {len(missing)} clip(s) needed a fallback with no {FALLBACK} "
              f"words; kept {PRIMARY}: {missing}")
    print(f"wrote -> {out_root}")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    return 0 if run(args.data) else 1


if __name__ == "__main__":
    sys.exit(main())
