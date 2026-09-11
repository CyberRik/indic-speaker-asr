#!/usr/bin/env python3
"""
Assemble the Drive folder that notebooks/end_to_end.ipynb reads.

    python pipeline/pack_drive.py [--out drive_upload/indic-speaker-asr] [--no-wav]

Upload the resulting indic-speaker-asr/ folder to the root of My Drive.

    indic-speaker-asr/
      youtube_segments_final.xlsx   the input spreadsheet
      manifest.jsonl                Stage 1 manifest of the reference run
      wav/                          99 clips, 16 kHz mono (1.4 GB)
      cache/                        the reference run's GPU outputs
        hyp/{pyannote31,sortformer,sortformer_stream}/
        asr/{indicconformer,indicconformer_free,whisper}/
        attrib/indicconformer__pyannote31+llm/
      expected/                     the committed tables, for the final check

Only GPU outputs are cached. Everything derived from them on CPU (the fallback
ASR system, attribution, the rule relabel, projections, every score) is
recomputed by the notebook.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

GPU_OUTPUTS = [
    "hyp/pyannote31", "hyp/sortformer", "hyp/sortformer_stream",
    "asr/indicconformer", "asr/indicconformer_free", "asr/whisper",
    "attrib/indicconformer__pyannote31+llm",
]
EXPECTED = ["diarization_per_clip.csv", "diarization_summary.csv",
            "asr_per_clip.csv", "asr_summary.csv",
            "results_per_video.csv", "results_table.md"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=ROOT / "drive_upload" / "indic-speaker-asr")
    ap.add_argument("--no-wav", action="store_true", help="skip the 1.4 GB of audio")
    args = ap.parse_args()
    out = args.out

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "youtube_segments_final.xlsx", out)
    shutil.copy(DATA / "manifest.jsonl", out)

    for sub in GPU_OUTPUTS:
        dst = out / "cache" / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(DATA / sub, dst)
        n = sum(1 for f in dst.rglob("*") if f.suffix in (".rttm", ".json"))
        print(f"cache/{sub}: {n} files")

    (out / "expected").mkdir(exist_ok=True)
    for name in EXPECTED:
        shutil.copy(DATA / "results" / name, out / "expected" / name)

    if not args.no_wav:
        wavs = sorted((DATA / "wav").glob("*.wav"))
        assert len(wavs) == 99, f"expected 99 wavs, found {len(wavs)}"
        (out / "wav").mkdir(exist_ok=True)
        for w in wavs:
            dst = out / "wav" / w.name
            if not dst.exists() or dst.stat().st_size != w.stat().st_size:
                shutil.copy2(w, dst)
        print(f"wav: {len(wavs)} files")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out}  ({size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
