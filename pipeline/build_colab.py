#!/usr/bin/env python3
"""
Generate the end-to-end Colab notebook: notebooks/end_to_end.ipynb.

    python pipeline/build_colab.py

Like build_notebooks.py, the pipeline/*.py scripts are the single source of
truth: each is embedded as a %%writefile cell at the start of the stage that
uses it, so the notebook cannot drift from the code.

One notebook, every stage in order, one switch:

    FULL_GPU_RUN = False   GPU stages run on SMOKE_CLIPS clips to prove the code
                           runs, and the full 99-clip outputs of the reference
                           run are loaded from Drive. Every CPU stage (parsing,
                           attribution, scoring, rule correction, projection,
                           results table) recomputes on all 99 clips, and the
                           final table is asserted equal to the committed one.
    FULL_GPU_RUN = True    the same cells run every GPU stage on all 99 clips.

Two rules keep one Colab session from breaking itself:

  * The kernel imports nothing but the standard library. pip replacing numpy on
    disk while the kernel holds the old one loaded makes the next in-kernel
    import of anything numpy-based fail (seen on Colab: pyannote.metrics ->
    "'numpy.ufunc' object has no attribute '__module__'"). Every stage and every
    check that needs numpy runs as a subprocess.
  * Each GPU stack is installed into its own venv that can see the system
    packages (torch, CUDA wheels) but installs into its own site-packages. NeMo,
    pyannote.audio, faster-whisper, onnxruntime-gpu and bitsandbytes never touch
    the environment the scoring stages run in, or each other.

The Drive folder it reads is assembled by pipeline/pack_drive.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_notebooks import code, embed, md

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "end_to_end.ipynb"


HEADER = """
# Indic speaker-attributed ASR — diarization + ASR, end to end

99 Indic YouTube clips (12.26 h, 9 scripts, 2–8 speakers). Every stage of the
pipeline, in order, in one notebook:

| stage | what | compute |
|---|---|---|
| 1 | audio: YouTube → 16 kHz mono WAV, trimmed sample-exact | network |
| 2 | reference parsing: spreadsheet → RTTM + speaker-attributed text | CPU |
| 3 | diarization: pyannote 3.1, Sortformer 4spk-v1 (offline + streaming) | **GPU** |
| 4a | ASR: IndicConformer-600M (ONNX, CTC), Whisper large-v3 | **GPU** |
| 4b | language-ID fallback, word → speaker attribution | CPU |
| 5 | speaker relabelling: rule baseline (CPU), Qwen2.5-7B (**GPU**) | mixed |
| 4c/5b | cpWER / WDER, projection back to RTTM, DER / JER | CPU |
| 6 | results table: baseline vs improved, per model, per video | CPU |

## How to run

1. **Runtime → Change runtime type → T4 GPU.**
2. Put the `indic-speaker-asr` data folder in **My Drive** (or set `DRIVE_DIR` below).
3. Add a Colab secret **`HF_TOKEN`** (key icon on the left), switch on **notebook access**, and
   run the setup cell — if the access dialog is missed, the cell reports a `TimeoutException`
   and both gated models are skipped; re-run that cell to recover. The token's account must
   have accepted the conditions of **three** gated repos:
   `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and
   `ai4bharat/indic-conformer-600m-multilingual`.
4. **Runtime → Run all.** To run again after a failure, use **Runtime → Disconnect and
   delete runtime** first, so the run starts from a clean machine.

## The one switch: `FULL_GPU_RUN`

A full GPU pass is over five T4 hours (Whisper 2.5 h, the LLM ~40 min per condition,
pyannote 50 min), which is longer than a free Colab session reliably lasts. So:

- **`FULL_GPU_RUN = False`** (default, ~45 min). Each GPU stage **runs live on the
  first `SMOKE_CLIPS` clips** and is compared with the reference run's output for those
  clips; then the reference run's **full 99-clip GPU outputs are loaded from Drive**.
  Every CPU stage recomputes on all 99 clips, and the final results table is
  **asserted identical** to the committed one.
- **`FULL_GPU_RUN = True`**: the same cells run every GPU stage on all 99 clips, and
  the results are compared (not asserted) against the committed table.

Stage 1 cannot run on Colab: YouTube bot-gates cloud IPs, even with cookies. Audio was
extracted from a home connection with the same script, and this notebook verifies it
(format, exact sample count, not silent) instead of downloading it.

The cached GPU outputs come from the per-stage Kaggle runs whose notebooks, with their
original outputs, are in `notebooks/`; each cached directory is byte-identical to the
corresponding Kaggle output dataset.

## How the notebook is built

- **Every stage runs as a subprocess** of a script written by the `%%writefile` cell at
  the start of its section — the same `pipeline/*.py` files as in the repository. The
  notebook kernel itself imports only the standard library.
- **Each GPU model gets its own Python environment** (a venv layered over Colab's
  packages). NeMo, pyannote.audio, faster-whisper and onnxruntime-gpu pin conflicting
  versions of numpy and cuDNN; isolated, none of them can break another or the scoring
  stack.
"""


CONFIG = r"""
FULL_GPU_RUN = False     # True: every GPU stage on all 99 clips (5+ T4 hours)
SMOKE_CLIPS  = 3         # clips each GPU stage runs live when FULL_GPU_RUN is False
DRIVE_DIR    = "/content/drive/MyDrive/indic-speaker-asr"
"""


ENV = r"""
# Standard library only: see "How the notebook is built" above.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

try:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    ON_COLAB = True
except ImportError:                      # any other Jupyter: point ASR_DRIVE at the folder
    ON_COLAB = False
    DRIVE_DIR = os.environ.get("ASR_DRIVE", DRIVE_DIR)

DRIVE = Path(DRIVE_DIR)
WORKDIR = Path("/content/indic-speaker-asr") if ON_COLAB else Path(os.environ.get("ASR_WORKDIR", "asr_run")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)                        # scripts are written here and run with --data data
DATA = WORKDIR / "data"
DATA.mkdir(exist_ok=True)
CACHE = DRIVE / "cache"                  # the reference run's GPU outputs
EXPECTED = DRIVE / "expected"            # the committed results tables

for p in (DRIVE / "youtube_segments_final.xlsx", DRIVE / "manifest.jsonl", DRIVE / "wav", CACHE, EXPECTED):
    assert p.exists(), f"missing {p} -- is the indic-speaker-asr folder in My Drive (DRIVE_DIR)?"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if ON_COLAB and not HF_TOKEN:
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception as exc:             # secret missing, or notebook access not granted
        # A TimeoutException means the "grant access" dialog was never answered --
        # easy to miss under Run all, and it silently costs two GPU stages.
        print(f"HF_TOKEN not available ({type(exc).__name__}). Both gated models will be skipped:")
        print("  pyannote/speaker-diarization-3.1  (accept its conditions, and segmentation-3.0)")
        print("  ai4bharat/indic-conformer-600m-multilingual")
        print("Add the secret with the key icon, switch on notebook access, then RE-RUN THIS CELL.")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

HAVE_GPU = (os.environ.get("ASR_GPU", "1") != "0"     # ASR_GPU=0: CPU half only
            and shutil.which("nvidia-smi") is not None
            and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0)
if FULL_GPU_RUN:
    assert HAVE_GPU, "FULL_GPU_RUN needs a GPU runtime"

REPORT = []

def run(label, args, *, may_fail=False, env=None, python=None):
    # Run one script in a fresh interpreter, streaming its output into the cell.
    print("$ python", " ".join(map(str, args)), flush=True)
    t0 = time.time()
    e = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
         "TQDM_DISABLE": "1", **(env or {})}
    p = subprocess.Popen([str(python or sys.executable)] + [str(a) for a in args], cwd=WORKDIR,
                         env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    mins = (time.time() - t0) / 60
    REPORT.append({"step": label, "status": "ok" if rc == 0 else "FAILED", "exit": rc, "min": mins})
    print(f"[{label}] exit {rc}, {mins:.1f} min", flush=True)
    if rc and not may_fail:
        raise RuntimeError(f"{label} failed (exit {rc}) -- see the log above")
    return rc

def skip(label, why):
    REPORT.append({"step": label, "status": "skipped: " + why, "exit": None, "min": 0.0})
    print(f"[{label}] skipped: {why}")

def gpu_env(name, *packages):
    # A venv for one GPU stack. Colab's own site-packages are appended to its path
    # through a .pth file, so torch and the CUDA wheels are reused rather than
    # downloaded again, while anything pip installs or upgrades here lands in the
    # venv and shadows the system copy for this environment only.
    # Returns the venv's python, or None if the install failed in smoke mode.
    root = WORKDIR / "_venvs" / name
    py = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    label = f"install env: {name}"
    t0 = time.time()
    try:
        if not py.exists():
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True)
            system_paths = json.loads(subprocess.run(
                [sys.executable, "-c", "import json, site; print(json.dumps(site.getsitepackages() + [site.getusersitepackages()]))"],
                capture_output=True, text=True, check=True).stdout)
            purelib = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                     capture_output=True, text=True, check=True).stdout.strip()
            Path(purelib).mkdir(parents=True, exist_ok=True)
            (Path(purelib) / "_system_site.pth").write_text("\n".join(system_paths) + "\n", encoding="utf-8")
        print(f"[{label}] pip install {' '.join(packages)}", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", *packages], check=True)
    except subprocess.CalledProcessError as exc:
        REPORT.append({"step": label, "status": "FAILED", "exit": exc.returncode, "min": (time.time() - t0) / 60})
        print(f"[{label}] FAILED (exit {exc.returncode})")
        if FULL_GPU_RUN:
            raise
        return None
    REPORT.append({"step": label, "status": "ok", "exit": 0, "min": (time.time() - t0) / 60})
    return py

def restore(sub):
    # Copy one directory of the reference run's GPU output into data/.
    src, dst = CACHE / sub, DATA / sub
    assert src.is_dir(), f"missing cached output {src}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for f in dst.rglob("*") if f.suffix in (".rttm", ".json"))
    print(f"restored {sub}: {n} files")

# A GPU stage that fails in smoke mode is recorded and the notebook carries on,
# because the cached full outputs do not depend on it. In a full run it is fatal.
GPU_MAY_FAIL = not FULL_GPU_RUN
GPU_LIMIT = [] if FULL_GPU_RUN else ["--limit", str(SMOKE_CLIPS)]
GPU_ROOT = "data" if FULL_GPU_RUN else "smoke"   # smoke output never mixes with the full data

print("Colab      :", ON_COLAB, "  Python", sys.version.split()[0])
print("GPU        :", HAVE_GPU)
print("HF_TOKEN   :", bool(HF_TOKEN))
print("mode       :", "FULL GPU RUN (99 clips)" if FULL_GPU_RUN else f"smoke {SMOKE_CLIPS} clips + cached GPU outputs")
print("drive      :", DRIVE)
print("workdir    :", WORKDIR)
"""


CPU_DEPS = r"""
# The CPU scoring stack, into the main environment. pyannote.metrics brings
# pyannote.core; meeteval is cpWER; rapidfuzz is the word alignment behind WDER;
# openpyxl writes the xlsx table. Nothing installs into this environment after here.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pyannote.metrics", "meeteval", "rapidfuzz", "openpyxl"], check=True)
"""


CHECKS_MD = """
### Notebook checks

The verification steps that need numpy, pandas or pyannote live in one small script,
run as subprocesses like the stages themselves: the audio check, the reference hash,
the smoke-vs-reference diarization comparison, and the final comparison with the
committed tables.
"""

CHECKS = r'''
"""Verification steps for end_to_end.ipynb, run as subprocesses."""
import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path


def audio(args):
    import numpy as np

    recs = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["clip_id"]] = r                  # append-only: last record per clip wins
    ok = [r for r in recs.values() if r["status"] == "ok"]
    failed = {r["video_id"]: r.get("error_class") for r in recs.values() if r["status"] != "ok"}

    broken, minutes = [], 0.0
    for r in ok:
        with wave.open(str(Path(args.wav_dir) / (r["clip_id"] + ".wav")), "rb") as w:
            fmt, n = (w.getframerate(), w.getnchannels(), w.getsampwidth()), w.getnframes()
            x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
        rms_db = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
        minutes += n / 16000 / 60
        # The corpus is ~94% speech, so a real clip sits far above -45 dBFS.
        if fmt != (16000, 1, 2) or n != r["expected_samples"] or rms_db < -45:
            broken.append((r["clip_id"], fmt, n - r["expected_samples"], round(rms_db, 1)))

    print(f"{len(ok)} clips, {minutes:.1f} min of audio; not extracted: {failed}")
    for b in broken:
        print("  BROKEN (clip, format, sample delta, rms dBFS):", b)
    if broken or len(ok) != 99 or failed != {"GUVrL5ltiP4": "unavailable"}:
        sys.exit("audio does not match the reference run")
    print("all 99: 16 kHz, mono, 16-bit, exact sample count, not silent")


EXPECTED_REF_SHA256 = "e505186dc1cd00ea79301effbc554dc01951d2aa2ec6d39dc4a739d2a4e36dc1"


def refs(args):
    # Stage 2 is a pure function of the xlsx, so unlike the audio it CAN be checked
    # byte for byte, against the ref/ every later stage consumed. clip_meta.csv is
    # left out: its has_audio column reads the Stage 1 manifest.
    import pandas as pd

    root = Path(args.data) / "ref"
    files = sorted(list((root / "rttm").glob("*.rttm")) + list((root / "segments").glob("*.json")),
                   key=lambda p: (p.parent.name, p.name))
    m = hashlib.sha256()
    for p in files:
        m.update(p.parent.name.encode() + b"/" + p.name.encode() + b"\0")
        m.update(p.read_bytes().replace(b"\r\n", b"\n"))   # the reference digest was taken on Windows
    print(f"this run : {m.hexdigest()}  ({len(files)} files)")
    print(f"expected : {EXPECTED_REF_SHA256}  (200 files)")
    if (m.hexdigest(), len(files)) != (EXPECTED_REF_SHA256, 200):
        sys.exit("ref/ differs from the reference run")
    print("ref/rttm + ref/segments: IDENTICAL to the reference run")

    meta = pd.read_csv(root / "clip_meta.csv")
    print(f"{len(meta)} clips, {meta.duration.sum() / 3600:.2f} h, {int(meta.has_audio.sum())} with audio; "
          f"overlap {meta.overlap_sec.sum() / meta.speech_sec.sum() * 100:.2f}% of speech")


def diar_smoke(args):
    # A diarizer on a GPU is not bit-reproducible, so this reports rather than
    # asserts: DER of the live RTTM, scored against the reference run's RTTM for
    # the same clip -- 0 means identical turns.
    from pyannote.metrics.diarization import DiarizationErrorRate
    from stage3_score import load_rttm

    for system in args.systems:
        live_dir = Path(args.smoke) / "hyp" / system / "rttm"
        for f in sorted(live_dir.glob("*.rttm")) if live_dir.is_dir() else []:
            ref = load_rttm(Path(args.cache) / "hyp" / system / "rttm" / f.name)
            if not ref:
                print(f"{system:18s} {f.stem}  (no reference-run RTTM: out of memory on Kaggle)")
                continue
            der = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, load_rttm(f))
            print(f"{system:18s} {f.stem}  DER against the reference run's output: {100 * der:5.2f}%")


def expected(args):
    import pandas as pd

    def canonical(df):
        # Row order follows the order systems were passed on the command line,
        # which differs between this notebook and the Kaggle runs; rows must not.
        keys = [c for c in df.columns if df[c].dtype == object]
        return df.sort_values(keys).reset_index(drop=True)

    ours_dir, exp_dir = Path(args.data) / "results", Path(args.expected)
    diffs = []
    for name in ("diarization_per_clip.csv", "diarization_summary.csv", "asr_per_clip.csv",
                 "asr_summary.csv", "results_per_video.csv"):
        ours, theirs = canonical(pd.read_csv(ours_dir / name)), canonical(pd.read_csv(exp_dir / name))
        try:
            pd.testing.assert_frame_equal(ours, theirs, check_exact=False, rtol=1e-6, atol=1e-6)
            print(f"{name:26s} identical to the committed table  {ours.shape}")
        except AssertionError as exc:
            diffs.append(name)
            print(f"{name:26s} DIFFERS: {str(exc)[:300]}")
    same_md = ((ours_dir / "results_table.md").read_text(encoding="utf-8")
               == (exp_dir / "results_table.md").read_text(encoding="utf-8"))
    print(f"{'results_table.md':26s} {'identical' if same_md else 'DIFFERS'}")
    if diffs or not same_md:
        if args.strict:
            sys.exit("the CPU stages did not reproduce the committed results")
        print("\nDifferences from the committed numbers: GPU non-determinism and model or library "
              "versions in a full GPU run, reported rather than asserted.")
    else:
        print("\nEvery CPU stage reproduced the committed results exactly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audio"); a.add_argument("--manifest"); a.add_argument("--wav-dir")
    r = sub.add_parser("refs"); r.add_argument("--data")
    d = sub.add_parser("diar-smoke"); d.add_argument("--smoke"); d.add_argument("--cache")
    d.add_argument("--systems", nargs="+")
    e = sub.add_parser("expected"); e.add_argument("--data"); e.add_argument("--expected")
    e.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    {"audio": audio, "refs": refs, "diar-smoke": diar_smoke, "expected": expected}[args.cmd](args)
'''


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

S1_MD = """
## Stage 1 — audio extraction

`stage1_extract.py` downloads each video once with yt-dlp, cuts every window with
ffmpeg **output seeking** (`-ss` after `-i`, which decodes up to the mark and is
sample-exact, where input seeking snaps to a keyframe and drifts by up to a second),
and writes 16 kHz mono 16-bit WAV. The manifest is append-only and resumable.

**It is not re-run here.** YouTube bot-gates Colab and Kaggle IPs; from Kaggle, adding
cookies moved the failure to "The page needs to be reloaded" (yt-dlp #17389), and the
`web_embedded` client then served no audio stream. The reference run, from a home
connection, extracted **99 of 100 clips**; `GUVrL5ltiP4` has been removed from YouTube.

What this notebook does instead is copy that audio from Drive to local disk and
**verify** it. There is no audio checksum, because YouTube can serve a different
encode to a different client; what the trim controls is length, and length is exact.

The command that produced the audio:

    python stage1_extract.py --input youtube_segments_final.xlsx --out data --workers 3
"""

S1_COPY = r"""
# Local disk, not the Drive mount: every GPU stage reads all 99 files, and the
# Drive FUSE mount is far slower for that than /content.
AUDIO = WORKDIR / "wav" if ON_COLAB else DRIVE / "wav"
if ON_COLAB and len(list(AUDIO.glob("*.wav"))) != 99:
    shutil.copytree(DRIVE / "wav", AUDIO, dirs_exist_ok=True)
shutil.copy(DRIVE / "manifest.jsonl", DATA / "manifest.jsonl")
print(AUDIO, len(list(AUDIO.glob("*.wav"))), "wavs")

run("stage 1: verify audio", ["nb_checks.py", "audio", "--manifest", DATA / "manifest.jsonl", "--wav-dir", AUDIO])
"""


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

S2_MD = """
## Stage 2 — reference parsing

A pure function of the spreadsheet, so unlike the audio it is checked **byte for
byte** against the `ref/` every later stage consumed. What it decides, not just
formats:

- the two label columns are joined by index, and a turn-count or timestamp
  disagreement stops the run rather than giving one speaker's words to another;
- turns are clipped to the window (88 truncated, 38 outside it dropped);
- overlap counts only time with two or more *distinct* speakers.

**Ground truth never enters the pipeline.** `ref/` is read by the scorers and by
diagnostics labelled *oracle*, never by a model or a correction step.
"""

S2_RUN = r"""
run("stage 2: parse references",
    ["stage2_parse_refs.py", "--input", DRIVE / "youtube_segments_final.xlsx",
     "--out", "data", "--manifest", "data/manifest.jsonl"])
run("stage 2: verify references", ["nb_checks.py", "refs", "--data", "data"])
"""


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

S3_MD = """
## Stage 3 — baseline diarization

- **pyannote 3.1** (`pyannote/speaker-diarization-3.1`, gated: needs `HF_TOKEN`).
- **Sortformer 4spk-v1**, offline and streaming. Offline attention is O(T²) and ran
  out of memory on the 25 longest clips on a T4; those count as total miss, so its row
  measures memory, not the model. The streaming mode (speaker cache + FIFO) fits.

**Metric policy:** `collar = 0`, **overlap scored**, UEM = the whole clip. Published
numbers usually use a 0.25 s collar and skip overlap; with that policy pyannote would
report 20.58 here instead of 27.34.
"""

S3_GPU = r"""
DIAR_SYSTEMS = ("pyannote31", "sortformer", "sortformer_stream")
envs = {}
if HAVE_GPU:
    envs["pyannote31"] = gpu_env("pyannote", "pyannote.audio")
    envs["sortformer"] = envs["sortformer_stream"] = gpu_env("nemo", "nemo_toolkit[asr]")

for system in DIAR_SYSTEMS:
    label = f"stage 3: diarize {system}"
    if not HAVE_GPU:
        skip(label, "no GPU")
    elif envs[system] is None:
        skip(label, "environment install failed")
    elif system == "pyannote31" and not HF_TOKEN:
        skip(label, "no HF_TOKEN (pyannote/speaker-diarization-3.1 is gated)")
    else:
        run(label, ["stage3_diarize.py", "--system", system, "--data", GPU_ROOT,
                    "--wav-dir", AUDIO] + GPU_LIMIT, python=envs[system], may_fail=GPU_MAY_FAIL)
"""

S3_COMPARE = r"""
if not FULL_GPU_RUN:
    if (WORKDIR / "smoke" / "hyp").is_dir():
        run("check: diarization smoke vs reference run",
            ["nb_checks.py", "diar-smoke", "--smoke", WORKDIR / "smoke", "--cache", CACHE,
             "--systems", *DIAR_SYSTEMS], may_fail=True)
    for system in DIAR_SYSTEMS:
        restore(f"hyp/{system}")
"""

S3_SCORE = r"""
# --diagnostic adds the metric-policy sensitivity table (collar, overlap) and the
# overlapped vs single-speaker error split quoted in the writeup.
run("stage 3: score diarizers",
    ["stage3_score.py", "--data", "data", "--diagnostic", "--systems", *DIAR_SYSTEMS])
"""


# ---------------------------------------------------------------------------
# Stage 4a
# ---------------------------------------------------------------------------

S4A_MD = """
## Stage 4a — ASR: transcribe the whole clip once, then assign words to speakers

ASR never sees a speaker label. Words with timestamps from the whole clip are
assigned to diarizer turns in Stage 4b, so every diarizer and every Stage 5
correction is scored on **identical words**, and a cpWER difference between them is
purely labelling. It also means 99 ASR calls instead of 12,809, and Whisper never has
to transcribe a sub-second fragment.

- **Whisper large-v3** via faster-whisper, `temperature=0.0`. The default temperature
  fallback samples **unseeded**: one clip gave 87, 84 and 100 words on three runs, and
  211 every time at temperature 0.
- **IndicConformer-600M** (AI4Bharat) via ONNX, CTC branch. Its CTC head is
  *multisoftmax*: one softmax per language, so logits from different languages are on
  incomparable scales. The decode picks the clip's language by frame vote and argmaxes
  within it (WER 93.66 → 78.82). `indicconformer_free`, the naive global argmax, is
  kept as a scored ablation.

**Separate environments are not optional here.** On Kaggle, installing `onnxruntime-gpu`
next to faster-whisper replaced the cuDNN that CTranslate2 needs, and Whisper fell
back to CPU with no error. Each runs in its own venv. `onnxruntime-gpu` is pinned to
1.20.2: newer releases are built for CUDA 13.
"""

S4A_WHISPER = r"""
label = "stage 4a: ASR whisper"
if not HAVE_GPU:
    skip(label, "no GPU")
elif (WHISPER_PY := gpu_env("whisper", "faster-whisper")) is None:
    skip(label, "environment install failed")
else:
    run(label, ["stage4_asr.py", "--system", "whisper", "--data", GPU_ROOT,
                "--wav-dir", AUDIO] + GPU_LIMIT, python=WHISPER_PY, may_fail=GPU_MAY_FAIL)
"""

S4A_INDIC = r"""
# The ONNX export is a gated repo as well, so without a token this fails mid-download
# with GatedRepoError rather than at the start.
INDIC_PY = gpu_env("indic", "onnxruntime-gpu==1.20.2", "librosa") if (HAVE_GPU and HF_TOKEN) else None
for system in ("indicconformer", "indicconformer_free"):
    label = f"stage 4a: ASR {system}"
    if not HAVE_GPU:
        skip(label, "no GPU")
    elif not HF_TOKEN:
        skip(label, "no HF_TOKEN (ai4bharat/indic-conformer-600m-multilingual is gated)")
    elif INDIC_PY is None:
        skip(label, "environment install failed")
    else:
        # The script prints its onnxruntime providers; CUDAExecutionProvider must be
        # among them, or this is the (slow, correct) CPU fallback.
        run(label, ["stage4_asr.py", "--system", system, "--data", GPU_ROOT,
                    "--wav-dir", AUDIO] + GPU_LIMIT, python=INDIC_PY, may_fail=GPU_MAY_FAIL)
"""

S4A_COMPARE = r"""
import difflib

def words(path):
    return [w["w"] for w in json.loads(path.read_text(encoding="utf-8"))["words"]]

if not FULL_GPU_RUN:
    for system in ("whisper", "indicconformer", "indicconformer_free"):
        live_dir = WORKDIR / "smoke" / "asr" / system / "words"
        for f in sorted(live_dir.glob("*.json")) if live_dir.is_dir() else []:
            live, ref = words(f), words(CACHE / "asr" / system / "words" / f.name)
            sim = difflib.SequenceMatcher(None, live, ref, autojunk=False).ratio()
            print(f"{system:20s} {f.stem}  {len(live):5d} words vs {len(ref):5d} in the reference run, "
                  f"{100 * sim:5.1f}% identical sequence")
    for system in ("whisper", "indicconformer", "indicconformer_free"):
        restore(f"asr/{system}")
"""


# ---------------------------------------------------------------------------
# Stage 4b
# ---------------------------------------------------------------------------

S4B_MD = """
## Stage 4b — language-ID fallback (the adopted improvement) and attribution

**Language-ID fallback.** On 13 clips IndicConformer's own vote picks Urdu (11) or
Nepali (2), outside the languages the task serves, and the whole clip comes out in the
wrong script: 100% WER whatever was heard. Spoken Hindi and Urdu are nearly identical
and differ mainly in script, so this is real ambiguity, not a decoder bug. The rule:
if IndicConformer's detected language is outside {hi, mr, bn, gu, kn, ml, or, pa, ta,
te}, take Whisper's words for that clip. **No reference is read.** It is written as a
fourth ASR system, `ic_lid_fallback`.

**Attribution rules.** A word goes to the turn it overlaps most (ties to the earlier
turn). A word with no turn is kept and flagged, never dropped: dropping it would
reward a diarizer for missing speech. `--diar ref` is an *oracle* diagnostic.

Expected: the offline `sortformer` conditions exit non-zero, with **25 clips** failing
`no turns in RTTM` — the out-of-memory clips from Stage 3. A clip with no hypothesis
is not a clip where nobody spoke, so it fails loudly instead of being written empty.
The next cell checks that those are the only failures.
"""

S4B_RUN = r"""
run("stage 4b: LID fallback", ["stage4_fallback.py", "--data", "data"])

ASR = ["indicconformer", "indicconformer_free", "whisper", "ic_lid_fallback"]
DIAR = list(DIAR_SYSTEMS)
run("stage 4b: attribute words", ["stage4_attribute.py", "--asr", *ASR,
                                  "--diar", *DIAR, "ref", "--data", "data"], may_fail=True)
"""

S4B_CHECK = r"""
from collections import Counter

failures = {}
for m in sorted((DATA / "attrib").glob("*/manifest.jsonl")):
    last = {}
    for line in m.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["clip_id"]] = r
    bad = [r for r in last.values() if r["status"] != "ok"]
    if bad:
        failures[m.parent.name] = Counter(r["error"] for r in bad)
for cond, errs in failures.items():
    print(cond, dict(errs))
assert set(failures) == {f"{a}__sortformer" for a in ASR}, "unexpected attribution failures"
assert all(errs == {"ValueError: no turns in RTTM": 25} for errs in failures.values())
for row in REPORT:
    if row["step"] == "stage 4b: attribute words" and row["exit"] == 1:
        row["status"] = "ok (expected: 25 OOM sortformer clips)"
print(f"{len(list((DATA / 'attrib').iterdir()))} conditions; the only failures are the "
      "25 out-of-memory sortformer clips, as expected")
"""


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------

S5_MD = """
## Stage 5 — speaker relabelling from the transcript (built, measured, not adopted)

After DiarizationLM and lexical speaker error correction. Transcripts are cut into
same-speaker runs split at pauses over 0.5 s. A unit may only be moved to an existing
speaker, so **the text cannot change and WER must not move** — asserted per clip.

- **Ceiling first** (`--audit`, an oracle diagnostic that edits nothing): even after
  pause splitting, 37.87% of words sit in units spanning two true speakers, where no
  relabel can help.
- **Rule baseline** (CPU): a unit under 1 s joins its nearer neighbour.
- **LLM** (GPU): Qwen2.5-7B-Instruct in 4-bit sees 25 units at a time and returns
  JSON edits with confidences; below 0.7 is dropped, and a clip where it tries to edit
  over 30% of units is discarded. It made WDER worse (20.12 → 20.96).
"""

S5_RULE = r"""
BASE = [f"{a}__{d}" for a in ASR for d in DIAR]
run("stage 5: relabel audit (oracle)", ["stage5_correct.py", "--cond", "indicconformer__pyannote31",
                                        "indicconformer__sortformer_stream", "--audit", "--data", "data"])
run("stage 5: rule relabel", ["stage5_correct.py", "--method", "rule", "--data", "data", "--cond", *BASE])
"""

S5_LLM = r"""
LLM_COND = "indicconformer__pyannote31"
label = "stage 5: LLM relabel"
if not HAVE_GPU:
    skip(label, "no GPU")
elif (LLM_PY := gpu_env("llm", "bitsandbytes", "accelerate")) is None:
    skip(label, "environment install failed")
else:
    if not FULL_GPU_RUN:
        # The smoke pass reads the attribution just computed, in its own root.
        shutil.copytree(DATA / "attrib" / LLM_COND, WORKDIR / "smoke" / "attrib" / LLM_COND,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("manifest.jsonl"))
    run(label, ["stage5_correct.py", "--cond", LLM_COND, "--method", "llm",
                "--data", GPU_ROOT] + GPU_LIMIT, python=LLM_PY, may_fail=GPU_MAY_FAIL)

if not FULL_GPU_RUN:
    live_dir = WORKDIR / "smoke" / "attrib" / f"{LLM_COND}+llm"
    for f in sorted(live_dir.glob("*.json")) if live_dir.is_dir() else []:
        live = json.loads(f.read_text(encoding="utf-8"))
        ref = json.loads((CACHE / "attrib" / f"{LLM_COND}+llm" / f.name).read_text(encoding="utf-8"))
        same = sum(a["spk"] == b["spk"] for a, b in zip(live["words"], ref["words"]))
        print(f"{f.stem}  {live['stage5']['n_applied']} edits live vs {ref['stage5']['n_applied']} "
              f"in the reference run; {100 * same / len(ref['words']):.1f}% of word labels identical")
    restore(f"attrib/{LLM_COND}+llm")
"""


# ---------------------------------------------------------------------------
# Stage 4c / 5b
# ---------------------------------------------------------------------------

S4C_MD = """
## Stage 4c — cpWER, WDER (and WER, DI-cpWER)

- **WER** ignores speakers, so it must be identical across every diarizer and every
  relabel of one ASR system — the tripwire for a leak between stages.
- **cpWER** (MeetEval): the best speaker permutation, then WER.
- **WDER**: hand-rolled after Shafey et al., since MeetEval has none.
- Corpus rates are **error-weighted** (total errors / total reference words).
"""

S4C_RUN = r"""
run("stage 4c: score ASR", ["stage4_score.py", "--data", "data"])
"""

S5B_MD = """
## Stage 5b — DER / JER for the word-level conditions

DER is measured over time, but Stage 5 and the fallback change word labels. Each
condition's word labels are projected back onto **the diarizer's own turns**
(duration-weighted majority), so boundaries never move and missed speech and false
alarm are identical to the baseline — any change is speaker confusion alone.

The projection is lossy by itself: with **zero edits** it raises pyannote's DER from
27.34 to 29.03. So the unedited conditions are projected too, and serve as the control
(`DER_ctrl`) that every corrected DER is read against.
"""

S5B_RUN = r"""
corrected = sorted(p.name for p in (DATA / "attrib").iterdir() if p.is_dir() and "+" in p.name)
run("stage 5b: project to RTTM", ["stage5_to_rttm.py", "--data", "data", "--cond", *corrected, *BASE])

# No --diagnostic here: it only prints, after the summaries are written, and on
# 28 systems it is the slowest CPU step in the notebook.
systems = sorted(p.name for p in (DATA / "hyp").iterdir() if p.is_dir())
run("stage 5b: score DER/JER", ["stage3_score.py", "--data", "data", "--systems", *systems])
"""


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------

S6_MD = """
## Stage 6 — results table: baseline vs improved

Regroups the per-clip error counts from Stages 3–5; runs no model. Before writing, it
refuses to proceed unless WER is identical across relabelling methods on every clip,
missed speech and false alarm are identical across baseline, control and corrected
RTTMs, every `ic_lid_fallback` row reproduces the system it took its words from, and
every corpus figure re-derives to the Stage 3 and Stage 4 summaries.

Writes `data/results/results_table.md`, `results_per_video.csv` and
`results_per_video.xlsx`.
"""

S6_RUN = r"""
run("stage 6: results table", ["stage6_report.py", "--data", "data"])
"""

S6_SHOW = r"""
from IPython.display import Markdown, display

table = (DATA / "results" / "results_table.md").read_text(encoding="utf-8")
# The headline section: the best combination and what was built on it.
display(Markdown(table.split("## 2.")[0]))
"""

S6_EXPECTED = r"""
# Smoke mode: the CPU stages ran on exactly the reference run's GPU outputs, so
# every table must match the committed one. A full GPU run is compared, not asserted.
run("check: results vs committed tables",
    ["nb_checks.py", "expected", "--data", "data", "--expected", EXPECTED]
    + ([] if FULL_GPU_RUN else ["--strict"]))
"""


REPORT_MD = """
## Run report

Every step this session ran, skipped or failed. In smoke mode a failed GPU stage
does not invalidate the tables above — they are built from the cached full outputs —
but it does mean that stage's code was not demonstrated in this environment.
"""

REPORT_SHOW = r"""
width = max(len(r["step"]) for r in REPORT)
print(f"{'step':<{width}}  {'status':<42} {'exit':>4} {'min':>6}")
for r in REPORT:
    ex = "" if r["exit"] is None else r["exit"]
    print(f"{r['step']:<{width}}  {r['status']:<42} {ex:>4} {r['min']:6.1f}")
"""


def build() -> None:
    cells = [
        md(HEADER),
        code(CONFIG),
        code(ENV),
        code(CPU_DEPS),
        md(CHECKS_MD),
        code("%%writefile nb_checks.py\n" + CHECKS.lstrip("\n")),

        md(S1_MD),
        embed("stage1_extract.py"),
        code(S1_COPY),

        md(S2_MD),
        embed("stage2_parse_refs.py"),
        code(S2_RUN),

        md(S3_MD),
        embed("stage3_diarize.py"),
        embed("stage3_score.py"),
        code(S3_GPU),
        code(S3_COMPARE),
        code(S3_SCORE),

        md(S4A_MD),
        embed("stage4_asr.py"),
        code(S4A_WHISPER),
        code(S4A_INDIC),
        code(S4A_COMPARE),

        md(S4B_MD),
        embed("stage4_attribute.py"),
        embed("stage4_fallback.py"),
        code(S4B_RUN),
        code(S4B_CHECK),

        md(S5_MD),
        embed("stage5_correct.py"),
        code(S5_RULE),
        code(S5_LLM),

        md(S4C_MD),
        embed("stage4_score.py"),
        code(S4C_RUN),

        md(S5B_MD),
        embed("stage5_to_rttm.py"),
        code(S5B_RUN),

        md(S6_MD),
        embed("stage6_report.py"),
        code(S6_RUN),
        code(S6_SHOW),
        code(S6_EXPECTED),

        md(REPORT_MD),
        code(REPORT_SHOW),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    build()
