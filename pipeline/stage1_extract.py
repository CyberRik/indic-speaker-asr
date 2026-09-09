#!/usr/bin/env python3
"""
Stage 1 -- YouTube audio extraction.

Reads the segment table, downloads the audio track for each unique video_id,
cuts each [start_sec, end_sec] window to a 16 kHz mono PCM WAV, and records one
line per clip in an append-only JSONL manifest.

Design notes:
  * Network work is keyed on video_id, cutting is keyed on clip_id, so a video
    contributing several windows is downloaded exactly once.
  * The trim uses ffmpeg OUTPUT seeking (-ss/-t placed after -i). ffmpeg decodes
    from zero and discards samples up to the mark, which is sample-exact.
    Input seeking (-ss before -i) snaps to a keyframe and can drift ~1s.
  * Every cut is verified against round((end-start) * 16000) samples. The
    verification is the deliverable, not the absence of a traceback.
  * The manifest is append-only. A killed session loses at most its last line.

Usage:
    python stage1_extract.py --input youtube_segments_final.xlsx --out /kaggle/working/data
    python stage1_extract.py --input ... --out ... --limit 10        # dev subset
    python stage1_extract.py --input ... --out ... --retry-permanent # force retry
    python stage1_extract.py --input ... --out ... --report-only     # summary only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TARGET_SR = 16_000
TARGET_CHANNELS = 1
SAMPLE_TOLERANCE = 16          # +/- 1 ms at 16 kHz; anything more is real drift
REQUIRED_COLUMNS = [
    "video_id", "youtube_link", "start_sec", "end_sec",
    "diarization_segments", "asr_segments",
]

# Which YouTube player clients yt-dlp should try. This is the single most
# version-sensitive knob in the whole stage -- what defeats bot-gating changes
# every few weeks. Override without editing code:
#     export YTDLP_PLAYER_CLIENTS="tv,web_safari,ios"
# Set to "" to let yt-dlp use its own defaults.
PLAYER_CLIENTS = os.environ.get("YTDLP_PLAYER_CLIENTS", "default,tv,web_safari")

# Optional Netscape-format cookie jar; the reliable answer to bot-gating.
#     export YTDLP_COOKIES=/kaggle/input/yt-cookies/cookies.txt
COOKIES_FILE = os.environ.get("YTDLP_COOKIES", "")

# How to invoke yt-dlp. Prefer the console script; fall back to the module, which
# is what you get when pip installs into a Scripts/bin dir that is not on PATH.
def _resolve_ytdlp() -> list[str]:
    override = os.environ.get("YTDLP_CMD")
    if override:
        return override.split()
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        probe = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass
    return []


# Failure classes we will NOT retry on a rerun -- the video is simply gone.
PERMANENT_FAILURES = {"unavailable", "private", "removed", "age_gated", "no_audio"}

# Populated by preflight().
YTDLP_CMD: list[str] = []


# --------------------------------------------------------------------------
# Manifest records
# --------------------------------------------------------------------------

@dataclass
class ClipRecord:
    clip_id: str
    video_id: str
    youtube_link: str
    start_sec: float
    end_sec: float
    status: str                      # ok | short_source | failed
    wav_path: str | None = None
    n_samples: int | None = None
    expected_samples: int | None = None
    sample_delta: int | None = None
    sample_rate: int | None = None
    duration_sec: float | None = None
    error_class: str | None = None
    error_msg: str | None = None
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    """Append-only JSONL manifest with an in-memory index for resume."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Truncated final line from a killed session. Expected.
                        continue
                    self.records[rec["clip_id"]] = rec

    def append(self, rec: ClipRecord) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.records[rec.clip_id] = asdict(rec)

    def should_skip(self, clip_id: str, retry_permanent: bool) -> bool:
        """True if this clip is already done (and still verifiably done)."""
        rec = self.records.get(clip_id)
        if rec is None:
            return False

        if rec["status"] in ("ok", "short_source"):
            # Trust but verify: the WAV must still exist with the right length.
            wav = Path(rec["wav_path"]) if rec.get("wav_path") else None
            if wav is None or not wav.exists():
                return False
            try:
                if wav_num_frames(wav) != rec.get("n_samples"):
                    return False
            except Exception:
                return False
            return True

        if rec["status"] == "failed":
            if retry_permanent:
                return False
            return rec.get("error_class") in PERMANENT_FAILURES

        return False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clip_id_for(video_id: str, start: float, end: float) -> str:
    """Stable per-window identity. Millisecond precision keeps it filename-safe."""
    return f"{video_id}__{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def wav_num_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


def wav_info(path: Path) -> tuple[int, int, int]:
    """(n_frames, sample_rate, n_channels)"""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes(), wf.getframerate(), wf.getnchannels()


def classify_error(text: str) -> str:
    """Map yt-dlp / ffmpeg stderr onto a failure class we can act on."""
    t = (text or "").lower()
    checks = [
        ("bot_gated",   [r"sign in to confirm", r"not a bot", r"confirm you.re not a bot"]),
        ("geo_blocked", [r"not available in your country", r"geo restrict",
                         r"blocked it in your country"]),
        ("private",     [r"private video", r"this video is private"]),
        ("removed",     [r"removed by the uploader", r"account associated with this video has been terminated",
                         r"video has been removed"]),
        ("unavailable", [r"video unavailable", r"is unavailable", r"does not exist",
                         r"has been deleted", r"members-only", r"join this channel"]),
        ("age_gated",   [r"age-restricted", r"inappropriate for some users"]),
        ("no_audio",    [r"requested format is not available", r"no audio",
                         r"only images are available"]),
        ("network",     [r"timed out", r"timeout", r"connection reset",
                         r"temporary failure in name resolution", r"unable to download",
                         r"http error 5\d\d", r"429", r"too many requests"]),
    ]
    for cls, patterns in checks:
        for p in patterns:
            if re.search(p, t):
                return cls
    return "unknown"


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


class StageError(Exception):
    def __init__(self, error_class: str, message: str):
        super().__init__(message)
        self.error_class = error_class
        self.message = message


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Load the segment table from .xlsx/.xls or .csv/.tsv, dispatching on suffix."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise SystemExit(f"Reading {suffix} needs openpyxl: pip install openpyxl  ({exc})")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_table(path: Path, limit: int | None, only_ids: set[str] | None) -> pd.DataFrame:
    df = read_table(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV is missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)}\n"
            "Stage 1 was written against the expected segment-table schema; "
            "if the real CSV differs, fix the mapping before going further."
        )

    n_before = len(df)
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    bad = df["start_sec"].isna() | df["end_sec"].isna() | (df["end_sec"] <= df["start_sec"])
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} row(s) with unusable start/end:", file=sys.stderr)
        for _, r in df[bad].head(10).iterrows():
            print(f"        {r['video_id']}  start={r['start_sec']}  end={r['end_sec']}", file=sys.stderr)
        df = df[~bad].copy()

    df["clip_id"] = [
        clip_id_for(str(r.video_id), float(r.start_sec), float(r.end_sec))
        for r in df.itertuples()
    ]

    dupes = int(df["clip_id"].duplicated().sum())
    if dupes:
        print(f"[warn] {dupes} duplicate clip_id row(s); keeping first of each", file=sys.stderr)
        df = df.drop_duplicates(subset="clip_id", keep="first").copy()

    if only_ids:
        df = df[df["video_id"].astype(str).isin(only_ids)].copy()
    if limit:
        df = df.head(limit).copy()

    print(f"[csv ] {n_before} rows in -> {len(df)} clips, "
          f"{df['video_id'].nunique()} unique videos, "
          f"{df['end_sec'].sub(df['start_sec']).sum() / 3600:.2f}h of audio requested "
          f"(~{df['end_sec'].sub(df['start_sec']).sum() * TARGET_SR * 2 / 1e9:.2f} GB of WAV)")
    return df


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_audio(video_id: str, url: str, raw_dir: Path, timeout: int) -> Path:
    """Fetch the best audio-only stream for one video. Returns the raw file path."""
    existing = sorted(raw_dir.glob(f"{video_id}.*"))
    if existing:
        return existing[0]

    out_tmpl = str(raw_dir / "%(id)s.%(ext)s")
    cmd = [
        *YTDLP_CMD,
        "--no-playlist",
        "--no-progress",
        "--no-warnings",
        "-f", "bestaudio/best",
        "--retries", "3",
        "--fragment-retries", "5",
        "--socket-timeout", "30",
        "--sleep-requests", "1",
        "-o", out_tmpl,
        # --print implies --simulate in current yt-dlp, so --no-simulate is
        # required to both download AND report the resulting path.
        "--no-simulate",
        "--print", "after_move:filepath",
    ]
    if PLAYER_CLIENTS:
        cmd += ["--extractor-args", f"youtube:player_client={PLAYER_CLIENTS}"]
    if COOKIES_FILE:
        cmd += ["--cookies", COOKIES_FILE]
    cmd.append(url)

    try:
        proc = run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise StageError("network", f"yt-dlp timed out after {timeout}s")

    if proc.returncode != 0:
        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        raise StageError(classify_error(blob), blob.strip()[-600:])

    # Preferred: the path yt-dlp printed. Fallback: glob, in case --print
    # semantics have shifted again.
    path = None
    for line in reversed((proc.stdout or "").splitlines()):
        cand = Path(line.strip())
        if line.strip() and cand.exists():
            path = cand
            break
    if path is None:
        found = sorted(raw_dir.glob(f"{video_id}.*"))
        path = found[0] if found else None
    if path is None:
        raise StageError("unknown", "yt-dlp exited 0 but produced no file")
    return path


# --------------------------------------------------------------------------
# Cut + verify
# --------------------------------------------------------------------------

def cut_clip(raw: Path, start: float, end: float, out_wav: Path, timeout: int) -> tuple[int, int, int]:
    """Cut [start, end) to 16 kHz mono PCM WAV. Returns (frames, sr, channels)."""
    duration = end - start
    tmp = out_wav.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(raw),
        # -ss/-t AFTER -i == output seeking == decode-and-discard == sample exact.
        "-ss", f"{start:.6f}",
        "-t", f"{duration:.6f}",
        "-map", "0:a:0",
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le",
        str(tmp),
    ]
    try:
        proc = run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        raise StageError("ffmpeg", f"ffmpeg timed out after {timeout}s")

    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise StageError("ffmpeg", (proc.stderr or "ffmpeg produced no output").strip()[-600:])

    try:
        frames, sr, ch = wav_info(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise StageError("ffmpeg", f"output WAV unreadable: {exc}")

    # Atomic publish: downstream stages never observe a half-written WAV.
    os.replace(tmp, out_wav)
    return frames, sr, ch


def process_clip(row, raw: Path, wav_dir: Path, timeout: int) -> ClipRecord:
    start, end = float(row.start_sec), float(row.end_sec)
    expected = int(round((end - start) * TARGET_SR))
    out_wav = wav_dir / f"{row.clip_id}.wav"

    frames, sr, ch = cut_clip(raw, start, end, out_wav, timeout)
    delta = frames - expected

    if sr != TARGET_SR or ch != TARGET_CHANNELS:
        raise StageError("ffmpeg", f"wrong format: sr={sr} ch={ch}")

    # A short result almost always means end_sec runs past the true video
    # duration. Keep the audio, but mark it so Stage 3+ can exclude or
    # renormalise rather than silently scoring against a truncated clip.
    status = "ok" if abs(delta) <= SAMPLE_TOLERANCE else "short_source"

    return ClipRecord(
        clip_id=row.clip_id,
        video_id=str(row.video_id),
        youtube_link=str(row.youtube_link),
        start_sec=start,
        end_sec=end,
        status=status,
        wav_path=str(out_wav),
        n_samples=frames,
        expected_samples=expected,
        sample_delta=delta,
        sample_rate=sr,
        duration_sec=round(frames / sr, 6),
        error_class=None if status == "ok" else "short_source",
        error_msg=None if status == "ok" else f"{delta:+d} samples vs expected {expected}",
    )


# --------------------------------------------------------------------------
# Per-video worker
# --------------------------------------------------------------------------

def process_video(video_id, rows, raw_dir, wav_dir, manifest, args) -> list[ClipRecord]:
    """Download one video once, then cut every window the CSV asks for from it."""
    out: list[ClipRecord] = []
    raw = None
    try:
        raw = download_audio(video_id, str(rows[0].youtube_link), raw_dir, args.download_timeout)
    except StageError as exc:
        for row in rows:
            rec = ClipRecord(
                clip_id=row.clip_id, video_id=str(video_id),
                youtube_link=str(row.youtube_link),
                start_sec=float(row.start_sec), end_sec=float(row.end_sec),
                status="failed", error_class=exc.error_class, error_msg=exc.message,
            )
            manifest.append(rec)
            out.append(rec)
        return out

    for row in rows:
        try:
            rec = process_clip(row, raw, wav_dir, args.ffmpeg_timeout)
        except StageError as exc:
            rec = ClipRecord(
                clip_id=row.clip_id, video_id=str(video_id),
                youtube_link=str(row.youtube_link),
                start_sec=float(row.start_sec), end_sec=float(row.end_sec),
                status="failed", error_class=exc.error_class, error_msg=exc.message,
            )
        manifest.append(rec)
        out.append(rec)

    if not args.keep_raw and raw is not None:
        raw.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(manifest: Manifest, df: pd.DataFrame) -> None:
    recs = [manifest.records[c] for c in df["clip_id"] if c in manifest.records]
    status_counts = Counter(r["status"] for r in recs)
    total = len(df)

    print("\n" + "=" * 68)
    print("STAGE 1 SUMMARY")
    print("=" * 68)
    print(f"  clips requested   : {total}")
    for st in ("ok", "short_source", "failed"):
        n = status_counts.get(st, 0)
        pct = f"  ({n / total * 100:5.1f}%)" if total else ""
        print(f"  {st:<18}: {n:>4}{pct}")
    missing = total - len(recs)
    if missing:
        print(f"  {'not attempted':<18}: {missing:>4}")

    fails = [r for r in recs if r["status"] == "failed"]
    if fails:
        print("\n  failure classes:")
        for cls, n in Counter(r["error_class"] for r in fails).most_common():
            tag = "permanent" if cls in PERMANENT_FAILURES else "retryable"
            print(f"    {cls:<14} {n:>4}   ({tag})")

    good = [r for r in recs if r["status"] in ("ok", "short_source")]
    if good:
        deltas = [abs(r["sample_delta"]) for r in good]
        exact = sum(1 for d in deltas if d <= SAMPLE_TOLERANCE)
        secs = sum(r["duration_sec"] for r in good)
        print(f"\n  trim accuracy     : {exact}/{len(good)} clips sample-exact "
              f"(|delta| <= {SAMPLE_TOLERANCE} samples = 1 ms)")
        print(f"  worst |delta|     : {max(deltas)} samples "
              f"({max(deltas) / TARGET_SR * 1000:.1f} ms)")
        print(f"  audio on disk     : {secs / 60:.1f} min across {len(good)} clips")

    drifted = [r for r in good if abs(r["sample_delta"]) > SAMPLE_TOLERANCE]
    if drifted:
        print(f"\n  [!] {len(drifted)} clip(s) off-length. Inspect before trusting Stage 3:")
        for r in sorted(drifted, key=lambda x: -abs(x["sample_delta"]))[:8]:
            print(f"      {r['clip_id']}  {r['sample_delta']:+d} samples "
                  f"({r['sample_delta'] / TARGET_SR:+.3f}s)")
    print("=" * 68)


# --------------------------------------------------------------------------

def preflight() -> None:
    global YTDLP_CMD
    YTDLP_CMD = _resolve_ytdlp()
    if not YTDLP_CMD:
        raise SystemExit(
            "yt-dlp not found (neither on PATH nor as an importable module).\n"
            "  pip install -qU yt-dlp\n"
            "If it is installed but not on PATH, set YTDLP_CMD, e.g.\n"
            '  export YTDLP_CMD="python -m yt_dlp"'
        )
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "'ffmpeg' not found on PATH. On Kaggle it is preinstalled; "
            "locally, install it and reopen the shell."
        )
    for label, cmd in (("yt-dlp", YTDLP_CMD + ["--version"]),
                       ("ffmpeg", ["ffmpeg", "-version"])):
        try:
            print(f"[env ] {label}: {run(cmd, timeout=60).stdout.splitlines()[0]}")
        except Exception:
            pass
    if COOKIES_FILE:
        print(f"[env ] cookies: {COOKIES_FILE} "
              f"({'found' if Path(COOKIES_FILE).exists() else 'MISSING'})")
    print(f"[env ] player_clients: {PLAYER_CLIENTS or '(yt-dlp default)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1: extract 16 kHz mono WAV clips from YouTube.")
    ap.add_argument("--input", "--csv", dest="input", required=True, type=Path,
                    help="segment table: .xlsx, .csv or .tsv")
    ap.add_argument("--out", required=True, type=Path, help="output root (wav/, raw/, manifest.jsonl)")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel video downloads; >4 raises bot-gating risk")
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (dev subset)")
    ap.add_argument("--only-ids", default=None, help="comma-separated video_ids")
    ap.add_argument("--keep-raw", action="store_true", help="do not delete source audio after cutting")
    ap.add_argument("--retry-permanent", action="store_true",
                    help="also retry failures classed as permanent")
    ap.add_argument("--download-timeout", type=int, default=900)
    ap.add_argument("--ffmpeg-timeout", type=int, default=600)
    ap.add_argument("--report-only", action="store_true",
                    help="print summary from manifest, do nothing else")
    args = ap.parse_args()

    out_root: Path = args.out
    wav_dir, raw_dir = out_root / "wav", out_root / "raw"
    for d in (out_root, wav_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    only_ids = set(args.only_ids.split(",")) if args.only_ids else None
    df = load_table(args.input, args.limit, only_ids)
    manifest = Manifest(out_root / "manifest.jsonl")

    if args.report_only:
        print_summary(manifest, df)
        return 0

    preflight()

    by_video: dict[str, list] = defaultdict(list)
    n_skipped = 0
    for row in df.itertuples():
        if manifest.should_skip(row.clip_id, args.retry_permanent):
            n_skipped += 1
            continue
        by_video[str(row.video_id)].append(row)

    if n_skipped:
        print(f"[run ] resuming: {n_skipped} clip(s) already done, skipping")
    if not by_video:
        print("[run ] nothing to do.")
        print_summary(manifest, df)
        return 0

    n_todo = sum(len(v) for v in by_video.values())
    print(f"[run ] {n_todo} clip(s) across {len(by_video)} video(s), {args.workers} worker(s)\n")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_video, vid, rows, raw_dir, wav_dir, manifest, args): vid
            for vid, rows in by_video.items()
        }
        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                recs = fut.result()
            except Exception as exc:                      # worker crash, not a clip failure
                print(f"[ERR ] {vid}: worker crashed: {exc!r}", file=sys.stderr)
                continue
            done += len(recs)
            for r in recs:
                if r.status == "ok":
                    print(f"[ ok ] {r.clip_id}  {r.duration_sec:.3f}s  ({r.sample_delta:+d} samp)")
                elif r.status == "short_source":
                    print(f"[shrt] {r.clip_id}  {r.error_msg}")
                else:
                    tail = (r.error_msg or "").splitlines()
                    print(f"[FAIL] {r.clip_id}  [{r.error_class}] "
                          f"{(tail[-1] if tail else '')[:110]}")
            print(f"       ---- {done}/{n_todo} clips, {time.time() - t0:.0f}s elapsed")

    print_summary(manifest, df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
