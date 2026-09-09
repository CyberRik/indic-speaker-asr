#!/usr/bin/env python3
"""
Stage 3a -- Baseline diarization inference.

Runs one diarization system over the extracted WAVs and writes hypothesis RTTMs.
Inference only; scoring lives in stage3_score.py so that a metric bug never costs
another GPU run.

    data/hyp/<system>/rttm/<clip_id>.rttm
    data/hyp/<system>/manifest.jsonl

Systems:
    pyannote31    pyannote/speaker-diarization-3.1   (gated; needs HF token)
    community1    pyannote/speaker-diarization-community-1  (pyannote.audio 4.x)
    sortformer    nvidia/diar_sortformer_4spk-v1     (NeMo; HARD CAP 4 speakers)

Resumability: a clip is skipped when its RTTM exists and the manifest records it
ok. Killed sessions resume at the next clip.

Usage:
    python stage3_diarize.py --system pyannote31 --data data --hf-token $HF_TOKEN
    python stage3_diarize.py --system sortformer --data data --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------

SUPPORTED = ("pyannote31", "community1", "sortformer", "sortformer_stream")

MODEL_IDS = {
    "pyannote31": "pyannote/speaker-diarization-3.1",
    "community1": "pyannote/speaker-diarization-community-1",
    "sortformer": "nvidia/diar_sortformer_4spk-v1",
    # Same checkpoint as `sortformer`, run through NeMo's streaming path. Kept as
    # a separate system so the offline and streaming hypotheses can be scored
    # side by side on the clips where both succeeded.
    "sortformer_stream": "nvidia/diar_sortformer_4spk-v1",
}

# Sortformer is architecturally limited to 4 speakers. Recorded so the Stage 6
# breakdown can separate "model got it wrong" from "model could not represent it".
SPEAKER_CAP = {"sortformer": 4, "sortformer_stream": 4}

# Sortformer attends over the whole session, so peak VRAM grows as O(duration^2):
# a 913 s clip asked for 7.8 GiB and an 1822 s clip for 30.9 GiB on a 14.6 GiB T4.
# Streaming mode bounds that by processing fixed-length chunks and carrying speaker
# identity forward in a speaker cache + FIFO queue, so labels stay consistent
# across chunk boundaries without any stitching on our side.
STREAMING_ATTRS = ("chunk_len", "chunk_left_context", "chunk_right_context",
                   "fifo_len", "spkcache_len", "spkcache_update_period")


@dataclass
class HypRecord:
    clip_id: str
    system: str
    status: str                       # ok | failed
    n_speakers_pred: int | None = None
    n_segments: int | None = None
    speech_sec: float | None = None
    runtime_sec: float | None = None
    rtf: float | None = None          # runtime / audio duration
    error: str | None = None
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records[r["clip_id"]] = r

    def append(self, rec: HypRecord) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.records[rec.clip_id] = asdict(rec)

    def done(self, clip_id: str, rttm_dir: Path) -> bool:
        r = self.records.get(clip_id)
        if not r or r["status"] != "ok":
            return False
        return (rttm_dir / f"{clip_id}.rttm").exists()


# --------------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def write_rttm(path: Path, clip_id: str, turns: list[tuple[float, float, str]]) -> None:
    """turns = [(start, end, speaker)]. Written atomically."""
    lines = [
        f"SPEAKER {clip_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {str(spk).replace(' ', '_')} <NA> <NA>"
        for s, e, spk in turns if e > s
    ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class PyannoteBackend:
    """pyannote 3.1 / community-1. Same API surface, different checkpoint."""

    def __init__(self, system: str, hf_token: str | None, device: str):
        import pyannote.audio
        from pyannote.audio import Pipeline
        import torch

        model_id = MODEL_IDS[system]
        print(f"[env ] pyannote.audio {pyannote.audio.__version__}")

        # The auth kwarg was renamed in pyannote.audio 4.0:
        #   3.x -> use_auth_token=...     4.x -> token=...
        # Try in that order rather than pinning a version, so the same script
        # runs on whichever the environment happens to give us.
        last_err = None
        self.pipeline = None
        attempts = ([{"token": hf_token}, {"use_auth_token": hf_token}]
                    if hf_token else [{}])
        for kwargs in attempts:
            try:
                self.pipeline = Pipeline.from_pretrained(model_id, **kwargs)
                break
            except TypeError as exc:
                last_err = exc                       # wrong kwarg name; try the other
                continue
        if self.pipeline is None and last_err is not None:
            raise SystemExit(
                f"could not call Pipeline.from_pretrained for {model_id}: {last_err}\n"
                "Neither token= nor use_auth_token= was accepted -- check the "
                "pyannote.audio version printed above."
            )
        if self.pipeline is None:
            raise SystemExit(
                f"Pipeline.from_pretrained returned None for {model_id}.\n"
                "That almost always means the HF token is missing/invalid, or you have not\n"
                "accepted the user conditions. For 3.1 you must accept BOTH:\n"
                "  https://hf.co/pyannote/speaker-diarization-3.1\n"
                "  https://hf.co/pyannote/segmentation-3.0"
            )
        self.pipeline.to(torch.device(device))

    def __call__(self, wav: Path, clip_id: str):
        out = self.pipeline(str(wav))

        # 3.x returns an Annotation directly. 4.x returns a DiarizeOutput with
        # BOTH .speaker_diarization (overlaps preserved) and
        # .exclusive_speaker_diarization (overlaps stripped for transcription).
        # We must take the former: we score with skip_overlap=False, so the
        # exclusive variant would silently flatten every overlap region and show
        # up as a large, entirely artificial miss rate.
        if hasattr(out, "itertracks"):
            ann = out
        elif hasattr(out, "speaker_diarization"):
            ann = out.speaker_diarization
        else:
            attrs = [a for a in dir(out) if not a.startswith("_")]
            raise RuntimeError(
                f"{clip_id}: cannot get an Annotation from {type(out).__name__}; "
                f"available attributes: {attrs}"
            )
        return [(seg.start, seg.end, label) for seg, _, label in ann.itertracks(yield_label=True)]


class SortformerBackend:
    """NeMo Sortformer.

    NOTE: NeMo's diarize() return shape has changed across releases. This handles
    the shapes seen in the wild and fails loudly with the actual repr if it meets
    something new -- better than silently writing an empty RTTM.
    """

    def __init__(self, system: str, hf_token: str | None, device: str,
                 streaming: bool = False, overrides: dict | None = None):
        from nemo.collections.asr.models import SortformerEncLabelModel
        import torch

        self.torch = torch
        self.model = SortformerEncLabelModel.from_pretrained(MODEL_IDS[system])
        self.model.eval()
        self.model.to(torch.device(device))

        sm = self.model.sortformer_modules
        if streaming:
            # NeMo exposes streaming as a model flag, not a diarize() argument:
            # _diarize_forward dispatches on it. The chunk parameters ship with the
            # checkpoint; we only override what was asked for on the command line.
            self.model.streaming_mode = True
            for key, val in (overrides or {}).items():
                if val is not None:
                    setattr(sm, key, val)

        # One output frame = window_stride * subsampling_factor seconds, so the
        # chunk parameters are reported in seconds too -- frames are meaningless
        # to read in a log.
        stride = float(self.model.cfg.get("preprocessor", {}).get("window_stride", 0.01))
        sub = float(getattr(sm, "subsampling_factor", 8) or 8)
        frame_sec = stride * sub
        print(f"[env ] streaming_mode={getattr(self.model, 'streaming_mode', None)}"
              f"  frame={frame_sec:.3f}s")
        for key in STREAMING_ATTRS:
            val = getattr(sm, key, None)
            if val is None:
                continue
            secs = f"  ({val * frame_sec:.1f}s)" if isinstance(val, (int, float)) else ""
            print(f"[env ]   {key} = {val}{secs}")

    @staticmethod
    def _parse(pred, clip_id: str):
        # Unwrap a per-file list wrapper.
        if isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], list):
            pred = pred[0]
        turns = []
        for item in pred:
            if isinstance(item, str):
                # "start end speaker_N" or an RTTM-ish line
                parts = item.split()
                if len(parts) >= 3 and parts[0] == "SPEAKER":
                    turns.append((float(parts[3]), float(parts[3]) + float(parts[4]), parts[7]))
                elif len(parts) >= 3:
                    turns.append((float(parts[0]), float(parts[1]), parts[2]))
                else:
                    raise ValueError(f"{clip_id}: unrecognised sortformer string: {item!r}")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                turns.append((float(item[0]), float(item[1]), str(item[2])))
            elif isinstance(item, dict):
                s = item.get("start", item.get("begin"))
                e = item.get("end", item.get("stop"))
                spk = item.get("speaker", item.get("label"))
                if s is None or e is None or spk is None:
                    raise ValueError(f"{clip_id}: unrecognised sortformer dict: {item!r}")
                turns.append((float(s), float(e), str(spk)))
            else:
                raise ValueError(f"{clip_id}: unrecognised sortformer item: {type(item)} {item!r}")
        return turns

    def __call__(self, wav: Path, clip_id: str):
        pred = self.model.diarize(audio=[str(wav)], batch_size=1)
        return self._parse(pred, clip_id)


def build_backend(system: str, hf_token: str | None, device: str,
                  overrides: dict | None = None):
    if system in ("pyannote31", "community1"):
        return PyannoteBackend(system, hf_token, device)
    if system in ("sortformer", "sortformer_stream"):
        return SortformerBackend(system, hf_token, device,
                                 streaming=(system == "sortformer_stream"),
                                 overrides=overrides)
    raise SystemExit(f"unknown system {system!r}; expected one of {SUPPORTED}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3a: run one diarization system over the clips.")
    ap.add_argument("--system", required=True, choices=SUPPORTED)
    ap.add_argument("--data", required=True, type=Path,
                    help="writable output root (holds ref/, hyp/, results/)")
    ap.add_argument("--wav-dir", type=Path, default=None,
                    help="where the WAVs live (default: <data>/wav). Point this at a "
                         "read-only Kaggle input mount so Save Version does not "
                         "re-snapshot 1.4 GB of audio on every commit.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="HuggingFace token (or set HF_TOKEN)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=None, help="first N clips only")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="skip clips longer than this many seconds (OOM guard)")
    for key in STREAMING_ATTRS:
        ap.add_argument(f"--{key.replace('_', '-')}", type=int, default=None,
                        help=f"sortformer_stream: override {key} "
                             "(frames; default = the checkpoint's own value)")
    args = ap.parse_args()
    overrides = {key: getattr(args, key) for key in STREAMING_ATTRS}

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    wav_dir = args.wav_dir or (args.data / "wav")
    out_dir = args.data / "hyp" / args.system
    rttm_dir = out_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no WAVs in {wav_dir} -- run Stage 1 first, "
                         "or pass --wav-dir if the audio lives elsewhere")
    if args.limit:
        wavs = wavs[:args.limit]

    manifest = Manifest(out_dir / "manifest.jsonl")
    todo = [w for w in wavs if not manifest.done(w.stem, rttm_dir)]

    print(f"[env ] system={args.system}  model={MODEL_IDS[args.system]}")
    print(f"[env ] wav_dir={wav_dir}")
    print(f"[env ] device={device}"
          + (f"  gpu={torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    if args.system in SPEAKER_CAP:
        print(f"[warn] {args.system} is capped at {SPEAKER_CAP[args.system]} speakers; "
              "clips above that cannot be solved and should be reported separately")
    print(f"[run ] {len(todo)} of {len(wavs)} clips to do "
          f"({len(wavs) - len(todo)} already done)\n")
    if not todo:
        print("nothing to do.")
        return 0

    backend = build_backend(args.system, args.hf_token or None, device, overrides)
    t_start = time.time()
    total_audio = 0.0

    for i, wav in enumerate(todo, 1):
        clip_id = wav.stem
        dur = wav_duration(wav)

        if args.max_duration and dur > args.max_duration:
            rec = HypRecord(clip_id, args.system, "failed",
                            error=f"skipped: {dur:.0f}s exceeds --max-duration")
            manifest.append(rec)
            print(f"[skip] {clip_id}  {dur:.0f}s > {args.max_duration:.0f}s")
            continue

        t0 = time.time()
        try:
            turns = backend(wav, clip_id)
            elapsed = time.time() - t0
            write_rttm(rttm_dir / f"{clip_id}.rttm", clip_id, turns)
            speakers = {spk for _, _, spk in turns}
            speech = sum(e - s for s, e, _ in turns if e > s)
            rec = HypRecord(clip_id, args.system, "ok",
                            n_speakers_pred=len(speakers), n_segments=len(turns),
                            speech_sec=round(speech, 3), runtime_sec=round(elapsed, 2),
                            rtf=round(elapsed / dur, 4) if dur else None)
            total_audio += dur
            print(f"[ ok ] {clip_id[:44]:<44} {dur:6.0f}s  "
                  f"{len(speakers)}spk {len(turns):4d}seg  {elapsed:6.1f}s "
                  f"(rtf {elapsed / dur:.3f})")
        except Exception as exc:
            # Never let one bad clip end the run -- record and continue.
            rec = HypRecord(clip_id, args.system, "failed",
                            runtime_sec=round(time.time() - t0, 2),
                            error=f"{type(exc).__name__}: {exc}"[:600])
            print(f"[FAIL] {clip_id[:44]:<44} {type(exc).__name__}: {str(exc)[:90]}",
                  file=sys.stderr)
        manifest.append(rec)

        # Several Sortformer failures asked for only ~1.7 GiB on a 14.6 GiB card:
        # that is fragmentation, not model size. Release between clips so one long
        # clip does not poison the ones after it.
        if device == "cuda":
            torch.cuda.empty_cache()

        if i % 10 == 0 or i == len(todo):
            el = time.time() - t_start
            print(f"       ---- {i}/{len(todo)} clips, {el / 60:.1f} min elapsed, "
                  f"{(len(todo) - i) * el / i / 60:.1f} min left (est)")

    ok = [r for r in manifest.records.values() if r["status"] == "ok"]
    fail = [r for r in manifest.records.values() if r["status"] != "ok"]
    print("\n" + "=" * 70)
    print(f"STAGE 3a SUMMARY -- {args.system}")
    print("=" * 70)
    print(f"  ok / failed        : {len(ok)} / {len(fail)}")
    if ok:
        rtfs = [r["rtf"] for r in ok if r.get("rtf")]
        print(f"  mean RTF           : {sum(rtfs) / len(rtfs):.4f} "
              f"({1 / (sum(rtfs) / len(rtfs)):.0f}x realtime)")
        preds = [r["n_speakers_pred"] for r in ok]
        from collections import Counter
        print(f"  predicted speakers : {sorted(Counter(preds).items())}")
    for r in fail[:10]:
        print(f"  [FAIL] {r['clip_id']}: {(r.get('error') or '')[:100]}")
    print(f"  wrote -> {rttm_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
