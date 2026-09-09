#!/usr/bin/env python3
"""
Stage 4 -- ASR (GPU-bound). Audio in, words with timestamps out.

This stage sees audio ONLY. It knows nothing about speakers, and nothing about
the diarization hypotheses from Stage 3. Speaker attribution is a separate
CPU-only stage (stage4_attribute.py), which means one ASR run is reused across
every diarization system and every Stage 5 correction -- so a cpWER delta
between them is attributable to the labelling, never to the ASR having seen a
different slice of audio.

    python stage4_asr.py --system whisper        --data data --wav-dir WAV
    python stage4_asr.py --system indicconformer --data data --wav-dir WAV

Writes data/asr/<system>/words/<clip_id>.json, one record per clip, and appends
to data/asr/<system>/manifest.jsonl. Re-running skips clips already marked ok,
so a dead Kaggle session costs only the clip that was in flight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

SUPPORTED = ("whisper", "indicconformer", "indicconformer_free")

WHISPER_MODEL = "large-v3"
INDIC_REPO = "ai4bharat/indic-conformer-600m-multilingual"

# A Conformer encoder attends over the whole input, so peak memory grows as
# O(T^2) -- the same trap that OOM'd offline Sortformer on the long clips in
# Stage 3. We chunk instead of discovering the limit at clip 74 of 99.
# CTC is a local, monotonic alignment, so chunks stitch by concatenation; there
# is no speaker identity to carry across boundaries the way Sortformer needed.
CHUNK_SEC = 30.0

OVERLAP_SEC = 2.0

SAMPLE_RATE = 16000

# Tokens per language block in the aggregate tokenizer (22 x 256 = 5632).
BLOCK_SIZE = 256

# SentencePiece word-boundary marker.
WORD_MARK = "▁"


# --------------------------------------------------------------------------
# manifest -- append-only, fsync'd, so a killed session leaves a readable file
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
        # Only "ok" counts. A failed clip is retried on the next run rather than
        # silently treated as complete -- an empty hypothesis scores as a perfect
        # miss and looks like a model result instead of a crash.
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


def read_wav(path: Path):
    """16 kHz mono PCM -> float32 numpy in [-1, 1]. Stage 1 guarantees the format."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(f"{path.name}: expected 16k mono, got "
                             f"{wf.getframerate()} Hz / {wf.getnchannels()} ch")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="int16").astype("float32") / 32768.0


# --------------------------------------------------------------------------
# mel frontend
# --------------------------------------------------------------------------

class MelFrontend:
    """
    NeMo's AudioToMelSpectrogramPreprocessor, reimplemented explicitly.

    Used only when AI4Bharat's shipped TorchScript frontend refuses to run. That
    graph was serialized against a torch whose `torch.stft` accepted an implicit
    `return_complex`; torch >= 2.1 raises instead, and the call is baked into the
    serialized code where it cannot be patched.

    Constants are not guesses. n_fft/hop/win are read straight off the failing
    call in the traceback (512 / 160 / 400 = 32 ms FFT, 10 ms hop, 25 ms window at
    16 kHz), 80 mel bins is forced by the encoder's declared input shape, and the
    rest are NeMo's Conformer defaults: Slaney-normalised mel bank, pre-emphasis
    0.97, power spectrum, log with an additive 2^-24 guard, per-feature mean/var
    normalisation over time.

    A wrong frontend degrades WER without ever raising, so the check that this is
    right is the decode itself: correct features give fluent native script, and
    subtly wrong ones give word salad in the right script.
    """

    N_FFT, HOP, WIN, N_MELS = 512, 160, 400, 80
    PREEMPH = 0.97
    LOG_GUARD = 2.0 ** -24

    def __init__(self, device: str):
        import librosa
        import torch

        self.torch = torch
        self.device = torch.device(device)
        fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=self.N_FFT,
                                 n_mels=self.N_MELS, fmin=0.0,
                                 fmax=SAMPLE_RATE / 2, norm="slaney", htk=False)
        self.fb = torch.from_numpy(fb).float().to(self.device)
        # periodic=False matches NeMo's FilterbankFeatures, not torch's default.
        self.window = torch.hann_window(self.WIN, periodic=False).to(self.device)

    def __call__(self, sig, length):
        torch = self.torch
        x = sig.to(self.device)
        x = torch.cat([x[:, :1], x[:, 1:] - self.PREEMPH * x[:, :-1]], dim=1)
        spec = torch.stft(x, n_fft=self.N_FFT, hop_length=self.HOP,
                          win_length=self.WIN, window=self.window,
                          center=True, pad_mode="reflect", return_complex=True)
        mel = torch.matmul(self.fb, spec.abs().pow(2.0))
        mel = torch.log(mel + self.LOG_GUARD)
        mean = mel.mean(dim=2, keepdim=True)
        std = mel.std(dim=2, keepdim=True).clamp_min(1e-5)
        mel = (mel - mean) / std
        n_frames = torch.div(length.to(self.device), self.HOP,
                             rounding_mode="floor") + 1
        return mel, n_frames.to(torch.int64)


# --------------------------------------------------------------------------
# backend: IndicConformer via ONNX, CTC branch
# --------------------------------------------------------------------------

class IndicConformerBackend:
    """
    AI4Bharat IndicConformer 600M, run through its exported ONNX graphs.

    Why ONNX rather than NeMo: the .nemo checkpoint declares
    `tokenizer.type: multilingual` (stock NeMo dispatches its aggregate
    tokenizer only on `agg`) and `multisoftmax: True` on both the RNNT and CTC
    decoders, which upstream RNNTDecoder/ConvASRDecoder do not accept. Loading
    it needs AI4Bharat's NeMo fork, which pins an older Python and torch than
    Kaggle provides, and which cannot coexist with the NeMo that Stage 3 needs
    for Sortformer. The ONNX export bakes those fork features into the graph, so
    it needs no fork at all.

    Why the CTC branch rather than RNNT: the RNNT joint ships one output head per
    language (joint_post_net_<lang>.onnx), so it would need a language decision
    per clip -- and the only cheap source of that decision is either another
    model or the reference transcript's script, the latter being ground truth
    leaking into the pipeline. The CTC head is a single 1024 -> 5632 projection
    over the whole aggregate vocabulary, so it is language-agnostic. Because the
    vocabulary is 22 per-language blocks concatenated in order, the argmax index
    also identifies the language as a byproduct. CTC gives frame-level alignment
    directly, which is what the word timestamps are built from.
    """

    name = "indicconformer"

    def __init__(self, device: str, lang_lock: bool = True):
        # lang_lock=False reproduces the unrestricted global argmax, which is
        # wrong for a multisoftmax head but is the ablation that demonstrates it.
        self.lang_lock = lang_lock
        import numpy as np
        import onnxruntime as ort
        import torch
        from huggingface_hub import snapshot_download

        self.np = np
        self.torch = torch
        self.device = device

        # snapshot, not hf_hub_download: encoder.onnx stores its weights as
        # external data resolved by relative path. Fetching the graph alone
        # loads without error and emits plausible garbage.
        local = Path(snapshot_download(INDIC_REPO,
                                       allow_patterns=["assets/*", "*.json"]))
        assets = local / "assets"

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(str(assets / "encoder.onnx"),
                                        providers=providers)
        self.ctc = ort.InferenceSession(str(assets / "ctc_decoder.onnx"),
                                        providers=providers)
        print(f"[env ] onnxruntime providers: {self.enc.get_providers()}")
        if device == "cuda" and "CUDAExecutionProvider" not in self.enc.get_providers():
            print("[warn] CUDA requested but onnxruntime fell back to CPU; "
                  "install onnxruntime-gpu matching the CUDA runtime")

        self.pre = self._load_frontend(assets / "preprocessor.ts")
        self._load_vocab(assets / "vocab.json")

    def _load_frontend(self, ts_path: Path):
        """
        Prefer AI4Bharat's own TorchScript frontend, and prove it runs before
        accepting it -- a one-second probe turns a per-clip failure into a startup
        decision, and picks a fallback once rather than 99 times.

        CPU is tried as well as the model device, because the graph bakes its Hann
        window in as a TorchScript constant which `.to(cuda)` does not move: the
        stft then sees a CUDA signal against a CPU window and raises. Mel
        extraction is a few milliseconds either way, so running the frontend on
        CPU costs nothing and keeps the authoritative feature pipeline rather than
        substituting my own reconstruction of it.
        """
        torch = self.torch
        devices = [self.device] + (["cpu"] if self.device != "cpu" else [])
        for dev in devices:
            try:
                ts = torch.jit.load(str(ts_path), map_location=torch.device(dev))
                ts.eval()
                ts.to(torch.device(dev))
                probe = torch.zeros(1, SAMPLE_RATE, device=dev)
                plen = torch.tensor([SAMPLE_RATE], dtype=torch.int64, device=dev)
                with torch.no_grad():
                    feats, _ = ts(probe, plen)
                if feats.shape[1] != MelFrontend.N_MELS:
                    raise RuntimeError(f"frontend emitted {feats.shape[1]} bins, "
                                       f"encoder wants {MelFrontend.N_MELS}")
                print(f"[env ] frontend: AI4Bharat TorchScript on {dev}")
                self.pre_device = dev
                return ts
            except Exception as exc:  # noqa: BLE001
                print(f"[env ] frontend: TorchScript on {dev} failed -- "
                      f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:160]}")

        print("[warn] falling back to a reimplemented frontend; verify the decode "
              "is real words, not same-script noise")
        self.pre_device = self.device
        return MelFrontend(self.device)

    def _load_vocab(self, path: Path) -> None:
        """
        vocab.json is {lang: <that language's tokens>}. The aggregate tokenizer
        concatenates the blocks in key order, so global id = block offset + local
        id, and blank is the final index.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.id2tok: list[str] = []
        self.id2lang: list[str] = []
        for lang, block in raw.items():
            if isinstance(block, dict):
                items = list(block.items())
                if all(str(v).lstrip("-").isdigit() for _, v in items):
                    toks = [k for k, _ in sorted(items, key=lambda kv: int(kv[1]))]
                else:
                    toks = [v for _, v in sorted(items, key=lambda kv: int(kv[0]))]
            else:
                toks = list(block)
            # vocab.json ships 257 entries per language while the aggregate is
            # 22 x 256 = 5632, so exactly one entry per block is surplus. It is
            # the *trailing* one: <unk> stays at local index 0. Verified on a
            # Marathi clip -- block[:256] decodes "namaskar mi gaurav joshi ani
            # mi amol kadkar ..." matching Whisper, while dropping the leading
            # <unk> shifts every token by one inside its block and yields
            # mixed-script salad with the right scripts but wrong characters.
            toks = toks[:BLOCK_SIZE]
            self.id2tok.extend(toks)
            self.id2lang.extend([lang] * len(toks))

        self.blank_id = len(self.id2tok)
        out_dim = self.ctc.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != self.blank_id + 1:
            raise SystemExit(f"vocab/graph mismatch: built {self.blank_id} tokens "
                             f"but the CTC head emits {out_dim} classes")
        print(f"[env ] vocab {self.blank_id} tokens over {len(set(self.id2lang))} "
              f"languages, blank id {self.blank_id}")

    def _forward(self, pcm):
        """One chunk of audio -> per-block scores, n_frames, sec/frame.

        Returns (best_idx, best_val, blank_val, n, frame_sec) where best_idx and
        best_val are [n, 22]: the winning token within each language block and
        its score. The global argmax is NOT taken here, because it is not a
        meaningful operation on this head -- see _decode_ids.
        """
        torch = self.torch
        # The frontend may live on a different device than the model (see
        # _load_frontend); ONNX takes numpy either way, so this costs one copy.
        dev = torch.device(self.pre_device)
        sig = torch.from_numpy(pcm).unsqueeze(0).to(dev)
        length = torch.tensor([pcm.shape[0]], dtype=torch.int64, device=dev)

        with torch.no_grad():
            feats, feat_len = self.pre(sig, length)

        enc_out, enc_len = self.enc.run(
            None,
            {"audio_signal": feats.cpu().numpy().astype("float32"),
             "length": feat_len.cpu().numpy().astype("int64")},
        )
        (logprobs,) = self.ctc.run(None, {"encoder_output": enc_out})

        import numpy as np

        n = int(enc_len[0])
        lp = logprobs[0, :n]
        n_lang = self.blank_id // BLOCK_SIZE
        per_block = lp[:, :self.blank_id].reshape(n, n_lang, BLOCK_SIZE)
        best_idx = per_block.argmax(-1).astype("int32")
        best_val = per_block.max(-1).astype("float32")
        blank_val = lp[:, self.blank_id].astype("float32")

        # Derived, not assumed: the subsampling factor is whatever makes the
        # encoder's frame count match the audio we actually fed it.
        frame_sec = (pcm.shape[0] / SAMPLE_RATE) / max(n, 1)
        return best_idx, best_val, blank_val, n, frame_sec

    def _decode_ids(self, best_idx, best_val, blank_val, lang_block: int | None):
        """Per-block scores -> one token id per frame.

        The CTC head was exported with `multisoftmax: True`: during training the
        softmax ran over ONE language's 256-token block, so the model was never
        asked to compare a Kannada logit against a Marathi one. Those scores are
        on incomparable scales, and a global argmax over all 5632 therefore picks
        a different block almost every frame -- which is exactly what we saw:
        phonetically correct words spelled in six scripts at once.

        With `lang_block` set, the argmax is restricted to that block, which is
        the calibrated comparison the head was actually trained to make. With it
        None, the old unrestricted behaviour is reproduced exactly, so the two
        can be scored against each other as an ablation rather than argued about.
        """
        import numpy as np

        if lang_block is None:
            chosen = best_val.argmax(-1)                       # per-frame block
            val = best_val[np.arange(len(chosen)), chosen]
            idx = best_idx[np.arange(len(chosen)), chosen]
        else:
            chosen = np.full(len(best_val), lang_block, dtype="int64")
            val = best_val[:, lang_block]
            idx = best_idx[:, lang_block]

        ids = chosen * BLOCK_SIZE + idx
        return np.where(blank_val >= val, self.blank_id, ids)

    def _tokens_to_words(self, ids, frame_sec: float, offset: float):
        """
        Greedy CTC collapse, then group BPE pieces into words on the
        SentencePiece word-boundary marker.

        Timing caveat, stated plainly: CTC gives the frame at which a token was
        *emitted*, not the interval it covers, and emission tends to lag acoustic
        onset. These boundaries are good to roughly a frame and are not a forced
        alignment. That is adequate for attributing a word to a speaker turn,
        which is all Stage 4 needs of them.
        """
        pieces = []
        prev = -1
        for t, idx in enumerate(ids):
            idx = int(idx)
            if idx != self.blank_id and idx != prev:
                pieces.append((self.id2tok[idx], self.id2lang[idx], t))
            prev = idx

        grouped = []
        cur, cur_langs, cur_start, cur_end = "", [], None, None
        for tok, lang, frame in pieces:
            starts_word = tok.startswith(WORD_MARK)
            text = tok[1:] if starts_word else tok
            if starts_word and cur:
                grouped.append((cur, cur_langs, cur_start, cur_end))
                cur, cur_langs, cur_start = "", [], None
            if cur_start is None:
                cur_start = frame
            cur += text
            cur_langs.append(lang)
            cur_end = frame
        if cur:
            grouped.append((cur, cur_langs, cur_start, cur_end))

        out = []
        for text, langs, f0, f1 in grouped:
            if not text:
                continue
            out.append({
                "w": text,
                "start": round(offset + f0 * frame_sec, 3),
                "end": round(offset + (f1 + 1) * frame_sec, 3),
                "lang": max(set(langs), key=langs.count),
            })
        return out

    def transcribe(self, pcm, duration: float) -> dict:
        n_chunk = int(CHUNK_SEC * SAMPLE_RATE)
        n_step = int((CHUNK_SEC - OVERLAP_SEC) * SAMPLE_RATE)

        import numpy as np

        # Pass 1: score every chunk, keeping only the per-block winners. The
        # language must be decided over the WHOLE clip -- deciding per chunk
        # would let a 30 s stretch of noise switch scripts mid-transcript, and
        # the corpus has one language per clip.
        chunks = []
        pos = 0
        while pos < pcm.shape[0]:
            seg = pcm[pos:pos + n_chunk]
            if seg.shape[0] < SAMPLE_RATE // 10:  # <100 ms tail, nothing to decode
                break
            best_idx, best_val, blank_val, _n, frame_sec = self._forward(seg)
            chunks.append((pos, best_idx, best_val, blank_val, frame_sec))
            if pos + n_chunk >= pcm.shape[0]:
                break
            pos += n_step

        # Language identification, from the model's own logits and nothing else:
        # count the frames each block would win, ignoring frames the blank takes.
        # Validated earlier -- on the Marathi clip this puts mr first at 48% and
        # the Devanagari blocks together at 73%, far from the 4.5% of noise.
        votes = np.zeros(self.blank_id // BLOCK_SIZE, dtype="int64")
        for _pos, _bi, best_val, blank_val, _fs in chunks:
            winner = best_val.argmax(-1)
            speech = best_val[np.arange(len(winner)), winner] > blank_val
            np.add.at(votes, winner[speech], 1)
        lang_block = int(votes.argmax()) if votes.sum() else 0
        lang_frames = {self.id2lang[b * BLOCK_SIZE]: int(v)
                       for b, v in enumerate(votes) if v}

        # Pass 2: decode each chunk under that decision.
        words = []
        for pos, best_idx, best_val, blank_val, frame_sec in chunks:
            offset = pos / SAMPLE_RATE
            ids = self._decode_ids(best_idx, best_val, blank_val,
                                   lang_block if self.lang_lock else None)
            chunk_words = self._tokens_to_words(ids, frame_sec, offset)

            # Keep only words whose midpoint falls in this chunk's core, so the
            # overlap region is claimed by exactly one chunk and a word split by
            # a cut is recovered whole from the neighbour that saw all of it.
            first = pos == 0
            last = pos + n_chunk >= pcm.shape[0]
            lo = offset if first else offset + OVERLAP_SEC / 2
            hi = offset + CHUNK_SEC - OVERLAP_SEC / 2
            for w in chunk_words:
                mid = (w["start"] + w["end"]) / 2
                if (first or mid >= lo) and (last or mid < hi):
                    words.append(w)

        words.sort(key=lambda w: w["start"])
        langs = [w["lang"] for w in words]
        return {
            "words": words,
            "lang": self.id2lang[lang_block * BLOCK_SIZE] if self.lang_lock
                    else (max(set(langs), key=langs.count) if langs else None),
            "lang_locked": self.lang_lock,
            # Frame votes per block: the model's own language ID, and a useful
            # confidence signal -- a clip whose top two blocks are close is one
            # to look at in the per-condition analysis.
            "lang_frames": dict(sorted(lang_frames.items(), key=lambda kv: -kv[1])),
            "lang_counts": {lg: langs.count(lg) for lg in sorted(set(langs))},
        }


# --------------------------------------------------------------------------
# backend: Whisper large-v3 via faster-whisper
# --------------------------------------------------------------------------

class WhisperBackend:
    """
    faster-whisper rather than WhisperX: WhisperX refines timestamps with
    per-language wav2vec2 alignment models, which do not exist for most of the
    nine Indic scripts in this corpus. Whisper's own cross-attention DTW word
    timestamps are coarser but exist for every language here, and a metric that
    silently degrades for some languages and not others is worse than one that
    is uniformly approximate.
    """

    name = "whisper"

    def __init__(self, device: str):
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
        print(f"[env ] faster-whisper {WHISPER_MODEL} ({compute})")

    def transcribe(self, pcm, duration: float) -> dict:
        segments, info = self.model.transcribe(
            pcm,
            word_timestamps=True,
            vad_filter=False,                   # diarization owns speech/non-speech
            condition_on_previous_text=False,   # stops a hallucination loop from
                                                # propagating across the whole clip
            temperature=0.0,                    # see below: greedy, and repeatable
        )
        # faster-whisper defaults to temperature [0, 0.2, ..., 1.0]: a segment
        # that trips the compression-ratio or avg-logprob check is re-decoded at
        # the next temperature, and above zero that samples instead of taking the
        # argmax, unseeded. Two costs, both measured on Tlha36rSd5o (318 ref
        # words). It is not reproducible -- three identical calls returned 87, 84
        # and 100 words. And it is worse: the fallback keeps a sampled draw over
        # the greedy decode it started from, so disabling it returned 211 words,
        # the same 211 every run. A benchmark number that no one can reproduce is
        # not a benchmark number, and here determinism was also the better decode.
        #
        # The fallback was the only guard against a degenerate repeat loop, so
        # `trips` below is now the sole monitor for one. Watch it in the manifest.

        words, trips = [], 0
        for seg in segments:
            # Flag rather than drop: a discarded segment is an invisible deletion
            # that inflates the miss rate for a reason nothing downstream records.
            if seg.compression_ratio > 2.4 or seg.no_speech_prob > 0.6:
                trips += 1
            for w in (seg.words or []):
                text = w.word.strip()
                if text:
                    words.append({"w": text,
                                  "start": round(w.start, 3),
                                  "end": round(w.end, 3)})

        return {
            "words": words,
            "lang": info.language,
            "lang_prob": round(float(info.language_probability), 4),
            "suspect_segments": trips,
        }


def build_backend(system: str, device: str):
    if system == "whisper":
        return WhisperBackend(device)
    if system == "indicconformer":
        return IndicConformerBackend(device, lang_lock=True)
    if system == "indicconformer_free":
        # The ablation: same weights, same words, global argmax across all 22
        # blocks. Kept as a first-class system so the cost of getting this wrong
        # is a measured WER delta in the results table, not a claim.
        return IndicConformerBackend(device, lang_lock=False)
    raise SystemExit(f"unknown system {system!r}; expected one of {SUPPORTED}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SUPPORTED)
    ap.add_argument("--data", default="data", help="output root")
    ap.add_argument("--wav-dir", required=True, help="16 kHz mono wavs from Stage 1")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N clips")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = args.device or "cpu"
    print(f"[env ] device={device}")

    wav_dir = Path(args.wav_dir)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {wav_dir}")

    out_root = Path(args.data) / "asr" / args.system
    words_dir = out_root / "words"
    manifest = Manifest(out_root / "manifest.jsonl")

    # Count completion from the manifest before --limit truncates the list;
    # deriving "done" from len(wavs) - len(todo) reports the limit instead of
    # actual progress, which reads as a finished run when nothing is finished.
    pending = [p for p in wavs if not manifest.done(p.stem)]
    todo = pending[:args.limit] if args.limit else pending
    print(f"[plan] {len(wavs)} clips: {len(wavs) - len(pending)} done, "
          f"{len(pending)} pending, running {len(todo)} now")
    if not todo:
        return

    backend = build_backend(args.system, device)

    n_ok = n_fail = 0
    total_audio = total_wall = 0.0
    for i, path in enumerate(todo, 1):
        clip_id = path.stem
        t0 = time.time()
        try:
            pcm = read_wav(path)
            duration = pcm.shape[0] / SAMPLE_RATE
            result = backend.transcribe(pcm, duration)
            wall = time.time() - t0

            write_json(words_dir / f"{clip_id}.json", {
                "clip_id": clip_id,
                "system": args.system,
                "duration": round(duration, 3),
                **result,
            })
            rec = {"clip_id": clip_id, "status": "ok",
                   "n_words": len(result["words"]),
                   "lang": result.get("lang"),
                   "duration": round(duration, 3),
                   "wall_sec": round(wall, 2),
                   "rtf": round(wall / duration, 4) if duration else None}
            n_ok += 1
            total_audio += duration
            total_wall += wall
        except Exception as exc:  # noqa: BLE001 -- one bad clip must not end the run
            wall = time.time() - t0
            rec = {"clip_id": clip_id, "status": "fail",
                   "error": f"{type(exc).__name__}: {exc}",
                   "wall_sec": round(wall, 2)}
            n_fail += 1
            print(f"[fail] {clip_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

        manifest.append(rec)
        print(f"[{i:3d}/{len(todo)}] {clip_id[:40]:40s} {rec['status']:4s} "
              f"{str(rec.get('n_words', '-')):>6} words  rtf={rec.get('rtf', '-')}")

        # Long clips fragment the allocator; releasing between clips stops one
        # from poisoning the clips after it (the Sortformer lesson from Stage 3).
        if device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\n[done] ok={n_ok} fail={n_fail}")
    if total_audio:
        print(f"[done] {total_audio / 3600:.2f} h audio in {total_wall / 60:.1f} min "
              f"(mean RTF {total_wall / total_audio:.4f})")


if __name__ == "__main__":
    main()
