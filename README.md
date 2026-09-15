# Indic Speaker-Attributed ASR

**Who said what, in multi-speaker Indic audio.** A pipeline that diarizes long
conversational recordings, transcribes them, and assigns every word to a speaker —
evaluated end to end on 12.26 h of labelled YouTube audio across 9 Indic scripts
and 2–8 speakers, under a deliberately strict metric policy.

| | |
|---|---|
| Diarization | pyannote 3.1 · Sortformer 4spk-v1 (offline and streaming) |
| ASR | IndicConformer-600M (AI4Bharat, ONNX, CTC) · Whisper large-v3 (faster-whisper) |
| Speaker relabelling | rule baseline · Qwen2.5-7B-Instruct (4-bit) |
| Metrics | DER / JER (pyannote.metrics) · WER / cpWER (MeetEval) · WDER |

A two-page write-up is in [`writeup/writeup.pdf`](writeup/writeup.pdf); the longer
version is [`WRITEUP.md`](WRITEUP.md).

## Results

Best combination: **IndicConformer-600M × pyannote 3.1**, then a language-ID
fallback on top of it.

| System (× pyannote 3.1) | WER | cpWER | WDER | DER | JER |
|---|---|---|---|---|---|
| IndicConformer, naive decode | 93.66 | 94.15 | 18.54 | 27.34 | 38.14 |
| IndicConformer, language-locked decode | 78.82 | 79.67 | 20.12 | 27.34 | 38.14 |
| **+ language-ID fallback to Whisper** | **71.71** | **72.96** | **9.32** | 27.34 | 38.14 |
| Whisper large-v3 | 85.70 | 86.67 | 21.27 | 27.34 | 38.14 |

Rates are error-weighted over all 99 recordings. DER and JER do not change with the
ASR system, because the diarizer never sees the words. Every model, every
diarizer, per-language and per-video breakdowns:
[`data/results/results_table.md`](data/results/results_table.md) and
`results_per_video.xlsx`.

## What is worth reading

- **A multi-softmax CTC head, decoded as if it were one softmax.** IndicConformer
  scores each language's 256-token block with its own softmax, so logits from
  different languages are not comparable. A global argmax spelled words across six
  scripts at once. Voting for the recording's language over its frames, then
  decoding inside that block, took **WER from 93.66 to 78.82**
  ([`pipeline/stage4_asr.py`](pipeline/stage4_asr.py), `_decode_ids`). The naive
  decode is kept as a scored ablation.
- **A non-deterministic benchmark, made deterministic.** faster-whisper's default
  temperature fallback samples without a seed: one recording returned 87, 84 and
  100 words on three identical runs, and 211 every time at temperature 0. Corpus
  WER improved 86.62 → 85.70 and runtime fell from 396 to 153 minutes.
- **The improvement that held.** On 13 recordings IndicConformer's own language
  vote chose Urdu or Nepali, so the whole transcript came out in the wrong script.
  Falling back to Whisper's words when the detected language is outside the target
  set gives **cpWER 79.67 → 72.96 with 0 of 99 recordings worse** under pyannote 3.1,
  and the gain holds on all three diarizers. No reference is read; every fallback row is checked to
  reproduce the per-recording scores of the system it took its words from.
- **The idea that did not.** LLM speaker relabelling in the spirit of
  DiarizationLM was built, guarded (confidence threshold, rogue-edit discard, a
  text-unchanged assertion) and measured: WDER went 20.12 → 20.96. An oracle audit
  explains why — even after pause splitting, **37.87% of words** sit in units that
  span two true speakers, out of reach of any relabel.
- **Where the error actually is.** Only 24% of pyannote's error seconds fall in
  overlapped speech; most of it is ordinary single-speaker confusion.

## Design

**Transcribe once, then assign words.** Each recording is transcribed a single
time and every word goes to the diarizer turn it overlaps most. Every diarizer and
every relabelling method is therefore scored on identical words, so a cpWER
difference between them is purely a labelling difference — and it is 99 ASR calls
instead of 12,809, with no sub-second fragments for Whisper to degrade on.

**Metric policy.** DER and JER at **collar 0 with overlapped speech scored**, over
the whole recording. A 0.25 s collar with overlap excluded would report pyannote at
20.58 instead of 27.34; `stage3_score.py --diagnostic` prints that sensitivity
table. Ground-truth labels are read only by the scorers and by diagnostics marked
*oracle*, never by a model.

**Invariants, asserted rather than assumed.** WER must be identical across every
relabelling of one ASR output (text never changes); missed speech and false alarm
must be identical between a diarizer's RTTM and its relabelled versions (boundaries
never move); every corpus figure in the final table is re-derived and matched
against the stage that produced it.

## Pipeline

| Stage | Script | Compute |
|---|---|---|
| 1 | `stage1_extract.py` — YouTube audio → 16 kHz mono WAV, sample-exact trims | network |
| 2 | `stage2_parse_refs.py` — labels → RTTM + speaker-attributed text | CPU |
| 3 | `stage3_diarize.py`, `stage3_score.py` — diarization and DER/JER | GPU, CPU |
| 4a | `stage4_asr.py` — whole-recording ASR with word timestamps | GPU |
| 4b | `stage4_fallback.py`, `stage4_attribute.py` — LID fallback, word → speaker | CPU |
| 5 | `stage5_correct.py` — relabelling (`--method rule`, `--method llm`, `--audit`) | CPU / GPU |
| 4c | `stage4_score.py` — WER, cpWER, DI-cpWER, WDER | CPU |
| 5b | `stage5_to_rttm.py` — relabelled words → RTTM, with a zero-edit control | CPU |
| 6 | `stage6_report.py` — results tables, per model and per video | CPU |

The download, model and attribution stages checkpoint to disk and resume from an
append-only manifest, so a lost session costs one recording, not a run; the scoring
stages are fast and rebuild their outputs from scratch.

```bash
python pipeline/stage1_extract.py    --input youtube_segments_final.xlsx --out data
python pipeline/stage2_parse_refs.py --input youtube_segments_final.xlsx --out data --manifest data/manifest.jsonl
python pipeline/stage3_diarize.py    --system pyannote31 --data data --wav-dir data/wav   # also sortformer, sortformer_stream
python pipeline/stage3_score.py      --data data --systems pyannote31 sortformer sortformer_stream --diagnostic
python pipeline/stage4_asr.py        --system indicconformer --data data --wav-dir data/wav   # also indicconformer_free, whisper
python pipeline/stage4_fallback.py   --data data
python pipeline/stage4_attribute.py  --asr indicconformer indicconformer_free whisper ic_lid_fallback \
                                     --diar pyannote31 sortformer sortformer_stream ref --data data
python pipeline/stage5_correct.py    --cond indicconformer__pyannote31 --method rule --data data
python pipeline/stage4_score.py      --data data
python pipeline/stage5_to_rttm.py    --data data --cond indicconformer__pyannote31+rule indicconformer__pyannote31
python pipeline/stage3_score.py      --data data --systems $(ls data/hyp)
python pipeline/stage6_report.py     --data data
```

**Environments.** NeMo, pyannote.audio, faster-whisper and onnxruntime-gpu pin
conflicting numpy and cuDNN versions — installing onnxruntime-gpu next to
faster-whisper silently pushed Whisper onto the CPU — so each GPU stack runs in its
own environment:

| Stack | Packages |
|---|---|
| diarization | `pyannote.audio` · `nemo_toolkit[asr]` |
| Whisper | `faster-whisper` |
| IndicConformer | `onnxruntime-gpu==1.20.2` (CUDA 12) · `librosa` |
| LLM | `transformers` · `bitsandbytes` · `accelerate` |
| scoring | `pyannote.metrics` · `meeteval` · `rapidfuzz` · `pandas` · `openpyxl` |

Stage 1 also needs `yt-dlp` and `ffmpeg`. Three Hugging Face repos are gated and
need an accepted-terms token in `HF_TOKEN`: `pyannote/speaker-diarization-3.1`,
`pyannote/segmentation-3.0` and `ai4bharat/indic-conformer-600m-multilingual`.

## Notebooks

- **`notebooks/end_to_end.ipynb`** — every stage in order on a Colab T4, generated
  from the pipeline scripts by `pipeline/build_colab.py`. With `FULL_GPU_RUN = False`
  each GPU model runs live on 3 recordings and is compared with the full run, the
  full GPU outputs are loaded from Drive, and every CPU stage recomputes over all 99
  recordings and asserts it reproduces the committed tables. The executed run is
  `end_to_end_output.ipynb`: 0.00% DER against the full run for all three
  diarizers and identical word sequences for all three ASR systems.
- **`notebooks/kaggle-*.ipynb`** — the full 99-recording GPU runs on Kaggle, with
  their original outputs.
- **`notebooks/stage*.ipynb`** — the per-stage Kaggle notebooks, generated by
  `pipeline/build_notebooks.py`.

## Data

The labelled segment table (YouTube IDs, time windows, reference speaker turns and
transcripts) is third-party and is **not redistributed here**; the pipeline expects
it as `youtube_segments_final.xlsx`. Audio is not included either — Stage 1 fetches
it from YouTube. One of the 100 source videos has since been removed, so results
cover 99 recordings. Note that YouTube blocks most cloud IPs, so Stage 1 needs a
residential connection.

## Limitations

- The reference labels were not hand-audited. Attributing words with the reference
  turns scores a worse WDER than with pyannote's (21.45 vs 20.12), which points at
  timing drift in the labels.
- The fallback's target-language list is a deployment assumption, not something
  learned.
- Offline Sortformer's O(T²) attention ran out of memory on the 25 longest
  recordings on a T4; its row measures memory, not the model. The streaming mode
  covers all 99.
- WDER is unstable near 100% WER, so part of the fallback's WDER gain is that effect.

## References

Background reading, grouped by stage, is in
[`writeup/papers/READING.md`](writeup/papers/READING.md); full citations are in
[`writeup/references.bib`](writeup/references.bib).
