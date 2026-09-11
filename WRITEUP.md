# Diarization + ASR on 99 Indic YouTube clips: benchmark and improvement

Ritankar Mondal · code, notebooks and all numbers: `pipeline/`, `notebooks/`, `data/results/`

**Summary.** The best benchmarked combination is **IndicConformer-600M × pyannote 3.1** (DER 27.34, JER 38.14, cpWER 79.67, WDER 20.12).
- **Adopted: a language-ID fallback to Whisper.** cpWER drops to **72.96** (−6.71) and WER to **71.71** (−7.11). Per clip it is 12 better, 87 unchanged, **0 worse**, and it holds on all three diarizers. DER and JER are unchanged by construction.
- **Rejected: LLM speaker relabelling.** It was built and measured, and it made cpWER and WDER worse (§3.2).

## 1. Data and metrics

- **Audio.** 99 of 100 segments were extracted (one video was removed from YouTube): 12.26 h of 16 kHz mono, trims verified sample-exact. The clips cover 9 scripts and 2–8 speakers; 7.61% of speech is overlapped.
- **Reference cleaning.**
  - 38 turns outside the window were dropped, 88 clipped, and 1 inverted and 3 empty turns removed.
  - 666 tags such as `<noise>` are stripped before scoring.
- **Metrics.**
  - **DER and JER:** pyannote.metrics, **collar 0, overlap scored**, whole clip.
  - **cpWER:** MeetEval.
  - **WDER:** hand-rolled, following Shafey et al.
  - **Aggregation:** all rates are error-weighted.
  - **Sensitivity:** a 0.25 s collar with overlap excluded would report pyannote at 20.58 instead of 27.34.
- **Ground truth never enters the pipeline.**

## 2. Baselines

| Diarizer | DER | JER | Miss | FA | Conf. | Spk-count acc |
|---|---|---|---|---|---|---|
| **pyannote 3.1** | **27.34** | **38.14** | 11.59 | 5.89 | 9.86 | **72.7%** |
| Sortformer 4spk-v1, streaming | 47.23 | 57.82 | 9.82 | 6.28 | 31.13 | 51.5% |
| Sortformer 4spk-v1, offline* | 74.85 | 65.29 | 65.14 | 2.19 | 7.52 | 43.4% |

\*Its O(T²) attention ran out of memory on the 25 longest clips on a T4, which then count as total miss. This row measures memory, not the model.

- **Speaker count.** pyannote holds flat from 2 to 5+ speakers (25.9–29.7). Streaming Sortformer climbs from 34.4 to 57.4 because of its 4-speaker cap.
- **Detection vs assignment.** Sortformer detects speech better but assigns speakers 3× worse.
- **Overlap is not where most of the error is.** pyannote's DER is 46.47 in overlapped speech and 20.41 elsewhere, but only **24%** of its error seconds are in overlap.

**ASR design: transcribe the whole clip once, then assign words to speakers.** Each word goes to the diarizer turn it overlaps most. Words with no turn are kept, since dropping them would reward a diarizer for missing speech.
- **Why not transcribe each diarized segment:** with identical words across diarizers, any cpWER difference is purely labelling. Whisper also degrades on sub-second fragments. And it's 99 ASR calls instead of 12,809.
- **Overlapped speech:** a word spoken during overlap goes to the speaker covering more of it. Given the 24% error share, source separation was not built.

| ASR (× pyannote 3.1) | WER | cpWER | WDER |
|---|---|---|---|
| **IndicConformer-600M** (AI4Bharat, ONNX, CTC) | **78.82** | **79.67** | **20.12** |
| Whisper large-v3 (faster-whisper, greedy) | 85.70 | 86.67 | 21.27 |

**Two decoding defects had to be fixed first.**
1. **IndicConformer's CTC head has one softmax per language.** Taking the argmax across all languages spelled words in several scripts at once. Picking the clip's language by frame vote, then decoding within it, takes **WER from 93.66 to 78.82**. The naive decode is kept as a scored ablation.
2. **faster-whisper's default temperature fallback samples without a seed.** One clip gave 87, 84 and 100 words on three runs; at temperature 0 it gives 211 every time. Corpus WER improved 86.62 → 85.70, and runtime fell from 396 to 153 min.

cpWER sits under 1 point above WER, so **recognition, not attribution, is the bottleneck.**

## 3. Improving the output

### 3.1 Language-ID fallback (adopted)

**Diagnosis.** IndicConformer decodes each clip in one language. On 13 clips its own vote chose **Urdu (11) or Nepali (2)**, and every word came out in the wrong script, so the whole clip scores 100% WER. Spoken Hindi and Urdu are nearly identical and differ mainly in script, so this is real acoustic ambiguity, not a decoder bug.

**Rule.** If IndicConformer's detected language is outside the served set {hi, mr, bn, gu, kn, ml, or, pa, ta, te}, take Whisper's words for that clip. No reference is read.

| IndicConformer × … | pyannote 3.1 | Sortformer stream | Sortformer (74) |
|---|---|---|---|
| WER | 78.82 → **71.71** | 78.82 → **71.71** | 75.41 → **67.69** |
| cpWER | 79.67 → **72.96** | 86.96 → **82.99** | 81.65 → **76.89** |
| WDER | 20.12 → **9.32** | 39.58 → **34.62** | 24.12 → **19.46** |
| cpWER per clip: better / same / worse | 12 / 87 / 0 | 12 / 86 / 1 | 8 / 66 / 0 |

- **Devanagari WER falls from 79.2 to 57.2,** better than either system alone (Whisper 69.4).
- **Evidence it wasn't fitted to scores:** sending clips detected as Hindi or Marathi to Whisper instead makes corpus WER *worse* (82.56). The gain comes from the out-of-set rule, not from a language preference.
- **Checked:** every fallback row reproduces exactly the per-clip scores of the system it took words from.
- **Caveats:**
  - The served-language list is knowledge about the task.
  - Part of the WDER drop is WDER misbehaving on clips near 100% WER.

### 3.2 Speaker relabelling from the transcript (not adopted)

**Design,** after DiarizationLM and lexical speaker error correction:
- **Units.** Transcripts are cut into same-speaker runs split at pauses over 0.5 s. A unit may only be moved to an existing speaker, so **text cannot change and WER must not move** (asserted per clip).
- **LLM.** Qwen2.5-7B-Instruct (4-bit, T4).
  - It sees windows of 25 units and returns JSON edits with confidences; edits below 0.7 are dropped.
  - A clip where it tries to edit over 30% of units is discarded.
- **Rule baseline.** A unit under 1 s joins its nearer neighbour.

| IndicConformer × pyannote 3.1 | cpWER | WDER | WDER per clip: better / same / worse |
|---|---|---|---|
| baseline | 79.67 | 20.12 | – |
| + rule | 79.70 | 20.01 | 30 / 31 / 38 |
| + LLM | 80.00 | 20.96 | 6 / 74 / 19 |

**Why it failed.**
1. **There's a ceiling.** Even after pause splitting, **37.87% of words** sit in units spanning two true speakers, where no relabel can help.
2. **At ~79% WER the text carries little evidence of who is speaking.** The LLM's confidences were always 0.8 or 0.9, and it did most damage on 2-speaker clips (WDER 7.54 → 12.16).
3. **The rule's effect depends on the ASR.** It helps all 6 IndicConformer conditions and hurts all 3 Whisper ones. CTC output fragments into more, shorter units (4,468 vs 3,132 on the same turns), where a sub-second unit is usually diarizer jitter.

**DER for a word-level fix.** Mapping word labels back onto the diarizer's turns with **zero edits** already raises DER from 27.34 to 29.03, so that projection is the fair control. Against it, the rule adds +1.52 DER and the LLM +0.87. The results table reports this control beside every DER.

## 4. Indic-specific observations

- **Script, not sound, decides the score.** Hindi/Urdu and Gujarati/Nepali language confusions turn a plausible transcript into 100% WER.
- **The two ASR systems fail in opposite places.**
  - IndicConformer: Kannada 64.9, Odia 68.5, Malayalam 71.4.
  - Whisper is at 96–101 on those three, going over 100 through insertions.
  - Whisper wins on Devanagari (69.4 vs 79.2) and Tamil (78.1 vs 92.1).
- **IndicConformer mostly deletes:** 0.42 output words per reference word, often dropping characters inside correct words.
- **Reference-turn attribution scores worse WDER than pyannote** (21.45 vs 20.12). Word timings drift past the tight reference turns, which suggests timing mismatch in the labels.

## 5. Limitations and next steps

- **Limitations.**
  - The reference labels were not hand-audited; the timing result above is the strongest hint they need it.
  - The served-language list is a deployment assumption.
  - Video was not used, although active-speaker detection targets exactly the dominant single-speaker confusion.
- **Next steps.**
  - Restrict IndicConformer's language vote to the served set.
  - Combine the two ASR systems at word level (ROVER-style).
  - Separate speakers in overlapped regions.
  - Rerun relabelling on the improved ASR.

**Reproducibility.** `notebooks/end_to_end.ipynb` runs every stage in order on Colab: it rebuilds the references from the spreadsheet byte-for-byte (SHA-256), re-runs each model live on a 3-clip subset, and recomputes attribution, scoring and the results table over all 99 clips, asserting they equal the committed tables. The subset reproduces the reference run exactly — **0.00% DER for all three diarizers, identical word sequences for all three ASR systems** — on different hardware and library versions. Full GPU runs are one switch away but take over 5 T4-hours, so they ran on Kaggle (notebooks with outputs in `notebooks/`); audio extraction ran from a home connection, because YouTube bot-gates Colab's and Kaggle's IPs even with cookies.

## References

Bredin (2023), pyannote.audio 2.1, *Interspeech* · Plaquet & Bredin (2023), powerset loss for diarization, *Interspeech* · Park et al. (2024), Sortformer, *arXiv:2409.06656* · Radford et al. (2023), Whisper, *ICML* · Javed et al. (2024), IndicVoices / IndicConformer, *Findings of ACL* · Wang et al. (2024), DiarizationLM, *Interspeech* · Paturi et al. (2023), lexical speaker error correction, *Interspeech* · Shafey et al. (2019), WDER, *Interspeech* · Watanabe et al. (2020), CHiME-6 / cpWER, *CHiME* · von Neumann et al. (2023), MeetEval, *CHiME* · Ryant et al. (2019), DIHARD II / JER, *Interspeech* · Fiscus (1997), ROVER, *ASRU* · Qwen Team (2024), Qwen2.5, *arXiv:2412.15115*.
