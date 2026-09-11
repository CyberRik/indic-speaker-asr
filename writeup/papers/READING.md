# Reading list

Blogs, docs and papers behind the terms and methods in this project, in pipeline
order. Start with the ★ items: they explain the ideas better. The 📄 papers are
for exact definitions worth quoting. Files named `NN_*.pdf` are in this folder
(see `README.md`).

Every link was checked on 2026-09-11. Two sources are left out because their
content could not be confirmed: OpenAI's Whisper announcement page (blocks
automated fetches) and the FFmpeg wiki page on seeking (behind a bot wall; the
official `-ss` docs below cover the same point).

**If you only have an hour:** Distill CTC, pyannote.metrics, the MeetEval README,
and the NVIDIA Streaming Sortformer blog.

## 1. Diarization, the big picture (Stage 3)

- ★ [NeMo speaker diarization intro](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html) -- cascaded (VAD → embeddings → clustering) vs end-to-end systems; the pyannote-vs-Sortformer split.
- ★ [Speaker Diarization With Pyannote In Production (Fora Soft)](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/pyannote-speaker-diarization-production) -- pyannote 3.1 step by step: powerset segmentation over ~10 s windows, embeddings that exclude overlapping speech, agglomerative clustering.
- 📄 `01_pyannote21_bredin2023.pdf`, `02_powerset_plaquet2023.pdf`
- 📄 [WeSpeaker, arXiv:2210.17016](https://arxiv.org/abs/2210.17016) -- the embedding model inside pyannote 3.1.

## 2. Diarization metrics: DER, JER, collar, RTTM

- ★ [pyannote.metrics reference](https://pyannote.github.io/pyannote-metrics/reference.html) -- DER's three components, what `collar` and `skip_overlap` do, Hungarian speaker mapping. The basis of the "collar 0, overlap scored" policy.
- ★ [dscore README](https://github.com/nryant/dscore) -- JER definition; RTTM and UEM file formats.
- 📄 `12_dihard2_jer_ryant2019.pdf`

## 3. Sortformer

- ★ [NVIDIA blog: Streaming Sortformer](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/) -- arrival-order speaker sorting, the speaker cache.
- ★ [diar_sortformer_4spk-v1 model card](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) -- "maximum of 4 speakers"; recording length limited by GPU memory ("around 12 minutes" on 48 GB). The documented cause of the 25 offline OOM clips.
- 📄 `03_sortformer_park2024.pdf`
- 📄 [Streaming Sortformer, arXiv:2507.18446](https://arxiv.org/abs/2507.18446)

## 4. ASR foundations: CTC, RNN-T, encoder-decoder (Stage 4)

- ★ [HF Audio Course: CTC architectures](https://huggingface.co/learn/audio-course/chapter3/ctc) -- the gentlest introduction.
- ★ [Distill: Sequence Modeling with CTC (Hannun)](https://distill.pub/2017/ctc/) -- the blank token, greedy vs beam decoding.
- ★ [Sequence-to-sequence learning with Transducers (Lugosch)](https://lorenlugosch.github.io/posts/2020/11/transducer/) -- RNN-T vs CTC; background for taking IndicConformer's CTC branch.
- ★ [HF Audio Course: Seq2Seq architectures](https://huggingface.co/learn/audio-course/chapter3/seq2seq) -- Whisper-style encoder-decoder.
- 📄 `06_conformer_gulati2020.pdf`
- 📄 [Sequence Transduction with RNNs (RNN-T), arXiv:1211.3711](https://arxiv.org/abs/1211.3711)

## 5. IndicConformer

- ★ [Model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) -- 22 languages, CTC and RNN-T decoding.
- **The multisoftmax head is not documented publicly.** The model card does not mention it; it was found by reading the `.nemo` config, and is implemented only in AI4Bharat's NeMo fork ([IndicConformerASR](https://github.com/AI4Bharat/IndicConformerASR) has the install pointer).
- 📄 `05_indicvoices_javed2024.pdf`
- 📄 Optional background: [Towards Building ASR Systems for the Next Billion Users, arXiv:2111.03945](https://arxiv.org/abs/2111.03945)

## 6. Whisper details

- ★ [whisper/transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py) -- read the parameter docstrings: default `temperature` `(0.0, 0.2, ..., 1.0)` "successively used upon failures" (the unseeded fallback), and `condition_on_previous_text` with its "failure loop" risk.
- ★ [faster-whisper README](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 needs CUDA 12 + cuDNN 9; the root of the onnxruntime-gpu clash.
- 📄 `04_whisper_radford2022.pdf`
- 📄 [Careless Whisper, arXiv:2402.08021](https://arxiv.org/abs/2402.08021) -- hallucination.
- 📄 [WhisperX, arXiv:2303.00747](https://arxiv.org/abs/2303.00747) -- for defending why it was not used.

## 7. ASR metrics: WER, cpWER, WDER

- ★ [HF Audio Course: Evaluation metrics for ASR](https://huggingface.co/learn/audio-course/chapter5/evaluation) -- WER from scratch, and how normalisation moves it.
- ★ [MeetEval README](https://github.com/fgnt/meeteval) -- cpWER, DI-cpWER, tcpWER, ORC-WER.
- 📄 `10_chime6_cpwer_watanabe2020.pdf`, `11_meeteval_vonneumann2023.pdf`, `09_wder_shafey2019.pdf`

## 8. LLM speaker correction (Stage 5)

- ★ [DiarizationLM README](https://github.com/google/speaker-id/tree/master/DiarizationLM) -- the `<spk:1> good morning <spk:2> ...` prompt format; WDER and cpWER implementations.
- ★ [HF blog: 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) -- NF4 loading with bitsandbytes; how a 7B model fits on a T4.
- 📄 `07_diarizationlm_wang2024.pdf`, `08_lexical_sec_paturi2023.pdf`, `13_qwen25_2024.pdf`
- 📄 [QLoRA, arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

## 9. Tooling

- ★ [yt-dlp wiki: Exporting YouTube cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) -- private window, `robots.txt`, close the window. The PO Token guide is on the same page.
- [yt-dlp wiki: EJS](https://github.com/yt-dlp/yt-dlp/wiki/EJS) -- why YouTube needs a JavaScript runtime; Deno by default, `--js-runtimes node` otherwise.
- [FFmpeg docs, `-ss`](https://ffmpeg.org/ffmpeg.html) -- input vs output seeking; basis of the sample-exact trim.
- [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) -- 1.27+ is CUDA 13; why 1.20.2 is pinned.
