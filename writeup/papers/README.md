# Papers referenced in the writeup

Each paper is listed with the reason it is cited. PDFs are the authors' open versions (arXiv, or the ISCA archive).

| File | Paper | Used for |
|---|---|---|
| `01_pyannote21_bredin2023.pdf` | Bredin (2023), *pyannote.audio 2.1 speaker diarization pipeline*, Interspeech | Baseline diarizer (pyannote 3.1 pipeline) |
| `02_powerset_plaquet2023.pdf` | Plaquet & Bredin (2023), *Powerset multi-class cross entropy loss for neural speaker diarization*, Interspeech | pyannote 3.x segmentation model; overlap-aware output |
| `03_sortformer_park2024.pdf` | Park et al. (2024), *Sortformer*, arXiv:2409.06656 | Second diarizer (offline and streaming); its 4-speaker cap |
| `04_whisper_radford2022.pdf` | Radford et al. (2023), *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML | ASR system 2 (Whisper large-v3) and the fallback |
| `05_indicvoices_javed2024.pdf` | Javed et al. (2024), *IndicVoices*, Findings of ACL | Training data behind AI4Bharat's IndicConformer (ASR system 1) |
| `06_conformer_gulati2020.pdf` | Gulati et al. (2020), *Conformer*, Interspeech | Architecture of IndicConformer |
| `07_diarizationlm_wang2024.pdf` | Wang et al. (2024), *DiarizationLM*, Interspeech | Design basis for LLM speaker relabelling (Stage 5) |
| `08_lexical_sec_paturi2023.pdf` | Paturi et al. (2023), *Lexical Speaker Error Correction*, Interspeech | Design basis for transcript-based speaker correction |
| `09_wder_shafey2019.pdf` | El Shafey et al. (2019), *Joint Speech Recognition and Speaker Diarization via Sequence Transduction*, Interspeech | Definition of WDER |
| `10_chime6_cpwer_watanabe2020.pdf` | Watanabe et al. (2020), *CHiME-6 Challenge*, CHiME Workshop | Definition of cpWER |
| `11_meeteval_vonneumann2023.pdf` | von Neumann et al. (2023), *MeetEval*, CHiME Workshop | cpWER / DI-cpWER implementation used for scoring |
| `12_dihard2_jer_ryant2019.pdf` | Ryant et al. (2019), *The Second DIHARD Diarization Challenge*, Interspeech | Definition of JER; strict DER scoring conventions |
| `13_qwen25_2024.pdf` | Qwen Team (2024), *Qwen2.5 Technical Report*, arXiv:2412.15115 | LLM used for relabelling (Qwen2.5-7B-Instruct) |

Not included: Fiscus (1997), *ROVER*, IEEE ASRU. It is cited as a next step (word-level system combination), and no open copy is available.
