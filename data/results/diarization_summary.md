# Baseline diarization

`collar=0.0`, `skip_overlap=False`, UEM = full clip. DER components are duration-weighted.

| System | DER | Miss | FA | Conf | JER | Spk acc | Spk MAE |
|---|---|---|---|---|---|---|---|
| indicconformer__pyannote31 | 29.03% | 11.59% | 5.89% | 11.56% | 39.47% | 72.7% | 0.33 |
| indicconformer__pyannote31+llm | 29.90% | 11.59% | 5.89% | 12.43% | 42.07% | 71.7% | 0.34 |
| indicconformer__pyannote31+rule | 30.55% | 11.59% | 5.89% | 13.08% | 41.07% | 73.7% | 0.32 |
| indicconformer__sortformer | 75.23% | 65.14% | 2.19% | 7.90% | 66.00% | 43.4% | 1.56 |
| indicconformer__sortformer_stream | 48.63% | 9.82% | 6.28% | 32.53% | 59.20% | 51.5% | 0.68 |
| indicconformer__sortformer_stream+rule | 49.68% | 9.82% | 6.28% | 33.58% | 60.23% | 52.5% | 0.67 |
| indicconformer__sortformer+rule | 75.48% | 65.14% | 2.19% | 8.16% | 66.70% | 43.4% | 1.56 |
| indicconformer_free__pyannote31 | 28.80% | 11.59% | 5.89% | 11.32% | 39.03% | 72.7% | 0.33 |
| indicconformer_free__pyannote31+rule | 30.08% | 11.59% | 5.89% | 12.61% | 40.41% | 72.7% | 0.33 |
| indicconformer_free__sortformer | 75.20% | 65.14% | 2.19% | 7.87% | 65.84% | 44.4% | 1.55 |
| indicconformer_free__sortformer_stream | 48.14% | 9.82% | 6.28% | 32.03% | 58.71% | 51.5% | 0.68 |
| indicconformer_free__sortformer_stream+rule | 48.87% | 9.82% | 6.28% | 32.76% | 59.64% | 54.5% | 0.66 |
| indicconformer_free__sortformer+rule | 75.44% | 65.14% | 2.19% | 8.11% | 66.39% | 44.4% | 1.55 |
| pyannote31 | 27.34% | 11.59% | 5.89% | 9.86% | 38.14% | 72.7% | 0.33 |
| sortformer | 74.85% | 65.14% | 2.19% | 7.52% | 65.29% | 43.4% | 1.55 |
| sortformer_stream | 47.23% | 9.82% | 6.28% | 31.13% | 57.82% | 51.5% | 0.68 |
| whisper__pyannote31 | 30.47% | 11.59% | 5.89% | 13.00% | 41.54% | 69.7% | 0.37 |
| whisper__pyannote31+rule | 30.65% | 11.59% | 5.89% | 13.18% | 41.81% | 69.7% | 0.36 |
| whisper__sortformer | 75.69% | 65.14% | 2.19% | 8.36% | 66.90% | 46.5% | 1.52 |
| whisper__sortformer_stream | 49.21% | 9.82% | 6.28% | 33.10% | 59.33% | 55.6% | 0.64 |
| whisper__sortformer_stream+rule | 49.31% | 9.82% | 6.28% | 33.20% | 59.42% | 55.6% | 0.64 |
| whisper__sortformer+rule | 75.64% | 65.14% | 2.19% | 8.32% | 66.98% | 46.5% | 1.52 |
