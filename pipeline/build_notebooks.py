#!/usr/bin/env python3
"""
Generate Kaggle-ready .ipynb files from the pipeline scripts.

The .py files under pipeline/ are the single source of truth. This embeds them
into notebooks as %%writefile cells, so a script fix means regenerating the
notebook rather than re-uploading a dataset -- and the notebook can never drift
from the code it is supposed to run.

    python pipeline/build_notebooks.py

Writes notebooks/*.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def embed(script_name: str) -> dict:
    """A %%writefile cell carrying the current contents of a pipeline script."""
    body = (PIPELINE / script_name).read_text(encoding="utf-8")
    return code(f"%%writefile {script_name}\n{body}")


def notebook(cells: list[dict], accelerator: str = "GPU") -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3",
                           "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": accelerator,
            "kaggle": {"accelerator": "nvidiaTeslaT4" if accelerator == "GPU" else "none",
                       "dataSources": [], "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "cells": cells,
    }


# --------------------------------------------------------------------------
# Shared setup cells
# --------------------------------------------------------------------------

# (DISCOVER cells intentionally omitted -- the notebook assumes inputs are already set up)

# (COPY_REF cells intentionally omitted -- the notebook assumes inputs are already set up)

# (HF_TOKEN cells intentionally omitted -- the notebook assumes inputs are already set up)


def build_stage3() -> None:
    cells = [
        md("""
# Stage 3 — Baseline Diarization

Append these cells to the notebook that already has your inputs set up.

**Assumes already set up by your existing cells:** `AUDIO`, `/kaggle/working/data/ref/`,
the scripts copied into `/kaggle/working/`, and `HF_TOKEN` in the environment.

The scripts come from the code dataset, so **re-upload `sarvam-diar-code` whenever
a script changes** — otherwise Kaggle runs a stale copy.

**Metric policy (deliberately unforgiving):** `collar=0.0`, `skip_overlap=False`,
UEM = the full clip. Overlapping speech **is** scored, and boundary errors get no
forgiveness. Expect DER well above published numbers — those almost always use
`collar=0.25` and skip overlap.

Corpus: 99 clips / 12.28 h / 9 Indic languages / 7.60% overlapped speech.
"""),
        code('# sanity check: the things the cells below depend on\n'
             'print("AUDIO :", AUDIO, f"({len(list(AUDIO.glob(\'*.wav\')))} wavs)")\n'
             'import os, pathlib\n'
             'print("ref   :", pathlib.Path("/kaggle/working/data/ref/clip_meta.csv").exists())\n'
             'print("token :", bool(os.environ.get("HF_TOKEN")))\n'
             'print("scripts:", sorted(p.name for p in '
             'pathlib.Path("/kaggle/working").glob("stage*.py")))'),
        md("## Smoke test (5 clips)\n\n"
           "Check before committing ~25 min of GPU:\n"
           "- `device=cuda` — if it says `cpu`, the accelerator is off\n"
           "- **RTF ≈ 0.03** (~30× realtime). Near 1.0 means it is silently on CPU\n"
           "- `pyannote.audio` version — 4.x means `community1` is available as System B"),
        code('!python stage3_diarize.py --system pyannote31 --data data '
             '--wav-dir {AUDIO} --limit 5'),
        md("## Full run — pyannote 3.1\n\n"
           "Resumable: re-run this cell after a session death and it skips completed clips."),
        code('!python stage3_diarize.py --system pyannote31 --data data --wav-dir {AUDIO}'),
        code('!python stage3_score.py --data data --systems pyannote31 --diagnostic'),
        md("""
## System B

Two options — try Sortformer first, fall back to community-1 if NeMo fights the
preinstalled torch.

**Sortformer is hard-capped at 4 speakers**, and 17 of our clips have ≥5. It
cannot represent those, so report its per-speaker-count breakdown rather than
only the headline DER — the cap is a model limitation, not a quality result.
"""),
        code('# Option B1: NeMo Sortformer. If this breaks torch, restart and use B2.\n'
             '!pip install -q "nemo_toolkit[asr]"'),
        code('!python stage3_diarize.py --system sortformer --data data '
             '--wav-dir {AUDIO} --limit 5'),
        code('!python stage3_diarize.py --system sortformer --data data --wav-dir {AUDIO}'),
        md("""
### Sortformer streaming — required for the long clips

Offline Sortformer attends over the whole session, so peak VRAM grows as
**O(duration²)**: a 913 s clip asked for 7.8 GiB, an 1822 s clip for 30.9 GiB.
On a 14.6 GiB T4 that OOM'd on 25 of 99 clips — and those 25 are systematically
the *longest* clips, which in this corpus also carry the most speakers and
overlap. Scoring the surviving 74 against pyannote's 99 would compare two
different corpora.

`sortformer_stream` is the same checkpoint with `streaming_mode = True`: fixed
chunks, with a speaker cache + FIFO queue carrying identity across boundaries,
so no stitching is needed on our side. It writes to its own `hyp/` directory,
so the offline hypotheses survive for the streaming-vs-offline comparison on the
74 clips where both ran.
"""),
        code('!python stage3_diarize.py --system sortformer_stream --data data '
             '--wav-dir {AUDIO} --limit 5'),
        code('!python stage3_diarize.py --system sortformer_stream --data data '
             '--wav-dir {AUDIO}'),
        md("### Option B2 — pyannote community-1 (fallback, needs pyannote.audio ≥ 4)"),
        code('# !python stage3_diarize.py --system community1 --data data '
             '--wav-dir {AUDIO} --limit 5\n'
             '# !python stage3_diarize.py --system community1 --data data --wav-dir {AUDIO}'),
        md("## Score everything"),
        code('!python stage3_score.py --data data '
             '--systems pyannote31 sortformer sortformer_stream --diagnostic'),
        code('import pandas as pd\n'
             'pd.read_csv("/kaggle/working/data/results/diarization_summary.csv")'),
        md("""
## Save

**Save Version → Quick Save** before the session ends. `/kaggle/working` is wiped
on timeout, and the hypothesis RTTMs represent the GPU time you just spent.

Next session: attach this notebook's output as a data source and the runs resume
from their manifests.
"""),
        code('!du -sh /kaggle/working/data/* 2>/dev/null\n'
             '!find /kaggle/working/data/hyp -name "*.rttm" | wc -l'),
    ]
    out = NOTEBOOKS / "stage3_diarization.ipynb"
    out.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


S4_SETUP = """
import pathlib, shutil

ROOT  = pathlib.Path("/kaggle/input")
CODE  = next(p.parent for p in ROOT.rglob("stage4_asr.py"))
AUDIO = next(p.parent for p in ROOT.rglob("*.wav"))
WORK  = pathlib.Path("/kaggle/working/data")
WORK.mkdir(parents=True, exist_ok=True)

for f in CODE.glob("*.py"):
    shutil.copy(f, "/kaggle/working/")

# Restore anything already produced -- Stage 3 RTTMs, and the other ASR system's
# words if its dataset is attached. Nothing here is required by this notebook;
# it is what lets stage4_attribute.py run later without re-attaching everything.
for src in sorted(ROOT.rglob("data")):
    if src.is_dir() and any((src / d).exists() for d in ("hyp", "ref", "asr")):
        shutil.copytree(src, WORK, dirs_exist_ok=True)
        print("restored", src)

print("CODE   :", CODE)
print("AUDIO  :", AUDIO, len(list(AUDIO.glob("*.wav"))), "wavs")
print("scripts:", sorted(p.name for p in pathlib.Path("/kaggle/working").glob("*.py")))
for sub in ("hyp", "asr"):
    d = WORK / sub
    if d.exists():
        for x in sorted(d.glob("*")):
            n = len(list(x.rglob("*.rttm"))) + len(list(x.rglob("*.json")))
            print(f"  {sub}/{x.name}: {n}")
"""


S4_FRESH = """
WIPE = True   # set False once a clean run exists and you want to resume it

import pathlib, shutil

src = pathlib.Path("/kaggle/working/stage4_asr.py").read_text(encoding="utf-8")
print("BLOCK_SIZE in the copied script:", src.count("BLOCK_SIZE"), "hits (expect 2)")
assert "toks = toks[:BLOCK_SIZE]" in src, (
    "stale stage4_asr.py -- re-upload sarvam-diar-code, restart the session, "
    "and re-run the setup cell above"
)

out = pathlib.Path("/kaggle/working/data/asr/indicconformer")
if WIPE and out.exists():
    n = len(list((out / "words").glob("*.json")))
    shutil.rmtree(out)
    print(f"wiped {n} clips of previous indicconformer output")
print("present now:", sorted(p.name for p in
      pathlib.Path("/kaggle/working/data/asr").glob("*")))
"""


S4W_FRESH = """
WIPE = True   # set False once a clean run exists and you want to resume it

import pathlib, shutil

# The run is resumable, so a leftover clip is silently kept. After a decode
# change that is the dangerous case: the corpus ends up half one setting and
# half the other, and nothing in the output records the split.
src = pathlib.Path("/kaggle/working/stage4_asr.py").read_text(encoding="utf-8")
assert "temperature=0.0" in src, (
    "stale stage4_asr.py -- this copy still uses the default temperature "
    "fallback, which samples unseeded. Re-upload sarvam-diar-code, restart the "
    "session, and re-run the setup cell above"
)
print("stage4_asr.py: temperature=0.0 present")

out = pathlib.Path("/kaggle/working/data/asr/whisper")
if WIPE and out.exists():
    n = len(list((out / "words").glob("*.json")))
    shutil.rmtree(out)
    print(f"wiped {n} clips of previous whisper output")
print("present now:", sorted(p.name for p in
      pathlib.Path("/kaggle/working/data/asr").glob("*")))
"""


def s4_inspect(system: str) -> str:
    return f"""
import json, glob

files = sorted(glob.glob("/kaggle/working/data/asr/{system}/words/*.json"))
print(len(files), "clips transcribed")
d = json.load(open(files[0], encoding="utf-8"))
w = d["words"]
print("clip     :", d["clip_id"][:44], f'{{d["duration"]:.1f}}s')
print("lang     :", d.get("lang"), d.get("lang_counts", ""))
print("words    :", len(w))
print("first    :", w[:6])
print("last     :", w[-3:])
print("text     :", " ".join(x["w"] for x in w[:40]))
print("span     :", w[0]["start"], "->", w[-1]["end"], "of", d["duration"], "s")
print("monotonic:", all(a["start"] <= b["start"] for a, b in zip(w, w[1:])))
print("in bounds:", w[-1]["end"] <= d["duration"] + 1)
"""


def s4_coverage(system: str) -> str:
    return f"""
import json, pathlib

mf = pathlib.Path("/kaggle/working/data/asr/{system}/manifest.jsonl")
recs = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()]
ok = [r for r in recs if r["status"] == "ok"]
print(f"{system}: {{len(ok)}} ok / {{len(recs)}} records, "
      f"{{sum(r['n_words'] for r in ok):,}} words")
for r in recs:
    if r["status"] != "ok":
        print("  fail:", r["clip_id"][:36], r.get("error", "")[:100])

rtfs = [r["rtf"] for r in ok if r.get("rtf")]
if rtfs:
    print(f"rtf: min {{min(rtfs):.4f}}  median {{sorted(rtfs)[len(rtfs)//2]:.4f}}  max {{max(rtfs):.4f}}")

langs = {{}}
for r in ok:
    langs[r.get("lang")] = langs.get(r.get("lang"), 0) + 1
print("languages:", dict(sorted(langs.items(), key=lambda kv: -kv[1])))
"""


SPLIT_NOTE = """
### Why this is its own notebook

Whisper and IndicConformer want incompatible CUDA stacks. `onnxruntime-gpu`
installs its own `nvidia-cudnn-cu12`, which replaces the cuDNN that CTranslate2
(the runtime behind faster-whisper) was built against. The result is not an
error: Whisper silently falls back to CPU and a run that should take minutes
sits on an idle GPU for a quarter of an hour saying nothing.

Rather than fight that, each system gets its own session. They share nothing at
runtime — separate manifests, separate output directories, neither reads the
other — so the split costs nothing and removes a whole class of silent failure.
Save each notebook's output as a dataset; `stage4_attribute.py` is CPU-only and
attaches both.
"""

HEADER = """
# Stage 4 — Baseline ASR ({title})

**Attach:** `sarvam-diar-code`, `sarvam-diar-audio`, `sarvam-diar-stage3`.
**Settings:** GPU (T4) on, Internet on. No `HF_TOKEN` — the model is public.

This stage turns audio into **words with timestamps**, and nothing else. It never
sees a speaker label and never sees a diarization hypothesis. Attribution is a
separate CPU stage, so a single ASR run is reused across every diarization system
and every Stage 5 correction — which is what makes a cpWER delta attributable to
the labelling rather than to the ASR having been fed different audio.
"""


S4B_SETUP = """
import pathlib, shutil, json

ROOT = pathlib.Path("/kaggle/input")
# Several datasets can contain a copy of the scripts: the attrib dataset is a
# snapshot of a working directory and carries whatever was current when it was
# saved. Taking the first rglob hit silently runs STALE code. Prefer the
# directory holding the most pipeline scripts, tie-broken toward a path named
# like the code dataset.
_cands = {p.parent for p in ROOT.rglob("stage4_attribute.py")}
CODE = max(_cands, key=lambda d: (len(list(d.glob("stage*.py"))),
                                  "code" in str(d).lower()))
if len(_cands) > 1:
    print("script copies found in:")
    for _c in sorted(map(str, _cands)):
        print("   ", _c, "  <-- using" if str(CODE) == _c else "")
WORK = pathlib.Path("/kaggle/working/data")
WORK.mkdir(parents=True, exist_ok=True)

for f in CODE.glob("*.py"):
    shutil.copy(f, "/kaggle/working/")

# Every attached dataset that carries a data/ tree is merged into one working
# copy: Stage 3's RTTMs, and one directory per ASR system from the two Stage 4a
# datasets. No audio is needed, so that dataset stays detached.
for src in sorted(ROOT.rglob("data")):
    if src.is_dir() and any((src / d).exists() for d in ("hyp", "ref", "asr")):
        shutil.copytree(src, WORK, dirs_exist_ok=True)
        print("restored", src)

# The code dataset carries ref/ at its top level, NOT inside a data/ tree, so
# the loop above misses it -- which is how ref/segments (the transcripts) can be
# absent while ref/rttm looks fine. Stage 4b never noticed; scoring needs them.
if (CODE / "ref").is_dir():
    shutil.copytree(CODE / "ref", WORK / "ref", dirs_exist_ok=True)
    print("restored", CODE / "ref")

print()
print("CODE:", CODE)
ASR = sorted(p.name for p in (WORK / "asr").glob("*") if p.is_dir())
DIAR = sorted(p.name for p in (WORK / "hyp").glob("*") if p.is_dir())
# Print the DECODE PROVENANCE, not just the count. Several attached datasets
# can carry an asr/ tree -- the attrib dataset is a snapshot of a whole working
# directory -- and copytree(dirs_exist_ok=True) lets a later one overwrite an
# earlier one. A stale words file is invisible in a file count and produces a
# full, plausible, wrong results table. `lang_locked` exists only in output
# from the multisoftmax-aware decode.
for a in ASR:
    files = sorted((WORK / "asr" / a / "words").glob("*.json"))
    d = json.loads(files[0].read_text(encoding="utf-8")) if files else {}
    locked = d.get("lang_locked", "n/a" if a.startswith("whisper") else "STALE")
    sample = " ".join(w["w"] for w in d.get("words", [])[:6])
    print(f"  asr/{a:20} {len(files):3} clips  lang_locked={locked}  {sample[:48]}")
    assert locked != "STALE", (
        f"{a}: words predate the language-mask fix. Detach sarvam-diar-attrib "
        f"(it carries an old asr/ tree that overwrites the new one) and re-attach "
        f"the current sarvam-diar-asr-indic version."
    )
for d in DIAR:
    print(f"  hyp/{d:16} {len(list((WORK / 'hyp' / d / 'rttm').glob('*.rttm'))):3} rttm")
print(f"  ref/rttm{'':12} {len(list((WORK / 'ref' / 'rttm').glob('*.rttm'))):3} rttm")
print(f"  ref/segments{'':8} {len(list((WORK / 'ref' / 'segments').glob('*.json'))):3} json"
      "   <- scoring needs these")
print()
print("ASR :", ASR)
print("DIAR:", DIAR, "+ ref (oracle)")
assert ASR, "no ASR words found -- attach the Stage 4a dataset(s)"
assert list((WORK / "ref" / "segments").glob("*.json")), (
    "no ref/segments -- Stage 4c cannot score without the reference transcripts"
)

# Built here rather than interpolated into the shell line below: IPython's {}
# expansion chokes on quotes inside the braces.
ASR_ARG, DIAR_ARG = " ".join(ASR), " ".join(DIAR)
"""


S4B_FALLBACK = """
!python stage4_fallback.py --data data

# Re-listed, not reused from the setup cell: ic_lid_fallback did not exist when
# that cell ran, and attribution only crosses the systems named in ASR_ARG.
ASR = sorted(p.name for p in (WORK / "asr").glob("*") if p.is_dir())
ASR_ARG = " ".join(ASR)
print("ASR :", ASR)
"""


S4B_RUN = """
!python stage4_attribute.py --asr {ASR_ARG} --diar {DIAR_ARG} ref --data data
"""


S4B_REPORT = """
import json, pathlib
import pandas as pd

rows = []
for cond in sorted((pathlib.Path("/kaggle/working/data/attrib")).glob("*")):
    mf = cond / "manifest.jsonl"
    if not mf.exists():
        continue
    # The manifest is append-only, so a retried clip has one line per attempt.
    # Collapse to the last record per clip or the failure counts multiply.
    recs = {}
    for l in mf.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            recs[r["clip_id"]] = r
    recs = list(recs.values())
    ok = [r for r in recs if r["status"] == "ok"]
    w = sum(r["n_words"] for r in ok) or 1
    asr, diar = cond.name.split("__")
    rows.append({
        "asr": asr,
        "diar": diar + (" (oracle)" if diar == "ref" else ""),
        "clips_ok": len(ok),
        "clips_fail": len(recs) - len(ok),
        "words": sum(r["n_words"] for r in ok),
        "orphan_%": round(100 * sum(r["n_orphan"] for r in ok) / w, 2),
        "overlap_%": round(100 * sum(r["n_overlap_words"] for r in ok) / w, 2),
        "boundary_%": round(100 * sum(r["n_boundary_words"] for r in ok) / w, 2),
        "mean_spk": round(sum(r["n_speakers"] for r in ok) / max(len(ok), 1), 2),
    })

df = pd.DataFrame(rows).sort_values(["asr", "diar"])
print(df.to_string(index=False))

# One clip, end to end, so the words are visibly attached to speakers.
cond = sorted(pathlib.Path("/kaggle/working/data/attrib").glob("*__ref"))[0]
clip = sorted(cond.glob("*.json"))[0]
d = json.load(open(clip, encoding="utf-8"))
print()
print(f"{d['clip_id'][:44]}  {d['lang']}  {d['n_words']} words, "
      f"{len(d['speakers'])} speakers, {d['n_orphan']} orphaned")
for spk, text in d["by_speaker"].items():
    print(f"  {spk:12} {text[:110]}")
"""


S4C_PROBE = """
import meeteval, meeteval.wer as mw
print("meeteval", getattr(meeteval, "__version__", "?"))
print("exposes:", sorted(n for n in dir(mw) if "error_rate" in n or n.endswith("wer")))
"""


S4C_REPORT = """
import pathlib
import pandas as pd

csv = pathlib.Path("/kaggle/working/data/results/asr_summary.csv")
assert csv.exists(), (
    "no asr_summary.csv -- the scoring cell above did not finish. Read its "
    "output: a meeteval binding problem prints the function names the installed "
    "version exposes, and that list is the fix."
)
s = pd.read_csv(csv)
for subset in ("common", "all"):
    print(f"--- {subset} ---")
    print(s[s.subset == subset].drop(columns=["subset"]).to_string(index=False))
    print()
"""


S4B_CLEAN = "\n".join([
    "import pathlib",
    "",
    "for _p in pathlib.Path('/kaggle/working').glob('*.py'):",
    "    _p.unlink()",
    "print('kept:', sorted(x.name for x in pathlib.Path('/kaggle/working').iterdir()))",
])


S4B_LID = "\n".join([
    "import json, pathlib",
    "import pandas as pd",
    "",
    "meta = pd.read_csv('/kaggle/working/data/ref/clip_meta.csv').set_index('clip_id')",
    "rows = []",
    "for a in sorted(pathlib.Path('/kaggle/working/data/asr').glob('*')):",
    "    for f in sorted((a / 'words').glob('*.json')):",
    "        d = json.loads(f.read_text(encoding='utf-8'))",
    "        cid = d['clip_id']",
    "        rows.append({",
    "            'asr': a.name,",
    "            'clip_id': cid,",
    "            'detected': d.get('lang'),",
    "            'script': meta.at[cid, 'language'] if cid in meta.index else '?',",
    "            'hyp_words': len(d['words']),",
    "            'ref_words': int(meta.at[cid, 'n_ref_words']) if cid in meta.index else 0,",
    "        })",
    "",
    "lid = pd.DataFrame(rows)",
    "for a, g in lid.groupby('asr'):",
    "    print('===', a, '===')",
    "    print(pd.crosstab(g['script'], g['detected']).to_string())",
    "    ratio = g['hyp_words'].sum() / max(g['ref_words'].sum(), 1)",
    "    print(f'hyp/ref word ratio: {ratio:.2f}')",
    "    print()",
])


def build_stage4_indic() -> None:
    cells = [
        md(HEADER.format(title="IndicConformer 600M")),
        md(SPLIT_NOTE),
        md("""
### Why ONNX rather than NeMo

The `.nemo` checkpoint declares `tokenizer.type: multilingual` (stock NeMo
dispatches its aggregate tokenizer only on `agg`) and `multisoftmax: True` on
**both** the RNNT and CTC decoders, which upstream NeMo cannot instantiate. That
needs AI4Bharat's NeMo fork, which pins an older Python and torch than Kaggle
provides. The ONNX export bakes those fork features into the graph, so no fork is
required.

We take the **CTC branch**, not RNNT. The RNNT joint ships one output head per
language (`joint_post_net_<lang>.onnx`), so it would need a language decision per
clip — and the cheap source of that decision is the reference transcript's
script, which is ground truth leaking into the pipeline. The CTC head is a single
1024 → 5632 projection over the whole aggregate vocabulary. Because that
vocabulary is 22 per-language blocks concatenated in order, the argmax index
identifies the language for free.
"""),
        md("""
### onnxruntime, carefully

Two traps, both of which land you silently on CPU:

- installing `onnxruntime-gpu` **alongside** the preinstalled `onnxruntime`
  leaves the CPU binaries in charge, so remove both first
- the latest `onnxruntime-gpu` (1.29) is built against CUDA 13 and dies with
  `libcublasLt.so.13: cannot open shared object file`. Kaggle ships CUDA 12

**Restart the kernel after this cell.** Do *not* add torch's NVIDIA libs to
`LD_LIBRARY_PATH` to force the provider — the version pin is what fixes it, and
the loader path breaks other CUDA consumers.
"""),
        code("!pip uninstall -y -q onnxruntime onnxruntime-gpu\n"
             '!pip install -q "onnxruntime-gpu==1.20.2" librosa'),
        code("import onnxruntime as ort\n"
             "print(ort.__version__, ort.get_available_providers())"),
        md("`CUDAExecutionProvider` must appear above. Without it the run is hours "
           "instead of minutes."),
        code(S4_SETUP),
        md("""
### Is the script actually the current one?

Re-uploading `sarvam-diar-code` does not refresh `/kaggle/working`; the copy cell
above must run after the new dataset version is attached, which needs a session
restart. The assert below fails loudly instead of silently re-running the old
vocabulary mapping.

The wipe matters just as much. `Manifest.done()` skips any clip already recorded,
and the setup cell restores `data/asr/` from attached datasets — so words written
by the old mapping count as finished, and a resume would transcribe nothing while
reporting success. Set `WIPE = False` once a clean run exists.
"""),
        code(S4_FRESH),
        md("""
## Smoke test — 3 clips

- `onnxruntime providers:` contains **CUDAExecutionProvider**, no fallback warning
- `frontend: AI4Bharat TorchScript on cuda` — the fallback reimplementation is a
  last resort, and its output should be checked harder if it is used
- `vocab 5632 tokens over 22 languages, blank id 5632` — 5654 means the
  257-entry blocks were not trimmed to 256 and every index is off
- **RTF around 0.007.** Anything near 0.2 is CPU
"""),
        code("!python stage4_asr.py --system indicconformer --data data "
             "--wav-dir {AUDIO} --limit 3"),
        md("""
### The check that actually matters

A clean summary line does not prove the decode is right. Read the `text` line:

- **it must be readable prose in ONE script, with no foreign characters at all.**
  On `0AEEA8NyVwY__000011000_000609000` the opening is
  `नमस्कार मी गौरव जोशी आणि मी अमोल कऱ्हाडकर …`. A single stray glyph means the
  language mask is not being applied
- **words that are phonetically right but spelled across several scripts**
  (`ನमस्कार`, `ଗౌरਵ`) is the multisoftmax failure. The CTC head was exported with
  `multisoftmax: True`: its softmax was trained over ONE language's 256-token
  block at a time, so logits from different blocks are on incomparable scales
  and a global argmax over all 5632 picks a different block nearly every frame.
  The decode must pick the clip's language from the frame votes and then argmax
  *within* that block. `--system indicconformer_free` reproduces the broken
  behaviour on purpose, as the ablation below
- **timestamps resetting every ~28 s** means the chunk offset is not applied, so
  every word after the first chunk is pinned to the wrong moment. WER would look
  fine and attribution would be destroyed
- **timestamps resetting every ~28 s** means the chunk offset is not applied, so
  every word after the first chunk is pinned to the wrong moment. WER would look
  fine and attribution would be destroyed
"""),
        code(s4_inspect("indicconformer")),
        md("## Full run\n\nResumable: clips already marked `ok` are skipped, so a "
           "dead session restarts at the clip that was in flight. At RTF 0.007 the "
           "whole corpus is about five minutes."),
        code("!python stage4_asr.py --system indicconformer --data data --wav-dir {AUDIO}"),
        md("""
### Ablation — the same model with the language mask off

`indicconformer_free` is the identical checkpoint, identical features, identical
words; the only difference is that the argmax runs over all 5632 tokens instead
of the chosen language's 256. That is what a multisoftmax head does when you
decode it as if it were a single softmax, and it turns fluent Marathi into
`ನमस्कार ଗౌरਵ` — right sounds, six scripts.

Another five minutes of GPU buys a measured WER delta for the writeup instead of
an assertion, and it is the evidence that the language mask is a correctness fix
rather than a tuning choice.
"""),
        code("!python stage4_asr.py --system indicconformer_free --data data "
             "--wav-dir {AUDIO}"),
        code(s4_coverage("indicconformer")),
        md("""
## Save

**Save Version → Quick Save**, then Output tab → **New dataset**, named
`sarvam-diar-asr-indic`. Notebook outputs re-point at the latest version, which
is how the Stage 3 RTTMs went missing; a dataset does not.
"""),
        code('!du -sh /kaggle/working/data/asr/* 2>/dev/null\n'
             '!find /kaggle/working/data/asr -name "*.json" | wc -l'),
    ]
    out = NOTEBOOKS / "stage4_asr_indicconformer.ipynb"
    out.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")



S4W_NSPROBE = "\n".join([
    "# Does faster-whisper's internal no-speech skip explain the word deficit?",
    "#",
    "# vad_filter=False turns off the *external* Silero VAD, but CTranslate2 still",
    "# applies no_speech_threshold=0.6 inside the decode loop: when a window looks",
    "# like non-speech AND its avg_logprob is low, the window is skipped and emits",
    "# nothing. That is invisible from outside -- the clip just comes back short.",
    "#",
    "# These four clips returned ~10-18% of the reference word count on the corpus",
    "# run while spending near-real-time on the GPU, which is the signature of",
    "# skipping rather than of fast, confident, wrong decoding.",
    "",
    "import soundfile as sf",
    "from faster_whisper import WhisperModel",
    "",
    "PROBE = {",
    "    'EmsKkNN2Me4__000035000_000095000':  180,",
    "    'PRAzUz0GANs__000223000_000283000':  197,",
    "    'Tlha36rSd5o__000240000_000374000':  318,",
    "    'BGAAfht5dYw__000000000_000612000': 2149,",
    "}",
    "",
    "m = WhisperModel('large-v3', device='cuda', compute_type='float16')",
    "",
    "def count(pcm, **kw):",
    "    segs, _ = m.transcribe(pcm, word_timestamps=True, vad_filter=False,",
    "                           condition_on_previous_text=False, **kw)",
    "    return sum(len([w for w in (s.words or []) if w.word.strip()]) for s in segs)",
    "",
    "hdr = ('clip', 'ref', 'default', 'ns=None', 'gain')",
    "print('%-34s %6s %8s %8s %6s' % hdr)",
    "for clip, ref in PROBE.items():",
    "    pcm, sr = sf.read(str(AUDIO / (clip + '.wav')), dtype='float32')",
    "    if pcm.ndim > 1:",
    "        pcm = pcm.mean(1)",
    "    a = count(pcm)",
    "    b = count(pcm, no_speech_threshold=None)",
    "    print('%-34s %6d %8d %8d %+6d' % (clip[:34], ref, a, b, b - a))",
    "",
    "# Reading it: if ns=None recovers most of the gap, the deficit is a decode",
    "# setting and the corpus run should be repeated with it off. If both columns",
    "# stay near the default, Whisper genuinely cannot transcribe these clips and",
    "# 86.68 WER is the honest number -- report it and move on.",
])


S4W_DETERMINISM = "\n".join([
    "# Is the decode even reproducible?",
    "#",
    "# The no-speech probe returned 47 and 117 words for two clips that the corpus",
    "# run scored at 18 and 34 -- same model, same audio (read_wav's int16/32768 is",
    "# bit-identical to soundfile float32), same three decode arguments. Nothing in",
    "# the pipeline explains a 3.4x swing, which leaves the decoder itself.",
    "#",
    "# faster-whisper's default temperature is [0, 0.2, 0.4, 0.6, 0.8, 1.0]. A",
    "# segment that trips the compression-ratio or avg-logprob threshold is",
    "# re-decoded at the next temperature, and above zero that samples rather than",
    "# taking the argmax -- unseeded. On clips where the fallback fires constantly",
    "# the transcript is a different draw every run.",
    "#",
    "# temperature=0.0 disables the fallback outright: greedy, deterministic,",
    "# repeatable. For a benchmark a reproducible number beats a slightly better",
    "# one that no one else can reproduce.",
    "",
    "import soundfile as sf",
    "from faster_whisper import WhisperModel",
    "",
    "CLIPS = ['Tlha36rSd5o__000240000_000374000',",
    "         'EmsKkNN2Me4__000035000_000095000']",
    "REPEATS = 3",
    "",
    "m = WhisperModel('large-v3', device='cuda', compute_type='float16')",
    "",
    "def count(pcm, **kw):",
    "    segs, _ = m.transcribe(pcm, word_timestamps=True, vad_filter=False,",
    "                           condition_on_previous_text=False, **kw)",
    "    return sum(len([w for w in (s.words or []) if w.word.strip()]) for s in segs)",
    "",
    "print('%-34s %-9s %s' % ('clip', 'setting', 'word counts over %d runs' % REPEATS))",
    "for clip in CLIPS:",
    "    pcm, _ = sf.read(str(AUDIO / (clip + '.wav')), dtype='float32')",
    "    if pcm.ndim > 1:",
    "        pcm = pcm.mean(1)",
    "    for label, kw in (('default', {}), ('temp=0', {'temperature': 0.0})):",
    "        runs = [count(pcm, **kw) for _ in range(REPEATS)]",
    "        spread = max(runs) - min(runs)",
    "        flag = 'STOCHASTIC' if spread else 'stable'",
    "        print('%-34s %-9s %-22s spread %4d  %s'",
    "              % (clip[:34], label, runs, spread, flag))",
    "",
    "# Reading it: 'default' spread > 0 confirms the fallback is sampling, and every",
    "# Whisper number in the results table inherits that variance. 'temp=0' must be",
    "# stable across all three runs -- if it is not, the non-determinism is coming",
    "# from somewhere else and changing temperature will not fix it.",
    "#",
    "# If temp=0 is stable AND its counts are not much worse than the default's",
    "# best draw, set temperature=0.0 in stage4_asr.py and rerun the corpus. That is",
    "# the only finding so far that justifies spending the 6.6 h again.",
])


def build_stage4_whisper() -> None:
    cells = [
        md(HEADER.format(title="Whisper large-v3")),
        md(SPLIT_NOTE),
        md("""
### Why faster-whisper rather than WhisperX

WhisperX refines timestamps with per-language wav2vec2 alignment models, which do
not exist for most of the nine Indic scripts in this corpus. Whisper's own
cross-attention DTW word timestamps are coarser but exist for every language
here, and a metric that silently degrades for some languages and not others is
worse than one that is uniformly approximate.

Three decode settings are deliberate. `condition_on_previous_text=False` stops a
single hallucinated segment from seeding a loop across the rest of a long clip.
`vad_filter=False` because diarization owns the speech/non-speech decision — an
ASR-side VAD would delete words that the attribution stage is meant to score.
Segments that trip the compression-ratio or no-speech thresholds are **counted
and logged, never dropped**: a silently discarded segment is an invisible
deletion that inflates the miss rate with nothing recording why.

`temperature=0.0` disables Whisper's temperature fallback, and that one is a
correction rather than a precaution — the first corpus run left the default in
place. The fallback re-decodes a suspect segment at rising temperature, and above
zero it samples unseeded, so the transcript is a different draw each run. On
`Tlha36rSd5o` (318 reference words) three identical calls returned 87, 84 and 100
words; with the fallback off, 211 words, the same 211 every time. It was both
irreproducible and worse, because the loop keeps a sampled draw in preference to
the greedy decode it started from. The diagnostics at the end of this notebook
are what found it.
"""),
        md("**Install only faster-whisper here.** Adding `onnxruntime-gpu` to this "
           "session replaces the cuDNN CTranslate2 needs and Whisper drops to CPU "
           "with no error."),
        code("!pip install -q faster-whisper"),
        code(S4_SETUP),
        md("""
## Smoke test — 2 clips

- `device=cuda` and `faster-whisper large-v3 (float16)`
- **RTF around 0.2–0.3.** Near 1.0, or a cell that sits for 15 minutes with an
  idle GPU, means CTranslate2 is on CPU
- native script in the words, not transliteration
"""),
        code("!python stage4_asr.py --system whisper --data data "
             "--wav-dir {AUDIO} --limit 2"),
        code(s4_inspect("whisper")),
        md("""
## Full run

The first corpus run measured mean RTF 0.547 over 12.28 hours — **6.6 hours of
T4 time**, the longest single job in the project. Expect this pass to be quicker:
`temperature=0.0` removes up to five re-decodes per suspect segment, and those
re-decodes were most of the variance (per-clip RTF ranged 0.09 to 1.42). Start it
with plenty of session left regardless.

It is resumable, so a timeout costs only the clip in flight. **That also means it
will skip every clip already on disk** — set `WIPE = True` in the fresh-start
cell before rerunning, or the old stochastic transcripts survive and you score a
mixture of two decode settings.
"""),
        code(S4W_FRESH),
        code("!python stage4_asr.py --system whisper --data data --wav-dir {AUDIO}"),
        code(s4_coverage("whisper")),
        md("""
## Save

**Save Version → Quick Save**, then Output tab → **New dataset**, named
`sarvam-diar-asr-whisper`.
"""),
        code('!du -sh /kaggle/working/data/asr/* 2>/dev/null\n'
             '!find /kaggle/working/data/asr -name "*.json" | wc -l'),
        md("""
## Diagnostic - the word deficit

Whisper returned 54,577 words against 126,583 in the reference (ratio 0.43,
median 0.36 per clip). Most of that is genuine error, but the deficit is not
uniform: a tail of clips came back at 0.10-0.18 while spending near real time on
the GPU. Slow *and* near-empty points at windows being skipped inside the decode
loop rather than at fast wrong answers.

This cell is **diagnostic only** -- it writes nothing to `data/asr` and costs
about a minute of GPU. Rerun the corpus with the setting changed only if the
gain here is large.
"""),
        code(S4W_NSPROBE),
        md("""
## Diagnostic - is the decode reproducible?

The probe above returned word counts well above what the corpus run recorded for
the same clips, on identical audio and identical settings. That is not a
threshold question, so this cell asks the prior one: does Whisper return the same
transcript twice?

Roughly two minutes of GPU. If the default decode turns out to be stochastic,
every Whisper figure in the results table carries unmeasured variance and the
table needs a footnote at minimum.
"""),
        code(S4W_DETERMINISM),
    ]
    out = NOTEBOOKS / "stage4_asr_whisper.ipynb"
    out.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


S5_SETUP = """
import pathlib, shutil

ROOT = pathlib.Path("/kaggle/input")
_cands = {p.parent for p in ROOT.rglob("stage5_correct.py")}
if not _cands:
    raise SystemExit("stage5_correct.py is in no attached dataset -- "
                     "re-upload sarvam-diar-code")
CODE = max(_cands, key=lambda d: (len(list(d.glob("stage*.py"))),
                                  "code" in str(d).lower()))
if len(_cands) > 1:
    print("script copies found in:")
    for _c in sorted(map(str, _cands)):
        print("   ", _c, "  <-- using" if str(CODE) == _c else "")

WORK = pathlib.Path("/kaggle/working/data")
WORK.mkdir(parents=True, exist_ok=True)
for f in CODE.glob("*.py"):
    shutil.copy(f, "/kaggle/working/")

# `attrib` is the one that matters here and it is absent from the Stage 3 and
# Stage 4a datasets, so it joins the list rather than replacing it: the audit
# needs ref/rttm and scoring needs ref/segments.
for src in sorted(ROOT.rglob("data")):
    if src.is_dir() and any((src / d).exists()
                            for d in ("attrib", "hyp", "ref", "asr")):
        shutil.copytree(src, WORK, dirs_exist_ok=True)
        print("restored", src)
if (CODE / "ref").is_dir():
    shutil.copytree(CODE / "ref", WORK / "ref", dirs_exist_ok=True)
    print("restored", CODE / "ref")

print()
print("CODE:", CODE)
CONDS = sorted(p.name for p in (WORK / "attrib").glob("*") if p.is_dir())
print("conditions:", len(CONDS))
for c in CONDS:
    print("   ", c, len(list((WORK / "attrib" / c).glob("*.json"))), "clips")
print("ref/rttm    :", len(list((WORK / "ref" / "rttm").glob("*.rttm"))))
print("ref/segments:", len(list((WORK / "ref" / "segments").glob("*.json"))))

_src = pathlib.Path("/kaggle/working/stage5_correct.py").read_text(encoding="utf-8")
assert "merged_words_maximal" in _src and "def close" in _src, (
    "stale stage5_correct.py -- this copy predates the audit and cleanup fixes. "
    "Re-upload sarvam-diar-code, confirm the new version is attached, and "
    "RESTART THE KERNEL (an already-imported module is not re-read)."
)
print("stage5_correct.py: current")

assert CONDS, "no Stage 4b conditions -- attach the sarvam-diar-attrib dataset"
assert (WORK / "ref" / "rttm").is_dir(), "no ref/rttm: --audit cannot run"
assert (WORK / "ref" / "segments").is_dir(), "no ref/segments: scoring cannot run"

# --- provenance -----------------------------------------------------------
# Kaggle attaches a SPECIFIC dataset version, so an old sarvam-diar-attrib can
# look perfectly healthy while carrying pre-fix words. It cost a full Stage 5
# run once: the directory named `indicconformer` held the unmasked decode, and
# every absolute number came out against the wrong ASR. The comparison between
# baseline/rule/llm was still internally valid, which is exactly what made it
# hard to notice.
#
# `indicconformer_free` was added as an ablation in the same change that fixed
# the multisoftmax decode, so its ABSENCE dates the dataset. That is a
# structural check, not a magic number.
ASRS = sorted({c.split("__")[0] for c in CONDS})
print()
print("ASR systems in attrib:", ASRS)
import json as _json
for c in CONDS:
    _n = sum(len(_json.loads(p.read_text(encoding="utf-8"))["words"])
             for p in sorted((WORK / "attrib" / c).glob("*.json")))
    print(f"   {c:38s} {_n:>7,} words")

if "indicconformer" in ASRS and "indicconformer_free" not in ASRS:
    raise SystemExit(
        "STALE sarvam-diar-attrib: `indicconformer` is present but the "
        "`indicconformer_free` ablation is not. They were produced by the same "
        "change, so this snapshot predates the multisoftmax fix and the words "
        "under `indicconformer` are the UNMASKED decode (WER ~93.7, not "
        "~78.9). Re-run stage4_attribute_score.ipynb, save its output as a new "
        "sarvam-diar-attrib version, and attach THAT version here."
    )

# Stage 5 output must not be corrected again, and the oracle must not be
# corrected at all. These are the conditions worth spending GPU on.
BASE = [c for c in CONDS if "+" not in c and not c.endswith("__ref")]
print()
print("correctable:", BASE)
"""


S5_PEEK = """
# What the model actually sees, and what it says back.
#
# Worth two minutes before spending hours: if the reply is not JSON, or the
# model is relabelling everything, the full run will produce a table of
# baselines and you will have paid GPU to learn it.

import importlib, json, pathlib, sys
sys.path.insert(0, "/kaggle/working")
import stage5_correct as S

# A module already imported in this kernel is NOT re-read when the dataset is
# re-uploaded and the setup cell copies a new file over it -- `import` returns
# the cached object and you debug a version that is no longer on disk.
S = importlib.reload(S)

COND = "indicconformer__pyannote31"
src = sorted((pathlib.Path("/kaggle/working/data/attrib") / COND).glob("*.json"))[0]
rec = json.loads(src.read_text(encoding="utf-8"))
units = S.build_units(rec["words"])
speakers = sorted({u["spk"] for u in units})
hi = min(S.WINDOW, len(units))
prompt = S.build_prompt(units, 0, hi, 0, speakers, rec.get("lang"))

print(f"clip {rec['clip_id']}  lang={rec.get('lang')}  "
      f"{len(rec['words'])} words -> {len(units)} units, {len(speakers)} speakers")
print("=" * 70)
print(prompt)
print("=" * 70)

llm = S.LLM(S.MODEL)
try:
    reply = llm(prompt)
    print("RAW REPLY:")
    print(reply)
    print("=" * 70)
    edits, bad = S.parse_edits(reply, range(0, hi), set(speakers), S.MIN_CONF)
    print("accepted:", edits)
    print("rejected:", {k: v for k, v in bad.items() if v})
    print(f"edit rate: {len(edits)}/{hi} units")
finally:
    # MUST run. The weights are ~8.8 GiB and this kernel holds them for its
    # whole life; the next cell shells out to a SEPARATE python process, which
    # then has ~5.7 GiB of a 14.6 GiB T4 and dies loading the same model.
    #
    # Done inline rather than via llm.close() so this still works against an
    # older copy of the script -- the cleanup must not itself depend on the
    # freshness of the thing it is cleaning up after.
    import gc

    import torch

    llm.model = None
    llm.tok = None
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"released GPU memory: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
"""


S5_REPORT = """
import json, pathlib, pandas as pd

rows = []
for m in sorted(pathlib.Path("/kaggle/working/data/attrib").glob("*+*")):
    recs = [json.loads(l) for l in
            (m / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    last = {r["clip_id"]: r for r in recs}
    ok = [r for r in last.values() if r.get("status") == "ok"]
    if not ok:
        continue
    rows.append({
        "condition": m.name,
        "clips": len(ok),
        "units": sum(r["n_units"] for r in ok),
        "proposed": sum(r["n_proposed"] for r in ok),
        "applied": sum(r["n_applied"] for r in ok),
        "words_relabelled": sum(r["n_words_relabelled"] for r in ok),
        "rogue_clips": sum(int(r["rogue"]) for r in ok),
        "low_conf": sum(r.get("low_conf", 0) for r in ok),
        "no_conf": sum(r.get("no_conf", 0) for r in ok),
        "parse_fail": sum(r.get("parse_fail", 0) for r in ok),
    })
df = pd.DataFrame(rows)
if len(df):
    df["edit_rate_%"] = (100 * df["applied"] / df["units"]).round(2)
    df["abstain_%"] = (100 * (df["low_conf"] + df["no_conf"])
                       / (df["applied"] + df["low_conf"] + df["no_conf"]).clip(lower=1)).round(1)
print(df.to_string(index=False))
"""


S5_DUAL = """
# Two GPUs, two clip shards, one condition.
#
# NOT model sharding: a 4-bit 7B fits in one T4, and device_map="auto" would
# split it pipeline-style so the halves run in SEQUENCE -- slower than one GPU,
# not faster. Data parallelism is where the 2x is: each worker pins itself to
# one GPU and takes disjoint clips, so neither can touch the other's output.

import subprocess, sys, pathlib, threading

COND    = "indicconformer__pyannote31"
METHOD  = "llm"
LIMIT   = []          # e.g. ["--limit", "25"] -- applies PER SHARD
N_GPU   = 2

def worker(i):
    cmd = [sys.executable, "stage5_correct.py", "--cond", COND,
           "--method", METHOD, "--data", "data",
           "--shard", f"{i}/{N_GPU}"] + LIMIT
    log = pathlib.Path(f"/kaggle/working/stage5_gpu{i}.log")
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd="/kaggle/working",
                             env={**__import__("os").environ,
                                  "CUDA_VISIBLE_DEVICES": str(i)})
        rc[i] = p.wait()

rc = {}
threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_GPU)]
for t in threads:
    t.start()
print(f"launched {N_GPU} workers; tail the logs below")
for t in threads:
    t.join()

for i in range(N_GPU):
    print("=" * 70)
    print(f"GPU {i}  (exit {rc[i]})")
    print(pathlib.Path(f"/kaggle/working/stage5_gpu{i}.log").read_text(
        encoding="utf-8", errors="replace")[-2500:])
"""


S5_RTTM = """
# Stage 5 relabels WORDS, but the results table needs baseline-vs-improved DER and
# JER, which are defined over TIME. This converts the corrected word labels back
# into RTTMs so stage3_score.py can score them.
#
# Boundaries are taken unchanged from the baseline diarizer and only the labels
# are revised, by duration-weighted majority of the corrected words in each
# turn. Rebuilding turns from word spans instead would drop the silence inside
# each turn, shrink hypothesis speech time, and move DER for a reason that has
# nothing to do with Stage 5.

#
# Read from disk, not from CONDS: the setup cell listed conditions BEFORE the
# rule and LLM passes above created theirs.
#
# BASE is converted too, with no edits applied. That round trip is the CONTROL:
# projecting word labels onto turns changes DER by itself (17% of pyannote31
# turns relabelled under IndicConformer words, 31% under Whisper), so a method's
# DER effect is `asr__diar+method` minus `asr__diar`, never minus the raw
# diarizer. Without these rows that cost is silently booked against Stage 5.
_attrib = pathlib.Path("/kaggle/working/data/attrib")
CORRECTED = sorted(p.name for p in _attrib.iterdir() if p.is_dir() and "+" in p.name)
RTTM_ARG = " ".join(CORRECTED + BASE)
print("converting:", RTTM_ARG)
"""


S6_REPORT = """
# Stage 6 -- CPU, seconds. Rebuilds every table from the per-clip CSVs above and
# refuses to write if WER moved between methods, if a boundary moved, or if a
# re-derived corpus figure disagrees with the Stage 3/4 summaries.
!python stage6_report.py --data data
"""


S5_DER = """
# Score the relabelled RTTMs against the baselines. Same metric policy as
# Stage 3: collar 0, overlap scored, UEM = the full clip.
import pathlib

SYSTEMS = sorted(p.name for p in pathlib.Path("/kaggle/working/data/hyp").glob("*")
                 if p.is_dir())
print("systems:", SYSTEMS)
DER_ARG = " ".join(SYSTEMS)
"""


def build_stage5() -> None:
    cells = [
        md("""
# Stage 5 — LLM speaker-label correction

**Attach:** `sarvam-diar-code`, `sarvam-diar-attrib` (Stage 4b output),
`sarvam-diar-stage3`.
**Settings:** GPU (T4) on, Internet on.

Stage 4 measured `attribution_cost` (cpWER − DI-cpWER) at 0.01–0.33 across all
twelve conditions. There is no cpWER headroom, so **this stage targets WDER**,
where the spread is real: 20.33 for pyannote31 against 39.53 for
sortformer_stream on identical IndicConformer words.

Three variants are scored side by side — **baseline**, **rule**, **llm** — so
"the LLM helped" has to beat a trivial heuristic, not merely beat doing nothing.
"""),
        md("""
### The edit space, and the invariant it buys

The model may only say *"unit 7 belongs to Speaker_A"*. No new speakers, no moved
boundaries, no edited text.

cpWER, DI-cpWER and WDER are functions of the word→speaker map alone, so relabel
is a **complete** edit space for every metric here — while WER, which ignores
speakers, **cannot move**. A Stage 5 run that shifts WER by 0.01 is broken, and
`check_text_unchanged` asserts it per clip rather than leaving it to be noticed
in the results table.

Units are same-speaker runs split further at pauses > 0.5 s. That split is what
makes false merges reachable at all: two speakers inside one unit cannot be
separated by relabelling it. The audit below measures how much that actually
buys instead of assuming it.
"""),
        code("# rapidfuzz is for stage4_score.py's WDER alignment at the end\n"
             "# of this notebook. It is not preinstalled on Kaggle, and\n"
             "# discovering that after the GPU work is done wastes a session.\n"
             "!pip install -q bitsandbytes accelerate rapidfuzz"),
        code(S5_SETUP),
        md("""
## First: the ceiling

`--audit` reads the reference RTTM and reports how many units still span two
true speakers after pause splitting — the false merges **no relabel can ever
fix** — against how many splitting rescued.

This is an **oracle diagnostic**. It writes no condition, never influences an
edit, and exists so the writeup can state the ceiling on relabel-only correction
rather than imply there isn't one. CPU, seconds.
"""),
        code('!python stage5_correct.py --cond indicconformer__pyannote31 '
             'indicconformer__sortformer_stream --audit --data data'),
        md("""
## The rule baseline — no GPU

Relabels any unit shorter than 1 s to the neighbour it is closer to, catching the
commonest diarization artefact: a sliver of turn dropped inside someone else's
speech. Run it on every correctable condition; it costs seconds and it is the
number the LLM has to beat.
"""),
        code('# BASE comes from the setup cell: every condition that is not an\n'
             '# oracle and not already a Stage 5 output.\n'
             'RULE_ARG = " ".join(BASE)\n'
             'print("correcting:", RULE_ARG)'),
        code("!python stage5_correct.py --method rule --data data --cond {RULE_ARG}"),
        md("""
## Look before you spend

Prints the exact prompt for one clip, the model's raw reply, and what survived
parsing and the confidence gate. Two minutes, and it is the cheapest way to find
out that the model returns prose instead of JSON, or wants to relabel everything.
"""),
        code(S5_PEEK),
        md("""
## Smoke test — 3 clips

Watch for: `units` in the tens not hundreds, a **low** edit count (conservative
is correct here), `0 clips rogue`, and no `[WARN]`. A run reporting many
`no confidence given` means the model is ignoring the reply schema — fix the
prompt before reading anything into the score.
"""),
        code('!python stage5_correct.py --cond indicconformer__pyannote31 '
             '--method llm --data data --limit 3'),
        md("""
## Full run

~25 s/clip, so **~40 min per condition**. The four below are ~2.7 h: three ASR
systems on the best diarizer for the results table, plus the worst diarizer on
IndicConformer to test whether correction helps more where there is more to fix.

Resumable — a dead session costs only the clip in flight.
"""),
        code('!python stage5_correct.py --data data --method llm --cond '
             'indicconformer__pyannote31 whisper__pyannote31 '
             'indicconformer_free__pyannote31 indicconformer__sortformer_stream'),
        md("""
## Two GPUs

Set the accelerator to **T4 x2** and this halves wall-clock. It is data
parallelism, not model sharding: a 4-bit 7B fits in a single T4, and
`device_map="auto"` would split it pipeline-style so the two halves run in
sequence -- slower than one GPU. Instead each worker pins itself to one GPU with
`CUDA_VISIBLE_DEVICES` and takes every other clip.

Shards are computed **before** the resume filter, so each worker owns a fixed
set no matter when it starts, and two workers can never be handed the same clip
-- they would otherwise race on the same output path and double-count in the
manifest.

Output is per-worker log files, printed when both finish; the notebook cell
itself shows nothing until then.
"""),
        code(S5_DUAL),
        md("### What the correction did"),
        code(S5_REPORT),
        md("""
## Score — baseline vs rule vs llm

Stage 4c needs no changes: it splits a condition name on the first `__`, so
`pyannote31+rule` and `pyannote31+llm` read as diarizers and sort directly under
their baseline.

**Check WER first.** A `+rule` or `+llm` row whose WER differs from its baseline
means the text invariant was violated and every other number in the row is void.
"""),
        code("!python stage4_score.py --data data"),
        code('import pandas as pd\n'
             'df = pd.read_csv("/kaggle/working/data/results/asr_summary.csv")\n'
             'df = df[df.subset == "all"] if "subset" in df else df\n'
             'print(df.sort_values(["asr", "diar"]).to_string(index=False))'),
        md("""
## DER and JER for the corrected output

The results table needs **baseline vs improved DER / JER / cpWER / WDER**. Stage 5
relabels words, so cpWER and WDER move but DER and JER cannot -- they are
defined over time, and nothing above has written a new RTTM.

This converts the corrected word labels back into RTTMs. Turn **boundaries are
taken unchanged** from the baseline diarizer; only the label is revised, by
duration-weighted majority of the corrected words inside each turn. So missed
speech, false alarm and total speech time are identical to the baseline by
construction, and any DER/JER delta is **speaker confusion alone** -- which is
the only thing a relabel-only stage can affect.

CPU, seconds.
"""),
        code(S5_RTTM),
        code("!python stage5_to_rttm.py --data data --cond {RTTM_ARG}"),
        code(S5_DER),
        code("!python stage3_score.py --data data --systems {DER_ARG} --diagnostic"),
        code('import pandas as pd\n'
             'print(pd.read_csv("/kaggle/working/data/results/'
             'diarization_summary.csv").to_string(index=False))'),
        md("""
## Stage 6 -- the results table

Baseline vs improved DER / JER / cpWER / WDER per model per video, with the
projection control beside every DER, per-clip win/loss counts, and breakdowns by
language, speaker count and overlap. Writes `results_table.md`,
`results_per_video.csv` and `results_per_video.xlsx` to `data/results/`.
"""),
        code(S6_REPORT.strip()),
        md("""
## Save

**Save Version → Quick Save**, then Output tab → **New dataset**, named
`sarvam-diar-stage5`.
"""),
        code('!du -sh /kaggle/working/data/attrib/* 2>/dev/null | tail -20'),
    ]
    out = NOTEBOOKS / "stage5_correct.ipynb"
    out.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


def build_stage4_attribute() -> None:
    cells = [
        md("""
# Stage 4b/4c — Attribution and scoring

**Attach:** `sarvam-diar-code`, `sarvam-diar-stage3`, `sarvam-diar-asr-indic`,
`sarvam-diar-asr-whisper`.
**Settings:** accelerator **None** — this stage is CPU-only and finishes in
seconds. Running it on a GPU session spends quota on nothing.

Stage 4a produced words with timestamps and no speaker. This stage crosses those
words with **one** diarization hypothesis at a time and writes one directory per
`(asr, diar)` condition. Because the words are identical across conditions, a
cpWER difference between two of them is attributable to the labelling — which is
the entire reason ASR and attribution are separate stages.

No audio is needed here, so the audio dataset stays detached.
"""),
        md("""
### The four rules, and why each one is a choice

1. **Maximum overlap.** A word goes to the turn sharing the most time with it.
   Assigning by midpoint is cheaper and throws away exactly the information that
   matters on words straddling a boundary. Equal overlap breaks toward the
   earlier turn, so the output is deterministic.
2. **Orphans are kept, not dropped.** A word landing where the diarizer heard
   nothing is given the nearest turn and flagged. Dropping it would delete it
   from the hypothesis and register as a cpWER deletion — quietly *rewarding* a
   system for missing speech. The flag is what lets Stage 6 separate this rule's
   cost from real labelling errors.
3. **Contested words are counted in two buckets.** `overlap` means two speakers
   genuinely active at once; `boundary` means a word crossing between two
   disjoint turns. Merging them would bury the first: the corpus is 7.60%
   overlapped, while every turn change makes boundary words.
4. **`--diar ref` is an oracle.** It attributes with the reference RTTM, giving a
   cpWER floor where labelling is perfect by construction, so every other
   condition reads as "ASR error + what this diarizer cost". Diagnostic only —
   nothing from it is fed back to any model, and it is labelled `oracle` in
   every table.
"""),
        code(S4B_SETUP),
        md("""
### What must be true before running

Attribution is silent about inputs it never sees: an ASR system whose dataset is
not attached simply produces no conditions, and the run still exits 0. The cell
above prints the inventory so a missing attachment is caught here rather than
discovered as a hole in the results table.

`sortformer` has 74 of 99 RTTMs — the long clips OOM'd in Stage 3. Those 25 clips
are expected to **fail loudly** below rather than be written as empty, which is
why the script's exit code is non-zero on that condition. That is the correct
outcome, not a bug to work around: a clip with no hypothesis is not a clip where
nobody spoke.
"""),
        md("""
### Language-ID fallback: one more ASR system, built from the two above

IndicConformer picks one language per clip from its own frame votes. On 13 clips
that vote lands outside the languages this task serves (11 Urdu, 2 Nepali), and
the whole clip is spelled in the wrong script: 100% WER regardless of what was
heard. `ic_lid_fallback` keeps IndicConformer's words everywhere else and takes
Whisper's on those clips.

No reference is read. The input is the model's own language decision and a fixed
list of served languages -- task configuration, not a per-clip label. CPU,
seconds, and written in Stage 4a's format so everything downstream treats it as
one more ASR system.
"""),
        code(S4B_FALLBACK),
        code(S4B_RUN),
        md("""
### Read the orphan rate before believing any cpWER

If a system orphans a few percent of words, rule 2 is a footnote. If it orphans
twenty, the rule is doing heavy lifting and the writeup has to say so before
quoting a single number.
"""),
        code(S4B_REPORT),
        md("""
### Which language did each clip get decoded in?

With the mask on, the language is now a **decision** taken once per clip from
frame votes, not something that emerges per token. A clip decoded in the wrong
block is ~100% WER by construction, so a handful of misidentified clips can
account for a large slice of the corpus WER — and this costs no GPU time to
check.

`clip_meta.csv` carries the reference script per clip. The diagonal should be
heavy; anything far off it is worth listening to.
"""),
        code(S4B_LID),
        md("""
# Stage 4c — Scoring

Same session: scoring is CPU-only too, and it needs exactly what 4b just wrote.

Four metrics, chosen so the errors decompose instead of piling into one number.
**WER** is speaker-agnostic and must be *identical* across every diar condition
of one ASR — it is the same words either way, so if it moves, something leaked
between stages. **cpWER** is the headline. **DI-cpWER** relaxes the speaker
constraint, so `cpWER − DI-cpWER` is what wrong attribution cost, which is
precisely the quantity Stage 5 sets out to reduce. **WDER** is hand-rolled:
meeteval has no WDER.

Corpus rates are error-weighted (sum of errors / sum of reference words), with
the unweighted mean beside them — that one lets a 50 s clip outweigh a 30 min
one, and is the number people publish by accident.
"""),
        code("\n".join([
            "# rapidfuzz is not on the Kaggle image, and WDER needs it for the word",
            "# alignment; a pure-Python DP would be minutes per clip at 3000 words.",
            "!pip install -q meeteval rapidfuzz",
        ])),
        md("""
The probe below matters: meeteval has moved these functions between
`meeteval.wer` and `meeteval.wer.wer.*` across releases, and DI-cpWER is recent.
`stage4_score.py` tries the known spellings and, if none match, prints what the
installed version actually exposes instead of dying on an AttributeError. If the
run fails, the list printed here is what to send back.
"""),
        code(S4C_PROBE),
        code("!python stage4_score.py --data data"),
        md("""
### Two subsets, and only one of them is a fair comparison

`sortformer` has 74 of 99 clips, and the 25 it lacks are all long ones — more
speakers, more turn changes than average. Its number over 74 clips is not
comparable to another system's over 99. The `common` table is the three-way
comparison; `all` is each condition over whatever it has.

Sanity checks worth making before believing any of it: **WER constant** across
the diar conditions of one ASR (it is the same words, so movement means a leak),
and `attribution_cost` non-negative everywhere — a large negative means the
meeteval binding is wrong, and the script says so loudly. A *small* negative is
the greedy DI-cpWER approximation and is expected.

The oracle is **not** guaranteed lowest on cpWER. When ASR error dominates, a
diarizer that merges speakers can score better than perfect diarization simply
by offering fewer ways to misattribute. If the oracle is not clearly best, that
is a statement about the ASR, not a bug in the scoring.
"""),
        code(S4C_REPORT),
        md("""
## Save

Strip the scripts from the working directory **before** saving. This dataset is
a snapshot of `/kaggle/working`, so it would otherwise carry its own copy of the
pipeline — and a later session's `rglob` can pick that stale copy over the code
dataset. That has already caused one silent run against old code and one against
old words.
"""),
        code(S4B_CLEAN),
        md("""
Then Output tab → **New Version** of `sarvam-diar-attrib` (not a new dataset —
Stages 5 and 6 attach this by name). It carries both the attribution and
`data/results/`, so they attach one thing.
"""),
    ]
    out = NOTEBOOKS / "stage4_attribute_score.ipynb"
    out.write_text(json.dumps(notebook(cells, accelerator="None"), indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    build_stage3()
    build_stage4_indic()
    build_stage4_whisper()
    build_stage4_attribute()
    build_stage5()


if __name__ == "__main__":
    main()
