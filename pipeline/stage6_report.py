#!/usr/bin/env python3
"""
Stage 6 -- the results table: baseline vs improved, per model, per video.

    python stage6_report.py --data data

Nothing here runs a model. Every number is a regrouping of per-clip error
counts that Stages 3-5 already wrote to data/results/, so this reruns in
seconds whenever anything upstream changes -- and it cannot quietly disagree
with the stage that produced a number, because it re-derives the corpus figures
and checks them against the Stage 3 and Stage 4 summaries before writing.

Writes to data/results/:
    results_per_video.csv    long: one row per clip x ASR x diarizer x method
    results_per_video.xlsx   summaries, plus one wide sheet per ASR x diarizer
    results_table.md         corpus table, consistency, breakdowns, and the
                             per-video table for the headline combination


MODEL AND METHOD
----------------
A "model" is an ASR x diarizer pair. A "method" is what ran on top of it:
    baseline   Stage 4 output: ASR words assigned to the diarizer's turns
    rule       Stage 5 heuristic: a sub-second unit takes its nearer neighbour
    llm        Stage 5 Qwen2.5-7B relabel (IndicConformer x pyannote31 only)

`indicconformer_free` is IndicConformer decoded with a naive argmax over the
whole multisoftmax head instead of the head for the detected language. It stays
in the table as a model in its own right: the gap between it and
`indicconformer` is the ASR-side improvement.


WHY THERE IS A SECOND DER COLUMN
--------------------------------
Stage 5 relabels words; DER is measured over time. stage5_to_rttm.py projects
word labels back onto the diarizer's own turns, and that projection is lossy on
its own: with ZERO edits it relabels 17% of pyannote31 turns under IndicConformer
words and 31% under Whisper words.

So every model carries two:
    DER        baseline: the diarizer's RTTM exactly as produced
               rule/llm: the relabelled RTTM
    DER_ctrl   the same round trip with no edits -- projection, no correction

A method's DER effect is DER - DER_ctrl. Comparing against the baseline DER
instead would charge the method for the projection. Boundaries never move, so
missed speech and false alarm are identical across all three; that is asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ASR_ORDER = ["indicconformer", "indicconformer_free", "whisper"]
DIAR_ORDER = ["pyannote31", "sortformer_stream", "sortformer"]
METHOD_ORDER = ["baseline", "rule", "llm"]
HEADLINE = ("indicconformer", "pyannote31")
NAIVE, LOCKED = "indicconformer_free", "indicconformer"

# Excel caps sheet names at 31 characters; `indicconformer_free__sortformer_stream`
# is 38.
SHORT = {"indicconformer": "IC", "indicconformer_free": "IC-naive",
         "whisper": "Whisper", "pyannote31": "pyannote",
         "sortformer": "sortformer", "sortformer_stream": "sf-stream"}

KEY = ["asr", "diar", "method", "clip_id"]
ASR_COLS = ["wer_errors", "wer_len", "cp_errors", "cp_len",
            "wder_errors", "wder_len"]
DIA_COLS = ["der", "jer", "err_miss_sec", "err_fa_sec", "err_conf_sec",
            "ref_speech_sec", "hyp_missing"]
EDIT_COLS = ["n_units", "n_applied", "n_words_relabelled", "rogue"]

# Missed speech / false alarm may differ by RTTM rounding (3 decimal places),
# never by more. Anything larger means a boundary moved.
BOUNDARY_TOL_SEC = 0.01


def rate(errors: pd.Series, length: pd.Series) -> float:
    """Error-weighted rate in percent: total errors over total reference."""
    n = length.sum()
    return 100.0 * errors.sum() / n if n else float("nan")


def der_of(g: pd.DataFrame, sfx: str = "") -> float:
    return rate(g[f"err_miss_sec{sfx}"] + g[f"err_fa_sec{sfx}"]
                + g[f"err_conf_sec{sfx}"], g[f"ref_speech_sec{sfx}"])


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ASR, diarizer, method in reading order rather than alphabet."""
    df = df.copy()
    rank = {c: {v: i for i, v in enumerate(o)} for c, o in
            (("asr", ASR_ORDER), ("diar", DIAR_ORDER), ("method", METHOD_ORDER))}
    cols = [c for c in ("asr", "diar", "method") if c in df]
    for c in cols:
        df[f"_{c}"] = df[c].map(rank[c]).fillna(99)
    by = [f"_{c}" for c in cols] + (["clip_id"] if "clip_id" in df else [])
    return (df.sort_values(by).drop(columns=[f"_{c}" for c in cols])
              .reset_index(drop=True))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_meta(data: Path) -> pd.DataFrame:
    meta = pd.read_csv(data / "ref" / "clip_meta.csv")
    meta = meta[meta.has_audio.astype(str).str.lower() == "true"].copy()
    meta["speakers"] = pd.Categorical(
        meta.n_speakers.map(lambda n: "5+" if n >= 5 else str(int(n))),
        ["2", "3", "4", "5+"], ordered=True)
    # Terciles over clips, not over rows of the long table: every clip counts
    # once regardless of how many models scored it.
    _, edges = pd.qcut(meta.overlap_frac_of_speech, 3, retbins=True)
    e1, e2 = edges[1], edges[2]
    meta["overlap"] = pd.qcut(
        meta.overlap_frac_of_speech, 3,
        labels=[f"low (<={e1:.1%})", f"mid ({e1:.1%}-{e2:.1%}]",
                f"high (>{e2:.1%})"])
    return meta


def load_edits(data: Path, models: pd.DataFrame) -> pd.DataFrame:
    """Per-clip edit counts from the Stage 5 manifests (last record wins)."""
    rows = []
    for a, d, m in models[models.method != "baseline"].itertuples(index=False):
        path = data / "attrib" / f"{a}__{d}+{m}" / "manifest.jsonl"
        if not path.is_file():
            continue
        last = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                last[rec["clip_id"]] = rec
        for rec in last.values():
            if rec.get("status") == "ok":
                rows.append({"asr": a, "diar": d, "method": m,
                             "clip_id": rec["clip_id"],
                             **{c: rec.get(c) for c in EDIT_COLS}})
    return pd.DataFrame(rows, columns=KEY + EDIT_COLS)


def build_long(data: Path) -> pd.DataFrame:
    res = data / "results"
    meta = load_meta(data)

    asr = pd.read_csv(res / "asr_per_clip.csv")
    # The `ref` condition assigns words using the REFERENCE diarization. It is
    # an oracle diagnostic, never a system result, so it has no row here.
    asr = asr[asr.oracle.astype(str).str.lower() != "true"].copy()
    parts = asr.diar.str.split("+", n=1, expand=True)
    asr["diar"] = parts[0]
    asr["method"] = parts[1].fillna("baseline") if 1 in parts else "baseline"

    # Every model x every clip, not just the clips a model produced words for:
    # Sortformer emitted nothing on 25 clips, and "per model per video" should
    # show those as scored-as-total-miss rows rather than silently omit them.
    models = asr[["asr", "diar", "method"]].drop_duplicates()
    df = models.merge(meta, how="cross")
    df = df.merge(asr[KEY + ASR_COLS], on=KEY, how="left")

    df["sys"] = [d if m == "baseline" else f"{a}__{d}+{m}"
                 for a, d, m in zip(df.asr, df.diar, df.method)]
    df["sys_ctrl"] = df.asr + "__" + df.diar

    dia = pd.read_csv(res / "diarization_per_clip.csv")
    missing = sorted((set(df.sys) | set(df.sys_ctrl)) - set(dia.system))
    if missing:
        raise SystemExit(
            f"no diarization scores for {missing}.\n"
            f"Run stage5_to_rttm.py for the Stage 5 conditions and their bare "
            f"`asr__diar` controls, then stage3_score.py over every system in "
            f"data/hyp, then rerun this.")
    dia = dia.set_index(["system", "clip_id"])[DIA_COLS]
    df = df.join(dia, on=["sys", "clip_id"])
    df = df.join(dia.add_suffix("_ctrl"), on=["sys_ctrl", "clip_id"])

    df["WER"] = 100 * df.wer_errors / df.wer_len
    df["cpWER"] = 100 * df.cp_errors / df.cp_len
    df["WDER"] = 100 * df.wder_errors / df.wder_len
    df["DER"] = 100 * df.der
    df["JER"] = 100 * df.jer
    df["DER_ctrl"] = 100 * df.der_ctrl
    df["JER_ctrl"] = 100 * df.jer_ctrl

    df = df.merge(load_edits(data, models), on=KEY, how="left")
    return ordered(df)


# --------------------------------------------------------------------------
# Checks -- run before anything is written
# --------------------------------------------------------------------------

def check_invariants(df: pd.DataFrame) -> float:
    # Stage 5 may only relabel. If WER differs between methods on any clip,
    # text changed, and every number in that row is void.
    scored = df.dropna(subset=["wer_len"])
    n = scored.groupby(["asr", "diar", "clip_id"])[["wer_errors", "wer_len"]].nunique()
    bad = n[(n > 1).any(axis=1)]
    if len(bad):
        raise SystemExit(f"TEXT INVARIANT BROKEN on {len(bad)} clip(s):\n{bad.head()}")

    # Boundaries never move, so miss/FA must match the raw diarizer in both
    # the corrected RTTM and the control.
    base = (df[df.method == "baseline"]
            .set_index(["asr", "diar", "clip_id"])[["err_miss_sec", "err_fa_sec"]]
            .add_suffix("_base"))
    j = df.join(base, on=["asr", "diar", "clip_id"])
    dev = max((j[f"{c}{s}"] - j[f"{c}_base"]).abs().max()
              for c in ("err_miss_sec", "err_fa_sec") for s in ("", "_ctrl"))
    if dev > BOUNDARY_TOL_SEC:
        raise SystemExit(f"BOUNDARY INVARIANT BROKEN: miss/FA moved by {dev:.3f} s")
    return float(dev)


def check_against_upstream(corp: pd.DataFrame, res: Path) -> None:
    """Re-derived corpus figures must equal what Stages 3 and 4 reported."""
    asum = pd.read_csv(res / "asr_summary.csv")
    asum = asum[(asum.subset == "all") & ~asum.diar.str.contains("oracle")]
    got = corp.assign(diar_full=[d if m == "baseline" else f"{d}+{m}"
                                 for d, m in zip(corp.diar, corp.method)])
    m = asum.rename(columns={"diar": "diar_full"}).merge(
        got, on=["asr", "diar_full"], suffixes=("_up", ""))
    if len(m) != len(asum):
        raise SystemExit(f"{len(asum) - len(m)} Stage 4 summary row(s) have no "
                         f"counterpart here")
    worst = max((m[f"{c}_up"] - m[c].round(2)).abs().max()
                for c in ("WER", "cpWER", "WDER"))
    if worst > 0.011:
        raise SystemExit(f"ASR figures disagree with asr_summary.csv by {worst:.3f}")

    dsum = pd.read_csv(res / "diarization_summary.csv").set_index("system").DER * 100
    worst = max((corp.DER - corp.sys.map(dsum)).abs().max(),
                (corp.DER_ctrl - corp.sys_ctrl.map(dsum)).abs().max())
    if not worst < 1e-6:
        raise SystemExit(f"DER disagrees with diarization_summary.csv by {worst}")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def corpus_table(df: pd.DataFrame, res: Path) -> pd.DataFrame:
    jer = pd.read_csv(res / "diarization_summary.csv").set_index("system")
    jer = jer.JER_pyannote_accum * 100
    rows = []
    for (a, d, m), g in df.groupby(["asr", "diar", "method"], sort=False):
        rows.append({
            "asr": a, "diar": d, "method": m,
            "sys": g.sys.iloc[0], "sys_ctrl": g.sys_ctrl.iloc[0],
            "clips": int(g.wer_len.notna().sum()),
            "WER": rate(g.wer_errors, g.wer_len),
            "cpWER": rate(g.cp_errors, g.cp_len),
            "WDER": rate(g.wder_errors, g.wder_len),
            # DER over all 99 clips, hypothesis-less ones scored as total
            # miss -- the same convention as the Stage 3 table.
            "DER": der_of(g),
            "JER": jer[g.sys.iloc[0]],
            "DER_ctrl": der_of(g, "_ctrl"),
            "JER_ctrl": jer[g.sys_ctrl.iloc[0]],
        })
    out = ordered(pd.DataFrame(rows))
    base = out[out.method == "baseline"].set_index(["asr", "diar"])
    idx = pd.MultiIndex.from_frame(out[["asr", "diar"]])
    is_base = out.method == "baseline"
    for c in ("cpWER", "WDER"):
        out[f"d_{c}"] = (out[c].values - base[c].reindex(idx).values)
        out.loc[is_base, f"d_{c}"] = float("nan")
    out["d_DER_vs_ctrl"] = (out.DER - out.DER_ctrl).where(~is_base)
    # On the baseline row, control minus raw diarizer is the price of the
    # projection itself, before any correction.
    out["projection_cost"] = (out.DER_ctrl - out.DER).where(is_base)
    return out


def consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per clip: did the method make it better, leave it, or make it worse?"""
    base = df[df.method == "baseline"].set_index(["asr", "diar", "clip_id"])
    rows = []
    for (a, d, m), g in df[df.method != "baseline"].groupby(
            ["asr", "diar", "method"], sort=False):
        g = g[g.wer_len.notna()]          # clips the method actually ran on
        b = base.loc[list(zip(g.asr, g.diar, g.clip_id))]
        row = {"asr": a, "diar": d, "method": m, "clips": len(g)}
        for name, new, old in (("WDER", g.WDER.values, b.WDER.values),
                               ("cpWER", g.cpWER.values, b.cpWER.values),
                               ("DER vs ctrl", g.DER.values, g.DER_ctrl.values)):
            delta = pd.Series(new - old)
            row[f"{name}: better"] = int((delta < -1e-9).sum())
            row[f"{name}: same"] = int((delta.abs() <= 1e-9).sum())
            row[f"{name}: worse"] = int((delta > 1e-9).sum())
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """The headline combination, grouped by one clip property."""
    a, d = HEADLINE
    h = df[(df.asr == a) & (df.diar == d)]
    methods = [m for m in METHOD_ORDER if m in set(h.method)]
    rows = []
    for key, g in h.groupby(by, observed=True, sort=True):
        b = g[g.method == "baseline"]
        row = {by: key, "clips": b.clip_id.nunique(),
               "hours": b.duration.sum() / 3600,
               "WER": rate(b.wer_errors, b.wer_len),
               "cpWER": rate(b.cp_errors, b.cp_len)}
        for m in methods:
            s = g[g.method == m]
            row[f"WDER {m}"] = rate(s.wder_errors, s.wder_len)
        row["DER"] = der_of(b)
        row["JER (clip mean)"] = b.JER.mean()
        row["DER ctrl"] = der_of(b, "_ctrl")
        for m in methods[1:]:
            row[f"DER {m}"] = der_of(g[g.method == m])
        rows.append(row)
    out = pd.DataFrame(rows)
    if by == "language":
        out = out.sort_values(["clips", "language"], ascending=[False, True])
    return out.reset_index(drop=True)


def asr_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """WER per language for each ASR. WER ignores speakers, so any diarizer
    gives the same number; pyannote31 is used because it covers all 99 clips."""
    b = df[(df.method == "baseline") & (df.diar == "pyannote31")]
    rows = []
    for lang, g in b.groupby("language"):
        row = {"language": lang, "clips": g.clip_id.nunique()}
        for a in ASR_ORDER:
            s = g[g.asr == a]
            if len(s):
                row[f"WER {a}"] = rate(s.wer_errors, s.wer_len)
        if f"WER {NAIVE}" in row and f"WER {LOCKED}" in row:
            row["decode gain"] = row[f"WER {NAIVE}"] - row[f"WER {LOCKED}"]
        rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values(["clips", "language"], ascending=[False, True])
              .reset_index(drop=True))


def decode_table(corp: pd.DataFrame) -> pd.DataFrame:
    """The ASR-side improvement: naive argmax -> language-locked decode."""
    b = corp[corp.method == "baseline"].set_index(["asr", "diar"])
    rows = []
    for d in DIAR_ORDER:
        if (NAIVE, d) not in b.index or (LOCKED, d) not in b.index:
            continue
        n, lk = b.loc[(NAIVE, d)], b.loc[(LOCKED, d)]
        row = {"diar": d, "clips": int(lk.clips)}
        for c in ("WER", "cpWER", "WDER"):
            row[f"{c} naive"] = n[c]
            row[f"{c} locked"] = lk[c]
            row[f"d_{c}"] = lk[c] - n[c]
        rows.append(row)
    return pd.DataFrame(rows)


def per_video(df: pd.DataFrame, a: str, d: str) -> pd.DataFrame:
    """One model, one row per clip, methods side by side."""
    g = df[(df.asr == a) & (df.diar == d)]
    methods = [m for m in METHOD_ORDER if m in set(g.method)]
    wide = g.pivot(index="clip_id", columns="method",
                   values=["DER", "JER", "cpWER", "WDER"])
    base = g[g.method == "baseline"].set_index("clip_id")

    out = base[["language", "duration", "n_speakers", "hyp_missing"]].copy()
    out["overlap_%"] = 100 * base.overlap_frac_of_speech
    out["WER"] = base.WER
    for metric in ("DER", "JER"):
        out[f"{metric} baseline"] = wide[(metric, "baseline")]
        out[f"{metric} ctrl"] = base[f"{metric}_ctrl"]
        for m in methods[1:]:
            out[f"{metric} {m}"] = wide[(metric, m)]
    for metric in ("cpWER", "WDER"):
        for m in methods:
            out[f"{metric} {m}"] = wide[(metric, m)]
    for m in methods[1:]:
        out[f"d_DER {m} vs ctrl"] = wide[("DER", m)] - base.DER_ctrl
        out[f"d_cpWER {m}"] = wide[("cpWER", m)] - wide[("cpWER", "baseline")]
        out[f"d_WDER {m}"] = wide[("WDER", m)] - wide[("WDER", "baseline")]
    return out.reset_index()


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def md(df: pd.DataFrame) -> str:
    # NaN -> None so tabulate prints a dash instead of "nan".
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_markdown(index=False, floatfmt=".2f", missingval="–")


CORPUS_COLS = ["asr", "diar", "method", "clips", "WER", "cpWER", "WDER",
               "d_cpWER", "d_WDER", "DER", "JER", "DER_ctrl", "JER_ctrl",
               "projection_cost", "d_DER_vs_ctrl"]

# The per-video table in the markdown is read, not filtered, so it carries the
# four brief metrics and the control; JER variants and deltas stay in the xlsx.
HEADLINE_MD_COLS = ["clip_id", "language", "duration", "n_speakers",
                    "overlap_%", "WER", "DER baseline", "DER ctrl", "DER rule",
                    "DER llm", "JER baseline", "cpWER baseline", "cpWER rule",
                    "cpWER llm", "WDER baseline", "WDER rule", "WDER llm"]


def write_markdown(path: Path, meta: pd.DataFrame, corp, cons, dec, lang_asr,
                   by_lang, by_spk, by_ovl, headline, max_dev: float) -> None:
    hours = meta.duration.sum() / 3600
    ov = meta.overlap_sec.sum() / meta.speech_sec.sum()
    a, d = HEADLINE
    L = [
        "# Results -- baseline vs improved",
        "",
        f"{len(meta)} clips, {hours:.2f} h, {meta.language.nunique()} scripts/"
        f"languages, {ov:.2%} of reference speech overlapped.",
        "",
        "**Metric policy.** DER/JER: `collar=0.0`, `skip_overlap=False`, UEM = "
        "full clip; overlap is scored. ASR rates are error-weighted (total "
        "errors / total reference words), DER is duration-weighted. JER is "
        "pyannote's accumulated JER in corpus tables and a per-clip mean in "
        "breakdowns.",
        "",
        "**Rows.** `asr` x `diar` is the model; `method` is what ran on top: "
        "`baseline` (Stage 4), `rule` (Stage 5 heuristic), `llm` (Stage 5 "
        "Qwen2.5-7B). `indicconformer_free` is IndicConformer with a naive "
        "argmax over all language heads; `indicconformer` uses the head for the "
        "detected language. The oracle (reference-diarization) condition is a "
        "diagnostic and is excluded.",
        "",
        "**DER_ctrl.** Stage 5 edits word labels; DER needs turns. Projecting "
        "words back onto the diarizer's turns with *no* edits already changes "
        "DER, so a method's DER effect is `DER - DER_ctrl`, and on the baseline "
        "row `projection_cost = DER_ctrl - DER` is the price of the projection "
        "alone.",
        "",
        "**Checks passed before writing:** WER identical across methods on every "
        f"clip (labels changed, text did not); missed speech and false alarm "
        f"identical across baseline, control and corrected RTTMs (max deviation "
        f"{max_dev:.3f} s); every corpus figure re-derived from per-clip counts "
        f"and matched against the Stage 3 and Stage 4 summaries.",
        "",
        "## 1. Corpus: every model, every method",
        "",
        "Sortformer produced no output on 25 clips: its ASR metrics cover the "
        "74 it did, its DER counts the other 25 as total miss.",
        "",
        md(corp[CORPUS_COLS]),
        "",
        "## 2. The ASR improvement: language-locked decoding",
        "",
        "Same model, same weights, same audio; only the decode differs.",
        "",
        md(dec),
        "",
        "### WER by language (pyannote31, baseline)",
        "",
        md(lang_asr),
        "",
        "## 3. Consistency: per-clip wins and losses",
        "",
        "Counts of clips where the method lowered, left unchanged, or raised the "
        "metric relative to baseline (DER relative to the projection control).",
        "",
        md(cons),
        "",
        f"## 4. Where it helps and hurts: `{a}` x `{d}`",
        "",
        "### By language",
        "",
        md(by_lang),
        "",
        "### By reference speaker count",
        "",
        md(by_spk),
        "",
        "### By overlap (tercile of overlapped-speech fraction)",
        "",
        md(by_ovl),
        "",
        f"## 5. Per video: `{a}` x `{d}`",
        "",
        "JER for the corrected RTTMs, the deltas, and every other model's "
        "per-video table are sheets in `results_per_video.xlsx`; all rows are "
        "in `results_per_video.csv`.",
        "",
        md(headline[[c for c in HEADLINE_MD_COLS if c in headline]]),
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


def write_excel(path: Path, df, corp, cons, dec, lang_asr, by_lang, by_spk,
                by_ovl) -> bool:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[skip] openpyxl not installed -- no .xlsx (CSV and MD still written)")
        return False
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheets = {"corpus": corp[CORPUS_COLS], "consistency": cons,
                  "asr_decode": dec, "asr_by_language": lang_asr,
                  "by_language": by_lang, "by_speakers": by_spk,
                  "by_overlap": by_ovl}
        for a, d in corp[["asr", "diar"]].drop_duplicates().itertuples(index=False):
            sheets[f"{SHORT.get(a, a)} x {SHORT.get(d, d)}"[:31]] = per_video(df, a, d)
        for name, table in sheets.items():
            table.round(2).to_excel(xw, sheet_name=name, index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    res = args.data / "results"

    meta = load_meta(args.data)
    df = build_long(args.data)
    max_dev = check_invariants(df)
    corp = corpus_table(df, res)
    check_against_upstream(corp, res)

    cons = consistency_table(df)
    dec = decode_table(corp)
    lang_asr = asr_by_language(df)
    by_lang = breakdown(df, "language")
    by_spk = breakdown(df, "speakers")
    by_ovl = breakdown(df, "overlap")
    headline = per_video(df, *HEADLINE)

    long_cols = (KEY + ["language", "duration", "n_speakers", "speakers",
                        "overlap_frac_of_speech", "overlap", "hyp_missing",
                        "WER", "cpWER", "WDER", "DER", "JER", "DER_ctrl",
                        "JER_ctrl"] + EDIT_COLS + ASR_COLS
                 + ["err_miss_sec", "err_fa_sec", "err_conf_sec",
                    "ref_speech_sec", "sys", "sys_ctrl"])
    df[long_cols].to_csv(res / "results_per_video.csv", index=False,
                         encoding="utf-8", float_format="%.4f")
    write_markdown(res / "results_table.md", meta, corp, cons, dec, lang_asr,
                   by_lang, by_spk, by_ovl, headline, max_dev)
    xlsx = write_excel(res / "results_per_video.xlsx", df, corp, cons, dec,
                       lang_asr, by_lang, by_spk, by_ovl)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("=" * 78)
    print("STAGE 6 -- BASELINE VS IMPROVED")
    print("=" * 78)
    print(corp[CORPUS_COLS].round(2).to_string(index=False))
    print("\nASR decode (naive argmax -> language-locked):")
    print(dec.round(2).to_string(index=False))
    print("\nPer-clip consistency:")
    print(cons.to_string(index=False))
    print(f"\nchecks: text invariant ok; boundary deviation {max_dev:.3f} s; "
          f"corpus figures match Stage 3/4 summaries")
    print(f"wrote -> {res / 'results_per_video.csv'}  ({len(df):,} rows)")
    print(f"wrote -> {res / 'results_table.md'}")
    if xlsx:
        print(f"wrote -> {res / 'results_per_video.xlsx'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
