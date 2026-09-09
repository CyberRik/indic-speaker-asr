#!/usr/bin/env python3
"""
Stage 3b -- Diarization scoring.

Scores hypothesis RTTMs against the Stage 2 references and writes per-clip and
per-system results. CPU-only and instant, so metric bugs cost seconds rather
than another GPU run.

    data/results/diarization_per_clip.csv
    data/results/diarization_summary.csv
    data/results/diarization_summary.md

Metric policy (deliberately unforgiving):
  * collar = 0.0        -- no boundary forgiveness
  * skip_overlap = False -- overlapping speech IS scored
  * an explicit UEM of [0, clip_duration] per clip, so the scoring region is the
    whole clip rather than the extent of reference-union-hypothesis. Without it,
    false alarms in leading/trailing silence are counted inconsistently.

Corpus numbers are DURATION-WEIGHTED (sum of errors / sum of reference speech),
not the mean of per-clip rates. Both are reported, because the unweighted mean
is the one people publish by accident: it lets a 50s clip outweigh a 30min one.

Usage:
    python stage3_score.py --data data --systems pyannote31 sortformer
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------

def load_rttm(path: Path):
    """Parse an RTTM file into a pyannote Annotation.

    Hand-rolled rather than using pyannote.database.util.load_rttm, which has
    moved between releases; the format is ten whitespace-separated fields.
    """
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=path.stem)
    if not path.exists():
        return ann
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        if dur <= 0:
            continue
        ann[Segment(start, start + dur)] = spk
    return ann


def subtract_timeline(a, b):
    """a minus b, as a new Timeline. pyannote.core's extrude() has moved across
    releases, so this is done explicitly with interval arithmetic."""
    from pyannote.core import Segment, Timeline

    cuts = []
    for seg in a:
        pieces = [(seg.start, seg.end)]
        for rem in b:
            nxt = []
            for lo, hi in pieces:
                if rem.end <= lo or rem.start >= hi:
                    nxt.append((lo, hi))
                    continue
                if rem.start > lo:
                    nxt.append((lo, rem.start))
                if rem.end < hi:
                    nxt.append((rem.end, hi))
            pieces = nxt
        cuts.extend(pieces)
    return Timeline([Segment(lo, hi) for lo, hi in cuts if hi - lo > 1e-6])


def load_pairs(system: str, data: Path, meta: pd.DataFrame) -> list[tuple]:
    """Load (meta_row, reference, hypothesis) once, for reuse across configs.

    The diagnostic runs six scoring passes; re-parsing ~200 RTTM files each time
    dominated the runtime.
    """
    ref_dir = data / "ref" / "rttm"
    hyp_dir = data / "hyp" / system / "rttm"
    pairs = []
    for _, m in meta.iterrows():
        if not m.has_audio:
            continue
        ref = load_rttm(ref_dir / f"{m.clip_id}.rttm")
        if not ref:
            continue
        pairs.append((m, ref, load_rttm(hyp_dir / f"{m.clip_id}.rttm")))
    return pairs


def score_pass(pairs: list[tuple], collar: float, skip_overlap: bool,
               region: str = "full") -> dict:
    """One scoring configuration over the whole corpus.

    region: 'full'    -- the whole clip
            'overlap' -- only where >=2 reference speakers are active
            'single'  -- reference speech with the overlap regions removed

    NOTE on the region split: DER applies an optimal speaker mapping, and that
    mapping is recomputed per scoring region. So the overlap/single numbers are
    a decomposition of *difficulty*, not a strictly additive decomposition of
    the full-clip error. Reported as such.
    """
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)

    err = tot = 0.0
    n = 0
    for m, ref, hyp in pairs:
        if region == "full":
            uem = Timeline([Segment(0.0, float(m.duration))], uri=m.clip_id)
        elif region == "overlap":
            uem = ref.get_overlap()
        elif region == "single":
            uem = subtract_timeline(ref.get_timeline().support(), ref.get_overlap())
        else:
            raise ValueError(region)

        if not list(uem):
            continue
        d = der(ref, hyp, uem=uem, detailed=True)
        if d["total"] <= 0:
            continue
        err += d["missed detection"] + d["false alarm"] + d["confusion"]
        tot += d["total"]
        n += 1

    return {"der": err / tot if tot else float("nan"),
            "err_sec": err, "ref_sec": tot, "n_clips": n}


def run_diagnostics(system: str, data: Path, meta: pd.DataFrame) -> None:
    """Quantify how much of the error comes from the metric policy vs the audio."""
    configs = [
        ("strict (reported)",      0.00, False, "full"),
        ("overlap not scored",     0.00, True,  "full"),
        ("collar 0.25",            0.25, False, "full"),
        ("collar 0.25 + no overlap", 0.25, True, "full"),
    ]
    pairs = load_pairs(system, data, meta)
    print("\n" + "=" * 78)
    print(f"DIAGNOSTIC -- {system}: metric policy sensitivity  ({len(pairs)} clips)")
    print("=" * 78)
    print(f"  {'configuration':<26}{'DER':>9}{'err (s)':>12}{'scored ref (s)':>16}")
    print("  " + "-" * 62)
    base = None
    for label, collar, skip, region in configs:
        r = score_pass(pairs, collar, skip, region)
        if base is None:
            base = r["der"]
        print(f"  {label:<26}{r['der']*100:>8.2f}%{r['err_sec']:>12.0f}{r['ref_sec']:>16.0f}")
    print("  " + "-" * 62)
    print("  The last row is roughly the configuration most published DER numbers use.")
    print("  Denominators differ between rows, so these are not subtractable.")

    print(f"\n  Error concentration by reference region:")
    print("  " + "-" * 62)
    for label, region in (("overlapped speech", "overlap"), ("single-speaker speech", "single")):
        r = score_pass(pairs, 0.0, False, region)
        print(f"  {label:<26}{r['der']*100:>8.2f}%{r['err_sec']:>12.0f}"
              f"{r['ref_sec']:>16.0f}  (n={r['n_clips']})")
    print("  " + "-" * 62)
    print("  Speaker mapping is recomputed per region, so these show where the")
    print("  difficulty concentrates -- they do not sum to the full-clip DER.")
    print("=" * 78)


def score_system(system: str, data: Path, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    ref_dir = data / "ref" / "rttm"
    hyp_dir = data / "hyp" / system / "rttm"
    if not hyp_dir.exists():
        raise SystemExit(f"no hypotheses for {system!r} at {hyp_dir} -- run stage3_diarize.py first")

    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)

    rows = []
    for _, m in meta.iterrows():
        clip_id = m.clip_id
        if not m.has_audio:
            continue                      # no audio -> no hypothesis is possible
        ref = load_rttm(ref_dir / f"{clip_id}.rttm")
        if not ref:
            continue

        hyp_path = hyp_dir / f"{clip_id}.rttm"
        # A missing/empty hypothesis is a legitimate total miss, NOT a skip.
        # Dropping it would flatter the system by removing its hardest clips.
        hyp = load_rttm(hyp_path)

        # Score the whole clip, explicitly.
        uem = Timeline([Segment(0.0, float(m.duration))], uri=clip_id)

        d = der(ref, hyp, uem=uem, detailed=True)
        j = jer(ref, hyp, uem=uem)

        n_ref_spk = len(ref.labels())
        n_hyp_spk = len(hyp.labels())
        total = d["total"]
        rows.append({
            "clip_id": clip_id,
            "system": system,
            "hyp_missing": not hyp_path.exists(),
            "der": d[der.name] if der.name in d else (d["missed detection"] + d["false alarm"]
                                                     + d["confusion"]) / total if total else 0.0,
            "miss": d["missed detection"] / total if total else 0.0,
            "false_alarm": d["false alarm"] / total if total else 0.0,
            "confusion": d["confusion"] / total if total else 0.0,
            "jer": float(j),
            "ref_speech_sec": total,
            "err_miss_sec": d["missed detection"],
            "err_fa_sec": d["false alarm"],
            "err_conf_sec": d["confusion"],
            "n_ref_speakers": n_ref_spk,
            "n_hyp_speakers": n_hyp_spk,
            "spk_count_correct": int(n_ref_spk == n_hyp_spk),
            "spk_count_err": n_hyp_spk - n_ref_spk,
            "duration": float(m.duration),
            "overlap_frac_of_speech": float(m.overlap_frac_of_speech),
            "language": m.language,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{system}: nothing scored")

    tot = df.ref_speech_sec.sum()
    summary = {
        "system": system,
        "n_clips": len(df),
        "n_hyp_missing": int(df.hyp_missing.sum()),
        # Duration-weighted -- the number to quote.
        "DER": (df.err_miss_sec.sum() + df.err_fa_sec.sum() + df.err_conf_sec.sum()) / tot,
        "miss": df.err_miss_sec.sum() / tot,
        "false_alarm": df.err_fa_sec.sum() / tot,
        "confusion": df.err_conf_sec.sum() / tot,
        # pyannote's own accumulated values, as a cross-check on the arithmetic.
        "DER_pyannote_accum": abs(der),
        "JER_pyannote_accum": abs(jer),
        # Unweighted, for contrast only.
        "DER_macro": df.der.mean(),
        "JER_macro": df.jer.mean(),
        "spk_count_acc": df.spk_count_correct.mean(),
        "spk_count_mae": df.spk_count_err.abs().mean(),
        "spk_count_bias": df.spk_count_err.mean(),
    }
    return df, summary


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3b: score diarization hypotheses.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--diagnostic", action="store_true",
                    help="also report metric-policy sensitivity and where error concentrates")
    args = ap.parse_args()

    meta = pd.read_csv(args.data / "ref" / "clip_meta.csv")
    out_dir = args.data / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_clip, summaries = [], []
    for system in args.systems:
        df, s = score_system(system, args.data, meta)
        per_clip.append(df)
        summaries.append(s)
        print(f"[ok] scored {system}: {len(df)} clips")

    per_clip_df = pd.concat(per_clip, ignore_index=True)
    per_clip_df.to_csv(out_dir / "diarization_per_clip.csv", index=False, encoding="utf-8")
    sm = pd.DataFrame(summaries)
    sm.to_csv(out_dir / "diarization_summary.csv", index=False, encoding="utf-8")

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("STAGE 3 -- BASELINE DIARIZATION   (collar=0.0, skip_overlap=False, UEM=full clip)")
    print("=" * 78)
    hdr = (f"{'system':<14}{'DER':>8}{'miss':>8}{'FA':>8}{'conf':>8}"
           f"{'JER':>8}{'spk acc':>9}{'spk MAE':>9}")
    print(hdr)
    print("-" * 78)
    for s in summaries:
        print(f"{s['system']:<14}{s['DER']*100:>7.2f}%{s['miss']*100:>7.2f}%"
              f"{s['false_alarm']*100:>7.2f}%{s['confusion']*100:>7.2f}%"
              f"{s['JER_pyannote_accum']*100:>7.2f}%{s['spk_count_acc']*100:>8.1f}%"
              f"{s['spk_count_mae']:>9.2f}")
    print("-" * 78)
    print("DER/miss/FA/conf are duration-weighted. Macro (per-clip mean) for contrast:")
    for s in summaries:
        print(f"  {s['system']:<14} DER_macro {s['DER_macro']*100:6.2f}%   "
              f"JER_macro {s['JER_macro']*100:6.2f}%   "
              f"(weighted DER {s['DER']*100:.2f}%)")
        if abs(s["DER"] - s["DER_pyannote_accum"]) > 1e-6:
            print(f"    [!] hand-computed DER {s['DER']*100:.4f}% != pyannote accumulated "
                  f"{s['DER_pyannote_accum']*100:.4f}% -- investigate")
        if s["n_hyp_missing"]:
            print(f"    [!] {s['n_hyp_missing']} clip(s) had NO hypothesis (scored as total miss)")

    # ---- per-condition breakdown (previews Stage 6) ----------------------
    print("\n" + "-" * 78)
    print("DER by reference speaker count (duration-weighted within each bucket):")
    print("-" * 78)
    piv = per_clip_df.copy()
    piv["err"] = piv.err_miss_sec + piv.err_fa_sec + piv.err_conf_sec
    g = (piv.groupby(["system", "n_ref_speakers"])
            .apply(lambda x: pd.Series({"DER": x.err.sum() / x.ref_speech_sec.sum(),
                                        "n": len(x)}), include_groups=False)
            .reset_index())
    for system in args.systems:
        sub = g[g.system == system]
        cells = "  ".join(f"{int(r.n_ref_speakers)}spk:{r.DER*100:5.1f}%(n={int(r.n)})"
                          for _, r in sub.iterrows())
        print(f"  {system:<14}{cells}")

    print("\nDER by overlap tercile:")
    print("-" * 78)
    piv["ov_bucket"] = pd.qcut(piv.overlap_frac_of_speech, 3,
                               labels=["low", "mid", "high"], duplicates="drop")
    g2 = (piv.groupby(["system", "ov_bucket"], observed=True)
             .apply(lambda x: pd.Series({"DER": x.err.sum() / x.ref_speech_sec.sum(),
                                         "n": len(x)}), include_groups=False)
             .reset_index())
    for system in args.systems:
        sub = g2[g2.system == system]
        cells = "  ".join(f"{r.ov_bucket}:{r.DER*100:5.1f}%(n={int(r.n)})" for _, r in sub.iterrows())
        print(f"  {system:<14}{cells}")

    # ---- markdown for the writeup ---------------------------------------
    md = ["# Baseline diarization", "",
          "`collar=0.0`, `skip_overlap=False`, UEM = full clip. "
          "DER components are duration-weighted.", "",
          "| System | DER | Miss | FA | Conf | JER | Spk acc | Spk MAE |",
          "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        md.append(f"| {s['system']} | {s['DER']*100:.2f}% | {s['miss']*100:.2f}% | "
                  f"{s['false_alarm']*100:.2f}% | {s['confusion']*100:.2f}% | "
                  f"{s['JER_pyannote_accum']*100:.2f}% | {s['spk_count_acc']*100:.1f}% | "
                  f"{s['spk_count_mae']:.2f} |")
    (out_dir / "diarization_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if args.diagnostic:
        for system in args.systems:
            run_diagnostics(system, args.data, meta)

    print(f"\nwrote -> {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
