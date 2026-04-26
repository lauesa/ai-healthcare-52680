from __future__ import annotations

import argparse
import json
import re
import statistics as _stat
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, wilcoxon as _scipy_wilcoxon

# ---------------------------------------------------------------------------
# Colour palette — accessible coral / steel-blue (not red/green)
# ---------------------------------------------------------------------------
_COL_MED  = "#E07B6A"   # MedGemma
_COL_GEM  = "#5B8DB8"   # Gemma-3
_PALETTE  = [_COL_MED, _COL_GEM]
_LABELS   = ["MedGemma-4B", "Gemma-3-4B"]

# Pipeline diagram box colours
_C_DATA  = "#D6EAF8"   # light blue   — data
_C_ETL   = "#D5F5E3"   # light green  — ETL
_C_INF   = "#FDEBD0"   # light orange — inference
_C_EVAL  = "#F9EBEA"   # light pink   — evaluation
_C_STAT  = "#EAE4F9"   # light purple — statistics
_C_EDGE  = "#555555"
_C_TEXT  = "#1a1a1a"


# ---------------------------------------------------------------------------
# Figure 0 — Pipeline flow diagram
# ---------------------------------------------------------------------------

def _fig0_pipeline(out_dir: Path, style: str) -> None:
    """Render the study pipeline as a flow diagram (fig0_pipeline).

    Five columns: Data → ETL → Inference → Evaluation → Statistics.
    Three task rows align horizontally across Inference and Evaluation.
    No hardware references appear in any label.
    """
    fs  = 7.0  if style == "academic" else 10.5
    hfs = 7.5  if style == "academic" else 11.0
    fw  = 7.0  if style == "academic" else 11.0
    fh  = 2.9  if style == "academic" else 4.4

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- Layout grid -------------------------------------------------------
    scale = fw / 7.0
    CX = [v * scale for v in [0.10, 1.35, 2.72, 4.08, 5.50]]
    BW = 1.15 * scale

    MARGIN_BOT = 0.52
    MARGIN_TOP = 0.30
    usable_h   = fh - MARGIN_BOT - MARGIN_TOP
    BH         = usable_h / 3 - 0.06
    ROW_GAP    = 0.06

    def row_y(r: int) -> float:
        """Bottom-edge y of row r (r=0 is topmost task)."""
        return fh - MARGIN_TOP - (r + 1) * BH - r * ROW_GAP

    # ---- Drawing primitives ------------------------------------------------
    def _box(x: float, y: float, w: float, h: float,
             label: str, colour: str) -> None:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04", linewidth=0.7,
            edgecolor=_C_EDGE, facecolor=colour,
            transform=ax.transData, zorder=2, clip_on=False,
        ))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fs,
                color=_C_TEXT, linespacing=1.4,
                transform=ax.transData, zorder=3, clip_on=False)

    def _arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color=_C_EDGE,
                                   lw=0.9, mutation_scale=8),
                    zorder=1, clip_on=False)

    def right_mid(col: int, y_bot: float, h: float) -> tuple[float, float]:
        return CX[col] + BW, y_bot + h / 2

    def left_mid(col: int, y_bot: float, h: float) -> tuple[float, float]:
        return CX[col], y_bot + h / 2

    # ---- Column headers ----------------------------------------------------
    for ci, hdr in enumerate(["Data", "ETL", "Inference", "Evaluation", "Statistics"]):
        ax.text(CX[ci] + BW / 2, fh - MARGIN_TOP + 0.06, hdr,
                ha="center", va="bottom", fontsize=hfs, fontweight="bold",
                color="#333333", transform=ax.transData, clip_on=False)

    # ---- Col 0: Data -------------------------------------------------------
    data_y0 = row_y(2)
    data_h  = row_y(0) + BH - data_y0
    _box(CX[0], data_y0, BW, data_h,
         "MIMIC-III\nNOTEEVENTS\nCHARTEVENTS\nLABEVENTS\nDIAGNOSES", _C_DATA)

    # ---- Col 1: ETL (primary box + Step 9 sub-box) -------------------------
    etl_h   = data_h * 0.58
    etl_y0  = data_y0 + data_h - etl_h
    _box(CX[1], etl_y0, BW, etl_h, "ETL\n(Polars)\nFHIR-lite JSONL", _C_ETL)

    step9_h  = data_h * 0.34
    step9_y0 = data_y0
    _box(CX[1], step9_y0, BW, step9_h, "Discharge Summary\nExtraction", _C_ETL)

    # ---- Col 2 & 3: Inference / Evaluation rows ----------------------------
    tasks = [
        ("Sub-task 1\nNote Generation\n(AARP format)", _C_INF),
        ("Sub-task 2\nAbnormal\nIdentification",        _C_INF),
        ("Sub-task 3\nAdmission\nSummary",              _C_INF),
    ]
    evals = [
        ("ROUGE-L\nEntity F1\nAARP Score",          _C_EVAL),
        ("Precision\nRecall\nF1",                    _C_EVAL),
        ("ROUGE-L\nBERTScore F1\nCoverage Metrics",  _C_EVAL),
    ]
    for r, ((tlabel, tcol), (elabel, ecol)) in enumerate(zip(tasks, evals)):
        y0 = row_y(r)
        _box(CX[2], y0, BW, BH, tlabel, tcol)
        _box(CX[3], y0, BW, BH, elabel, ecol)

    # ---- Col 4: Statistics -------------------------------------------------
    _box(CX[4], data_y0, BW, data_h,
         "Paired\nWilcoxon\nSigned-Rank\nComparison", _C_STAT)

    # ---- Arrows ------------------------------------------------------------
    _arrow(*right_mid(0, etl_y0, etl_h), *left_mid(1, etl_y0, etl_h))
    for r in (0, 1):
        y0 = row_y(r)
        _arrow(CX[1] + BW, etl_y0 + etl_h / 2, CX[2], y0 + BH / 2)
    _arrow(CX[1] + BW * 0.5, etl_y0, CX[1] + BW * 0.5, step9_y0 + step9_h)
    _arrow(*right_mid(1, step9_y0, step9_h), *left_mid(2, row_y(2), BH))
    for r in range(3):
        y0 = row_y(r)
        _arrow(*right_mid(2, y0, BH), *left_mid(3, y0, BH))
    fan_ys = [data_y0 + data_h * f for f in (0.72, 0.50, 0.28)]
    for r in range(3):
        sx, sy = right_mid(3, row_y(r), BH)
        _arrow(sx, sy, CX[4], fan_ys[r])

    # ---- Legend ------------------------------------------------------------
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=c, edgecolor=_C_EDGE, label=lbl)
            for c, lbl in [(_C_DATA, "Data"), (_C_ETL, "ETL"),
                           (_C_INF, "Inference"), (_C_EVAL, "Evaluation"),
                           (_C_STAT, "Statistics")]
        ],
        loc="lower center", ncol=5, fontsize=fs - 0.5, frameon=False,
        bbox_to_anchor=(fw / 2, -0.02), bbox_transform=ax.transData,
    )

    fig.tight_layout(pad=0.1)
    _save(fig, out_dir / "fig0_pipeline", style)


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

def _apply_theme(style: str) -> None:
    """Set rcParams for *style* (``'academic'`` or ``'presentation'``)."""
    plt.rcdefaults()
    if style == "academic":
        sns.set_theme(style="whitegrid", font_scale=0.85)
        plt.rcParams.update({
            "font.family":      "DejaVu Sans",
            "font.size":        8,
            "axes.titlesize":   9,
            "axes.labelsize":   8,
            "xtick.labelsize":  7,
            "ytick.labelsize":  7,
            "legend.fontsize":  7,
            "figure.dpi":       150,
            "savefig.dpi":      300,
        })
    else:  # presentation
        sns.set_theme(style="white", font_scale=1.4)
        plt.rcParams.update({
            "font.family":      "DejaVu Sans",
            "font.size":        14,
            "axes.titlesize":   15,
            "axes.labelsize":   13,
            "xtick.labelsize":  12,
            "ytick.labelsize":  12,
            "legend.fontsize":  12,
            "figure.dpi":       150,
            "savefig.dpi":      300,
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })


def _save(fig: plt.Figure, path: Path, style: str) -> None:
    """Save *fig* as PDF (academic) and PNG (both styles)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    if style == "academic":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    recs: list[dict] = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file; return empty dict if missing."""
    return json.loads(path.read_text()) if path.exists() else {}


def _load_data(data_dir: Path) -> dict[str, Any]:
    """Load all evaluation, inference, summary, and comparison files.

    All statistical values used in figure annotations are derived from the
    loaded files at runtime — nothing is hardcoded in figure functions.
    """
    def p(name: str) -> Path:
        return data_dir / name

    def _try_jsonl(name: str) -> list[dict]:
        fp = p(name)
        return _load_jsonl(fp) if fp.exists() else []

    # Per-record eval files
    n1_med = _load_jsonl(p("eval_notes_medgemma_val637opa.jsonl"))
    n1_gem = _load_jsonl(p("eval_notes_gemma3_val637opa.jsonl"))
    n2_med = _load_jsonl(p("eval_abnormal_medgemma_val637opa.jsonl"))
    n2_gem = _load_jsonl(p("eval_abnormal_gemma3_val637opa.jsonl"))

    # Raw inference (needed for fig3 heatmap)
    inf_med = _load_jsonl(p("inference_abnormal_medgemma_val637opa.jsonl"))
    inf_gem = _load_jsonl(p("inference_abnormal_gemma3_val637opa.jsonl"))

    # Sub-task 3 (summary) — optional
    n3_med = _try_jsonl("eval_summary_medgemma_val637opa.jsonl")
    n3_gem = _try_jsonl("eval_summary_gemma3_val637opa.jsonl")

    # Corpus summary JSONs (mean ± std per metric)
    sum_notes_med    = _load_json(p("summary_notes_medgemma_val637opa.json"))
    sum_notes_gem    = _load_json(p("summary_notes_gemma3_val637opa.json"))
    sum_abnormal_med = _load_json(p("summary_abnormal_medgemma_val637opa.json"))
    sum_abnormal_gem = _load_json(p("summary_abnormal_gemma3_val637opa.json"))

    # Comparison JSONs (Wilcoxon p, effect r, significance flag)
    cmp_notes    = _load_json(p("comparison_notes_medgemma_val637opa_vs_notes_gemma3_val637opa.json"))
    cmp_abnormal = _load_json(p("comparison_abnormal_medgemma_val637opa_vs_abnormal_gemma3_val637opa.json"))
    cmp_summary  = _load_json(p("comparison_summary_medgemma_val637opa_vs_summary_gemma3_val637opa.json"))

    return dict(
        n1_med=n1_med, n1_gem=n1_gem,
        n2_med=n2_med, n2_gem=n2_gem,
        inf_med=inf_med, inf_gem=inf_gem,
        n3_med=n3_med, n3_gem=n3_gem,
        sum_notes_med=sum_notes_med,       sum_notes_gem=sum_notes_gem,
        sum_abnormal_med=sum_abnormal_med, sum_abnormal_gem=sum_abnormal_gem,
        cmp_notes=cmp_notes,
        cmp_abnormal=cmp_abnormal,
        cmp_summary=cmp_summary,
    )


def _cmp_lookup(comparison: dict[str, Any], metric: str) -> dict[str, Any]:
    """Return the metric row from a comparison JSON, or an empty dict."""
    for m in comparison.get("metrics", []):
        if m["metric"] == metric:
            return m
    return {}


def _p_label(p: float, r: float) -> str:
    """Format p-value + effect r for figure annotations.

    Args:
        p: Wilcoxon p-value.
        r: Rank-biserial correlation.

    Returns:
        String like ``"p<0.001 ***\\nr=0.828"`` or ``"p=0.017 *\\nr=0.975"``.
    """
    if p < 0.001:
        p_str = "p<0.001 ***"
    elif p < 0.01:
        p_str = f"p={p:.3f} **"
    elif p < 0.05:
        p_str = f"p={p:.3f} *"
    else:
        p_str = f"p={p:.3f} (n.s.)"
    return f"{p_str}\nr={r:.3f}"


# ---------------------------------------------------------------------------
# Figure 1 — Distribution violins: ROUGE-L, Entity F1, AARP Score
# ---------------------------------------------------------------------------

def _fig1_distributions(
    data: dict, out_dir: Path, style: str,
) -> None:
    """Split violin + strip for ROUGE-L, Entity F1, AARP Score."""
    _apply_theme(style)

    figw = 7.0 if style == "academic" else 11.0
    figh = 3.0 if style == "academic" else 5.0

    metrics = [
        ("rouge_l",    "ROUGE-L"),
        ("entity_f1",  "Entity F1"),
        ("aarp_score", "AARP Score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(figw, figh))
    fig.subplots_adjust(wspace=0.38)

    for ax, (key, title) in zip(axes, metrics):
        med_vals = np.array([r[key] for r in data["n1_med"] if key in r])
        gem_vals = np.array([r[key] for r in data["n1_gem"] if key in r])

        # Build tidy arrays for seaborn
        vals  = np.concatenate([med_vals, gem_vals])
        model = (["MedGemma-4B"] * len(med_vals) +
                 ["Gemma-3-4B"]  * len(gem_vals))

        df = pd.DataFrame({"Score": vals, "Model": model})

        sns.violinplot(
            data=df, x="Model", y="Score", hue="Model",
            palette=dict(zip(["MedGemma-4B", "Gemma-3-4B"], _PALETTE)),
            inner=None, cut=0, linewidth=0.8, ax=ax, legend=False,
            order=["MedGemma-4B", "Gemma-3-4B"],
        )
        # Strip overlay
        sns.stripplot(
            data=df, x="Model", y="Score", hue="Model",
            palette=dict(zip(["MedGemma-4B", "Gemma-3-4B"], _PALETTE)),
            size=1.6 if style == "academic" else 2.5,
            alpha=0.18, jitter=True, ax=ax, legend=False,
            order=["MedGemma-4B", "Gemma-3-4B"],
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MedGemma", "Gemma-3"], rotation=0)

        # Annotate medians
        for xi, vals_arr in enumerate([med_vals, gem_vals]):
            med = float(np.median(vals_arr))
            ax.annotate(
                f"{med:.3f}",
                xy=(xi, med), xytext=(xi + 0.22, med),
                fontsize=6 if style == "academic" else 10,
                color="black", va="center",
            )

    n_pairs = data["cmp_notes"].get("n_pairs", len(data["n1_med"]))
    fig.suptitle(
        f"Sub-task 1: Score Distributions by Model  (n={n_pairs} pairs)",
        fontsize=9 if style == "academic" else 14,
        fontweight="bold", y=1.02,
    )

    # Legend patches
    patches = [
        mpatches.Patch(color=_COL_MED, label="MedGemma-4B-IT"),
        mpatches.Patch(color=_COL_GEM, label="Gemma-3-4B-IT"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.06),
               frameon=False)

    _save(fig, out_dir / "fig1_distributions", style)


# ---------------------------------------------------------------------------
# Figure 2 — Paired ROUGE-L difference histogram
# ---------------------------------------------------------------------------

def _fig2_rouge_diff(
    data: dict, out_dir: Path, style: str,
) -> None:
    """Histogram of per-record (MedGemma − Gemma-3) ROUGE-L differences."""
    _apply_theme(style)

    figw = 4.5 if style == "academic" else 7.5
    figh = 3.2 if style == "academic" else 5.0

    med_by_id = {r["note_id"]: r for r in data["n1_med"]}
    gem_by_id = {r["note_id"]: r for r in data["n1_gem"]}
    common = sorted(set(med_by_id) & set(gem_by_id))
    diffs = np.array([
        med_by_id[i]["rouge_l"] - gem_by_id[i]["rouge_l"]
        for i in common
    ])

    n_med_better = int((diffs > 0).sum())
    n_gem_better = int((diffs < 0).sum())

    # Retrieve Wilcoxon stats from pre-computed comparison file
    cmp_row = _cmp_lookup(data["cmp_notes"], "rouge_l")
    w_p   = cmp_row.get("wilcoxon_p", float("nan"))
    w_r   = cmp_row.get("effect_r",   float("nan"))
    n_pairs = data["cmp_notes"].get("n_pairs", len(common))
    p_display = "p<0.001" if w_p < 0.001 else f"p={w_p:.3f}"

    fig, ax = plt.subplots(figsize=(figw, figh))

    bins = np.linspace(diffs.min() - 0.001, diffs.max() + 0.001, 42)
    ax.hist(diffs[diffs >= 0], bins=bins[bins >= 0],
            color=_COL_MED, alpha=0.75, label=f"MedGemma better  (n={n_med_better})")
    ax.hist(diffs[diffs < 0],  bins=bins[bins <= 0],
            color=_COL_GEM, alpha=0.75, label=f"Gemma-3 better   (n={n_gem_better})")

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)

    # KDE overlay
    kde = gaussian_kde(diffs, bw_method=0.3)
    x_kde = np.linspace(diffs.min() - 0.005, diffs.max() + 0.005, 300)
    y_kde = kde(x_kde)
    # Scale KDE to histogram height
    bin_width = bins[1] - bins[0]
    scale = len(diffs) * bin_width
    ax.plot(x_kde, y_kde * scale, color="black", linewidth=1.2, zorder=6, label="KDE")

    ax.set_xlabel("ROUGE-L  (MedGemma − Gemma-3)  per record")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Sub-task 1: Paired ROUGE-L Differences  (n={n_pairs} pairs,  {p_display})",
        fontweight="bold",
    )

    # Wilcoxon annotation
    txt_fs = 7 if style == "academic" else 11
    sig_str = "n.s." if w_p >= 0.05 else ("***" if w_p < 0.001 else ("**" if w_p < 0.01 else "*"))
    ax.text(
        0.98, 0.95,
        f"Wilcoxon {p_display}  ({sig_str})\nEffect r = {w_r:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=txt_fs,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax.legend(frameon=False, fontsize=7 if style == "academic" else 11)
    fig.tight_layout()

    _save(fig, out_dir / "fig2_rouge_diff", style)


# ---------------------------------------------------------------------------
# Figure 3 — Per-vital miss-rate heatmap
# ---------------------------------------------------------------------------

def _fig3_vital_heatmap(
    data: dict, out_dir: Path, style: str,
) -> None:
    """Heatmap of miss rate (%) per vital sign × model."""
    _apply_theme(style)

    figw = 5.5 if style == "academic" else 9.0
    figh = 4.0 if style == "academic" else 6.5

    def _compute_miss(inf_records: list) -> dict[str, tuple[int, int]]:
        """Returns {metric: (n_missed, n_truly_abnormal)}."""
        missed:  dict[str, int] = defaultdict(int)
        total:   dict[str, int] = defaultdict(int)
        for rec in inf_records:
            gen_lower = rec["generated_text"].lower()
            for flag in rec.get("gt_flags", []):
                if not flag["is_abnormal"]:
                    continue
                metric = flag["metric"]
                label  = flag["label"].lower()
                total[metric] += 1
                mentioned = bool(re.search(re.escape(label), gen_lower))
                if not mentioned:
                    mentioned = bool(re.search(re.escape(metric.lower()), gen_lower))
                if not mentioned:
                    missed[metric] += 1
        return {m: (missed.get(m, 0), total[m]) for m in total}

    miss_med = _compute_miss(data["inf_med"])
    miss_gem = _compute_miss(data["inf_gem"])

    # Union of all vitals, sorted by n_truly_abnormal descending
    all_vitals = sorted(
        set(miss_med) | set(miss_gem),
        key=lambda v: -(miss_med.get(v, (0, 0))[1]),
    )

    # Build rate matrix: rows=vitals, cols=[MedGemma, Gemma-3]
    rate_matrix = np.zeros((len(all_vitals), 2))
    annot_matrix = [[""] * 2 for _ in all_vitals]

    for ri, vital in enumerate(all_vitals):
        for ci, miss_dict in enumerate([miss_med, miss_gem]):
            n_miss, n_total = miss_dict.get(vital, (0, 0))
            rate = 100.0 * n_miss / n_total if n_total > 0 else 0.0
            rate_matrix[ri, ci] = rate
            annot_matrix[ri][ci] = f"{n_miss}/{n_total}\n({rate:.0f}%)"

    df_heat = pd.DataFrame(
        rate_matrix,
        index=all_vitals,
        columns=["MedGemma-4B", "Gemma-3-4B"],
    )

    fig, ax = plt.subplots(figsize=(figw, figh))

    sns.heatmap(
        df_heat,
        annot=annot_matrix,
        fmt="",
        cmap="YlOrRd",
        vmin=0, vmax=55,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Miss rate (%)", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 6 if style == "academic" else 10},
    )

    ax.set_title(
        "Sub-task 2: Abnormal Vital Miss Rate by Model\n"
        "(% of truly abnormal values not identified)",
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Vital Sign")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, out_dir / "fig3_vital_heatmap", style)


# ---------------------------------------------------------------------------
# Figure 4 — Abnormal recall & F1 box plots
# ---------------------------------------------------------------------------

def _fig4_abnormal_recall(
    data: dict, out_dir: Path, style: str,
) -> None:
    """Box plots for Abnormal Recall and F1 with significance annotations."""
    _apply_theme(style)

    figw = 5.5 if style == "academic" else 9.0
    figh = 3.5 if style == "academic" else 5.5

    metric_keys = [
        ("abnormal_recall", "Abnormal Recall"),
        ("abnormal_f1",     "Abnormal F1"),
    ]
    # Build sig_txt from the comparison file for each metric
    metrics = []
    for key, title in metric_keys:
        row = _cmp_lookup(data["cmp_abnormal"], key)
        sig_txt = _p_label(row.get("wilcoxon_p", 1.0), row.get("effect_r", 0.0))
        metrics.append((key, title, sig_txt))

    n_pairs = data["cmp_abnormal"].get("n_pairs", len(data["n2_med"]))

    fig, axes = plt.subplots(1, 2, figsize=(figw, figh), sharey=False)
    fig.subplots_adjust(wspace=0.38)

    for ax, (key, title, sig_txt) in zip(axes, metrics):
        med_vals = np.array([r[key] for r in data["n2_med"]])
        gem_vals = np.array([r[key] for r in data["n2_gem"]])

        vals  = np.concatenate([med_vals, gem_vals])
        model = (["MedGemma-4B"] * len(med_vals) +
                 ["Gemma-3-4B"]  * len(gem_vals))
        df = pd.DataFrame({"Score": vals, "Model": model})

        sns.boxplot(
            data=df, x="Model", y="Score",
            hue="Model",
            palette=dict(zip(["MedGemma-4B", "Gemma-3-4B"], _PALETTE)),
            width=0.45,
            fliersize=0,
            linewidth=0.9 if style == "academic" else 1.4,
            order=["MedGemma-4B", "Gemma-3-4B"],
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=df, x="Model", y="Score",
            hue="Model",
            palette=dict(zip(["MedGemma-4B", "Gemma-3-4B"], _PALETTE)),
            size=1.8 if style == "academic" else 3.0,
            alpha=0.2, jitter=True,
            order=["MedGemma-4B", "Gemma-3-4B"],
            legend=False,
            ax=ax,
        )

        # Significance bracket
        y_max = max(med_vals.max(), gem_vals.max())
        y_br  = y_max + 0.04
        bar_h = 0.015
        ax.plot([0, 0, 1, 1], [y_br, y_br + bar_h, y_br + bar_h, y_br],
                lw=1.0, color="black")
        ax.text(0.5, y_br + bar_h + 0.005, sig_txt,
                ha="center", va="bottom",
                fontsize=6 if style == "academic" else 10,
                color="black")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, y_br + bar_h + 0.10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MedGemma", "Gemma-3"], rotation=0)

    fig.suptitle(
        f"Sub-task 2: Abnormal Vital Identification  (n={n_pairs} pairs)",
        fontsize=9 if style == "academic" else 14,
        fontweight="bold", y=1.02,
    )

    patches = [
        mpatches.Patch(color=_COL_MED, label="MedGemma-4B-IT"),
        mpatches.Patch(color=_COL_GEM, label="Gemma-3-4B-IT"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)

    _save(fig, out_dir / "fig4_abnormal_recall", style)


# ---------------------------------------------------------------------------
# Figure 5 — Summary horizontal bar chart (significant results only)
# ---------------------------------------------------------------------------

def _fig5_summary_bars(
    data: dict, out_dir: Path, style: str,
) -> None:
    """Horizontal grouped bars for all statistically significant metrics.

    Metric means come from the per-task corpus summary JSONs; p-values and
    effect sizes come from the comparison JSONs.  Only significant metrics
    (p < 0.05) are displayed.
    """
    _apply_theme(style)

    figw = 6.0 if style == "academic" else 10.0
    figh = 3.5 if style == "academic" else 5.5

    # Metric display config: (metric_key, display_label, sum_med_key, sum_gem_key, cmp_key)
    # sum_*_key selects which corpus summary dict to pull std from.
    _METRIC_CFG = [
        ("aarp_score",        "AARP Score\n(Sub-task 1)",        "sum_notes_med",    "sum_notes_gem",    "cmp_notes"),
        ("rouge_l",           "ROUGE-L\n(Sub-task 1)",           "sum_notes_med",    "sum_notes_gem",    "cmp_notes"),
        ("entity_f1",         "Entity F1\n(Sub-task 1)",         "sum_notes_med",    "sum_notes_gem",    "cmp_notes"),
        ("entity_f1_vitals",  "Entity F1 – Vitals\n(Sub-task 1)","sum_notes_med",    "sum_notes_gem",    "cmp_notes"),
        ("entity_f1_drugs",   "Entity F1 – Drugs\n(Sub-task 1)", "sum_notes_med",    "sum_notes_gem",    "cmp_notes"),
        ("abnormal_precision","Abnormal Precision\n(Sub-task 2)","sum_abnormal_med", "sum_abnormal_gem", "cmp_abnormal"),
        ("abnormal_recall",   "Abnormal Recall\n(Sub-task 2)",   "sum_abnormal_med", "sum_abnormal_gem", "cmp_abnormal"),
        ("abnormal_f1",       "Abnormal F1\n(Sub-task 2)",       "sum_abnormal_med", "sum_abnormal_gem", "cmp_abnormal"),
    ]

    rows: list[tuple] = []
    for metric_key, display_label, sum_med_k, sum_gem_k, cmp_k in _METRIC_CFG:
        cmp_row = _cmp_lookup(data[cmp_k], metric_key)
        if not cmp_row or not cmp_row.get("significant", False):
            continue
        sum_med = data[sum_med_k].get(metric_key, {})
        sum_gem = data[sum_gem_k].get(metric_key, {})
        mm  = cmp_row["model_a_mean"]
        gm  = cmp_row["model_b_mean"]
        ms  = sum_med.get("std", 0.0)
        gs  = sum_gem.get("std", 0.0)
        p_lbl = _p_label(cmp_row["wilcoxon_p"], cmp_row["effect_r"]).replace("\n", "  ")
        rows.append((display_label, mm, ms, gm, gs, p_lbl))

    if not rows:
        return  # nothing significant to display

    n_pairs = data["cmp_notes"].get("n_pairs", "?")

    n_rows = len(rows)
    y_pos  = np.arange(n_rows)
    bar_h  = 0.32

    fig, ax = plt.subplots(figsize=(figw, max(figh, n_rows * 0.9)))

    for i, (label, mm, ms, gm, gs, p_lbl) in enumerate(rows):
        y_med = y_pos[i] + bar_h / 2
        y_gem = y_pos[i] - bar_h / 2

        ax.barh(y_med, mm, height=bar_h, color=_COL_MED,
                xerr=ms, capsize=3, error_kw={"linewidth": 0.8},
                label="MedGemma-4B-IT" if i == 0 else "_nolegend_")
        ax.barh(y_gem, gm, height=bar_h, color=_COL_GEM,
                xerr=gs, capsize=3, error_kw={"linewidth": 0.8},
                label="Gemma-3-4B-IT" if i == 0 else "_nolegend_")

        # p-value annotation at end of longer bar
        x_ann = max(mm + ms, gm + gs) + 0.01
        ax.text(x_ann, y_pos[i], p_lbl,
                va="center", ha="left",
                fontsize=6 if style == "academic" else 9,
                color="#444444")

        # Delta annotation inside the bars
        delta = gm - mm
        sign  = "+" if delta > 0 else ""
        ax.text(
            max(mm, gm) * 0.5, y_pos[i],
            f"Δ={sign}{delta:.3f}",
            va="center", ha="center",
            fontsize=5 if style == "academic" else 8,
            color="white", fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Mean Score  (±1 SD error bars)")
    ax.set_title(
        f"Significant Differences: MedGemma-4B vs Gemma-3-4B  (n={n_pairs} pairs)",
        fontweight="bold",
    )
    ax.set_xlim(0, 1.25)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Vertical reference line at 0
    ax.axvline(0, color="black", linewidth=0.6)

    ax.legend(frameon=False, loc="lower right",
              fontsize=7 if style == "academic" else 11)
    fig.tight_layout()

    _save(fig, out_dir / "fig5_summary_bars", style)



# ---------------------------------------------------------------------------
# Figure 6 — Sub-task 3: Admission Summary comparison
# ---------------------------------------------------------------------------

def _fig6_summary_task(data: dict[str, Any], out_dir: Path, style: str) -> None:
    """Three-panel Sub-task 3 comparison.

    Panel A: ROUGE-L violin (filtered-reference)
    Panel B: BERTScore F1 violin
    Panel C: Grouped bar chart — Problem Coverage, Key Lab Coverage,
             Medication Accuracy

    All p-values and significance markers are derived from the comparison JSON.
    """
    n3_med: list[dict] = data.get("n3_med", [])
    n3_gem: list[dict] = data.get("n3_gem", [])
    if not n3_med or not n3_gem:
        return  # data not yet available; skip silently

    _apply_theme(style)

    cmp = data.get("cmp_summary", {})
    n_pairs = cmp.get("n_pairs", len(n3_med))
    fs_ann = 6 if style == "academic" else 10

    fig, axes = plt.subplots(1, 3, figsize=(11, 4) if style == "academic" else (14, 5))

    # --- Panel A: ROUGE-L violin -------------------------------------------
    ax = axes[0]
    rouge_med = [r["rouge_l"] for r in n3_med]
    rouge_gem = [r["rouge_l"] for r in n3_gem]
    _vp = ax.violinplot(
        [rouge_med, rouge_gem], positions=[0, 1],
        showmedians=True, widths=0.6,
    )
    for pc, col in zip(_vp["bodies"], _PALETTE):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    for part in ("cmedians", "cmaxes", "cmins", "cbars"):
        _vp[part].set_color("black")
        _vp[part].set_linewidth(0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(_LABELS, rotation=12, ha="right")
    ax.set_ylabel("ROUGE-L (filtered ref.)")
    ax.set_title("A  ROUGE-L")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    row_rl = _cmp_lookup(cmp, "rouge_l")
    rl_p = row_rl.get("wilcoxon_p", float("nan"))
    rl_p_str = "p<0.001" if rl_p < 0.001 else f"p={rl_p:.3f}"
    sig_mark = "*" if row_rl.get("significant", False) else "n.s."
    med_m = float(np.median(rouge_med))
    med_g = float(np.median(rouge_gem))
    ax.annotate(
        f"med: {med_m:.3f} vs {med_g:.3f}\n{rl_p_str} {sig_mark}",
        xy=(0.5, 0.94), xycoords="axes fraction",
        ha="center", va="top", fontsize=fs_ann,
    )

    # --- Panel B: BERTScore F1 violin ------------------------------------
    ax = axes[1]
    bs_med = [r["bertscore_f1"] for r in n3_med]
    bs_gem = [r["bertscore_f1"] for r in n3_gem]
    _vp2 = ax.violinplot(
        [bs_med, bs_gem], positions=[0, 1],
        showmedians=True, widths=0.6,
    )
    for pc, col in zip(_vp2["bodies"], _PALETTE):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    for part in ("cmedians", "cmaxes", "cmins", "cbars"):
        _vp2[part].set_color("black")
        _vp2[part].set_linewidth(0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(_LABELS, rotation=12, ha="right")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("B  BERTScore F1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    row_bs = _cmp_lookup(cmp, "bertscore_f1")
    bs_p = row_bs.get("wilcoxon_p", float("nan"))
    bs_p_str = "p<0.001" if bs_p < 0.001 else f"p={bs_p:.3f}"
    bs_sig = "*" if row_bs.get("significant", False) else "n.s."
    med_bm = float(np.median(bs_med))
    med_bg = float(np.median(bs_gem))
    ax.annotate(
        f"med: {med_bm:.3f} vs {med_bg:.3f}\n{bs_p_str} {bs_sig}",
        xy=(0.5, 0.94), xycoords="axes fraction",
        ha="center", va="top", fontsize=fs_ann,
    )

    # --- Panel C: Coverage grouped bar chart --------------------------------
    ax = axes[2]
    coverage_metrics = [
        ("Problem\nCoverage",    "problem_coverage"),
        ("Key Lab\nCoverage",    "key_lab_coverage"),
        ("Medication\nAccuracy", "medication_accuracy"),
    ]
    labels_c  = [lbl for lbl, _ in coverage_metrics]
    means_med = [_stat.mean(r[k] for r in n3_med) for _, k in coverage_metrics]
    means_gem = [_stat.mean(r[k] for r in n3_gem) for _, k in coverage_metrics]
    # Significance markers derived from comparison file
    sig_marks = [
        "*" if _cmp_lookup(cmp, k).get("significant", False) else "n.s."
        for _, k in coverage_metrics
    ]

    x = np.arange(len(labels_c))
    width = 0.35
    ax.bar(x - width / 2, means_med, width, color=_COL_MED, alpha=0.85, label=_LABELS[0])
    ax.bar(x + width / 2, means_gem, width, color=_COL_GEM, alpha=0.85, label=_LABELS[1])

    # Annotate significance above the higher bar
    for i, sig in enumerate(sig_marks):
        top = max(means_med[i], means_gem[i])
        ax.text(i, top + 0.01, sig, ha="center", va="bottom",
                fontsize=8 if style == "academic" else 12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontsize=6 if style == "academic" else 10)
    ax.set_ylabel("Mean coverage score")
    ax.set_title("C  Clinical Coverage")
    ax.set_ylim(0, max(max(means_med), max(means_gem)) + 0.12)
    ax.legend(fontsize=6 if style == "academic" else 10, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        f"Sub-task 3: Admission Summary — MedGemma-4B vs Gemma-3-4B (n={n_pairs})",
        fontsize=9 if style == "academic" else 13,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir / "fig6_summary_task", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for MedGemma vs Gemma-3 study."
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("figures"),
        help="Root output directory (default: figures/).",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/processed"),
        help="Directory containing eval and inference JSONL files.",
    )
    args = parser.parse_args(argv)

    print("Loading data …")
    data = _load_data(args.data_dir)

    for style in ("academic", "presentation"):
        out_dir = args.figures_dir / style
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {style} figures → {out_dir}/")

        _fig0_pipeline(out_dir, style)
        print("  fig0_pipeline       done")

        _fig1_distributions(data, out_dir, style)
        print(f"  fig1_distributions  done")

        _fig2_rouge_diff(data, out_dir, style)
        print(f"  fig2_rouge_diff     done")

        _fig3_vital_heatmap(data, out_dir, style)
        print(f"  fig3_vital_heatmap  done")

        _fig4_abnormal_recall(data, out_dir, style)
        print(f"  fig4_abnormal_recall done")

        _fig5_summary_bars(data, out_dir, style)
        print(f"  fig5_summary_bars   done")

        _fig6_summary_task(data, out_dir, style)
        print(f"  fig6_summary_task   done")

    print("All figures written.")


if __name__ == "__main__":
    main()
