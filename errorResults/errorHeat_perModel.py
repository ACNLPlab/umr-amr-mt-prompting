"""
This program is to make a per-model heatmap, which calculates error rate at a given shot count and outputs one PNG per model.

Rows = languages
Cols = prompting methods 
Cells = error rate at CONFIG_SHOT-shot for the given (language, method, graph).

Configurable for any (error type, shot count) pair via CONFIG_ERROR and CONFIG_SHOT below.

Reads:  {CONFIG_ERROR}_errors_all_models.json
Writes: one PNG per model under
        {--out-dir}/{CONFIG_LINEUP}/perModel/
        named {CONFIG_ERROR}_errors_{CONFIG_SHOT}shot_{model}_heatmap.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns


# ---------------------------------------------------------------------------
# Toggles 
# Error types: graph = graph-output errors; explanation = parenthetical explanation errors; romanization = parenthetical romanization errors; slash = slash variant errors
# ---------------------------------------------------------------------------
CONFIG_ERROR = "graph" # one of: "graph", "explanation", "romanization", "slash"
CONFIG_SHOT  = "0" # one of: "0", "1", "3", "5"
CONFIG_LINEUP = "all" # one of: "all", "noMistral"

ERROR_DISPLAY = {
    "graph":        ("Graph-output",          "graph-output"),
    "explanation":  ("Parenthetical-explanation", "parenthetical-explanation"),
    "romanization": ("Parenthetical-romanization", "parenthetical-romanization"),
    "slash":        ("Slash-variant",         "slash-variant"),
}
ERROR_TITLE, ERROR_LABEL = ERROR_DISPLAY[CONFIG_ERROR]

COUNT_KEY = {
    "graph": "amr_umr_graph",
    "explanation": "parenthetical_explanation",
    "romanization": "parenthetical_romanization",
    "slash": "slash_variant",
}[CONFIG_ERROR]


LATIN_LANGS = ["catalan", "czech", "dutch", "german", "latvian", "portuguese"]
NON_LATIN_LANGS = ["hindi", "japanese", "mandarin", "ukrainian"]
LANGUAGE_ORDER = LATIN_LANGS + NON_LATIN_LANGS

LANGUAGE_DISPLAY = {
    "catalan": "Catalan",
    "czech": "Czech",
    "dutch": "Dutch",
    "german": "German",
    "latvian": "Latvian",
    "portuguese": "Portuguese",
    "hindi": "Hindi",
    "japanese": "Japanese",
    "mandarin": "Mandarin",
    "ukrainian": "Ukrainian",
}

MODEL_ORDER_ALL = ["aya", "gemma_4b", "gemma_12b", "gemma_27b", "llama", "mistral", "qwen"]
MODEL_ORDER_NO_MISTRAL = [m for m in MODEL_ORDER_ALL if m != "mistral"]

LINEUP_TO_MODELS = {
    "all": MODEL_ORDER_ALL,
    "noMistral": MODEL_ORDER_NO_MISTRAL,
}
if CONFIG_LINEUP not in LINEUP_TO_MODELS:
    raise ValueError(
        f"CONFIG_LINEUP={CONFIG_LINEUP!r} not recognized; "
        f"expected one of {list(LINEUP_TO_MODELS)}"
    )
MODEL_ORDER = LINEUP_TO_MODELS[CONFIG_LINEUP]

MODEL_DISPLAY = {
    "aya": "Aya-23-8B",
    "gemma_4b": "Gemma-3-4B-IT",
    "gemma_12b": "Gemma-3-12B-IT",
    "gemma_27b": "Gemma-3-27B-IT",
    "llama": "Llama-3.1-8B-Instruct",
    "mistral": "Mistral-7B-Instruct",
    "qwen": "Qwen2.5-7B-Instruct",
}


# mapping the method names used in the experiment to the method names we want to display in the heatmap

COLUMNS = [
    ("No SR",        "Baseline", "direct",            "none"),
    ("No SR",        "SENSE",    "sense",             "none"),
    ("SR-Direct",    "Graph",    "direct",            "amr"),
    ("SR-Direct",    "Graph",    "direct",            "umr"),
    ("SR-Direct",    "NLD",      "srllm_direct",      "amr"),
    ("SR-Direct",    "NLD",      "srllm_direct",      "umr"),
    ("SR-Grounded",  "Graph",    "instruct_grounded", "amr"),
    ("SR-Grounded",  "Graph",    "instruct_grounded", "umr"),
    ("SR-Grounded",  "NLD",      "srllm_grounded",    "amr"),
    ("SR-Grounded",  "NLD",      "srllm_grounded",    "umr"),
    ("SR-Assisted",  "Graph",    "instruct_assisted", "amr"),
    ("SR-Assisted",  "Graph",    "instruct_assisted", "umr"),
    ("SR-Assisted",  "NLD",      "srllm_assisted",    "amr"),
    ("SR-Assisted",  "NLD",      "srllm_assisted",    "umr"),
]

GRAPH_DISPLAY = {"none": "—", "amr": "AMR", "umr": "UMR"}


def build_matrix(data: dict, model: str, languages: list[str]) -> pd.DataFrame:
    """Build a (languages x COLUMNS) DataFrame of error rate at CONFIG_SHOT
    for a single model. Cells with no data become NaN."""
    n_rows = len(languages)
    n_cols = len(COLUMNS)
    M = np.full((n_rows, n_cols), np.nan)

    for i, language in enumerate(languages):
        for j, (_cat, _var, method, graph) in enumerate(COLUMNS):
            try:
                shots = data[model][method][graph][language]
            except KeyError:
                continue
            cell = shots.get(CONFIG_SHOT)
            if cell is None or cell.get("pct") is None:
                continue
            M[i, j] = float(cell["pct"])

    df = pd.DataFrame(
        M,
        index=[LANGUAGE_DISPLAY.get(l, l) for l in languages],
        columns=[GRAPH_DISPLAY[g] for (_c, _v, _m, g) in COLUMNS],
    )
    return df


def discover_languages_in_order(data: dict) -> list[str]:
    """Use LANGUAGE_ORDER but filter to languages actually present in the JSON.
    Any language found in the data but missing from LANGUAGE_ORDER is appended
    at the end so nothing gets silently dropped."""
    present = set()
    for model in data.values():
        for method in model.values():
            for graph in method.values():
                present.update(graph.keys())
    ordered = [l for l in LANGUAGE_ORDER if l in present]
    extras = sorted(present - set(LANGUAGE_ORDER))
    return ordered + extras


def _draw_group_headers(ax, columns: list[tuple], panel_title: str) -> None:
    n_cols = len(columns)

    def spans(key_fn):
        out = []
        start = 0
        cur = key_fn(columns[0])
        for i in range(1, n_cols):
            k = key_fn(columns[i])
            if k != cur:
                out.append((cur, start, i - 1))
                start = i
                cur = k
        out.append((cur, start, n_cols - 1))
        return out

    variant_spans  = spans(lambda c: (c[0], c[1]))
    category_spans = spans(lambda c: c[0])

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    y_variant_text     = 1.04
    y_variant_bracket  = 1.02
    y_category_text    = 1.11
    y_category_bracket = 1.09
    y_panel_title      = 1.20

    for (_cat, var), lo, hi in variant_spans:
        mid = (lo + hi + 1) / 2.0
        span_width = hi - lo + 1

        var_fs = 11 if span_width == 1 else 14
        ax.text(mid, y_variant_text, var,
                transform=trans, ha="center", va="bottom",
                fontsize=var_fs, fontweight="normal", clip_on=False)
        ax.plot([lo + 0.1, hi + 0.9], [y_variant_bracket, y_variant_bracket],
                transform=trans, color="black", lw=1.0, clip_on=False)

    for cat, lo, hi in category_spans:
        mid = (lo + hi + 1) / 2.0
        ax.text(mid, y_category_text, cat,
                transform=trans, ha="center", va="bottom",
                fontsize=16, fontweight="normal", clip_on=False)
        ax.plot([lo + 0.1, hi + 0.9], [y_category_bracket, y_category_bracket],
                transform=trans, color="black", lw=1.2, clip_on=False)

    mid_panel = n_cols / 2.0
    ax.text(mid_panel, y_panel_title, panel_title,
            transform=trans, ha="center", va="bottom",
            fontsize=22, fontweight="bold", clip_on=False)


def plot_one_model(df: pd.DataFrame, model: str, out_path: Path) -> None:
    """Render and save a single-model heatmap with per-model autoscaled vmax."""
    if df.dropna(how="all").empty:
        print(f"  [skip] {model}: no data")
        return

    vmax = float(np.nanmax(df.values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vmin = 0.0

    fig_h = max(6, 0.7 * len(df.index) + 3)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": f"{ERROR_TITLE} error rate (%)"},
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("")
    ax.set_ylabel("Language", fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    panel_title = (
        f"{MODEL_DISPLAY[model]} — {ERROR_TITLE} error rate at {CONFIG_SHOT}-shot"
    )
    _draw_group_headers(ax, COLUMNS, panel_title)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"  saved {out_path}  (vmax={vmax:.1f})")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=here / f"{CONFIG_ERROR}_errors_all_models.json",
        help=f"Path to {CONFIG_ERROR}_errors_all_models.json "
             f"(default: alongside this script)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "errorHeat",
        help="Base directory for output (default: ./errorHeat next to this script). "
             "Per-model PNGs go into {out-dir}/{CONFIG_LINEUP}/perModel/",
    )
    args = parser.parse_args()

    if not args.json.exists():
        parser.error(f"input JSON not found: {args.json}")

    out_dir = args.out_dir / CONFIG_LINEUP / "perModel"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.json.read_text())
    languages = discover_languages_in_order(data)

    print(f"CONFIG_ERROR={CONFIG_ERROR!r}  CONFIG_SHOT={CONFIG_SHOT!r}  "
          f"CONFIG_LINEUP={CONFIG_LINEUP!r}")
    print(f"Languages (in row order): {languages}")
    print(f"Models to render: {MODEL_ORDER}")

    for model in MODEL_ORDER:
        if model not in data:
            print(f"  [skip] {model}: not in JSON")
            continue
        df = build_matrix(data, model, languages)
        out_png = out_dir / f"{CONFIG_ERROR}_errors_{CONFIG_SHOT}shot_{model}_heatmap.png"
        plot_one_model(df, model, out_png)


if __name__ == "__main__":
    main()