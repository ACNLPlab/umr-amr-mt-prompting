"""
This program is to make a two-panel heatmap comparing error rates across models and methods, grouping languages by Latin vs non-Latin script.

Rows = models
Cols = prompting methods 
Cells = mean error rate at CONFIG_SHOT-shot across the languages in that script group.

Configurable for any (error type, shot count) pair via CONFIG_ERROR and CONFIG_SHOT below.


Reads:  {CONFIG_ERROR}_errors_all_models.json
Writes: {CONFIG_ERROR}_errors_{CONFIG_SHOT}_heatmap.png in --out-dir
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
CONFIG_LINEUP = "all" # one of: "all", "noMistral" sine mistral results are substantially worse than the others and make the heatmap harder to read

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

NON_LATIN = {"mandarin", "hindi", "japanese", "ukrainian"}

MODEL_ORDER_ALL = ["aya", "gemma_4b", "gemma_12b", "gemma_27b", "llama", "mistral", "qwen"]
MODEL_ORDER_NO_MISTRAL = [m for m in MODEL_ORDER_ALL if m != "mistral"]
LINEUP_TO_MODELS = {
    "all": MODEL_ORDER_ALL,
    "noMistral": MODEL_ORDER_NO_MISTRAL,
}
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


def build_matrix(data: dict, languages: set[str]) -> pd.DataFrame:
    n_rows = len(MODEL_ORDER)
    n_cols = len(COLUMNS)
    M = np.full((n_rows, n_cols), np.nan)

    for i, model in enumerate(MODEL_ORDER):
        for j, (_cat, _var, method, graph) in enumerate(COLUMNS):
            try:
                lang_block = data[model][method][graph]
            except KeyError:
                continue
            pcts = []
            for lang, shots in lang_block.items():
                if lang not in languages:
                    continue
                cell = shots.get(CONFIG_SHOT)
                if cell is None or cell.get("pct") is None:
                    continue
                pcts.append(cell["pct"])
            if pcts:
                M[i, j] = float(np.mean(pcts))

    df = pd.DataFrame(
        M,
        index=[MODEL_DISPLAY[m] for m in MODEL_ORDER],
        columns=[GRAPH_DISPLAY[g] for (_c, _v, _m, g) in COLUMNS],
    )
    return df


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
        # Narrow groups (1 column) need a smaller font to avoid overlap.
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

    # Panel title sits above both bands.
    mid_panel = n_cols / 2.0
    ax.text(mid_panel, y_panel_title, panel_title,
            transform=trans, ha="center", va="bottom",
            fontsize=22, fontweight="normal", clip_on=False)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=here / f"{CONFIG_ERROR}_errors_all_models.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "errorHeat",
    )
    args = parser.parse_args()

    if not args.json.exists():
        parser.error(f"input JSON not found: {args.json}")

    out_dir = args.out_dir / CONFIG_LINEUP
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.json.read_text())

    all_langs = set()
    for model in data.values():
        for method in model.values():
            for graph in method.values():
                all_langs.update(graph.keys())
    latin = all_langs - NON_LATIN
    non_latin = all_langs & NON_LATIN

    df_latin = build_matrix(data, latin)
    df_nl = build_matrix(data, non_latin)

    vmax = float(np.nanmax([df_latin.values, df_nl.values]))
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(32, 13))

    for ax, df_panel, panel_title in [
        (axes[0], df_latin, f"Latin-script languages (n={len(latin)})"),
        (axes[1], df_nl,    f"Non-Latin-script languages (n={len(non_latin)})"),
    ]:
        sns.heatmap(
            df_panel,
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
        ax.set_ylabel("Model", fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)

        _draw_group_headers(ax, COLUMNS, panel_title)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.subplots_adjust(wspace=0.25)

    out_png = out_dir / f"{CONFIG_ERROR}_errors_{CONFIG_SHOT}shot_heatmap.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.4)
    print(f"Saved {out_png}")
    print(f"  CONFIG_ERROR = {CONFIG_ERROR!r}, CONFIG_SHOT = {CONFIG_SHOT!r}")
    print(f"  vmax used: {vmax:.1f}")
    plt.close()


if __name__ == "__main__":
    main()