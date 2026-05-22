"""
This program is to extract the prediction for a specific sentence from models with evaluation metric scores

Reads:  {model_dir}/{LANGUAGE}/*.txt
Writes: sentences_line{SENTENCE_LINE}.json in OUTPUT_DIR
"""
import json
import re
import sys
from pathlib import Path

import evaluate
from evaluate import load
from comet import download_model, load_from_checkpoint

SENTENCE_LINE = 43 # 1-indexed line number to extract (e.g., 440 or 816)

LANGUAGE = "japanese"
LANG_CODE = "ja" # for BERTScore

RESULTS_ROOT = Path("/home/common/ACNLP/umr_applications/results")
GOLD_REF_FILE = Path("/home/common/ACNLP/umr_applications/lpp/japa_experiment_data/jap_sent.txt")
OUTPUT_DIR = Path("/home/common/ACNLP/umr_applications/weixin_experiments/case_study")

# ============================================================================
# FILENAME PARSING
# ============================================================================

# Pattern: {model}_{shot}_shot_{method}[_{variant}].txt
# - model: first token
# - shot: second token (digit)
# - "shot": literal third token
# - method: everything after "shot" except trailing amr/umr
# - variant: "amr", "umr", or "base" (when no suffix)

VARIANT_SUFFIXES = {"amr", "umr"}


def parse_filename(filename):
    """
    Parse a prediction filename into (model, shot, method, variant).
    Returns None if the filename doesn't match the expected pattern.
    """
    stem = Path(filename).stem  # strip .txt
    tokens = stem.split("_")

    if len(tokens) < 4:
        return None
    if tokens[2] != "shot":
        return None
    if not tokens[1].isdigit():
        return None

    model = tokens[0]
    shot = int(tokens[1])

    # Check if last token is a variant suffix
    if tokens[-1] in VARIANT_SUFFIXES:
        variant = tokens[-1]
        method_tokens = tokens[3:-1]
    else:
        variant = "base"
        method_tokens = tokens[3:]

    if not method_tokens:
        return None

    method = "_".join(method_tokens)
    return {
        "model": model,
        "shot": shot,
        "method": method,
        "variant": variant,
    }


# ============================================================================
# FILE READING
# ============================================================================

def read_line(filepath, line_num_1indexed):
    """Read a specific 1-indexed line from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    idx = line_num_1indexed - 1
    if idx < 0 or idx >= len(lines):
        raise IndexError(
            f"Line {line_num_1indexed} out of range for {filepath} "
            f"(file has {len(lines)} lines)"
        )
    return lines[idx]


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_metrics(records, gold_reference):
    """
    Given a list of records (each with a 'prediction' field) and a single
    gold reference sentence, compute chrF, BERTScore, and COMET for each
    record. Mutates records in place by adding a 'scores' dict.
    """
    predictions = [r["prediction"] for r in records]
    references = [gold_reference] * len(predictions)

    # --- chrF ---
    print("Computing chrF...", flush=True)
    chrf = load("chrf")
    chrf_scores = []
    for pred, ref in zip(predictions, references):
        result = chrf.compute(predictions=[pred], references=[ref])
        chrf_scores.append(result["score"])

    # --- BERTScore ---
    print("Computing BERTScore...", flush=True)
    bertscore = load("bertscore")
    bs_result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=LANG_CODE,
        model_type="xlm-roberta-base",
    )
    bertscore_f1 = bs_result["f1"]

    # --- COMET ---
    print("Computing COMET...", flush=True)
    eng_ref_file = GOLD_REF_FILE.parent / "eng_sent.txt"
    if eng_ref_file.exists():
        source = read_line(eng_ref_file, SENTENCE_LINE)
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": source, "mt": pred, "ref": gold_reference}
            for pred in predictions
        ]
        comet_result = comet_model.predict(comet_data, batch_size=8, gpus=1)
        comet_scores = comet_result["scores"]
    else:
        print(f"WARNING: {eng_ref_file} not found; skipping COMET.", flush=True)
        comet_scores = [None] * len(predictions)

    # Attach scores to records
    for r, chrf_s, bs_s, comet_s in zip(records, chrf_scores, bertscore_f1, comet_scores):
        r["scores"] = {
            "chrf": float(chrf_s),
            "bertscore": float(bs_s),
            "comet": float(comet_s) if comet_s is not None else None,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not RESULTS_ROOT.exists():
        print(f"ERROR: results root not found: {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)
    if not GOLD_REF_FILE.exists():
        print(f"ERROR: gold reference not found: {GOLD_REF_FILE}", file=sys.stderr)
        sys.exit(1)

    # Read the gold reference for this sentence
    gold_reference = read_line(GOLD_REF_FILE, SENTENCE_LINE)
    print(f"Gold reference (line {SENTENCE_LINE}): {gold_reference}", flush=True)

    # Walk all model directories
    records = []
    skipped = []

    model_dirs = sorted([d for d in RESULTS_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}", flush=True)

    for model_dir in model_dirs:
        lang_dir = model_dir / LANGUAGE
        if not lang_dir.exists():
            print(f"  Skipping {model_dir.name}: no {LANGUAGE}/ subdir", flush=True)
            continue

        pred_files = sorted(lang_dir.glob("*.txt"))
        print(f"  {model_dir.name}/{LANGUAGE}: {len(pred_files)} files", flush=True)

        for pred_file in pred_files:
            parsed = parse_filename(pred_file.name)
            if parsed is None:
                skipped.append(pred_file.name)
                continue

            try:
                prediction = read_line(pred_file, SENTENCE_LINE)
            except IndexError as e:
                print(f"    WARNING: {e}", flush=True)
                skipped.append(pred_file.name)
                continue

            records.append({
                "model": parsed["model"],
                "shot": parsed["shot"],
                "method": parsed["method"],
                "variant": parsed["variant"],
                "filename": pred_file.name,
                "prediction": prediction,
            })

    print(f"\nExtracted {len(records)} records.", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)

    if not records:
        print("No records to score. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    compute_metrics(records, gold_reference)

    # Build output payload
    payload = {
        "sentence_line": SENTENCE_LINE,
        "language": LANGUAGE,
        "lang_code": LANG_CODE,
        "gold_reference": gold_reference,
        "num_records": len(records),
        "records": records,
    }

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"sentences_line{SENTENCE_LINE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {output_file}", flush=True)


if __name__ == "__main__":
    main()