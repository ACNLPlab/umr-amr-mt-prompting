import os
import json
import argparse
from tqdm import tqdm
import evaluate

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

##### DATA PRE-PROCESSING #####
def load_lpp_data(lang):
    """Load the English and target-language sentences of `[lang]_experiment_data`."""
    data_dir = os.path.join(DATA_ROOT, f"{lang}_experiment_data")
    with open(os.path.join(data_dir, "eng_sent.txt"), "r", encoding="utf-8") as f:
        lpp_eng_lines = f.readlines()
    with open(os.path.join(data_dir, f"{lang[:3]}_sent.txt"), "r", encoding="utf-8") as f:
        lpp_tgt_lines = f.readlines()
    assert len(lpp_eng_lines) == len(lpp_tgt_lines), "eng_sent.txt and target sentence file are not parallel"
    return data_dir, lpp_eng_lines

##### NEAREST NEIGHBORS FOR FIVE SHOT #####
chrf = evaluate.load("chrf")

def get_nearest_indices(n, input_sent, lpp_eng_lines):
    """Indices of the n English sentences most similar (chrF) to input_sent."""
    chrf_scores = []
    for item in lpp_eng_lines:
        results = chrf.compute(predictions=[item], references=[input_sent])
        chrf_scores.append(results["score"])

    # the top-scoring sentence is dropped as it is assumed to be input_sent itself
    indices = [x[0] for x in sorted(enumerate(chrf_scores), key=lambda x: x[1])[-(n+1):]]
    return indices[:-1]

def generate_five_shot_json(lpp_eng_lines, output_file):
    """Generate JSON file with 5 nearest neighbor indices for each sentence"""
    all_indices = []

    for input_sent in tqdm(lpp_eng_lines):
        all_indices.append(get_nearest_indices(5, input_sent, lpp_eng_lines))

    with open(output_file, "w") as f:
        json.dump(all_indices, f)

##### MAIN #####
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="chin",
                        help="first 4 letters of the language, e.g. chin or port (default: chin)")
    args = parser.parse_args()

    data_dir, lpp_eng_lines = load_lpp_data(args.lang)
    generate_five_shot_json(lpp_eng_lines, os.path.join(data_dir, "five_shot.json"))

if __name__ == "__main__":
    main()
