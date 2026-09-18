# Results and Evaluation

This folder should hold the translations produced by the experiments (should you decide to run them). Right now, it only has the scripts used to evaluate the translations.

## Evaluation data

The scripts read the source and reference sentences from `../lpp/<first 4 letters of the language>_experiment_data/` (`eng_sent.txt` and `[first 3 letters]_sent.txt`; Mandarin uses `chin`). See [../lpp/README.md](../lpp/README.md) for the format. Only Chinese and Portuguese data are included in this repository.

## Running the evaluation

Run from this folder. All scripts find their files relative to their own location.

### 1. BLEU, BERTScore, chrF++, COMET: `eval.py`

```bash
pip install evaluate sacrebleu bert_score unbabel-comet scipy tqdm
python eval.py --model gemma_4b --language mandarin
```

For all `*.txt` files of one model and language, it computes:
- **BLEU** (sacreBLEU, corpus level; `zh` and `ja-mecab` tokenizers for Mandarin and Japanese), **BERTScore** (`xlm-roberta-base`), **chrF++** and **COMET** (`Unbabel/wmt22-comet-da`, needs a GPU).
- Two-tailed paired t-tests (BERTScore, chrF++, COMET), separately for each of 0/1/3/5 shots, between every other setting and the `direct` baseline.

### 2. BLEU significance tests: `eval_bleu_ttest.py`

`eval.py` reports BLEU at corpus level, which cannot be used in a paired t-test. This script computes sentence-level BLEU.

```bash
python eval_bleu_ttest.py --model gemma_4b --language mandarin
```

Appends to `bleu_average_results.csv` and `bleu_ttest_results.csv`. Japanese BLEU needs MeCab (`pip install mecab-python3 unidic-lite`); without it, the script falls back to default tokenization.

### 3. MetricX: `metricx_ttest.py`

We only provide the code for the MetricX t-tests (`metricx_ttest.py`). The MetricX scores themselves are calculated by cloning Google's [metricx](https://github.com/google-research/metricx) repository and following the instructions there. It is not part of this repository. We used MetricX-24 hybrid XL (`google/metricx-24-hybrid-xl-v2p6`).