# Le Petit Prince (LPP) Data

This folder contains the full data for Chinese and Portuguese, as examples of what the data should look like for anyone who would like to replicate our experiments. We extracted them from the publicly available Chinese and Portuguese AMR data.

## Contents

```
lpp/
├── all_eng_amrs.txt
├── all_eng_sent.txt
├── all_eng_umrs.txt
├── chin_experiment_data/
│   ├── eng_amrs.txt
│   ├── eng_umrs.txt
│   ├── eng_sent.txt
│   └── chi_sent.txt
└── port_experiment_data/
    ├── eng_amrs.txt
    ├── eng_umrs.txt
    ├── eng_sent.txt
    └── por_sent.txt
```

- `all_eng_*.txt`: English sentences with their AMRs and UMRs, before selecting the subset available for each language.
- `chin_experiment_data/`, `port_experiment_data/`: the data used in the experiments for Chinese and Portuguese.

## Data format for replicating our experiments

The data of each language should be in a folder named `[first 4 letters of the language]_experiment_data` (e.g. `chin_experiment_data`, `port_experiment_data`). Each folder must contain:

| File | Description |
|---|---|
| `eng_amrs.txt` | English AMRs |
| `eng_umrs.txt` | English UMRs |
| `eng_sent.txt` | English sentences |
| `[first 3 letters of the language]_sent.txt` | Sentences in the target language (e.g. `chi_sent.txt`, `por_sent.txt`) |

The four data files are parallel: the *n*-th entry in each of them corresponds to the same sentence.

The experiment scripts also read a `five_shot.json` file from this folder, which stores the indices of the shots to use for each sentence so they do not have to be selected again every time an experiment is run. It is not included: generate it with `collect_shot.py` (see below) before running the experiments. It is a list of lists, one list of five indices for each sentence to be translated.

## Collecting the shots

`collect_shot.py` generates `five_shot.json` for a language. For each English sentence, it computes the chrF score against every English sentence in `eng_sent.txt` and keeps the five most similar ones (excluding the top match, which is assumed to be the sentence itself) as its shots.

```bash
pip install evaluate sacrebleu tqdm
python collect_shot.py --lang chin   # reads chin_experiment_data/, writes chin_experiment_data/five_shot.json
python collect_shot.py --lang port
```

`--lang` is the first 4 letters of the language, i.e. the name of its `[...]_experiment_data` folder.

## Validating the alignment

Before running the experiments, we ran `validate.py` as a further check that each target-language sentence is a one-to-one translation of its English sentence. For every line, it embeds both sentences with `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers) and computes their cosine similarity. Lines with a similarity below 0.5 are reported as potential misalignments.

```bash
pip install sentence-transformers
python validate.py Chinese   # or the first 4 letters of the language, e.g. chin
```

We used this report to manually identify and remove the sentences that were indeed not one-to-one with the English sentence. This is how we arrived at the final size of each dataset.
