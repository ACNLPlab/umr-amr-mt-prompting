import re
import sys
import warnings
import evaluate
import argparse
import json
import csv
import os
from datetime import datetime
from evaluate import load
from pathlib import Path
from comet import download_model, load_from_checkpoint
from contextlib import redirect_stdout, redirect_stderr
from scipy import stats
from itertools import combinations
from tqdm import tqdm

RESULTS_ROOT = Path(__file__).resolve().parent  # this folder: predictions in, CSVs out
LPP_ROOT = RESULTS_ROOT.parent / 'lpp'

warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

def load_texts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip().splitlines()

def bertscore(ref_file, pred_file, lang):
    bertscore = load('bertscore')

    references = load_texts(ref_file)
    predictions = load_texts(pred_file)

    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang,
        model_type='xlm-roberta-base' 
    )

    return results['f1']

def bleu_score(ref_file, pred_file, lang):
    bleu = evaluate.load("sacrebleu")

    references = load_texts(ref_file)
    predictions = load_texts(pred_file)
    
    if lang == "zh":
        results = bleu.compute(predictions=predictions, references=references, tokenize=lang)
    elif lang == "ja":
        results = bleu.compute(predictions=predictions, references=references, tokenize="ja-mecab")
    else:
        results = bleu.compute(predictions=predictions, references=references)
    return results["score"] / 100

def chrf_score(ref_file, pred_file, lang):
    references = load_texts(ref_file)
    predictions = load_texts(pred_file)
    
    chrf = load('chrf')

    scores = []
    for pred, ref in zip(predictions, references):
        result = chrf.compute(predictions=[pred], references=[ref])
        scores.append(result["score"])
    return scores

def comet_score_batch(src_file, ref_file, mt_files, lang, model=None):
    if model is None:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
    
    with open(src_file, 'r', encoding='utf-8') as f:
        sources = f.read().strip().split('\n')
    
    references = load_texts(ref_file)
    all_results = []
    
    for mt_file in mt_files:
        translations = load_texts(mt_file)
        
        data = [
            {
                "src": src,
                "mt": mt,
                "ref": ref
            }
            for src, mt, ref in zip(sources, translations, references)
        ]
        
        results = model.predict(data, batch_size=8, gpus=1)
        all_results.append(results["scores"])
    
    return all_results, model

def all_scores(src_file, ref_file, pred_files, lang):
    scores = {
        'bertscore': [],
        'bleu': [],
        'chrf': [],
        'comet': []
    }

    file_names = [p.name for p in pred_files]
    
    # Evaluate COMET in batch
    comet_scores, _ = comet_score_batch(src_file, ref_file, pred_files, lang)
    
    for i, pred_file in enumerate(tqdm(pred_files, desc="Processing files", unit="file")):
        scores['bertscore'].append(bertscore(ref_file, pred_file, lang))
        scores['bleu'].append(bleu_score(ref_file, pred_file, lang))
        scores['chrf'].append(chrf_score(ref_file, pred_file, lang))
        scores['comet'].append(comet_scores[i])
    
    return scores, file_names

def simplify_filename(filename):
    # Remove .txt extension
    name = filename.replace('.txt', '')
    
    # Find _shot_ and take everything after it
    if '_shot_' in name:
        return name.split('_shot_', 1)[1]
    
    return name

def ttest(scores, file_names, file_pairs):
    score_types = ['bertscore', 'chrf', 'comet']
    results = {}
    
    for score_type in score_types:
        print(f"\n{score_type.upper()} - Pairwise T-Tests:")
        results[score_type] = []
        
        score_list = scores[score_type]
        
        for i, j in file_pairs:
            file1 = file_names[i]
            file2 = file_names[j]
            
            # Keep only pairs involving the _direct.txt baseline
            if '_direct.txt' not in file1 and '_direct.txt' not in file2:
                continue
            
            # Put the baseline second (after the other variants)
            if '_direct.txt' in file1 and '_direct.txt' not in file2:
                file1, file2 = file2, file1
                i, j = j, i
            
            t_stat, p_value = stats.ttest_rel(
                score_list[i], 
                score_list[j]
            )
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            results[score_type].append({
                'file1': file1,
                'file2': file2,
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': significance
            })
        
        if results[score_type]:
            print(f"{'Everything vs Baseline':<40} {'t-statistic':<15} {'p-value':<15}")
            for result in results[score_type]:
                file1_simple = simplify_filename(result['file1'])
                file2_simple = simplify_filename(result['file2'])
                comparison = f"{file1_simple} vs {file2_simple}"
                p_value_str = f"{result['p_value']:.4f}{result['significance']}"
                print(f"{comparison:<40} {result['t_statistic']:<15.4f} {p_value_str:<15}")
        
        print()
    
    return results

def print_average_scores(scores, file_names):
    score_types = ['bleu', 'bertscore', 'chrf', 'comet']
    
    # Calculate averages for non-bleu metrics
    avg_scores = {
        'bleu': scores['bleu'],
        'bertscore': [sum(vals) / len(vals) for vals in scores['bertscore']],
        'chrf': [sum(vals) / len(vals) for vals in scores['chrf']],
        'comet': [sum(vals) / len(vals) for vals in scores['comet']]
    }
    
    # Print header
    print(f"\n{'File':<40} {'BLEU':<15} {'BERTscore':<15} {'chrF++':<15} {'COMET':<15}")
    
    # Print each file's scores
    for file_name, bleu, bert, chrf, comet in zip(
        file_names,
        avg_scores['bleu'],
        avg_scores['bertscore'],
        avg_scores['chrf'],
        avg_scores['comet']
    ):
        print(f"{file_name:<40} {bleu:<15.4f} {bert:<15.4f} {chrf:<15.4f} {comet:<15.4f}")
    
    return avg_scores

def save_average_results_to_csv(model, language, scores, file_names, output_dir=str(RESULTS_ROOT)):
    # Calculate average scores
    avg_scores = {
        'bleu': scores['bleu'],
        'bertscore': [sum(vals) / len(vals) for vals in scores['bertscore']],
        'chrf': [sum(vals) / len(vals) for vals in scores['chrf']],
        'comet': [sum(vals) / len(vals) for vals in scores['comet']]
    }
    
    # Create per-file results
    file_results = []
    for i, file_name in enumerate(file_names):
        file_results.append({
            'file': file_name,
            'bleu': avg_scores['bleu'][i],
            'bertscore': avg_scores['bertscore'][i],
            'chrf': avg_scores['chrf'][i],
            'comet': avg_scores['comet'][i]
        })
    
    timestamp_str = datetime.now().isoformat()
    
    # Save to CSV file
    csv_file = os.path.join(output_dir, 'average_results.csv')
    csv_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['model', 'language', 'file', 'bleu', 'bertscore', 'chrf', 'comet', 'timestamp'])
        
        # Write data rows
        for result in file_results:
            writer.writerow([
                model,
                language,
                result['file'],
                f"{result['bleu']:.4f}",
                f"{result['bertscore']:.4f}",
                f"{result['chrf']:.4f}",
                f"{result['comet']:.4f}",
                timestamp_str
            ])

def save_ttest_results_to_csv(all_ttest_results, model, language, output_dir=str(RESULTS_ROOT)):
    csv_file = os.path.join(output_dir, 'ttest_results.csv')
    csv_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['model', 'language', 'shot_type', 'score_type', 'file1', 'file2', 't_statistic', 'p_value', 'significance', 'timestamp'])
        
        timestamp_str = datetime.now().isoformat()
        
        # Write data rows
        for shot_type, score_results in all_ttest_results.items():
            for score_type, test_results in score_results.items():
                for result in test_results:
                    writer.writerow([
                        model,
                        language,
                        shot_type,
                        score_type,
                        result['file1'],
                        result['file2'],
                        f"{result['t_statistic']:.4f}",
                        f"{result['p_value']:.6f}",
                        result['significance'],
                        timestamp_str
                    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    
    args = parser.parse_args()
    language = "Upper Sorbian" if args.language == "upper sorbian" else args.language.capitalize()
    if language == "Mandarin":
        prefix = "chin"
    else:
        prefix = language.lower()[:4]
    base_path = str(LPP_ROOT / f'{prefix}_experiment_data')

    codes = {
        'Mandarin': 'zh',
        'Japanese': 'ja',
        'Hindi': 'hi',
        'German': 'de',
        'Czech': 'cs',
        'Dutch': 'nl',
        'Portuguese': 'pt',
        'Ukrainian': 'uk',
        'Catalan': 'ca',
        'Latvian': 'lv',
        'Belarusian': 'be',
        'Upper Sorbian': 'hs'
    }

    src_file = f'{base_path}/eng_sent.txt'
    ref_file = f'{base_path}/{prefix[:3]}_sent.txt'
    results = RESULTS_ROOT / args.model / language.lower()
    pred_files = sorted([p for p in results.glob('*.txt')])

    scores, file_names = all_scores(src_file, ref_file, pred_files, codes[language])
    print_average_scores(scores, file_names)

    shot_types = ['0_shot', '1_shot', '3_shot', '5_shot']
    all_ttest_results = {}
    
    for shot_idx, shot_type in enumerate(shot_types, 1):
        # Find indices of files matching current shot type
        shot_indices = [i for i, name in enumerate(file_names) if shot_type in name]
        
        # Create all pairs within this shot type
        shot_pairs = list(combinations(shot_indices, 2))
        
        print()
        print(f"T-Test Results for {shot_type.upper()}:")
        shot_results = ttest(scores, file_names, file_pairs=shot_pairs)
        all_ttest_results[shot_type] = shot_results
    
    # Save results to CSV file
    save_average_results_to_csv(args.model, language.lower(), scores, file_names)
    save_ttest_results_to_csv(all_ttest_results, args.model, language.lower())