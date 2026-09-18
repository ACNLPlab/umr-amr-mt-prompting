import re
import sys
import os
import warnings

# Set MeCab dictionary path (used for Japanese BLEU) BEFORE any other imports that might use MeCab
try:
    import unidic_lite
    os.environ['MECABRC'] = os.path.join(unidic_lite.DICDIR, 'mecabrc')
    os.environ['MECAB_PATH'] = unidic_lite.DICDIR
except ImportError:
    pass  # falls back to default tokenization for Japanese (see bleu_scores_per_sentence)

import evaluate
import argparse
import json
import csv
from datetime import datetime
from evaluate import load
from pathlib import Path
from scipy import stats
from itertools import combinations
from tqdm import tqdm

RESULTS_ROOT = Path(__file__).resolve().parent  # this folder: predictions in, CSVs out
LPP_ROOT = RESULTS_ROOT.parent / 'lpp'

warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

def load_texts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip().splitlines()

def bleu_score(ref_file, pred_file, lang):
    bleu = evaluate.load("sacrebleu")

    references = load_texts(ref_file)
    predictions = load_texts(pred_file)
    
    try:
        if lang == "zh":
            results = bleu.compute(predictions=predictions, references=references, tokenize=lang)
        elif lang == "ja":
            results = bleu.compute(predictions=predictions, references=references, tokenize="ja-mecab")
        else:
            results = bleu.compute(predictions=predictions, references=references)
    except RuntimeError as e:
        # If MeCab fails for Japanese, fall back to no tokenization
        if lang == "ja" and "MeCab" in str(e):
            print(f"Warning: MeCab tokenizer failed, falling back to default tokenization")
            results = bleu.compute(predictions=predictions, references=references)
        else:
            raise
    
    # Return scores for each sentence to enable paired t-tests
    return results["score"] / 100

def bleu_scores_per_sentence(ref_file, pred_file, lang):
    """Compute BLEU score for each sentence individually for paired t-tests"""
    bleu = evaluate.load("sacrebleu")
    
    references = load_texts(ref_file)
    predictions = load_texts(pred_file)
    
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            if lang == "zh":
                result = bleu.compute(predictions=[pred], references=[ref], tokenize=lang)
            elif lang == "ja":
                result = bleu.compute(predictions=[pred], references=[ref], tokenize="ja-mecab")
            else:
                result = bleu.compute(predictions=[pred], references=[ref])
            scores.append(result["score"] / 100)
        except RuntimeError as e:
            # If MeCab fails for Japanese, fall back to no tokenization
            if lang == "ja" and "MeCab" in str(e):
                print(f"Warning: MeCab tokenizer failed, falling back to default tokenization")
                result = bleu.compute(predictions=[pred], references=[ref])
                scores.append(result["score"] / 100)
            else:
                raise
    
    return scores

def all_bleu_scores(src_file, ref_file, pred_files, lang):
    """Compute BLEU scores for all prediction files"""
    scores = []
    file_names = [p.name for p in pred_files]
    
    for pred_file in tqdm(pred_files, desc="Computing BLEU scores", unit="file"):
        bleu_scores = bleu_scores_per_sentence(ref_file, pred_file, lang)
        scores.append(bleu_scores)
    
    return scores, file_names

def simplify_filename(filename):
    # Remove .txt extension
    name = filename.replace('.txt', '')
    
    # Find _shot_ and take everything after it
    if '_shot_' in name:
        return name.split('_shot_', 1)[1]
    
    return name

def ttest_bleu(bleu_scores, file_names, file_pairs):
    """Perform paired t-tests on BLEU scores"""
    print(f"\nBLEU - Pairwise T-Tests:")
    results = []
    
    for i, j in file_pairs:
        file1 = file_names[i]
        file2 = file_names[j]
        
        t_stat, p_value = stats.ttest_rel(
            bleu_scores[i], 
            bleu_scores[j]
        )
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        result = {
            'file1': file1,
            'file2': file2,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance
        }
        
        results.append(result)
    
    # Print results
    print(f"{'File 1':<40} {'File 2':<40} {'t-statistic':<15} {'p-value':<15}")
    for result in results:
        file1_simple = simplify_filename(result['file1'])
        file2_simple = simplify_filename(result['file2'])
        p_value_str = f"{result['p_value']:.4f}{result['significance']}"
        print(f"{file1_simple:<40} {file2_simple:<40} {result['t_statistic']:<15.4f} {p_value_str:<15}")
    
    return results

def print_average_bleu_scores(bleu_scores, file_names):
    """Print average BLEU scores for each file"""
    print(f"\n{'File':<40} {'BLEU':<15}")
    
    for file_name, scores in zip(file_names, bleu_scores):
        avg_bleu = sum(scores) / len(scores)
        print(f"{file_name:<40} {avg_bleu:<15.4f}")

def save_bleu_ttest_results_to_csv(all_ttest_results, model, language, output_dir=str(RESULTS_ROOT)):
    """Save BLEU t-test results to CSV file"""
    csv_file = os.path.join(output_dir, 'bleu_ttest_results.csv')
    csv_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['model', 'language', 'shot_type', 'file1', 'file2', 't_statistic', 'p_value', 'significance', 'timestamp'])
        
        timestamp_str = datetime.now().isoformat()
        
        # Write data rows
        for shot_type, test_results in all_ttest_results.items():
            for result in test_results:
                writer.writerow([
                    model,
                    language,
                    shot_type,
                    result['file1'],
                    result['file2'],
                    f"{result['t_statistic']:.4f}",
                    f"{result['p_value']:.6f}",
                    result['significance'],
                    timestamp_str
                ])

def save_bleu_average_results_to_csv(model, language, bleu_scores, file_names, output_dir=str(RESULTS_ROOT)):
    """Save average BLEU results to CSV file"""
    csv_file = os.path.join(output_dir, 'bleu_average_results.csv')
    csv_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['model', 'language', 'file', 'bleu', 'timestamp'])
        
        timestamp_str = datetime.now().isoformat()
        
        # Write data rows
        for file_name, scores in zip(file_names, bleu_scores):
            avg_bleu = sum(scores) / len(scores)
            writer.writerow([
                model,
                language,
                file_name,
                f"{avg_bleu:.4f}",
                timestamp_str
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BLEU score statistical significance tests")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--language', type=str, default='japanese', help='Language')
    
    args = parser.parse_args()
    
    # Map model folder names to file prefixes (e.g., gemma_4b folder -> gemma4b prefix)
    file_prefix_map = {
        'gemma_4b': 'gemma4b',
        'gemma_12b': 'gemma12b',
        'gemma_27b': 'gemma27b'
    }
    file_prefix = file_prefix_map.get(args.model, args.model)
    
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
    
    # Filter to only srllm_direct and direct files (0_shot and 5_shot only)
    all_files = sorted([p for p in results.glob('*.txt')])
    pred_files = [f for f in all_files if (('_srllm_direct_amr.txt' in f.name or '_srllm_direct_umr.txt' in f.name or 
                                            ('_direct.txt' in f.name and '_direct_amr' not in f.name and '_direct_umr' not in f.name and '_srllm' not in f.name)) and
                                           ('0_shot' in f.name or '5_shot' in f.name))]
    
    if not pred_files:
        print(f"Error: Could not find srllm_direct or direct files in {results}")
        print(f"Available files: {[f.name for f in all_files[:5]]}")
        sys.exit(1)

    # Compute BLEU scores for selected files
    bleu_scores, file_names = all_bleu_scores(src_file, ref_file, pred_files, codes[language])
    
    # Print average scores
    print_average_bleu_scores(bleu_scores, file_names)

    # Perform t-tests comparing srllm_direct variants to direct baseline
    all_ttest_results = {}
    
    # Group files by shot type
    shot_types = {}
    for i, name in enumerate(file_names):
        # Extract shot type (0_shot, 5_shot only)
        shot_match = re.search(r'(0_shot|5_shot)', name)
        if shot_match:
            shot_type = shot_match.group(1)
            if shot_type not in shot_types:
                shot_types[shot_type] = {'srllm_amr': [], 'srllm_umr': [], 'direct': []}
            
            if '_srllm_direct_amr.txt' in name:
                shot_types[shot_type]['srllm_amr'].append(i)
            elif '_srllm_direct_umr.txt' in name:
                shot_types[shot_type]['srllm_umr'].append(i)
            elif '_direct.txt' in name and '_srllm' not in name:
                shot_types[shot_type]['direct'].append(i)
    
    # Create pairs comparing srllm variants vs direct baseline for each shot type
    for shot_type, files in shot_types.items():
        if files['direct']:
            print()
            print(f"T-Test Results: {shot_type.upper()} SRLLM_DIRECT variants vs DIRECT")
            pairs = []
            
            # Compare srllm_amr against direct
            for amr_idx in files['srllm_amr']:
                for direct_idx in files['direct']:
                    pairs.append((amr_idx, direct_idx))
            
            # Compare srllm_umr against direct
            for umr_idx in files['srllm_umr']:
                for direct_idx in files['direct']:
                    pairs.append((umr_idx, direct_idx))
            
            if pairs:
                shot_results = ttest_bleu(bleu_scores, file_names, file_pairs=pairs)
                all_ttest_results[shot_type] = shot_results
    
    # Save results to CSV files
    if all_ttest_results:
        save_bleu_ttest_results_to_csv(all_ttest_results, args.model, language.lower())