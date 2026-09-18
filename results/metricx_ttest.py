import json
import os
import argparse
import csv
from datetime import datetime
from pathlib import Path
from scipy import stats
from itertools import combinations
from tqdm import tqdm
import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parent  # this folder: predictions in, CSVs out
LPP_ROOT = RESULTS_ROOT.parent / 'lpp'

OUTPUT_DIR = str(RESULTS_ROOT / "metricx_outputs")

def load_metricx_scores(output_file):
    """Load metricx scores from JSONL file"""
    scores = []
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'prediction' in data:
                        scores.append(data['prediction'])
        return scores
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return []

def load_all_metricx_scores(model, language, pred_files):
    """Load metricx scores for all prediction files"""
    scores = []
    file_names = [p.name for p in pred_files]
    
    for pred_file in tqdm(pred_files, desc="Loading metricx scores", unit="file"):
        file_base = pred_file.stem
        output_file = os.path.join(OUTPUT_DIR, model, language, f"{file_base}.jsonl")
        
        if os.path.exists(output_file):
            metricx_scores = load_metricx_scores(output_file)
            scores.append(metricx_scores)
        else:
            print(f"Warning: {output_file} not found")
            scores.append([])
    
    return scores, file_names

def simplify_filename(filename):
    """Simplify filename for display"""
    name = filename.replace('.txt', '')
    
    if '_shot_' in name:
        return name.split('_shot_', 1)[1]
    
    return name

def ttest_metricx(metricx_scores, file_names, file_pairs):
    """Perform paired t-tests on metricx scores"""
    print(f"\nMetricX - Pairwise T-Tests:")
    results = []
    
    for i, j in file_pairs:
        file1 = file_names[i]
        file2 = file_names[j]
        
        # Skip if either file has no scores
        if not metricx_scores[i] or not metricx_scores[j]:
            continue
        
        # Keep only pairs involving the _direct.txt baseline
        if '_direct.txt' not in file1 and '_direct.txt' not in file2:
            continue
        
        # Put the baseline second (after the other variants)
        if '_direct.txt' in file1 and '_direct.txt' not in file2:
            file1, file2 = file2, file1
            i, j = j, i
        
        t_stat, p_value = stats.ttest_rel(
            metricx_scores[i], 
            metricx_scores[j]
        )
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        results.append({
            'file1': file1,
            'file2': file2,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance
        })
    
    if results:
        print(f"{'Everything vs Baseline':<40} {'t-statistic':<15} {'p-value':<15}")
        for result in results:
            file1_simple = simplify_filename(result['file1'])
            file2_simple = simplify_filename(result['file2'])
            comparison = f"{file1_simple} vs {file2_simple}"
            p_value_str = f"{result['p_value']:.4f}{result['significance']}"
            print(f"{comparison:<40} {result['t_statistic']:<15.4f} {p_value_str:<15}")
    
    print()
    
    return results

def print_average_metricx_scores(metricx_scores, file_names):
    """Print average metricx scores for each file"""
    print(f"\n{'File':<40} {'MetricX':<15}")
    
    for file_name, scores in zip(file_names, metricx_scores):
        if scores:
            avg_score = np.mean(scores)
            print(f"{file_name:<40} {avg_score:<15.4f}")
        else:
            print(f"{file_name:<40} {'N/A':<15}")

def save_metricx_ttest_results_to_csv(all_ttest_results, model, language, output_dir=str(RESULTS_ROOT)):
    """Save MetricX t-test results to CSV file"""
    csv_file = os.path.join(output_dir, 'metricx_ttest_results.csv')
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

def save_metricx_average_results_to_csv(model, language, metricx_scores, file_names, output_dir=str(RESULTS_ROOT)):
    """Save average MetricX results to CSV file"""
    csv_file = os.path.join(output_dir, 'metricx_average_results.csv')
    csv_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a' if csv_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not csv_exists:
            writer.writerow(['model', 'language', 'file', 'metricx', 'timestamp'])
        
        timestamp_str = datetime.now().isoformat()
        
        # Write data rows
        for file_name, scores in zip(file_names, metricx_scores):
            if scores:
                avg_score = np.mean(scores)
                writer.writerow([
                    model,
                    language,
                    file_name,
                    f"{avg_score:.4f}",
                    timestamp_str
                ])
            else:
                writer.writerow([
                    model,
                    language,
                    file_name,
                    "N/A",
                    timestamp_str
                ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MetricX score statistical significance tests")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--language', type=str, required=True, help='Language')
    
    args = parser.parse_args()
    language = "Upper Sorbian" if args.language == "upper sorbian" else args.language.capitalize()
    
    results = RESULTS_ROOT / args.model / language.lower()
    pred_files = sorted([p for p in results.glob('*.txt')])
    
    if not pred_files:
        print(f"No prediction files found in {results}")
        exit(1)

    # Load MetricX scores for all files
    metricx_scores, file_names = load_all_metricx_scores(args.model, language.lower(), pred_files)
    
    # Print average scores
    print_average_metricx_scores(metricx_scores, file_names)

    # Perform t-tests grouped by shot type
    shot_types = ['0_shot', '1_shot', '3_shot', '5_shot']
    all_ttest_results = {}
    
    for shot_type in shot_types:
        # Find indices of files matching current shot type
        shot_indices = [i for i, name in enumerate(file_names) if shot_type in name]
        
        # Create all pairs within this shot type
        shot_pairs = list(combinations(shot_indices, 2))
        
        if shot_pairs:
            print()
            print(f"T-Test Results for {shot_type.upper()}:")
            shot_results = ttest_metricx(metricx_scores, file_names, file_pairs=shot_pairs)
            all_ttest_results[shot_type] = shot_results
    
    # Save results to CSV files
    save_metricx_ttest_results_to_csv(all_ttest_results, args.model, language.lower())
    save_metricx_average_results_to_csv(args.model, language.lower(), metricx_scores, file_names)