#!/usr/bin/env python3
"""
Analyze veracity predictions from baseline, positive, and negative fact-checking systems.
Compare label shifts, quality changes, and potential bias patterns.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
from pathlib import Path


def load_predictions(file_path):
    """Load prediction file (JSON format)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(file_path):
    """Load ground truth labels"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def align_predictions(baseline_preds, positive_preds, negative_preds, ground_truth):
    """Align predictions by claim_id and merge with ground truth"""
    # Create dictionaries for quick lookup
    baseline_dict = {pred['claim_id']: pred for pred in baseline_preds}
    positive_dict = {pred['claim_id']: pred for pred in positive_preds}
    negative_dict = {pred['claim_id']: pred for pred in negative_preds}
    gt_dict = {i: example for i, example in enumerate(ground_truth)}

    aligned_data = []

    # Find common claim_ids
    all_claim_ids = set(baseline_dict.keys()) & set(positive_dict.keys()) & set(negative_dict.keys())

    for claim_id in sorted(all_claim_ids):
        if claim_id in gt_dict:
            aligned_data.append({
                'claim_id': claim_id,
                'claim': baseline_dict[claim_id]['claim'],
                'ground_truth': gt_dict[claim_id]['label'],
                'baseline_pred': baseline_dict[claim_id]['pred_label'],
                'positive_pred': positive_dict[claim_id]['pred_label'],
                'negative_pred': negative_dict[claim_id]['pred_label'],
                'baseline_output': baseline_dict[claim_id].get('llm_output', ''),
                'positive_output': positive_dict[claim_id].get('llm_output', ''),
                'negative_output': negative_dict[claim_id].get('llm_output', '')
            })

    return pd.DataFrame(aligned_data)


def analyze_label_shifts(df):
    """Analyze how labels shift between systems"""
    print("\n=== LABEL SHIFT ANALYSIS ===")

    # Overall label distributions
    print("\nLabel Distributions:")
    print("Ground Truth:", dict(df['ground_truth'].value_counts()))
    print("Baseline:    ", dict(df['baseline_pred'].value_counts()))
    print("Positive:    ", dict(df['positive_pred'].value_counts()))
    print("Negative:    ", dict(df['negative_pred'].value_counts()))

    # Transition matrices
    systems = ['baseline_pred', 'positive_pred', 'negative_pred']
    system_names = ['Baseline', 'Positive', 'Negative']

    for i, (system, name) in enumerate(zip(systems, system_names)):
        print(f"\n{name} vs Ground Truth Confusion Matrix:")
        cm = confusion_matrix(df['ground_truth'], df[system])
        labels = sorted(df['ground_truth'].unique())
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df)

    # Pairwise system comparisons
    print("\n=== SYSTEM COMPARISONS ===")
    comparisons = [
        ('baseline_pred', 'positive_pred', 'Baseline vs Positive'),
        ('baseline_pred', 'negative_pred', 'Baseline vs Negative'),
        ('positive_pred', 'negative_pred', 'Positive vs Negative')
    ]

    for sys1, sys2, name in comparisons:
        agreement = (df[sys1] == df[sys2]).sum()
        total = len(df)
        print(f"\n{name} Agreement: {agreement}/{total} ({agreement / total:.3f})")

        # Disagreement analysis
        disagreements = df[df[sys1] != df[sys2]]
        if len(disagreements) > 0:
            print(f"Disagreements by ground truth:")
            for gt_label in sorted(df['ground_truth'].unique()):
                gt_disagreements = disagreements[disagreements['ground_truth'] == gt_label]
                if len(gt_disagreements) > 0:
                    transitions = gt_disagreements[[sys1, sys2]].apply(
                        lambda x: f"{x[sys1]} -> {x[sys2]}", axis=1
                    ).value_counts()
                    print(f"  {gt_label}: {dict(transitions)}")


def compute_quality_metrics(df):
    """Compute accuracy and F1 scores for each system"""
    print("\n=== QUALITY METRICS ===")

    systems = ['baseline_pred', 'positive_pred', 'negative_pred']
    system_names = ['Baseline', 'Positive', 'Negative']

    results = {}

    for system, name in zip(systems, system_names):
        accuracy = accuracy_score(df['ground_truth'], df[system])
        f1_macro = f1_score(df['ground_truth'], df[system], average='macro')
        f1_micro = f1_score(df['ground_truth'], df[system], average='micro')

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (micro): {f1_micro:.4f}")

        # Per-class metrics
        print(f"  Classification Report:")
        report = classification_report(df['ground_truth'], df[system],
                                       output_dict=True, zero_division=0)
        for label in sorted(df['ground_truth'].unique()):
            if label in report:
                print(f"    {label}: P={report[label]['precision']:.3f}, "
                      f"R={report[label]['recall']:.3f}, F1={report[label]['f1-score']:.3f}")

        results[name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'report': report
        }

    return results


def analyze_bias_patterns(df):
    """Analyze potential bias patterns in the systems"""
    print("\n=== BIAS ANALYSIS ===")

    # Define supporting vs refuting labels
    supporting_labels = {'Supported'}
    refuting_labels = {'Refuted'}
    neutral_labels = {'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking'}

    def categorize_prediction(pred):
        if pred in supporting_labels:
            return 'Supporting'
        elif pred in refuting_labels:
            return 'Refuting'
        else:
            return 'Neutral'

    # Add categorized predictions
    df['gt_category'] = df['ground_truth'].apply(categorize_prediction)
    df['baseline_category'] = df['baseline_pred'].apply(categorize_prediction)
    df['positive_category'] = df['positive_pred'].apply(categorize_prediction)
    df['negative_category'] = df['negative_pred'].apply(categorize_prediction)

    # Bias towards supporting/refuting
    print("\nPrediction Category Distribution:")
    for system in ['ground_truth', 'baseline_pred', 'positive_pred', 'negative_pred']:
        category_col = system.replace('_pred', '_category') if '_pred' in system else 'gt_category'
        dist = df[category_col].value_counts(normalize=True)
        system_name = system.replace('_pred', '').replace('_', ' ').title()
        print(f"{system_name:12}: {dict(dist)}")

    # Directional bias analysis
    print("\nDirectional Bias Analysis:")

    # Positive system bias
    positive_more_supporting = ((df['positive_category'] == 'Supporting') &
                                (df['baseline_category'] != 'Supporting')).sum()
    positive_more_refuting = ((df['positive_category'] == 'Refuting') &
                              (df['baseline_category'] != 'Refuting')).sum()

    print(f"Positive system vs Baseline:")
    print(f"  More supporting: {positive_more_supporting} cases")
    print(f"  More refuting:   {positive_more_refuting} cases")

    # Negative system bias
    negative_more_supporting = ((df['negative_category'] == 'Supporting') &
                                (df['baseline_category'] != 'Supporting')).sum()
    negative_more_refuting = ((df['negative_category'] == 'Refuting') &
                              (df['baseline_category'] != 'Refuting')).sum()

    print(f"Negative system vs Baseline:")
    print(f"  More supporting: {negative_more_supporting} cases")
    print(f"  More refuting:   {negative_more_refuting} cases")

    # Performance by ground truth category
    print("\nPerformance by Ground Truth Category:")
    for gt_cat in ['Supporting', 'Refuting', 'Neutral']:
        subset = df[df['gt_category'] == gt_cat]
        if len(subset) > 0:
            baseline_acc = (subset['baseline_pred'] == subset['ground_truth']).mean()
            positive_acc = (subset['positive_pred'] == subset['ground_truth']).mean()
            negative_acc = (subset['negative_pred'] == subset['ground_truth']).mean()

            print(f"{gt_cat} claims ({len(subset)} total):")
            print(f"  Baseline: {baseline_acc:.3f}")
            print(f"  Positive: {positive_acc:.3f}")
            print(f"  Negative: {negative_acc:.3f}")


def analyze_error_patterns(df):
    """Analyze specific error patterns and interesting cases"""
    print("\n=== ERROR PATTERN ANALYSIS ===")

    # Cases where positive/negative both differ from baseline
    both_differ = df[(df['positive_pred'] != df['baseline_pred']) &
                     (df['negative_pred'] != df['baseline_pred'])]

    print(f"\nCases where both positive and negative differ from baseline: {len(both_differ)}")

    if len(both_differ) > 0:
        # Analyze these cases
        both_differ_correct = both_differ[
            (both_differ['positive_pred'] == both_differ['ground_truth']) |
            (both_differ['negative_pred'] == both_differ['ground_truth'])
            ]
        baseline_correct = both_differ[both_differ['baseline_pred'] == both_differ['ground_truth']]

        print(f"  Cases where pos/neg are correct but baseline wrong: {len(both_differ_correct)}")
        print(f"  Cases where baseline is correct but pos/neg wrong: {len(baseline_correct)}")

    # Cases where positive and negative give opposite predictions
    opposite_preds = df[(df['positive_pred'] == 'Supported') & (df['negative_pred'] == 'Refuted') |
                        (df['positive_pred'] == 'Refuted') & (df['negative_pred'] == 'Supported')]

    print(f"\nCases where positive and negative give opposite predictions: {len(opposite_preds)}")

    if len(opposite_preds) > 0:
        print("Ground truth distribution for opposite predictions:")
        print(dict(opposite_preds['ground_truth'].value_counts()))

        # Show some examples
        print("\nExample claims with opposite predictions:")
        for i, row in opposite_preds.head(3).iterrows():
            print(f"\nClaim {row['claim_id']}: {row['claim'][:100]}...")
            print(f"  Ground truth: {row['ground_truth']}")
            print(f"  Positive: {row['positive_pred']}")
            print(f"  Negative: {row['negative_pred']}")
            print(f"  Baseline: {row['baseline_pred']}")


def create_visualizations(df, results, output_dir):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8')

    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    systems = ['Baseline', 'Positive', 'Negative']
    accuracies = [results[sys]['accuracy'] for sys in systems]
    f1_scores = [results[sys]['f1_macro'] for sys in systems]

    x = np.arange(len(systems))
    width = 0.35

    ax.bar(x - width / 2, accuracies, width, label='Accuracy')
    ax.bar(x + width / 2, f1_scores, width, label='F1 (macro)')

    ax.set_xlabel('System')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison Across Systems')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Label distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    systems_data = [
        ('Ground Truth', df['ground_truth']),
        ('Baseline', df['baseline_pred']),
        ('Positive', df['positive_pred']),
        ('Negative', df['negative_pred'])
    ]

    for idx, (name, data) in enumerate(systems_data):
        ax = axes[idx // 2, idx % 2]
        counts = data.value_counts()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax.set_title(f'{name} Label Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'label_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    systems = ['baseline_pred', 'positive_pred', 'negative_pred']
    system_names = ['Baseline', 'Positive', 'Negative']

    labels = sorted(df['ground_truth'].unique())

    for idx, (system, name) in enumerate(zip(systems, system_names)):
        cm = confusion_matrix(df['ground_truth'], df[system], labels=labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f'{name} Confusion Matrix (Normalized)')

        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, f'{cm_normalized[i, j]:.2f}',
                               horizontalalignment="center",
                               color="white" if cm_normalized[i, j] > thresh else "black")

        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticks(range(len(labels)))
        axes[idx].set_yticks(range(len(labels)))
        axes[idx].set_xticklabels(labels, rotation=45)
        axes[idx].set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}")


def save_detailed_analysis(df, output_dir):
    """Save detailed analysis to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save aligned predictions
    df.to_csv(output_dir / 'aligned_predictions.csv', index=False)

    # Save disagreement cases
    disagreements = df[
        (df['baseline_pred'] != df['positive_pred']) |
        (df['baseline_pred'] != df['negative_pred']) |
        (df['positive_pred'] != df['negative_pred'])
        ]
    disagreements.to_csv(output_dir / 'disagreement_cases.csv', index=False)

    # Save error cases for each system
    for system in ['baseline_pred', 'positive_pred', 'negative_pred']:
        errors = df[df[system] != df['ground_truth']]
        system_name = system.replace('_pred', '')
        errors.to_csv(output_dir / f'{system_name}_errors.csv', index=False)

    print(f"\nDetailed analysis files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze veracity predictions across systems')
    parser.add_argument('--baseline', required=True, help='Baseline predictions JSON file')
    parser.add_argument('--positive', required=True, help='Positive system predictions JSON file')
    parser.add_argument('--negative', required=True, help='Negative system predictions JSON file')
    parser.add_argument('--ground-truth', required=True, help='Ground truth JSON file')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory for results')

    args = parser.parse_args()

    print("Loading prediction files...")
    baseline_preds = load_predictions(args.baseline)
    positive_preds = load_predictions(args.positive)
    negative_preds = load_predictions(args.negative)
    ground_truth = load_ground_truth(args.ground_truth)

    print("Aligning predictions...")
    df = align_predictions(baseline_preds, positive_preds, negative_preds, ground_truth)

    print(f"Aligned {len(df)} predictions across all systems")

    # Run analyses
    analyze_label_shifts(df)
    results = compute_quality_metrics(df)
    analyze_bias_patterns(df)
    analyze_error_patterns(df)

    # Create visualizations and save detailed results
    create_visualizations(df, results, args.output_dir)
    save_detailed_analysis(df, args.output_dir)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
