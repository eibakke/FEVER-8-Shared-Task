import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import os
import torch
from tqdm import tqdm
import time


def load_predictions(file_path):
    """Load prediction data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dataframe(perspective_preds, baseline_preds, perspective_name, gold_data=None):
    """Create a pandas DataFrame for analysis comparing predictions."""
    # Create mapping from claim_id to predictions
    perspective_dict = {item['claim_id']: item for item in perspective_preds}
    baseline_dict = {item['claim_id']: item for item in baseline_preds}

    # Get all unique claim IDs
    all_ids = sorted(list(set(perspective_dict.keys()) | set(baseline_dict.keys())))

    # Create dataframe rows
    rows = []
    for claim_id in all_ids:
        perspective = perspective_dict.get(claim_id, {})
        baseline = baseline_dict.get(claim_id, {})

        # Skip if we don't have both predictions
        if not perspective or not baseline:
            continue

        row = {
            'claim_id': claim_id,
            'claim': perspective.get('claim', ''),
            f'{perspective_name}_label': perspective.get('pred_label', ''),
            'baseline_label': baseline.get('pred_label', ''),
            f'{perspective_name}_output': perspective.get('llm_output', ''),
            'baseline_output': baseline.get('llm_output', ''),
            f'{perspective_name}_evidence_count': len(perspective.get('evidence', [])),
            'baseline_evidence_count': len(baseline.get('evidence', [])),
        }

        # Add ground truth if available
        if gold_data:
            gold_item = gold_data[claim_id] if claim_id < len(gold_data) else None
            if gold_item:
                row['gold_label'] = gold_item.get('label', '')

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived columns
    df['agreement'] = df[f'{perspective_name}_label'] == df['baseline_label']

    if 'gold_label' in df.columns:
        df[f'{perspective_name}_correct'] = df[f'{perspective_name}_label'] == df['gold_label']
        df['baseline_correct'] = df['baseline_label'] == df['gold_label']
        df['improvement'] = ~df[f'{perspective_name}_correct'] & df['baseline_correct']
        df['degradation'] = df[f'{perspective_name}_correct'] & ~df['baseline_correct']

    return df


def calculate_agreement_metrics(df, perspective_name):
    """Calculate agreement metrics between perspective and baseline predictions."""
    results = {}

    # Overall agreement
    results['overall_agreement'] = df['agreement'].mean()

    # Agreement by perspective label
    agreement_by_perspective = df.groupby(f'{perspective_name}_label')['agreement'].agg(['count', 'mean'])
    # Convert to a properly structured dictionary
    agreement_by_perspective_dict = {}
    for label in agreement_by_perspective.index:
        agreement_by_perspective_dict[label] = {
            'count': int(agreement_by_perspective.loc[label, 'count']),
            'mean': float(agreement_by_perspective.loc[label, 'mean'])
        }
    results[f'agreement_by_{perspective_name}_label'] = agreement_by_perspective_dict

    # Agreement by baseline label
    agreement_by_baseline = df.groupby('baseline_label')['agreement'].agg(['count', 'mean'])
    # Convert to a properly structured dictionary
    agreement_by_baseline_dict = {}
    for label in agreement_by_baseline.index:
        agreement_by_baseline_dict[label] = {
            'count': int(agreement_by_baseline.loc[label, 'count']),
            'mean': float(agreement_by_baseline.loc[label, 'mean'])
        }
    results['agreement_by_baseline_label'] = agreement_by_baseline_dict

    # Transition matrix (from perspective to baseline)
    transition_matrix = pd.crosstab(
        df[f'{perspective_name}_label'],
        df['baseline_label'],
        normalize='index'
    ).round(3)

    # Convert transition matrix to proper dictionary structure
    transition_dict = {}
    for label in transition_matrix.index:
        transition_dict[label] = {}
        for col in transition_matrix.columns:
            if col in transition_matrix.columns:
                transition_dict[label][col] = float(transition_matrix.loc[label, col])

    results['transition_matrix'] = transition_dict

    # Count specific transitions (for major label changes)
    transitions = df.groupby([f'{perspective_name}_label', 'baseline_label']).size().reset_index(name='count')
    total_changes = len(df[~df['agreement']])

    if total_changes > 0:
        # Calculate percentages of label flips
        for _, row in transitions.iterrows():
            if row[f'{perspective_name}_label'] != row['baseline_label']:
                flip_key = f"flip_{row[f'{perspective_name}_label']}_to_{row['baseline_label']}"
                results[flip_key] = {
                    'count': int(row['count']),
                    'percentage_of_changes': round(row['count'] / total_changes * 100, 2)
                }

    # Overall flip rate
    results['overall_flip_rate'] = 1.0 - results['overall_agreement']

    return results


def calculate_correctness_metrics(df, perspective_name):
    """Calculate metrics related to correctness and knowledge impact."""
    if 'gold_label' not in df.columns:
        return {"error": "Gold data not available for correctness metrics"}

    results = {}

    # Overall accuracy
    results[f'{perspective_name}_accuracy'] = df[f'{perspective_name}_correct'].mean()
    results['baseline_accuracy'] = df['baseline_correct'].mean()
    results['accuracy_delta'] = results['baseline_accuracy'] - results[f'{perspective_name}_accuracy']

    # Correction analysis
    correction_opportunities = len(df[~df[f'{perspective_name}_correct']])
    corrections_made = len(df[~df[f'{perspective_name}_correct'] & df['baseline_correct']])

    results['correction_opportunities'] = correction_opportunities
    results['corrections_made'] = corrections_made

    if correction_opportunities > 0:
        results['correction_rate'] = corrections_made / correction_opportunities
    else:
        results['correction_rate'] = 0

    # Error introduction analysis
    error_opportunities = len(df[df[f'{perspective_name}_correct']])
    errors_introduced = len(df[df[f'{perspective_name}_correct'] & ~df['baseline_correct']])

    results['error_opportunities'] = error_opportunities
    results['errors_introduced'] = errors_introduced

    if error_opportunities > 0:
        results['error_introduction_rate'] = errors_introduced / error_opportunities
    else:
        results['error_introduction_rate'] = 0

    # Net impact
    results['net_corrections'] = corrections_made - errors_introduced

    # Label-specific correctness
    correctness_by_label = df.groupby('gold_label').agg({
        f'{perspective_name}_correct': 'mean',
        'baseline_correct': 'mean'
    }).reset_index()

    results['correctness_by_label'] = correctness_by_label.to_dict()

    # Accuracy for cases where perspective and baseline agree vs. disagree
    agree_cases = df[df['agreement']]
    disagree_cases = df[~df['agreement']]

    if len(agree_cases) > 0:
        results['accuracy_when_agree'] = {
            'count': len(agree_cases),
            f'{perspective_name}_accuracy': agree_cases[f'{perspective_name}_correct'].mean(),
            'baseline_accuracy': agree_cases['baseline_correct'].mean()
        }

    if len(disagree_cases) > 0:
        results['accuracy_when_disagree'] = {
            'count': len(disagree_cases),
            f'{perspective_name}_accuracy': disagree_cases[f'{perspective_name}_correct'].mean(),
            'baseline_accuracy': disagree_cases['baseline_correct'].mean()
        }

    return results


def analyze_evidence_impact(df, perspective_name):
    """Analyze how evidence quantity impacts prediction changes."""
    results = {}

    # Average evidence counts
    results[f'avg_{perspective_name}_evidence'] = df[f'{perspective_name}_evidence_count'].mean()
    results['avg_baseline_evidence'] = df['baseline_evidence_count'].mean()

    # Evidence count for agreement vs disagreement cases
    agree_cases = df[df['agreement']]
    disagree_cases = df[~df['agreement']]

    if len(agree_cases) > 0:
        results['evidence_when_agree'] = {
            'count': len(agree_cases),
            'avg_baseline_evidence': agree_cases['baseline_evidence_count'].mean()
        }

    if len(disagree_cases) > 0:
        results['evidence_when_disagree'] = {
            'count': len(disagree_cases),
            'avg_baseline_evidence': disagree_cases['baseline_evidence_count'].mean()
        }

    # Correlation between evidence count and prediction change
    results['correlation_evidence_disagreement'] = np.corrcoef(
        df['baseline_evidence_count'],
        ~df['agreement']
    )[0, 1]

    # Evidence analysis by label transition
    evidence_by_transition = df.groupby([f'{perspective_name}_label', 'baseline_label']).agg({
        'baseline_evidence_count': 'mean',
        'claim_id': 'count'
    }).reset_index()

    results['evidence_by_transition'] = evidence_by_transition.to_dict()

    return results


def analyze_justification_similarity(df, perspective_name, embedder, batch_size=32, use_gpu=True):
    """Analyze similarity between perspective and baseline justifications."""
    # Check for GPU availability
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if use_gpu:
            print("GPU requested but not available. Using CPU instead.")
        else:
            print("Using CPU for embeddings")

    # Function to clean and extract justification text
    def extract_justification(text):
        # Remove formatting and extract any justification section
        text = text.replace('<|start_header_id|>assistant<|end_header_id|>', '')

        # Look for justification section
        justification_match = re.search(r'justification:\s*(.*?)(?:verdict:|$)', text, re.DOTALL)
        if justification_match:
            return justification_match.group(1).strip()

        # If no explicit justification, return the whole text
        return text.strip()

    # Sample up to 100 examples to avoid memory issues with embeddings
    if len(df) > 100:
        sample_df = df.sample(100, random_state=42)
    else:
        sample_df = df

    # Extract justifications
    print(f"Extracting justifications for {perspective_name}...")
    sample_df[f'{perspective_name}_justification'] = sample_df[f'{perspective_name}_output'].apply(
        extract_justification)
    sample_df['baseline_justification'] = sample_df['baseline_output'].apply(extract_justification)

    # Move model to GPU if available
    embedder = embedder.to(device)

    # Compute embeddings in batches
    print("Computing embeddings in batches...")
    perspective_texts = sample_df[f'{perspective_name}_justification'].tolist()
    baseline_texts = sample_df['baseline_justification'].tolist()

    # Function to compute embeddings in batches
    def compute_embeddings_batched(texts, batch_size=batch_size):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    # Compute embeddings for both sets of texts
    perspective_embeddings = compute_embeddings_batched(perspective_texts)
    baseline_embeddings = compute_embeddings_batched(baseline_texts)

    # Compute similarities
    print("Computing similarities...")
    similarities = []
    for i in range(len(perspective_embeddings)):
        similarity = 1 - cosine(perspective_embeddings[i], baseline_embeddings[i])
        similarities.append(similarity)

    sample_df['justification_similarity'] = similarities

    results = {
        'mean_justification_similarity': np.mean(similarities),
        'median_justification_similarity': np.median(similarities),
        'min_justification_similarity': min(similarities),
        'max_justification_similarity': max(similarities)
    }

    # Analyze similarity for agreement vs disagreement cases
    agree_similarities = sample_df[sample_df['agreement']]['justification_similarity'].tolist()
    disagree_similarities = sample_df[~sample_df['agreement']]['justification_similarity'].tolist()

    if agree_similarities:
        results['mean_similarity_when_agree'] = np.mean(agree_similarities)

    if disagree_similarities:
        results['mean_similarity_when_disagree'] = np.mean(disagree_similarities)

    return results


def analyze_label_distribution(df, perspective_name):
    """Analyze how label distribution shifts between perspective and baseline predictions."""
    results = {}

    # Count labels
    perspective_counts = df[f'{perspective_name}_label'].value_counts().to_dict()
    baseline_counts = df['baseline_label'].value_counts().to_dict()

    results[f'{perspective_name}_label_counts'] = perspective_counts
    results['baseline_label_counts'] = baseline_counts

    # Calculate percentages
    total = len(df)
    perspective_percentages = {k: v / total for k, v in perspective_counts.items()}
    baseline_percentages = {k: v / total for k, v in baseline_counts.items()}

    results[f'{perspective_name}_label_percentages'] = perspective_percentages
    results['baseline_label_percentages'] = baseline_percentages

    # Calculate shifts
    labels = sorted(list(set(perspective_counts.keys()) | set(baseline_counts.keys())))
    shifts = {}

    for label in labels:
        perspective_pct = perspective_percentages.get(label, 0)
        baseline_pct = baseline_percentages.get(label, 0)
        shifts[label] = {
            'absolute_shift': baseline_pct - perspective_pct,
            'relative_shift': (baseline_pct / perspective_pct - 1) if perspective_pct > 0 else float('inf')
        }

    results['label_shifts'] = shifts

    return results


def generate_visualizations(df, perspective_name, results, output_dir):
    """Generate visualizations from the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Confusion matrix between perspective and baseline predictions
    plt.figure(figsize=(10, 8))
    labels = sorted(list(set(df[f'{perspective_name}_label'].unique()) | set(df['baseline_label'].unique())))
    cm = confusion_matrix(
        df[f'{perspective_name}_label'],
        df['baseline_label'],
        labels=labels
    )

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Baseline Predictions')
    plt.ylabel(f'{perspective_name.capitalize()} Predictions')
    plt.title(f'Prediction Changes: {perspective_name.capitalize()} vs. Baseline')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{perspective_name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. Label distribution comparison
    plt.figure(figsize=(12, 6))
    labels = sorted(list(set(df[f'{perspective_name}_label'].unique()) | set(df['baseline_label'].unique())))

    perspective_counts = [df[df[f'{perspective_name}_label'] == label].shape[0] for label in labels]
    baseline_counts = [df[df['baseline_label'] == label].shape[0] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, perspective_counts, width, label=perspective_name.capitalize())
    plt.bar(x + width / 2, baseline_counts, width, label='Baseline')

    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(f'Label Distribution Comparison: {perspective_name.capitalize()} vs. Baseline')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{perspective_name}_label_distribution.png'), dpi=300)
    plt.close()

    # 3. Evidence count vs agreement
    plt.figure(figsize=(10, 6))
    evidence_counts = df.groupby('baseline_evidence_count')['agreement'].agg(['count', 'mean']).reset_index()
    evidence_counts = evidence_counts[evidence_counts['count'] >= 5]  # Filter out rare counts

    plt.scatter(evidence_counts['baseline_evidence_count'], evidence_counts['mean'],
                s=evidence_counts['count'] * 5, alpha=0.6)

    for _, row in evidence_counts.iterrows():
        plt.annotate(f"n={row['count']}",
                     (row['baseline_evidence_count'], row['mean']),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Evidence Count')
    plt.ylabel('Agreement Rate')
    plt.title(f'Agreement Rate by Evidence Count: {perspective_name.capitalize()} vs. Baseline')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{perspective_name}_evidence_vs_agreement.png'), dpi=300)
    plt.close()

    # 4. If gold data is available, accuracy comparison
    if 'gold_label' in df.columns:
        plt.figure(figsize=(10, 6))

        labels = sorted(df['gold_label'].unique())
        perspective_accs = [df[df['gold_label'] == label][f'{perspective_name}_correct'].mean() for label in labels]
        baseline_accs = [df[df['gold_label'] == label]['baseline_correct'].mean() for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, perspective_accs, width, label=perspective_name.capitalize())
        plt.bar(x + width / 2, baseline_accs, width, label='Baseline')

        plt.xlabel('True Label')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Label: {perspective_name.capitalize()} vs. Baseline')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{perspective_name}_accuracy_by_label.png'), dpi=300)
        plt.close()

        # 5. Improvements vs Degradations
        if 'improvement' in df.columns and 'degradation' in df.columns:
            plt.figure(figsize=(10, 6))

            # Count improvements and degradations by gold label
            improvements_by_label = df[df['improvement']].groupby('gold_label').size()
            degradations_by_label = df[df['degradation']].groupby('gold_label').size()

            # Ensure all labels are represented
            for label in labels:
                if label not in improvements_by_label:
                    improvements_by_label[label] = 0
                if label not in degradations_by_label:
                    degradations_by_label[label] = 0

            # Sort by label
            improvements_by_label = improvements_by_label.reindex(labels, fill_value=0)
            degradations_by_label = degradations_by_label.reindex(labels, fill_value=0)

            x = np.arange(len(labels))
            width = 0.35

            plt.bar(x - width / 2, improvements_by_label, width, label='Improvements', color='green')
            plt.bar(x + width / 2, degradations_by_label, width, label='Degradations', color='red')

            plt.xlabel('True Label')
            plt.ylabel('Count')
            plt.title(f'Baseline Impact by Label: {perspective_name.capitalize()} vs. Baseline')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{perspective_name}_impact_by_label.png'), dpi=300)
            plt.close()

    return


def write_metrics_to_file(results, perspective_name, output_path):
    """Write metrics to a formatted text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Comparison Analysis: {perspective_name.capitalize()} vs. Baseline Prediction\n\n")

        # Write agreement metrics
        f.write("## 1. Agreement Metrics\n\n")
        f.write(f"Overall agreement rate: {results['agreement']['overall_agreement']:.2%}\n")
        f.write(f"Overall flip rate: {results['agreement']['overall_flip_rate']:.2%}\n\n")

        f.write(f"### Agreement by {perspective_name} prediction label:\n")
        for label, stats in results['agreement'][f'agreement_by_{perspective_name}_label'].items():
            f.write(f"- {label}: {stats['mean']:.2%} (n={stats['count']})\n")
        f.write("\n")

        # Write transition matrix info
        f.write(f"### Transition Matrix (rows={perspective_name}, columns=baseline):\n")
        for perspective_label in results['agreement']['transition_matrix']:
            f.write(f"- From {perspective_label}:\n")
            for baseline_label, value in results['agreement']['transition_matrix'][perspective_label].items():
                f.write(f"  - To {baseline_label}: {value:.2%}\n")
        f.write("\n")

        # Write flip details
        f.write("### Major label flips:\n")
        for key, value in results['agreement'].items():
            if key.startswith('flip_'):
                _, from_label, _, to_label = key.split('_')
                f.write(
                    f"- {from_label} → {to_label}: {value['count']} cases ({value['percentage_of_changes']:.1f}% of all changes)\n")
        f.write("\n")

        # If correctness metrics are available
        if 'correctness' in results and 'error' not in results['correctness']:
            f.write("## 2. Correctness Impact\n\n")
            f.write(
                f"{perspective_name.capitalize()} prediction accuracy: {results['correctness'][f'{perspective_name}_accuracy']:.2%}\n")
            f.write(f"Baseline accuracy: {results['correctness']['baseline_accuracy']:.2%}\n")
            f.write(f"Accuracy delta: {results['correctness']['accuracy_delta']:.2%}\n\n")

            f.write("### Correction analysis:\n")
            f.write(
                f"- Correction rate: {results['correctness']['correction_rate']:.2%} ({results['correctness']['corrections_made']} of {results['correctness']['correction_opportunities']} opportunities)\n")
            f.write(
                f"- Error introduction rate: {results['correctness']['error_introduction_rate']:.2%} ({results['correctness']['errors_introduced']} of {results['correctness']['error_opportunities']} opportunities)\n")
            f.write(f"- Net corrections: {results['correctness']['net_corrections']}\n\n")

            f.write("### Accuracy when predictions agree vs. disagree:\n")
            if 'accuracy_when_agree' in results['correctness']:
                f.write(
                    f"- When agree (n={results['correctness']['accuracy_when_agree']['count']}): {results['correctness']['accuracy_when_agree']['baseline_accuracy']:.2%}\n")
            if 'accuracy_when_disagree' in results['correctness']:
                f.write(f"- When disagree (n={results['correctness']['accuracy_when_disagree']['count']}):\n")
                f.write(
                    f"  - {perspective_name.capitalize()}: {results['correctness']['accuracy_when_disagree'][f'{perspective_name}_accuracy']:.2%}\n")
                f.write(
                    f"  - Baseline: {results['correctness']['accuracy_when_disagree']['baseline_accuracy']:.2%}\n\n")

        # Write evidence impact metrics
        f.write("## 3. Evidence Analysis\n\n")
        f.write(
            f"Average evidence count in {perspective_name} predictions: {results['evidence'][f'avg_{perspective_name}_evidence']:.2f}\n")
        f.write(
            f"Average evidence count in baseline predictions: {results['evidence']['avg_baseline_evidence']:.2f}\n\n")

        f.write("### Evidence impact on prediction changes:\n")
        if 'evidence_when_agree' in results['evidence']:
            f.write(
                f"- Average evidence when predictions agree (n={results['evidence']['evidence_when_agree']['count']}): {results['evidence']['evidence_when_agree']['avg_baseline_evidence']:.2f}\n")
        if 'evidence_when_disagree' in results['evidence']:
            f.write(
                f"- Average evidence when predictions disagree (n={results['evidence']['evidence_when_disagree']['count']}): {results['evidence']['evidence_when_disagree']['avg_baseline_evidence']:.2f}\n")
        f.write(
            f"- Correlation between evidence count and disagreement: {results['evidence']['correlation_evidence_disagreement']:.3f}\n\n")

        # Write justification similarity metrics if available
        if 'justification' in results:
            f.write("## 4. Justification Analysis\n\n")
            f.write(
                f"Mean semantic similarity between justifications: {results['justification']['mean_justification_similarity']:.3f}\n")
            f.write(f"Median similarity: {results['justification']['median_justification_similarity']:.3f}\n")
            f.write(
                f"Range: {results['justification']['min_justification_similarity']:.3f} - {results['justification']['max_justification_similarity']:.3f}\n\n")

            if 'mean_similarity_when_agree' in results['justification']:
                f.write(
                    f"- Mean similarity when predictions agree: {results['justification']['mean_similarity_when_agree']:.3f}\n")
            if 'mean_similarity_when_disagree' in results['justification']:
                f.write(
                    f"- Mean similarity when predictions disagree: {results['justification']['mean_similarity_when_disagree']:.3f}\n\n")

        # Write label distribution metrics
        f.write("## 5. Label Distribution Analysis\n\n")
        f.write("### Label counts:\n")
        f.write(f"| Label | {perspective_name.capitalize()} | Baseline | Absolute Shift | Relative Shift |\n")
        f.write("|-------|--------|----------|----------------|---------------|\n")

        all_labels = sorted(list(set(results['distribution'][f'{perspective_name}_label_counts'].keys()) |
                                 set(results['distribution']['baseline_label_counts'].keys())))

        for label in all_labels:
            perspective_count = results['distribution'][f'{perspective_name}_label_counts'].get(label, 0)
            baseline_count = results['distribution']['baseline_label_counts'].get(label, 0)
            abs_shift = results['distribution']['label_shifts'][label]['absolute_shift']
            rel_shift = results['distribution']['label_shifts'][label]['relative_shift']

            if rel_shift == float('inf'):
                rel_shift_str = "∞"
            else:
                rel_shift_str = f"{rel_shift:.2%}"

            f.write(f"| {label} | {perspective_count} | {baseline_count} | {abs_shift:.2%} | {rel_shift_str} |\n")


def make_json_serializable(obj):
    """Convert NumPy types to native Python types to make JSON serializable."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # For datetime objects
        return obj.isoformat()
    else:
        return obj


def analyze_perspective(perspective_file, baseline_file, perspective_name, output_dir, gold_data=None, embedder=None,
                        args=None):
    """Analyze a single perspective against baseline."""
    print(f"\n=== Analyzing {perspective_name.capitalize()} vs. Baseline ===")

    # Load data
    perspective_preds = load_predictions(perspective_file)
    baseline_preds = load_predictions(baseline_file)

    # Create dataframe
    df = create_dataframe(perspective_preds, baseline_preds, perspective_name, gold_data)

    # Calculate metrics
    results = {}

    print("Calculating agreement metrics...")
    results['agreement'] = calculate_agreement_metrics(df, perspective_name)

    if gold_data:
        print("Calculating correctness metrics...")
        results['correctness'] = calculate_correctness_metrics(df, perspective_name)

    print("Analyzing evidence impact...")
    results['evidence'] = analyze_evidence_impact(df, perspective_name)

    print("Analyzing label distribution...")
    results['distribution'] = analyze_label_distribution(df, perspective_name)

    # Optional justification analysis
    if embedder:
        try:
            print("Analyzing justification similarity...")
            results['justification'] = analyze_justification_similarity(
                df, perspective_name, embedder, batch_size=args.batch_size, use_gpu=args.use_gpu
            )
        except Exception as e:
            print(f"Skipping justification analysis due to error: {e}")

    # Create perspective-specific output directory
    perspective_output_dir = os.path.join(output_dir, perspective_name)
    os.makedirs(perspective_output_dir, exist_ok=True)

    # Save dataframe
    df.to_csv(os.path.join(perspective_output_dir, f'{perspective_name}_comparison_data.csv'), index=False)

    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, perspective_name, results, perspective_output_dir)

    # Write metrics
    print("Writing metrics to file...")
    write_metrics_to_file(results, perspective_name,
                          os.path.join(perspective_output_dir, f'{perspective_name}_comparison_metrics.md'))

    # Save raw results
    with open(os.path.join(perspective_output_dir, f'{perspective_name}_raw_metrics.json'), 'w', encoding='utf-8') as f:
        serializable_results = make_json_serializable(results)
        json.dump(serializable_results, f, indent=2)

    return results


def create_summary_comparison(positive_results, negative_results, objective_results, output_dir):
    """Create a summary comparison across all three perspectives."""
    summary_path = os.path.join(output_dir, 'multi_perspective_summary.md')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Perspective Analysis Summary\n\n")
        f.write("## Overall Agreement Rates\n\n")
        f.write("| Perspective | Agreement Rate | Flip Rate |\n")
        f.write("|-------------|----------------|----------|\n")

        for name, results in [('Positive', positive_results), ('Negative', negative_results),
                              ('Objective', objective_results)]:
            agreement = results['agreement']['overall_agreement']
            flip_rate = results['agreement']['overall_flip_rate']
            f.write(f"| {name} | {agreement:.2%} | {flip_rate:.2%} |\n")

        f.write("\n## Accuracy Comparison (if gold data available)\n\n")

        # Check if correctness data is available
        if 'correctness' in positive_results and 'error' not in positive_results['correctness']:
            f.write("| Perspective | Perspective Acc. | Baseline Acc. | Accuracy Delta |\n")
            f.write("|-------------|------------------|---------------|----------------|\n")

            for name, results in [('Positive', positive_results), ('Negative', negative_results),
                                  ('Objective', objective_results)]:
                perspective_acc = results['correctness'][f'{name.lower()}_accuracy']
                baseline_acc = results['correctness']['baseline_accuracy']
                delta = results['correctness']['accuracy_delta']
                f.write(f"| {name} | {perspective_acc:.2%} | {baseline_acc:.2%} | {delta:.2%} |\n")

        f.write("\n## Evidence Impact\n\n")
        f.write("| Perspective | Avg. Evidence Count | Correlation with Disagreement |\n")
        f.write("|-------------|---------------------|--------------------------------|\n")

        for name, results in [('Positive', positive_results), ('Negative', negative_results),
                              ('Objective', objective_results)]:
            avg_evidence = results['evidence'][f'avg_{name.lower()}_evidence']
            correlation = results['evidence']['correlation_evidence_disagreement']
            f.write(f"| {name} | {avg_evidence:.2f} | {correlation:.3f} |\n")

        # Justification similarity if available
        f.write("\n## Justification Similarity (if computed)\n\n")
        if 'justification' in positive_results:
            f.write("| Perspective | Mean Similarity | Similarity When Agree | Similarity When Disagree |\n")
            f.write("|-------------|-----------------|----------------------|-------------------------|\n")

            for name, results in [('Positive', positive_results), ('Negative', negative_results),
                                  ('Objective', objective_results)]:
                if 'justification' in results:
                    mean_sim = results['justification']['mean_justification_similarity']
                    agree_sim = results['justification'].get('mean_similarity_when_agree', 'N/A')
                    disagree_sim = results['justification'].get('mean_similarity_when_disagree', 'N/A')

                    agree_str = f"{agree_sim:.3f}" if agree_sim != 'N/A' else 'N/A'
                    disagree_str = f"{disagree_sim:.3f}" if disagree_sim != 'N/A' else 'N/A'

                    f.write(f"| {name} | {mean_sim:.3f} | {agree_str} | {disagree_str} |\n")


def main():
    parser = argparse.ArgumentParser(description='Compare multi-perspective and baseline prediction systems')
    parser.add_argument('--positive_file', required=True, help='Path to positive perspective results JSON')
    parser.add_argument('--negative_file', required=True, help='Path to negative perspective results JSON')
    parser.add_argument('--objective_file', required=True, help='Path to objective perspective results JSON')
    parser.add_argument('--baseline_file', required=True, help='Path to baseline results JSON')
    parser.add_argument('--gold_file', help='Path to gold data for correctness evaluation (optional)')
    parser.add_argument('--output_dir', default='multi_perspective_results', help='Directory for output files')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computing embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for computing embeddings')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2',
                        help='Name of the sentence transformer model to use')
    parser.add_argument('--skip_justification', action='store_true', help='Skip justification similarity analysis')
    args = parser.parse_args()

    # Time the execution
    start_time = time.time()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load gold data if provided
    gold_data = None
    if args.gold_file:
        print("Loading gold data...")
        gold_data = load_predictions(args.gold_file)

    # Initialize embedder if justification analysis is requested
    embedder = None
    if not args.skip_justification:
        try:
            print(f"Loading sentence transformer model: {args.embedding_model}...")
            embedder = SentenceTransformer(args.embedding_model)
        except Exception as e:
            print(f"Failed to load embedder: {e}. Skipping justification analysis.")

    # Analyze each perspective
    perspectives = [
        (args.positive_file, 'positive'),
        (args.negative_file, 'negative'),
        (args.objective_file, 'objective')
    ]

    all_results = {}

    for perspective_file, perspective_name in perspectives:
        results = analyze_perspective(
            perspective_file, args.baseline_file, perspective_name,
            args.output_dir, gold_data, embedder, args
        )
        all_results[perspective_name] = results

    # Create summary comparison
    print("\nCreating summary comparison...")
    create_summary_comparison(
        all_results['positive'],
        all_results['negative'],
        all_results['objective'],
        args.output_dir
    )

    # Save combined results
    with open(os.path.join(args.output_dir, 'all_perspectives_raw_metrics.json'), 'w', encoding='utf-8') as f:
        serializable_results = make_json_serializable(all_results)
        json.dump(serializable_results, f, indent=2)

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nMulti-perspective analysis complete! Results saved to {args.output_dir}")
    print(f"Total execution time: {execution_time:.2f} seconds")

    # Print summary statistics
    print("\n=== Summary ===")
    for perspective_name in ['positive', 'negative', 'objective']:
        agreement = all_results[perspective_name]['agreement']['overall_agreement']
        print(f"{perspective_name.capitalize()} agreement with baseline: {agreement:.2%}")


if __name__ == "__main__":
    main()