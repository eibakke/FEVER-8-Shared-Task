import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
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


def create_dataframe(direct_preds, baseline_preds, gold_data=None):
    """Create a pandas DataFrame for analysis comparing predictions."""
    # Create mapping from claim_id to predictions
    direct_dict = {item['claim_id']: item for item in direct_preds}
    baseline_dict = {item['claim_id']: item for item in baseline_preds}

    # Get all unique claim IDs
    all_ids = sorted(list(set(direct_dict.keys()) | set(baseline_dict.keys())))

    # Create dataframe rows
    rows = []
    for claim_id in all_ids:
        direct = direct_dict.get(claim_id, {})
        baseline = baseline_dict.get(claim_id, {})

        # Skip if we don't have both predictions
        if not direct or not baseline:
            continue

        row = {
            'claim_id': claim_id,
            'claim': direct.get('claim', ''),
            'direct_label': direct.get('pred_label', ''),
            'baseline_label': baseline.get('pred_label', ''),
            'direct_output': direct.get('llm_output', ''),
            'baseline_output': baseline.get('llm_output', ''),
            'direct_evidence_count': len(direct.get('evidence', [])),
            'baseline_evidence_count': len(baseline.get('evidence', [])),
        }

        # Add ground truth if available
        if gold_data:
            gold_item = next((item for item in gold_data if item['claim_id'] == claim_id), None)
            if gold_item:
                row['gold_label'] = gold_item.get('label', '')

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived columns
    df['agreement'] = df['direct_label'] == df['baseline_label']

    if 'gold_label' in df.columns:
        df['direct_correct'] = df['direct_label'] == df['gold_label']
        df['baseline_correct'] = df['baseline_label'] == df['gold_label']
        df['improvement'] = ~df['direct_correct'] & df['baseline_correct']
        df['degradation'] = df['direct_correct'] & ~df['baseline_correct']

    return df


def calculate_agreement_metrics(df):
    """Calculate agreement metrics between direct and baseline predictions."""
    results = {}

    # Overall agreement
    results['overall_agreement'] = df['agreement'].mean()

    # Agreement by direct label
    agreement_by_direct = df.groupby('direct_label')['agreement'].agg(['count', 'mean'])
    results['agreement_by_direct_label'] = agreement_by_direct.to_dict()

    # Agreement by baseline label
    agreement_by_baseline = df.groupby('baseline_label')['agreement'].agg(['count', 'mean'])
    results['agreement_by_baseline_label'] = agreement_by_baseline.to_dict()

    # Transition matrix (from direct to baseline)
    labels = sorted(list(set(df['direct_label'].unique()) | set(df['baseline_label'].unique())))
    transition_matrix = pd.crosstab(
        df['direct_label'],
        df['baseline_label'],
        normalize='index'
    ).round(3)

    results['transition_matrix'] = transition_matrix.to_dict()

    # Count specific transitions (for major label changes)
    transitions = df.groupby(['direct_label', 'baseline_label']).size().reset_index(name='count')
    total_changes = len(df[~df['agreement']])

    if total_changes > 0:
        # Calculate percentages of label flips
        for _, row in transitions.iterrows():
            if row['direct_label'] != row['baseline_label']:
                flip_key = f"flip_{row['direct_label']}_to_{row['baseline_label']}"
                results[flip_key] = {
                    'count': int(row['count']),
                    'percentage_of_changes': round(row['count'] / total_changes * 100, 2)
                }

    # Overall flip rate
    results['overall_flip_rate'] = 1.0 - results['overall_agreement']

    return results


def calculate_correctness_metrics(df):
    """Calculate metrics related to correctness and knowledge impact."""
    if 'gold_label' not in df.columns:
        return {"error": "Gold data not available for correctness metrics"}

    results = {}

    # Overall accuracy
    results['direct_accuracy'] = df['direct_correct'].mean()
    results['baseline_accuracy'] = df['baseline_correct'].mean()
    results['accuracy_delta'] = results['baseline_accuracy'] - results['direct_accuracy']

    # Correction analysis
    correction_opportunities = len(df[~df['direct_correct']])
    corrections_made = len(df[~df['direct_correct'] & df['baseline_correct']])

    results['correction_opportunities'] = correction_opportunities
    results['corrections_made'] = corrections_made

    if correction_opportunities > 0:
        results['correction_rate'] = corrections_made / correction_opportunities
    else:
        results['correction_rate'] = 0

    # Error introduction analysis
    error_opportunities = len(df[df['direct_correct']])
    errors_introduced = len(df[df['direct_correct'] & ~df['baseline_correct']])

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
        'direct_correct': 'mean',
        'baseline_correct': 'mean'
    }).reset_index()

    results['correctness_by_label'] = correctness_by_label.to_dict()

    # Accuracy for cases where direct and baseline agree vs. disagree
    agree_cases = df[df['agreement']]
    disagree_cases = df[~df['agreement']]

    if len(agree_cases) > 0:
        results['accuracy_when_agree'] = {
            'count': len(agree_cases),
            'direct_accuracy': agree_cases['direct_correct'].mean(),
            'baseline_accuracy': agree_cases['baseline_correct'].mean()
        }

    if len(disagree_cases) > 0:
        results['accuracy_when_disagree'] = {
            'count': len(disagree_cases),
            'direct_accuracy': disagree_cases['direct_correct'].mean(),
            'baseline_accuracy': disagree_cases['baseline_correct'].mean()
        }

    return results


def analyze_evidence_impact(df):
    """Analyze how evidence quantity impacts prediction changes."""
    results = {}

    # Average evidence counts
    results['avg_direct_evidence'] = df['direct_evidence_count'].mean()
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
    evidence_by_transition = df.groupby(['direct_label', 'baseline_label']).agg({
        'baseline_evidence_count': 'mean',
        'claim_id': 'count'
    }).reset_index()

    results['evidence_by_transition'] = evidence_by_transition.to_dict()

    return results


def analyze_justification_similarity(df, embedder, batch_size=32, use_gpu=True):
    """Analyze similarity between direct and baseline justifications."""
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
    print("Extracting justifications...")
    sample_df['direct_justification'] = sample_df['direct_output'].apply(extract_justification)
    sample_df['baseline_justification'] = sample_df['baseline_output'].apply(extract_justification)

    # Move model to GPU if available
    embedder = embedder.to(device)

    # Compute embeddings in batches
    print("Computing embeddings in batches...")
    direct_texts = sample_df['direct_justification'].tolist()
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
    direct_embeddings = compute_embeddings_batched(direct_texts)
    baseline_embeddings = compute_embeddings_batched(baseline_texts)

    # Compute similarities
    print("Computing similarities...")
    similarities = []
    for i in range(len(direct_embeddings)):
        similarity = 1 - cosine(direct_embeddings[i], baseline_embeddings[i])
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


def analyze_label_distribution(df):
    """Analyze how label distribution shifts between direct and baseline predictions."""
    results = {}

    # Count labels
    direct_counts = df['direct_label'].value_counts().to_dict()
    baseline_counts = df['baseline_label'].value_counts().to_dict()

    results['direct_label_counts'] = direct_counts
    results['baseline_label_counts'] = baseline_counts

    # Calculate percentages
    total = len(df)
    direct_percentages = {k: v / total for k, v in direct_counts.items()}
    baseline_percentages = {k: v / total for k, v in baseline_counts.items()}

    results['direct_label_percentages'] = direct_percentages
    results['baseline_label_percentages'] = baseline_percentages

    # Calculate shifts
    labels = sorted(list(set(direct_counts.keys()) | set(baseline_counts.keys())))
    shifts = {}

    for label in labels:
        direct_pct = direct_percentages.get(label, 0)
        baseline_pct = baseline_percentages.get(label, 0)
        shifts[label] = {
            'absolute_shift': baseline_pct - direct_pct,
            'relative_shift': (baseline_pct / direct_pct - 1) if direct_pct > 0 else float('inf')
        }

    results['label_shifts'] = shifts

    return results


def generate_visualizations(df, results, output_dir):
    """Generate visualizations from the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Confusion matrix between direct and baseline predictions
    plt.figure(figsize=(10, 8))
    labels = sorted(list(set(df['direct_label'].unique()) | set(df['baseline_label'].unique())))
    cm = confusion_matrix(
        df['direct_label'],
        df['baseline_label'],
        labels=labels
    )

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Baseline Predictions')
    plt.ylabel('Direct Predictions')
    plt.title('Prediction Changes: Direct vs. Knowledge-Based')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. Label distribution comparison
    plt.figure(figsize=(12, 6))
    labels = sorted(list(set(df['direct_label'].unique()) | set(df['baseline_label'].unique())))

    direct_counts = [df[df['direct_label'] == label].shape[0] for label in labels]
    baseline_counts = [df[df['baseline_label'] == label].shape[0] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, direct_counts, width, label='Direct')
    plt.bar(x + width / 2, baseline_counts, width, label='Knowledge-Based')

    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution Comparison')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=300)
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
    plt.title('Agreement Rate by Evidence Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evidence_vs_agreement.png'), dpi=300)
    plt.close()

    # 4. If gold data is available, accuracy comparison
    if 'gold_label' in df.columns:
        plt.figure(figsize=(10, 6))

        labels = sorted(df['gold_label'].unique())
        direct_accs = [df[df['gold_label'] == label]['direct_correct'].mean() for label in labels]
        baseline_accs = [df[df['gold_label'] == label]['baseline_correct'].mean() for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, direct_accs, width, label='Direct')
        plt.bar(x + width / 2, baseline_accs, width, label='Knowledge-Based')

        plt.xlabel('True Label')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Label')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_label.png'), dpi=300)
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
            plt.title('Knowledge Store Impact by Label')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'knowledge_impact_by_label.png'), dpi=300)
            plt.close()

    return


def write_metrics_to_file(results, output_path):
    """Write metrics to a formatted text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Comparison Analysis: Direct vs. Knowledge-Based Prediction\n\n")

        # Write agreement metrics
        f.write("## 1. Agreement Metrics\n\n")
        f.write(f"Overall agreement rate: {results['agreement']['overall_agreement']:.2%}\n")
        f.write(f"Overall flip rate: {results['agreement']['overall_flip_rate']:.2%}\n\n")

        f.write("### Agreement by direct prediction label:\n")
        for label, stats in results['agreement']['agreement_by_direct_label'].items():
            f.write(f"- {label}: {stats['mean']:.2%} (n={stats['count']})\n")
        f.write("\n")

        # Write transition matrix info
        f.write("### Transition Matrix (rows=direct, columns=knowledge-based):\n")
        for direct_label in results['agreement']['transition_matrix']:
            f.write(f"- From {direct_label}:\n")
            for baseline_label, value in results['agreement']['transition_matrix'][direct_label].items():
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
            f.write(f"Direct prediction accuracy: {results['correctness']['direct_accuracy']:.2%}\n")
            f.write(f"Knowledge-based accuracy: {results['correctness']['baseline_accuracy']:.2%}\n")
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
                f.write(f"  - Direct: {results['correctness']['accuracy_when_disagree']['direct_accuracy']:.2%}\n")
                f.write(
                    f"  - Knowledge-based: {results['correctness']['accuracy_when_disagree']['baseline_accuracy']:.2%}\n\n")

        # Write evidence impact metrics
        f.write("## 3. Evidence Analysis\n\n")
        f.write(f"Average evidence count in direct predictions: {results['evidence']['avg_direct_evidence']:.2f}\n")
        f.write(
            f"Average evidence count in knowledge-based predictions: {results['evidence']['avg_baseline_evidence']:.2f}\n\n")

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
        f.write("| Label | Direct | Knowledge-Based | Absolute Shift | Relative Shift |\n")
        f.write("|-------|--------|----------------|----------------|---------------|\n")

        all_labels = sorted(list(set(results['distribution']['direct_label_counts'].keys()) |
                                 set(results['distribution']['baseline_label_counts'].keys())))

        for label in all_labels:
            direct_count = results['distribution']['direct_label_counts'].get(label, 0)
            baseline_count = results['distribution']['baseline_label_counts'].get(label, 0)
            abs_shift = results['distribution']['label_shifts'][label]['absolute_shift']
            rel_shift = results['distribution']['label_shifts'][label]['relative_shift']

            if rel_shift == float('inf'):
                rel_shift_str = "∞"
            else:
                rel_shift_str = f"{rel_shift:.2%}"

            f.write(f"| {label} | {direct_count} | {baseline_count} | {abs_shift:.2%} | {rel_shift_str} |\n")


def main():
    parser = argparse.ArgumentParser(description='Compare direct and knowledge-based prediction systems')
    parser.add_argument('--direct_file', required=True, help='Path to direct prediction results JSON')
    parser.add_argument('--baseline_file', required=True, help='Path to baseline (knowledge-based) results JSON')
    parser.add_argument('--gold_file', help='Path to gold data for correctness evaluation (optional)')
    parser.add_argument('--output_dir', default='comparison_results', help='Directory for output files')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computing embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for computing embeddings')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2',
                        help='Name of the sentence transformer model to use')
    args = parser.parse_args()

    # Time the execution
    start_time = time.time()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading prediction data...")
    direct_preds = load_predictions(args.direct_file)
    baseline_preds = load_predictions(args.baseline_file)

    gold_data = None
    if args.gold_file:
        print("Loading gold data...")
        gold_data = load_predictions(args.gold_file)

    print("Creating analysis dataframe...")
    df = create_dataframe(direct_preds, baseline_preds, gold_data)

    # Calculate all metrics
    results = {}

    print("Calculating agreement metrics...")
    results['agreement'] = calculate_agreement_metrics(df)

    if gold_data:
        print("Calculating correctness metrics...")
        results['correctness'] = calculate_correctness_metrics(df)

    print("Analyzing evidence impact...")
    results['evidence'] = analyze_evidence_impact(df)

    print("Analyzing label distribution...")
    results['distribution'] = analyze_label_distribution(df)

    # Optional justification analysis with GPU support
    try:
        print(f"Loading sentence transformer model: {args.embedding_model}...")
        embedder = SentenceTransformer(args.embedding_model)

        print("Analyzing justification similarity...")
        results['justification'] = analyze_justification_similarity(
            df, embedder, batch_size=args.batch_size, use_gpu=args.use_gpu
        )
    except Exception as e:
        print(f"Skipping justification analysis due to error: {e}")

    # Save dataframe for further analysis
    df.to_csv(os.path.join(args.output_dir, 'comparison_data.csv'), index=False)

    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, results, args.output_dir)

    # Write metrics to file
    print("Writing metrics to file...")
    write_metrics_to_file(results, os.path.join(args.output_dir, 'comparison_metrics.md'))

    # Save raw results
    with open(os.path.join(args.output_dir, 'raw_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
