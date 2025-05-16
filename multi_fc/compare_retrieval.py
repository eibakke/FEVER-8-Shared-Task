import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from urllib.parse import urlparse
from tqdm import tqdm


def load_retrieval_data(file_paths):
    """Load retrieval data from multiple bias conditions."""
    data = {}
    for bias_type, file_path in file_paths.items():
        print(f"Loading {bias_type} retrieval data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data[bias_type] = json.load(f)
            except json.JSONDecodeError:
                # Try reading line by line for JSONL format
                data[bias_type] = []
                f.seek(0)
                for line in f:
                    data[bias_type].append(json.loads(line))
    return data


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"


def create_comparison_dataframe(retrieval_data, top_k=10):
    """Create DataFrame for comparing retrieved documents across bias conditions."""
    claim_ids = set()
    for bias_type, data in retrieval_data.items():
        for item in data:
            claim_ids.add(item['claim_id'])

    records = []
    for claim_id in tqdm(sorted(claim_ids), desc="Creating comparison DataFrame"):
        for bias_type, data in retrieval_data.items():
            item = next((item for item in data if item['claim_id'] == claim_id), None)
            if item is None:
                continue

            claim = item['claim']
            top_k_key = next((key for key in item if key.startswith('top_')), None)

            if top_k_key is None:
                continue

            # Get top k documents
            retrieved_docs = item[top_k_key][:top_k]

            for rank, doc in enumerate(retrieved_docs):
                records.append({
                    'claim_id': claim_id,
                    'claim': claim,
                    'bias_type': bias_type,
                    'rank': rank + 1,
                    'sentence': doc['sentence'],
                    'url': doc['url'],
                    'domain': extract_domain(doc['url'])
                })

    return pd.DataFrame(records)


def calculate_document_overlap(df, output_dir):
    """Calculate document overlap between bias conditions."""
    print("Calculating document overlap...")

    # Calculate overlap metrics for each claim
    claim_ids = df['claim_id'].unique()
    overlap_stats = []

    bias_types = df['bias_type'].unique()

    for claim_id in tqdm(claim_ids, desc="Analyzing document overlap"):
        claim_df = df[df['claim_id'] == claim_id]

        # Create sets of documents for each bias type
        doc_sets = {}
        for bias_type in bias_types:
            bias_docs = claim_df[claim_df['bias_type'] == bias_type]
            if len(bias_docs) > 0:
                doc_sets[bias_type] = set(bias_docs['sentence'].values)

        # Calculate pairwise Jaccard similarity
        for i, bias1 in enumerate(bias_types):
            if bias1 not in doc_sets:
                continue

            for bias2 in bias_types[i + 1:]:
                if bias2 not in doc_sets:
                    continue

                set1 = doc_sets[bias1]
                set2 = doc_sets[bias2]

                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard = intersection / union if union > 0 else 0

                overlap_stats.append({
                    'claim_id': claim_id,
                    'bias1': bias1,
                    'bias2': bias2,
                    'intersection': intersection,
                    'union': union,
                    'jaccard': jaccard
                })

    overlap_df = pd.DataFrame(overlap_stats)

    # Calculate average Jaccard across all claims
    avg_jaccard = overlap_df.groupby(['bias1', 'bias2'])['jaccard'].mean().reset_index()

    # Plot average Jaccard similarity as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(avg_jaccard.apply(lambda x: f"{x['bias1']}-{x['bias2']}", axis=1), avg_jaccard['jaccard'])
    plt.title('Average Jaccard Similarity Between Bias Conditions')
    plt.xlabel('Bias Pair')
    plt.ylabel('Jaccard Similarity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jaccard_similarity.png'), dpi=300)
    plt.close()

    return overlap_df


def calculate_rank_correlation(df, output_dir):
    """Calculate rank correlation for documents across bias conditions."""
    print("Calculating rank correlation...")

    bias_types = df['bias_type'].unique()
    claim_ids = df['claim_id'].unique()

    # Calculate rank correlation for each claim
    corr_stats = []

    for claim_id in tqdm(claim_ids, desc="Analyzing rank correlation"):
        claim_df = df[df['claim_id'] == claim_id]

        # Get documents shared between different bias types
        for i, bias1 in enumerate(bias_types):
            if bias1 not in claim_df['bias_type'].values:
                continue

            for bias2 in bias_types[i + 1:]:
                if bias2 not in claim_df['bias_type'].values:
                    continue

                docs1 = claim_df[claim_df['bias_type'] == bias1]
                docs2 = claim_df[claim_df['bias_type'] == bias2]

                # Find common documents
                common_docs = set(docs1['sentence']) & set(docs2['sentence'])

                if len(common_docs) < 3:  # Need at least 3 for meaningful correlation
                    continue

                # Get ranks for common documents
                ranks1 = []
                ranks2 = []

                for doc in common_docs:
                    rank1 = docs1[docs1['sentence'] == doc]['rank'].iloc[0]
                    rank2 = docs2[docs2['sentence'] == doc]['rank'].iloc[0]
                    ranks1.append(rank1)
                    ranks2.append(rank2)

                # Calculate Spearman and Kendall rank correlations
                spearman_corr, spearman_p = spearmanr(ranks1, ranks2)
                kendall_corr, kendall_p = kendalltau(ranks1, ranks2)

                corr_stats.append({
                    'claim_id': claim_id,
                    'bias1': bias1,
                    'bias2': bias2,
                    'common_docs': len(common_docs),
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'kendall_corr': kendall_corr,
                    'kendall_p': kendall_p
                })

    if not corr_stats:
        print("No sufficient overlap found for rank correlation analysis.")
        return None

    corr_df = pd.DataFrame(corr_stats)

    # Calculate average correlations
    avg_corr = corr_df.groupby(['bias1', 'bias2'])[['spearman_corr', 'kendall_corr']].mean().reset_index()

    # Plot average Spearman correlation
    plt.figure(figsize=(10, 6))
    plt.bar(avg_corr.apply(lambda x: f"{x['bias1']}-{x['bias2']}", axis=1), avg_corr['spearman_corr'])
    plt.title('Average Spearman Rank Correlation Between Bias Conditions')
    plt.xlabel('Bias Pair')
    plt.ylabel('Spearman Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spearman_correlation.png'), dpi=300)
    plt.close()

    return corr_df


def analyze_domain_distribution(df, output_dir):
    """Analyze the distribution of domains in retrieved documents."""
    print("Analyzing domain distribution...")

    # Count domains per bias type
    domain_counts = df.groupby(['bias_type', 'domain']).size().reset_index(name='count')

    # Get top 10 domains for each bias type
    top_domains = {}
    for bias_type in df['bias_type'].unique():
        bias_domains = domain_counts[domain_counts['bias_type'] == bias_type]
        top_domains[bias_type] = bias_domains.nlargest(10, 'count')

    # Plot domain distribution for each bias type
    for bias_type, domains in top_domains.items():
        plt.figure(figsize=(12, 6))
        plt.bar(domains['domain'], domains['count'])
        plt.title(f'Top 10 Domains for {bias_type.capitalize()} Bias')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'domains_{bias_type}.png'), dpi=300)
        plt.close()

    return top_domains


def main():
    parser = argparse.ArgumentParser(description='Compare retrieved documents from different bias conditions')
    parser.add_argument('--positive_file', required=True, help='Path to positive bias retrieval results')
    parser.add_argument('--negative_file', required=True, help='Path to negative bias retrieval results')
    parser.add_argument('--objective_file', required=True, help='Path to objective bias retrieval results')
    parser.add_argument('--output_dir', default='retrieval_comparison_results', help='Directory for output files')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top documents to analyze')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # File paths for different bias conditions
    file_paths = {
        'positive': args.positive_file,
        'negative': args.negative_file,
        'objective': args.objective_file
    }

    # Load retrieval data
    retrieval_data = load_retrieval_data(file_paths)

    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(retrieval_data, top_k=args.top_k)

    # Save the comparison dataframe
    comparison_df.to_csv(os.path.join(args.output_dir, 'retrieval_comparison.csv'), index=False)

    # Generate statistics
    overlap_df = calculate_document_overlap(comparison_df, args.output_dir)
    rank_corr_df = calculate_rank_correlation(comparison_df, args.output_dir)
    domain_analysis = analyze_domain_distribution(comparison_df, args.output_dir)

    # Write summary report
    with open(os.path.join(args.output_dir, 'retrieval_comparison_report.md'), 'w') as f:
        f.write("# Retrieval Comparison Report\n\n")

        f.write("## Overview\n\n")
        total_claims = comparison_df['claim_id'].nunique()
        total_documents = len(comparison_df)
        f.write(f"Total claims analyzed: {total_claims}\n")
        f.write(f"Total documents retrieved: {total_documents}\n\n")

        f.write("## Document Overlap\n\n")
        if overlap_df is not None:
            avg_overlap = overlap_df.groupby(['bias1', 'bias2'])['jaccard'].mean().reset_index()
            f.write("Average Jaccard similarity between bias conditions:\n\n")
            f.write("| Bias 1 | Bias 2 | Jaccard Similarity |\n")
            f.write("|--------|--------|--------------------|\n")
            for _, row in avg_overlap.iterrows():
                f.write(f"| {row['bias1']} | {row['bias2']} | {row['jaccard']:.4f} |\n")

        f.write("\n## Rank Correlation\n\n")
        if rank_corr_df is not None:
            avg_corr = rank_corr_df.groupby(['bias1', 'bias2'])[['spearman_corr', 'kendall_corr']].mean().reset_index()
            f.write("Average rank correlation between bias conditions:\n\n")
            f.write("| Bias 1 | Bias 2 | Spearman | Kendall |\n")
            f.write("|--------|--------|----------|----------|\n")
            for _, row in avg_corr.iterrows():
                f.write(
                    f"| {row['bias1']} | {row['bias2']} | {row['spearman_corr']:.4f} | {row['kendall_corr']:.4f} |\n")

        f.write("\n## Domain Analysis\n\n")
        f.write("Top domains by bias type:\n\n")
        for bias_type, domains in domain_analysis.items():
            f.write(f"### {bias_type.capitalize()}\n\n")
            f.write("| Domain | Count |\n")
            f.write("|--------|-------|\n")
            for _, row in domains.iterrows():
                f.write(f"| {row['domain']} | {row['count']} |\n")
            f.write("\n")

        f.write("\n## Conclusion\n\n")
        f.write(
            "This report provides a basic analysis of how different bias conditions in hypothetical document generation affect document retrieval. ")
        f.write(
            "The key findings include differences in document overlap, ranking patterns, and domain distributions across bias conditions.")

    print(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
