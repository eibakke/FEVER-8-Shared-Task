import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from venn import venn
import torch
from scipy.stats import spearmanr, kendalltau
from wordcloud import WordCloud
from transformers import pipeline
from urllib.parse import urlparse

# Download NLTK dependencies
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


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

    # Create heatmap of average Jaccard similarity
    plt.figure(figsize=(10, 8))
    heatmap_data = pd.pivot_table(
        avg_jaccard,
        values='jaccard',
        index='bias1',
        columns='bias2'
    )

    # Ensure symmetry in the heatmap
    for bias1 in bias_types:
        for bias2 in bias_types:
            if bias1 == bias2:
                if (bias1, bias2) not in heatmap_data.values:
                    heatmap_data.loc[bias1, bias2] = 1.0
            elif (bias1, bias2) in avg_jaccard[['bias1', 'bias2']].values:
                jaccard_val = avg_jaccard[(avg_jaccard['bias1'] == bias1) &
                                          (avg_jaccard['bias2'] == bias2)]['jaccard'].values[0]
                if pd.isna(heatmap_data.loc[bias1, bias2]):
                    heatmap_data.loc[bias1, bias2] = jaccard_val
                if bias2 in heatmap_data.index and bias1 in heatmap_data.columns:
                    heatmap_data.loc[bias2, bias1] = jaccard_val

    # Fill diagonal with 1.0 (perfect similarity with self)
    for bias in bias_types:
        if bias in heatmap_data.index and bias in heatmap_data.columns:
            heatmap_data.loc[bias, bias] = 1.0

    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('Average Jaccard Similarity Between Bias Conditions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jaccard_similarity.png'), dpi=300)
    plt.close()

    # Create venn diagrams for a sample of claims
    sample_claims = min(5, len(claim_ids))
    sampled_claim_ids = np.random.choice(claim_ids, sample_claims, replace=False)

    for claim_id in sampled_claim_ids:
        claim_df = df[df['claim_id'] == claim_id]
        claim_text = claim_df['claim'].iloc[0]

        # Create sets of documents for each bias type
        doc_sets = {}
        for bias_type in bias_types:
            bias_docs = claim_df[claim_df['bias_type'] == bias_type]
            if len(bias_docs) > 0:
                doc_sets[bias_type] = set(bias_docs['sentence'].values)

        if len(doc_sets) < 2:
            continue

        plt.figure(figsize=(12, 8))
        venn_plot = venn(doc_sets, cmap=plt.cm.tab10)
        plt.title(f'Document Overlap for Claim {claim_id}:\n{claim_text[:100]}...')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'venn_claim_{claim_id}.png'), dpi=300)
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

    # Create heatmap of average Spearman correlation
    plt.figure(figsize=(10, 8))
    spearman_heatmap = pd.pivot_table(
        avg_corr,
        values='spearman_corr',
        index='bias1',
        columns='bias2'
    )

    # Ensure symmetry in the heatmap
    for bias1 in bias_types:
        for bias2 in bias_types:
            if bias1 == bias2:
                if pd.isna(spearman_heatmap.loc[
                               bias1, bias2]) if bias1 in spearman_heatmap.index and bias2 in spearman_heatmap.columns else True:
                    try:
                        spearman_heatmap.loc[bias1, bias2] = 1.0
                    except:
                        pass
            elif (bias1, bias2) in avg_corr[['bias1', 'bias2']].values:
                corr_val = avg_corr[(avg_corr['bias1'] == bias1) &
                                    (avg_corr['bias2'] == bias2)]['spearman_corr'].values[0]
                try:
                    if pd.isna(spearman_heatmap.loc[bias1, bias2]):
                        spearman_heatmap.loc[bias1, bias2] = corr_val
                    if bias2 in spearman_heatmap.index and bias1 in spearman_heatmap.columns:
                        spearman_heatmap.loc[bias2, bias1] = corr_val
                except:
                    pass

    # Fill diagonal with 1.0 (perfect correlation with self)
    for bias in bias_types:
        if bias in spearman_heatmap.index and bias in spearman_heatmap.columns:
            spearman_heatmap.loc[bias, bias] = 1.0

    sns.heatmap(spearman_heatmap, annot=True, cmap="YlGnBu", vmin=-1, vmax=1)
    plt.title('Average Spearman Rank Correlation Between Bias Conditions')
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
        ax = sns.barplot(x='domain', y='count', data=domains, palette='viridis')
        plt.title(f'Top 10 Domains for {bias_type.capitalize()} Bias')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'domains_{bias_type}.png'), dpi=300)
        plt.close()

    # Calculate domain overlap between bias types
    bias_types = df['bias_type'].unique()
    domain_sets = {}

    for bias_type in bias_types:
        bias_domains = df[df['bias_type'] == bias_type]['domain'].unique()
        domain_sets[bias_type] = set(bias_domains)

    # Create Venn diagram of domain overlap
    plt.figure(figsize=(10, 8))
    venn_plot = venn(domain_sets, cmap=plt.cm.tab10)
    plt.title('Domain Overlap Between Bias Conditions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_overlap.png'), dpi=300)
    plt.close()

    return top_domains


def analyze_content_similarity(df, output_dir, use_gpu=True):
    """Analyze semantic similarity of retrieved content."""
    print("Analyzing content similarity...")

    # Initialize sentence transformer model
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Sample claims for detailed analysis (to avoid processing all claims)
    sample_size = min(50, df['claim_id'].nunique())
    sampled_claims = np.random.choice(df['claim_id'].unique(), sample_size, replace=False)

    # Calculate content similarity for sampled claims
    similarity_stats = []

    for claim_id in tqdm(sampled_claims, desc="Computing semantic similarity"):
        claim_df = df[df['claim_id'] == claim_id]
        claim_text = claim_df['claim'].iloc[0]
        bias_types = claim_df['bias_type'].unique()

        # Group sentences by bias type
        bias_sentences = {}
        for bias_type in bias_types:
            bias_docs = claim_df[claim_df['bias_type'] == bias_type]
            bias_sentences[bias_type] = bias_docs['sentence'].tolist()

        # Calculate embeddings
        all_sentences = []
        bias_indices = {}
        current_idx = 0

        for bias_type, sentences in bias_sentences.items():
            bias_indices[bias_type] = (current_idx, current_idx + len(sentences))
            all_sentences.extend(sentences)
            current_idx += len(sentences)

        # Embed all sentences (batched processing)
        batch_size = 32
        embeddings = []

        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)

        if embeddings:
            all_embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

            # Calculate similarity between bias conditions
            for i, bias1 in enumerate(bias_types):
                start1, end1 = bias_indices[bias1]
                if end1 <= start1:  # Skip if empty
                    continue

                embeddings1 = all_embeddings[start1:end1]

                for bias2 in bias_types[i + 1:]:
                    start2, end2 = bias_indices[bias2]
                    if end2 <= start2:  # Skip if empty
                        continue

                    embeddings2 = all_embeddings[start2:end2]

                    # Calculate average cosine similarity
                    sim_matrix = cosine_similarity(embeddings1, embeddings2)
                    avg_sim = np.mean(sim_matrix)
                    max_sim = np.max(sim_matrix)

                    similarity_stats.append({
                        'claim_id': claim_id,
                        'bias1': bias1,
                        'bias2': bias2,
                        'avg_similarity': avg_sim,
                        'max_similarity': max_sim
                    })

    if not similarity_stats:
        print("No content similarity statistics could be calculated.")
        return None

    similarity_df = pd.DataFrame(similarity_stats)

    # Calculate average similarity across all claims
    avg_similarity = similarity_df.groupby(['bias1', 'bias2'])[
        ['avg_similarity', 'max_similarity']].mean().reset_index()

    # Create heatmap of average similarity
    plt.figure(figsize=(10, 8))
    similarity_heatmap = pd.pivot_table(
        avg_similarity,
        values='avg_similarity',
        index='bias1',
        columns='bias2'
    )

    # Ensure a symmetric heatmap
    bias_types = df['bias_type'].unique()
    for bias1 in bias_types:
        for bias2 in bias_types:
            if bias1 == bias2:
                try:
                    similarity_heatmap.loc[bias1, bias2] = 1.0
                except:
                    pass
            elif (bias1, bias2) in avg_similarity[['bias1', 'bias2']].values:
                sim_val = avg_similarity[(avg_similarity['bias1'] == bias1) &
                                         (avg_similarity['bias2'] == bias2)]['avg_similarity'].values[0]
                try:
                    similarity_heatmap.loc[bias1, bias2] = sim_val
                    if bias2 in similarity_heatmap.index and bias1 in similarity_heatmap.columns:
                        similarity_heatmap.loc[bias2, bias1] = sim_val
                except:
                    pass

    sns.heatmap(similarity_heatmap, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('Average Content Similarity Between Bias Conditions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'content_similarity.png'), dpi=300)
    plt.close()

    return similarity_df


def analyze_sentiment(df, output_dir, sample_size=100):
    """Analyze sentiment of retrieved documents."""
    print("Analyzing sentiment...")

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )

    # Sample sentences for sentiment analysis
    if len(df) > sample_size:
        sampled_df = df.groupby('bias_type').apply(
            lambda x: x.sample(min(len(x), sample_size // len(df['bias_type'].unique()))))
        sampled_df = sampled_df.reset_index(drop=True)
    else:
        sampled_df = df

    # Process in batches
    batch_size = 32
    sentiment_results = []

    for bias_type in df['bias_type'].unique():
        bias_sentences = sampled_df[sampled_df['bias_type'] == bias_type]['sentence'].tolist()

        for i in tqdm(range(0, len(bias_sentences), batch_size), desc=f"Analyzing {bias_type} sentiment"):
            batch = bias_sentences[i:i + batch_size]
            try:
                results = sentiment_pipeline(batch)

                for j, result in enumerate(results):
                    idx = i + j
                    if idx < len(bias_sentences):
                        score = result['score']
                        if result['label'] == 'NEGATIVE':
                            score = -score

                        sentiment_results.append({
                            'bias_type': bias_type,
                            'sentence': bias_sentences[idx][:100] + "...",
                            'sentiment': result['label'],
                            'score': score
                        })
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                continue

    sentiment_df = pd.DataFrame(sentiment_results)

    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='bias_type', y='score', data=sentiment_df, palette='viridis')
    plt.title('Sentiment Distribution by Bias Type')
    plt.xlabel('Bias Type')
    plt.ylabel('Sentiment Score (negative = NEGATIVE, positive = POSITIVE)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    plt.close()

    # Calculate average sentiment by bias type
    avg_sentiment = sentiment_df.groupby('bias_type')['score'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='bias_type', y='score', data=avg_sentiment, palette='viridis')
    plt.title('Average Sentiment by Bias Type')
    plt.xlabel('Bias Type')
    plt.ylabel('Average Sentiment Score')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_sentiment.png'), dpi=300)
    plt.close()

    return sentiment_df


def generate_document_wordclouds(df, output_dir):
    """Generate wordclouds for each bias type."""
    print("Generating wordclouds...")

    for bias_type in df['bias_type'].unique():
        bias_text = ' '.join(df[df['bias_type'] == bias_type]['sentence'].values)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(bias_text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {bias_type.capitalize()} Bias')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'wordcloud_{bias_type}.png'), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare retrieved documents from different bias conditions')
    parser.add_argument('--positive_file', required=True, help='Path to positive bias retrieval results')
    parser.add_argument('--negative_file', required=True, help='Path to negative bias retrieval results')
    parser.add_argument('--objective_file', required=True, help='Path to objective bias retrieval results')
    parser.add_argument('--output_dir', default='retrieval_comparison_results', help='Directory for output files')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top documents to analyze')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for semantic analysis')
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

    # Generate statistics and visualizations
    overlap_df = calculate_document_overlap(comparison_df, args.output_dir)
    rank_corr_df = calculate_rank_correlation(comparison_df, args.output_dir)
    domain_analysis = analyze_domain_distribution(comparison_df, args.output_dir)

    # More intensive analyses - can be disabled if needed
    content_sim_df = analyze_content_similarity(comparison_df, args.output_dir, use_gpu=args.use_gpu)
    sentiment_df = analyze_sentiment(comparison_df, args.output_dir)
    generate_document_wordclouds(comparison_df, args.output_dir)

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

        f.write("\n## Content Similarity\n\n")
        if content_sim_df is not None:
            avg_sim = content_sim_df.groupby(['bias1', 'bias2'])[
                ['avg_similarity', 'max_similarity']].mean().reset_index()
            f.write("Average semantic similarity between bias conditions:\n\n")
            f.write("| Bias 1 | Bias 2 | Average Similarity | Max Similarity |\n")
            f.write("|--------|--------|-------------------|----------------|\n")
            for _, row in avg_sim.iterrows():
                f.write(
                    f"| {row['bias1']} | {row['bias2']} | {row['avg_similarity']:.4f} | {row['max_similarity']:.4f} |\n")

        f.write("\n## Sentiment Analysis\n\n")
        if sentiment_df is not None:
            avg_sent = sentiment_df.groupby('bias_type')['score'].mean().reset_index()
            f.write("Average sentiment by bias type:\n\n")
            f.write("| Bias Type | Average Sentiment |\n")
            f.write("|-----------|-------------------|\n")
            for _, row in avg_sent.iterrows():
                f.write(f"| {row['bias_type']} | {row['score']:.4f} |\n")

        f.write("\n## Conclusion\n\n")
        f.write(
            "This report provides a comprehensive analysis of how different bias conditions in hypothetical document generation affect document retrieval. ")
        f.write(
            "The key findings include differences in document overlap, ranking patterns, domain distributions, content similarity, and sentiment across bias conditions.")

    print(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()