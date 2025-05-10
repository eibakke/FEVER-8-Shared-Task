#!/usr/bin/env python3
import os
import argparse
import json
import time
import pickle
import nltk
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi


# Ensure NLTK data is available
def download_nltk_data(package_name, download_dir='nltk_data'):
    os.makedirs(download_dir, exist_ok=True)
    nltk.data.path.append(download_dir)
    try:
        nltk.data.find(f'tokenizers/{package_name}')
        print(f"Package '{package_name}' is already downloaded")
    except LookupError:
        print(f"Downloading {package_name}...")
        nltk.download(package_name, download_dir=download_dir)
        print(f"Successfully downloaded {package_name}")


# Load sentences from knowledge file
def combine_all_sentences(knowledge_file):
    sentences, urls = [], []
    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            sentences.extend(data["url2text"])
            urls.extend([data["url"] for _ in range(len(data["url2text"]))])
    return sentences, urls, i + 1


# Remove duplicate sentences
def remove_duplicates(sentences, urls):
    df = pd.DataFrame({"document_in_sentences": sentences, "sentence_urls": urls})
    df['sentences'] = df['document_in_sentences'].str.strip().str.lower()
    df = df.drop_duplicates(subset="sentences").reset_index()
    return df['document_in_sentences'].tolist(), df['sentence_urls'].tolist()


# Precompute BM25 components for faster retrieval
def precompute_bm25(knowledge_store_dir, output_dir):
    """Precompute and save tokenized documents and BM25 components for all files."""
    os.makedirs(output_dir, exist_ok=True)

    # Get all knowledge files
    knowledge_files = [f for f in os.listdir(knowledge_store_dir) if f.endswith('.json')]

    for file_name in tqdm(knowledge_files, desc="Precomputing BM25 data"):
        file_id = file_name.split('.')[0]
        output_file = os.path.join(output_dir, f"{file_id}.pkl")

        # Skip if already processed
        if os.path.exists(output_file):
            continue

        # Load sentences
        knowledge_file = os.path.join(knowledge_store_dir, file_name)
        sentences, urls, _ = combine_all_sentences(knowledge_file)
        sentences, urls = remove_duplicates(sentences, urls)

        # Tokenize sentences
        tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in sentences]

        # Create BM25 object
        bm25 = BM25Okapi(tokenized_docs)

        # Save preprocessed data
        with open(output_file, 'wb') as f:
            pickle.dump({
                'sentences': sentences,
                'urls': urls,
                'tokenized_docs': tokenized_docs,
                'bm25_avgdl': bm25.avgdl,
                'bm25_corpus_size': bm25.corpus_size,
                'bm25_doc_freqs': bm25.doc_freqs,
                'bm25_doc_len': bm25.doc_len,
                'bm25_idf': bm25.idf
            }, f)

    print(f"BM25 preprocessing completed. Data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute data for BM25 and dense retrieval")
    parser.add_argument("--knowledge_store_dir", type=str, required=True,
                        help="Directory containing knowledge store files")
    parser.add_argument("--bm25_output_dir", type=str, default="precomputed_bm25",
                        help="Output directory for precomputed BM25 data")

    args = parser.parse_args()

    # Download required NLTK data
    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')

    # Precompute BM25 data
    start_time = time.time()
    print(f"Starting BM25 preprocessing for files in {args.knowledge_store_dir}...")
    precompute_bm25(args.knowledge_store_dir, args.bm25_output_dir)
    print(f"BM25 preprocessing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
