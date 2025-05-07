#!/usr/bin/env python3
import json
import argparse
import os
import csv
import random
from collections import defaultdict
import pprint

def load_json_file(filepath):
    """Load a JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Try loading line by line if the file contains JSON lines
        results = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {filepath}")
                    continue
        return results
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None

def analyze_pipeline_outputs(system_name, split, data_store):
    """Analyze outputs from each step of the pipeline."""
    results = {}
    
    # Define file paths for each step
    file_paths = {
        "hyde_fc": f"{data_store}/{system_name}/{split}_hyde_fc.json",
        "retrieval": f"{data_store}/{system_name}/{split}_retrieval_top_k.json",
        "reranking": f"{data_store}/{system_name}/{split}_reranking_top_k.json",
        "questions": f"{data_store}/{system_name}/{split}_top_k_qa.json",
        "veracity": f"{data_store}/{system_name}/{split}_veracity_prediction.json"
    }
    
    # Load each file
    for step, path in file_paths.items():
        data = load_json_file(path)
        if data:
            results[step] = data
            print(f"Loaded {len(data)} entries from {step}")
        else:
            print(f"Warning: Could not load data from {step}")
    
    return results

def get_claim_by_id(claim_id, pipeline_results, verbose=False):
    """Get all information for a specific claim across pipeline stages."""
    claim_info = {}
    
    # Extract information for the claim from each step
    for step, data in pipeline_results.items():
        # Find the entry for this claim
        entry = next((item for item in data if str(item.get('claim_id', '')) == str(claim_id)), None)
        
        # If no entry with claim_id, try using just the index in hyde_fc
        if entry is None and step == 'hyde_fc':
            # In hyde_fc, sometimes claims don't have claim_id - they're just indexed
            try:
                index = int(claim_id)
                if index < len(data):
                    entry = data[index]
            except (ValueError, IndexError):
                pass
        
        if entry:
            claim_info[step] = entry
            if verbose:
                print(f"\nFound {step} entry for claim {claim_id}")
                print(f"Keys: {list(entry.keys())}")
                if 'hypo_fc_docs' in entry:
                    print(f"  hypo_fc_docs is present at top level with {len(entry['hypo_fc_docs'])} items")
    
    return claim_info

def get_sorted_claims_by_label(pipeline_results):
    """Get claims sorted by their verdict label."""
    if 'veracity' not in pipeline_results:
        return {}
    
    claims_by_label = defaultdict(list)
    
    for entry in pipeline_results['veracity']:
        claim_id = entry.get('claim_id')
        label = entry.get('pred_label', 'Unknown')
        claims_by_label[label].append(claim_id)
    
    return claims_by_label

def find_hypo_fc_docs(claim_info, verbose=False):
    """Find hypothetical FC documents wherever they may be in the claim data."""
    # First, check if hypo_fc_docs is in hyde_fc
    if 'hyde_fc' in claim_info and 'hypo_fc_docs' in claim_info['hyde_fc']:
        if verbose:
            print(f"Found hypo_fc_docs in hyde_fc with {len(claim_info['hyde_fc']['hypo_fc_docs'])} items")
        return claim_info['hyde_fc']['hypo_fc_docs']
    
    # Then check if hypo_fc_docs is in retrieval
    if 'retrieval' in claim_info and 'hypo_fc_docs' in claim_info['retrieval']:
        if verbose:
            print(f"Found hypo_fc_docs in retrieval with {len(claim_info['retrieval']['hypo_fc_docs'])} items")
        return claim_info['retrieval']['hypo_fc_docs']
    
    # Finally, just search for it anywhere in the claim_info
    for stage, data in claim_info.items():
        if isinstance(data, dict) and 'hypo_fc_docs' in data:
            if verbose:
                print(f"Found hypo_fc_docs in {stage} with {len(data['hypo_fc_docs'])} items")
            return data['hypo_fc_docs']
    
    if verbose:
        print("Could not find hypo_fc_docs anywhere in claim data")
    return []

def generate_analysis_csv(pipeline_results, output_file, samples_per_label=5, verbose=False):
    """Generate a CSV file with detailed analysis of sampled claims."""
    claims_by_label = get_sorted_claims_by_label(pipeline_results)
    
    # Ensure all output directories exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'claim_id', 
            'label', 
            'claim_text',
            'hyde_fc_doc_1',
            'hyde_fc_doc_2',
            'reranked_sentence_1',
            'reranked_sentence_2',
            'reranked_sentence_3',
            'question_1',
            'answer_1',
            'question_2',
            'answer_2',
            'question_3',
            'answer_3',
            'justification'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sample claims from each label category
        all_samples = []
        for label, claim_ids in claims_by_label.items():
            num_samples = min(samples_per_label, len(claim_ids))
            if num_samples > 0:
                sampled_ids = random.sample(claim_ids, num_samples)
                for claim_id in sampled_ids:
                    claim_info = get_claim_by_id(claim_id, pipeline_results, verbose)
                    if claim_info:
                        all_samples.append((label, claim_id, claim_info))
        
        # Write samples to CSV
        for label, claim_id, claim_info in all_samples:
            # Extract claim text from any available source
            claim_text = ""
            for stage in ["hyde_fc", "retrieval", "reranking", "questions", "veracity"]:
                if stage in claim_info and "claim" in claim_info[stage]:
                    claim_text = claim_info[stage]["claim"]
                    break
            
            # Find and extract hypo_fc_docs wherever they may be
            hyde_fc_docs = find_hypo_fc_docs(claim_info, verbose)
            
            hyde_fc_doc_1 = hyde_fc_docs[0] if len(hyde_fc_docs) > 0 else ""
            hyde_fc_doc_2 = hyde_fc_docs[1] if len(hyde_fc_docs) > 1 else ""
            
            # Extract reranked sentences
            reranked_sentences = []
            if 'reranking' in claim_info and f'top_10' in claim_info['reranking']:
                for sent in claim_info['reranking'][f'top_10'][:3]:
                    reranked_sentences.append(sent['sentence'])
            
            while len(reranked_sentences) < 3:
                reranked_sentences.append("")
            
            # Extract questions and answers
            qa_pairs = []
            if 'questions' in claim_info and 'evidence' in claim_info['questions']:
                for qa in claim_info['questions']['evidence'][:3]:
                    qa_pairs.append((qa['question'], qa['answer']))
            
            while len(qa_pairs) < 3:
                qa_pairs.append(("", ""))
            
            # Extract justification
            justification = claim_info.get('veracity', {}).get('llm_output', '')
            
            # Write row
            row = {
                'claim_id': claim_id,
                'label': label,
                'claim_text': claim_text,
                'hyde_fc_doc_1': hyde_fc_doc_1,
                'hyde_fc_doc_2': hyde_fc_doc_2,
                'reranked_sentence_1': reranked_sentences[0],
                'reranked_sentence_2': reranked_sentences[1],
                'reranked_sentence_3': reranked_sentences[2],
                'question_1': qa_pairs[0][0],
                'answer_1': qa_pairs[0][1],
                'question_2': qa_pairs[1][0],
                'answer_2': qa_pairs[1][1],
                'question_3': qa_pairs[2][0],
                'answer_3': qa_pairs[2][1],
                'justification': justification
            }
            
            writer.writerow(row)
    
    print(f"Analysis CSV file generated: {output_file}")

def generate_analysis_markdown(pipeline_results, output_file, samples_per_label=5, verbose=False):
    """Generate a markdown file with detailed analysis of sampled claims."""
    claims_by_label = get_sorted_claims_by_label(pipeline_results)
    
    # Ensure all output directories exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as mdfile:
        mdfile.write("# AVeriTeC Pipeline Qualitative Analysis\n\n")
        
        # Add table of contents
        mdfile.write("## Table of Contents\n\n")
        for label in claims_by_label.keys():
            mdfile.write(f"- [{label}](#{label.lower().replace(' ', '-').replace('/', '-')})\n")
        mdfile.write("\n")
        
        # Sample claims from each label category
        for label, claim_ids in claims_by_label.items():
            mdfile.write(f"## {label}\n\n")
            
            num_samples = min(samples_per_label, len(claim_ids))
            if num_samples > 0:
                sampled_ids = random.sample(claim_ids, num_samples)
                
                for i, claim_id in enumerate(sampled_ids):
                    claim_info = get_claim_by_id(claim_id, pipeline_results, verbose)
                    if not claim_info:
                        continue
                    
                    # Extract claim text from any available source
                    claim_text = ""
                    for stage in ["hyde_fc", "retrieval", "reranking", "questions", "veracity"]:
                        if stage in claim_info and "claim" in claim_info[stage]:
                            claim_text = claim_info[stage]["claim"]
                            break
                    
                    mdfile.write(f"### Sample {i+1}: Claim {claim_id}\n\n")
                    mdfile.write(f"**Claim**: {claim_text}\n\n")
                    
                    # Find and extract hypo_fc_docs wherever they may be
                    mdfile.write("#### Hypothetical Fact-Checking Documents\n\n")
                    hyde_fc_docs = find_hypo_fc_docs(claim_info, verbose)
                    
                    if hyde_fc_docs:
                        for j, doc in enumerate(hyde_fc_docs[:3]):
                            mdfile.write(f"**Document {j+1}**:\n\n```\n{doc}\n```\n\n")
                    else:
                        mdfile.write("No hypothetical FC documents available.\n\n")
                    
                    # Write reranked sentences
                    mdfile.write("#### Top Reranked Sentences\n\n")
                    if 'reranking' in claim_info and f'top_10' in claim_info['reranking']:
                        for j, sent in enumerate(claim_info['reranking'][f'top_10'][:5]):
                            mdfile.write(f"**Sentence {j+1}**: {sent['sentence']}\n\n")
                            mdfile.write(f"Source: {sent['url']}\n\n")
                    else:
                        mdfile.write("No reranked sentences available.\n\n")
                    
                    # Write questions and answers
                    mdfile.write("#### Generated Questions and Answers\n\n")
                    if 'questions' in claim_info and 'evidence' in claim_info['questions']:
                        for j, qa in enumerate(claim_info['questions']['evidence'][:5]):
                            mdfile.write(f"**Q{j+1}**: {qa['question']}\n\n")
                            mdfile.write(f"**A{j+1}**: {qa['answer']}\n\n")
                            mdfile.write(f"Source: {qa.get('url', 'N/A')}\n\n")
                    else:
                        mdfile.write("No questions and answers available.\n\n")
                    
                    # Write veracity prediction
                    mdfile.write("#### Veracity Prediction\n\n")
                    mdfile.write(f"**Label**: {label}\n\n")
                    mdfile.write("**Justification**:\n\n")
                    mdfile.write(f"```\n{claim_info.get('veracity', {}).get('llm_output', 'No justification provided')}\n```\n\n")
                    
                    mdfile.write("---\n\n")
            else:
                mdfile.write(f"No claims with label '{label}' found.\n\n")
    
    print(f"Analysis markdown file generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze fact-checking pipeline outputs')
    parser.add_argument('--system', default='baseline', help='System name (default: baseline)')
    parser.add_argument('--split', default='dev', help='Data split (default: dev)')
    parser.add_argument('--data-store', default='./data_store', help='Path to data store directory')
    parser.add_argument('--output', default='analysis_results.md', help='Output file path')
    parser.add_argument('--format', choices=['csv', 'markdown'], default='markdown', help='Output format')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples per label')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--dump-entry', action='store_true', help='Dump first entry from each dataset')
    
    args = parser.parse_args()
    
    # Analyze pipeline outputs
    pipeline_results = analyze_pipeline_outputs(args.system, args.split, args.data_store)
    
    if not pipeline_results:
        print("No results to analyze. Check file paths and try again.")
        return
    
    # Debug option to print structure of first item in each dataset
    if args.verbose:
        print("\nDEBUG: Data structure samples")
        for step, data in pipeline_results.items():
            if data and len(data) > 0:
                print(f"\n{step} structure:")
                keys = list(data[0].keys())
                print(f"  Keys: {keys}")
                # Print hypo_fc_docs if it exists
                if 'hypo_fc_docs' in keys:
                    print(f"  hypo_fc_docs is at the top level (correct)")
                    if args.dump_entry:
                        print(f"  First hypo_fc_doc: {data[0]['hypo_fc_docs'][0][:200]}...")
    
    # Generate analysis file
    if args.format == 'csv':
        output_file = args.output if args.output.endswith('.csv') else args.output + '.csv'
        generate_analysis_csv(pipeline_results, output_file, args.samples, args.verbose)
    else:
        output_file = args.output if args.output.endswith('.md') else args.output + '.md'
        generate_analysis_markdown(pipeline_results, output_file, args.samples, args.verbose)

if __name__ == "__main__":
    main()

