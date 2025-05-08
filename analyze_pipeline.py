#!/usr/bin/env python3
import json
import argparse
import os
import csv
import random
from collections import defaultdict


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


def load_pipeline_data(system_name, split, data_store):
    """Load data from a specific pipeline."""
    results = {}

    # Define file paths for each step
    file_paths = {
        "hyde_fc": f"{data_store}/{system_name}/{split}_hyde_fc.json",
        "multi_hyde_fc": f"{data_store}/{system_name}/{split}_multi_hyde_fc.json",  # For multi-perspective system
        "retrieval": f"{data_store}/{system_name}/{split}_retrieval_top_k.json",
        "reranking": f"{data_store}/{system_name}/{split}_reranking_top_k.json",
        "questions": f"{data_store}/{system_name}/{split}_top_k_qa.json",
        "merged_qa": f"{data_store}/{system_name}/{split}_merged_qa.json",  # For multi-perspective system
        "veracity": f"{data_store}/{system_name}/{split}_veracity_prediction.json"
    }

    # Load each file
    for step, path in file_paths.items():
        data = load_json_file(path)
        if data:
            results[step] = data
            print(f"Loaded {len(data)} entries from {system_name}/{step}")
        else:
            print(f"Note: File not found or empty: {path}")

    return results


def analyze_pipeline_outputs(systems, split, data_store, reference_file=None):
    """Analyze outputs from multiple pipeline systems."""
    results = {}

    # Load data for each system
    for system_name in systems:
        results[system_name] = load_pipeline_data(system_name, split, data_store)

    # Load reference data if provided
    if reference_file:
        reference_data = load_json_file(reference_file)
        if reference_data:
            results["reference"] = {"data": reference_data}
            print(f"Loaded {len(reference_data)} entries from reference file")
        else:
            print(f"Warning: Could not load reference data from {reference_file}")

    return results


def get_claim_by_id(claim_id, pipeline_results, verbose=False):
    """Get information for a specific claim across all systems."""
    claim_info = {}

    # Process each system
    for system_name, system_data in pipeline_results.items():
        if system_name == "reference":
            # For reference data, match by index rather than claim_id
            try:
                index = int(claim_id)
                if index < len(system_data["data"]):
                    claim_info[system_name] = system_data["data"][index]
            except (ValueError, IndexError):
                pass
            continue

        # For system data, extract from each step
        claim_info[system_name] = {}

        for step, data in system_data.items():
            # Find the entry for this claim
            entry = next((item for item in data if str(item.get('claim_id', '')) == str(claim_id)), None)

            # If no entry with claim_id, try using just the index
            if entry is None and step in ['hyde_fc', 'multi_hyde_fc']:
                try:
                    index = int(claim_id)
                    if index < len(data):
                        entry = data[index]
                except (ValueError, IndexError):
                    pass

            if entry:
                claim_info[system_name][step] = entry
                if verbose and step in ['hyde_fc', 'multi_hyde_fc']:
                    print(f"\nFound {system_name}/{step} entry for claim {claim_id}")
                    if 'hypo_fc_docs' in entry:
                        count = len(entry['hypo_fc_docs'])
                        print(f"  hypo_fc_docs has {count} items")
                    elif step == 'multi_hyde_fc':
                        for fc_type in ['hypo_fc_positive', 'hypo_fc_negative', 'hypo_fc_objective']:
                            if fc_type in entry:
                                count = len(entry[fc_type])
                                print(f"  {fc_type} has {count} items")

    return claim_info


def get_unique_claims_by_label(pipeline_results):
    """Get unique claim IDs from all systems, grouped by predicted label."""
    claims_by_label = defaultdict(set)

    for system_name, system_data in pipeline_results.items():
        if system_name == "reference":
            continue

        if "veracity" in system_data:
            for entry in system_data["veracity"]:
                claim_id = entry.get('claim_id')
                label = entry.get('pred_label', 'Unknown')
                claims_by_label[label].add(claim_id)

    # Convert sets to lists for easier handling
    return {label: list(claims) for label, claims in claims_by_label.items()}


def find_hypo_fc_docs(claim_info, system_name, verbose=False):
    """Find hypothetical FC documents for a specific system."""
    system_data = claim_info.get(system_name, {})

    # Check for multi-perspective FC docs
    if 'multi_hyde_fc' in system_data:
        fc_docs = {}
        entry = system_data['multi_hyde_fc']

        for fc_type in ['hypo_fc_positive', 'hypo_fc_negative', 'hypo_fc_objective']:
            if fc_type in entry:
                fc_docs[fc_type] = entry[fc_type]

        if fc_docs:
            return fc_docs

    # Check for standard FC docs
    if 'hyde_fc' in system_data and 'hypo_fc_docs' in system_data['hyde_fc']:
        return {'hypo_fc_docs': system_data['hyde_fc']['hypo_fc_docs']}

    # Check for FC docs in retrieval
    if 'retrieval' in system_data and 'hypo_fc_docs' in system_data['retrieval']:
        return {'hypo_fc_docs': system_data['retrieval']['hypo_fc_docs']}

    # Search elsewhere
    for step, data in system_data.items():
        if isinstance(data, dict) and 'hypo_fc_docs' in data:
            return {'hypo_fc_docs': data['hypo_fc_docs']}

    if verbose:
        print(f"No hypo_fc_docs found for {system_name}")
    return {}


def extract_reference_qa_pairs(reference_data):
    """Extract question-answer pairs from reference data."""
    qa_pairs = []

    if 'questions' in reference_data:
        for q_item in reference_data['questions']:
            question = q_item.get('question', '')
            answers = q_item.get('answers', [])

            if answers:
                for answer in answers:
                    answer_text = answer.get('answer', '')

                    # Add boolean explanation if available
                    if answer.get('answer_type') == 'Boolean' and 'boolean_explanation' in answer:
                        answer_text += f" - {answer['boolean_explanation']}"

                    qa_pairs.append((question, answer_text))
            else:
                # If no answers, add question with empty answer
                qa_pairs.append((question, ""))

    return qa_pairs


def extract_system_qa_pairs(system_data):
    """Extract question-answer pairs from system data."""
    qa_pairs = []

    # First check for merged QA (multi-perspective)
    if 'merged_qa' in system_data:
        if 'evidence' in system_data['merged_qa']:
            for qa in system_data['merged_qa']['evidence']:
                qa_pairs.append((qa.get('question', ''), qa.get('answer', '')))
    # Then check for regular QA
    elif 'questions' in system_data:
        if 'evidence' in system_data['questions']:
            for qa in system_data['questions']['evidence']:
                qa_pairs.append((qa.get('question', ''), qa.get('answer', '')))

    return qa_pairs


def get_veracity_info(claim_info, system_name):
    """Get veracity prediction info for a specific system."""
    system_data = claim_info.get(system_name, {})

    if system_name == "reference":
        return {
            "label": claim_info.get(system_name, {}).get("label", "Unknown"),
            "justification": claim_info.get(system_name, {}).get("justification", "")
        }

    if 'veracity' in system_data:
        return {
            "label": system_data['veracity'].get('pred_label', 'Unknown'),
            "justification": system_data['veracity'].get('llm_output', '')
        }

    return {"label": "Unknown", "justification": ""}


def generate_analysis_markdown(pipeline_results, output_file, samples_per_label=5, verbose=False):
    """Generate a markdown file with detailed analysis comparing multiple systems."""
    claims_by_label = get_unique_claims_by_label(pipeline_results)
    systems = [s for s in pipeline_results.keys() if s != "reference"]

    # Ensure all output directories exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as mdfile:
        mdfile.write("# AVeriTeC Multi-System Comparison Analysis\n\n")

        # Add table of contents
        mdfile.write("## Table of Contents\n\n")
        for label in claims_by_label.keys():
            mdfile.write(f"- [{label}](#{label.lower().replace(' ', '-').replace('/', '-')})\n")
        mdfile.write("\n")

        # Add systems summary
        mdfile.write("## Systems Compared\n\n")
        for system in systems:
            mdfile.write(f"- **{system}**\n")
        if "reference" in pipeline_results:
            mdfile.write(f"- **reference** (ground truth)\n")
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
                    for system in systems:
                        system_data = claim_info.get(system, {})
                        for step in ["hyde_fc", "multi_hyde_fc", "retrieval", "questions", "veracity"]:
                            if step in system_data and "claim" in system_data[step]:
                                claim_text = system_data[step]["claim"]
                                break
                        if claim_text:
                            break

                    mdfile.write(f"### Sample {i + 1}: Claim {claim_id}\n\n")
                    mdfile.write(f"**Claim**: {claim_text}\n\n")

                    # Write veracity predictions comparison
                    mdfile.write("#### Veracity Predictions\n\n")

                    # Create table for veracity comparison
                    mdfile.write("| System | Verdict | Justification |\n")
                    mdfile.write("|--------|---------|---------------|\n")

                    # Add reference if available
                    if "reference" in claim_info:
                        ref_info = get_veracity_info(claim_info, "reference")
                        mdfile.write(f"| Reference | {ref_info['label']} | {ref_info['justification'][:100]}... |\n")

                    # Add each system
                    for system in systems:
                        sys_info = get_veracity_info(claim_info, system)
                        justification_preview = sys_info['justification'][:100] + "..." if len(
                            sys_info['justification']) > 100 else sys_info['justification']
                        mdfile.write(f"| {system} | {sys_info['label']} | {justification_preview} |\n")

                    # Write hypothetical fact-checking documents comparison
                    mdfile.write("\n#### Hypothetical Fact-Checking Documents\n\n")

                    # For each system, display FC docs
                    for system in systems:
                        mdfile.write(f"**{system} FC documents**:\n\n")

                        fc_docs = find_hypo_fc_docs(claim_info, system, verbose)

                        if not fc_docs:
                            mdfile.write("No hypothetical FC documents available.\n\n")
                            continue

                        # Handle multi-perspective FC docs specially
                        for fc_type, docs in fc_docs.items():
                            if fc_type.startswith('hypo_fc_'):
                                perspective = fc_type.replace('hypo_fc_', '')
                                mdfile.write(f"*{perspective.capitalize()} Perspective*:\n\n")

                            # Display up to 2 documents per type to keep it manageable
                            for j, doc in enumerate(docs[:2]):
                                if not doc:
                                    continue
                                mdfile.write(f"```\n{doc}\n```\n\n")

                    # Write questions and answers comparison
                    mdfile.write("#### Questions and Answers Comparison\n\n")

                    # Get QA pairs from all systems
                    all_qa_pairs = {}

                    # Get reference QA pairs if available
                    if "reference" in claim_info:
                        all_qa_pairs["reference"] = extract_reference_qa_pairs(claim_info["reference"])

                    # Get QA pairs for each system
                    for system in systems:
                        all_qa_pairs[system] = extract_system_qa_pairs(claim_info.get(system, {}))

                    # Determine max number of QA pairs to show
                    max_pairs = 5
                    for system, qa_pairs in all_qa_pairs.items():
                        max_pairs = max(max_pairs, min(5, len(qa_pairs)))

                    # Build table headers
                    headers = ["", "Question", "Answer"]
                    separator = ["-" * 2, "-" * 40, "-" * 40]

                    # Write each system's QA pairs
                    for system, qa_pairs in all_qa_pairs.items():
                        mdfile.write(f"**{system} Questions and Answers:**\n\n")

                        # Create a table for this system's QA pairs
                        mdfile.write("| # | Question | Answer |\n")
                        mdfile.write("|---|----------|--------|\n")

                        for j in range(min(max_pairs, len(qa_pairs))):
                            if j < len(qa_pairs):
                                q, a = qa_pairs[j]
                                # Truncate long questions/answers for readability
                                q_short = (q[:75] + "...") if len(q) > 75 else q
                                a_short = (a[:75] + "...") if len(a) > 75 else a
                                mdfile.write(f"| {j + 1} | {q_short} | {a_short} |\n")

                        mdfile.write("\n")

                    # Write detailed justifications
                    mdfile.write("#### Detailed Veracity Justifications\n\n")

                    # Reference justification if available
                    if "reference" in claim_info:
                        mdfile.write("**Reference Justification**:\n\n")
                        ref_just = claim_info["reference"].get("justification", "No reference justification available")
                        mdfile.write(f"```\n{ref_just}\n```\n\n")

                    # Each system's justification
                    for system in systems:
                        mdfile.write(f"**{system} Justification**:\n\n")
                        sys_info = get_veracity_info(claim_info, system)
                        mdfile.write(f"```\n{sys_info['justification']}\n```\n\n")

                    mdfile.write("---\n\n")
            else:
                mdfile.write(f"No claims with label '{label}' found.\n\n")

    print(f"Analysis markdown file generated: {output_file}")


def generate_comparison_csv(pipeline_results, output_file, samples_per_label=5, verbose=False):
    """Generate a CSV file with detailed comparison of multiple systems."""
    claims_by_label = get_unique_claims_by_label(pipeline_results)
    systems = [s for s in pipeline_results.keys() if s != "reference"]

    # Ensure all output directories exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Create fieldnames with system-specific columns
        fieldnames = ['claim_id', 'claim_text']

        # Add reference fields if available
        if "reference" in pipeline_results:
            fieldnames.extend(['reference_label', 'reference_justification',
                               'reference_q1', 'reference_a1', 'reference_q2', 'reference_a2'])

        # Add system-specific fields
        for system in systems:
            fieldnames.extend([
                f'{system}_label',
                f'{system}_justification',
                f'{system}_fc_doc',
                f'{system}_q1',
                f'{system}_a1',
                f'{system}_q2',
                f'{system}_a2'
            ])

            # Add multi-perspective fields if applicable
            if system == "multi_perspective" or system == "multi_fc":
                fieldnames.extend([
                    f'{system}_fc_positive',
                    f'{system}_fc_negative',
                    f'{system}_fc_objective'
                ])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sample claims from each label category
        all_samples = []
        for label, claim_ids in claims_by_label.items():
            num_samples = min(samples_per_label, len(claim_ids))
            if num_samples > 0:
                sampled_ids = random.sample(claim_ids, num_samples)
                all_samples.extend(sampled_ids)

        # Process each sampled claim
        for claim_id in all_samples:
            claim_info = get_claim_by_id(claim_id, pipeline_results, verbose)
            if not claim_info:
                continue

            # Extract claim text from any available source
            claim_text = ""
            for system in systems:
                system_data = claim_info.get(system, {})
                for step in ["hyde_fc", "multi_hyde_fc", "retrieval", "questions", "veracity"]:
                    if step in system_data and "claim" in system_data[step]:
                        claim_text = system_data[step]["claim"]
                        break
                if claim_text:
                    break

            # Initialize row with claim info
            row = {'claim_id': claim_id, 'claim_text': claim_text}

            # Add reference data if available
            if "reference" in claim_info:
                ref_info = get_veracity_info(claim_info, "reference")
                row['reference_label'] = ref_info['label']
                row['reference_justification'] = ref_info['justification']

                # Add reference QA pairs
                ref_qa_pairs = extract_reference_qa_pairs(claim_info["reference"])
                for i, (q, a) in enumerate(ref_qa_pairs[:2]):
                    row[f'reference_q{i + 1}'] = q
                    row[f'reference_a{i + 1}'] = a

            # Add system-specific data
            for system in systems:
                # Add veracity info
                sys_info = get_veracity_info(claim_info, system)
                row[f'{system}_label'] = sys_info['label']
                row[f'{system}_justification'] = sys_info['justification']

                # Add FC docs
                fc_docs = find_hypo_fc_docs(claim_info, system, verbose)
                if fc_docs:
                    # Standard FC docs
                    if 'hypo_fc_docs' in fc_docs and fc_docs['hypo_fc_docs']:
                        row[f'{system}_fc_doc'] = fc_docs['hypo_fc_docs'][0]

                    # Multi-perspective FC docs
                    if (system == "multi_perspective" or system == "multi_fc"):
                        for fc_type in ['hypo_fc_positive', 'hypo_fc_negative', 'hypo_fc_objective']:
                            if fc_type in fc_docs and fc_docs[fc_type]:
                                row[f'{system}_{fc_type.replace("hypo_fc_", "fc_")}'] = fc_docs[fc_type][0]

                # Add QA pairs
                qa_pairs = extract_system_qa_pairs(claim_info.get(system, {}))
                for i, (q, a) in enumerate(qa_pairs[:2]):
                    row[f'{system}_q{i + 1}'] = q
                    row[f'{system}_a{i + 1}'] = a

            # Write the row
            writer.writerow(row)

    print(f"Comparison CSV file generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple fact-checking pipeline systems')
    parser.add_argument('--systems', nargs='+', default=['baseline', 'multi_perspective'],
                        help='System names to compare (default: baseline and multi_perspective)')
    parser.add_argument('--split', default='dev', help='Data split (default: dev)')
    parser.add_argument('--data-store', default='./data_store', help='Path to data store directory')
    parser.add_argument('--output', default='system_comparison.md', help='Output file path')
    parser.add_argument('--format', choices=['csv', 'markdown'], default='markdown', help='Output format')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples per label')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--reference', default=None, help='Path to reference data file')

    args = parser.parse_args()

    # Analyze pipeline outputs from multiple systems
    pipeline_results = analyze_pipeline_outputs(args.systems, args.split, args.data_store, args.reference)

    if not pipeline_results:
        print("No results to analyze. Check file paths and try again.")
        return

    # Debug option to print structure of first item in each dataset
    if args.verbose:
        print("\nDEBUG: Data structure samples")
        for system, system_data in pipeline_results.items():
            if system == "reference":
                continue

            print(f"\nSystem: {system}")
            for step, data in system_data.items():
                if data and len(data) > 0:
                    print(f"  {step} keys: {list(data[0].keys())}")

                    # Print multi-hyde info if available
                    if step == 'multi_hyde_fc':
                        for fc_type in ['hypo_fc_positive', 'hypo_fc_negative', 'hypo_fc_objective']:
                            if fc_type in data[0]:
                                print(f"    {fc_type} has {len(data[0][fc_type])} items")

    # Generate analysis file
    if args.format == 'csv':
        output_file = args.output if args.output.endswith('.csv') else args.output + '.csv'
        generate_comparison_csv(pipeline_results, output_file, args.samples, args.verbose)
    else:
        output_file = args.output if args.output.endswith('.md') else args.output + '.md'
        generate_analysis_markdown(pipeline_results, output_file, args.samples, args.verbose)


if __name__ == "__main__":
    main()