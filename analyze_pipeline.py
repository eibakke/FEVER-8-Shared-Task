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
        print(f"Warning: File not found: {filepath}")
        return None


def load_pipeline_data(system_name, split, data_store, verbose=False):
    """Load data from a specific pipeline, accounting for different file structures."""
    results = {}

    # Check if this is the multi-perspective system
    is_multi_perspective = system_name in ["multi_perspective", "multi_fc"]

    # Define file paths for each step
    file_paths = {}

    # Standard file paths (for baseline or regular systems)
    standard_paths = {
        "hyde_fc": f"{data_store}/{system_name}/{split}_hyde_fc.json",
        "retrieval": f"{data_store}/{system_name}/{split}_retrieval_top_k.json",
        "reranking": f"{data_store}/{system_name}/{split}_reranking_top_k.json",
        "questions": f"{data_store}/{system_name}/{split}_top_k_qa.json",
        "veracity": f"{data_store}/{system_name}/{split}_veracity_prediction.json"
    }

    # Multi-perspective specific paths
    multi_paths = {
        "multi_hyde_fc": f"{data_store}/{system_name}/{split}_multi_hyde_fc.json",
        "merged_qa": f"{data_store}/{system_name}/{split}_merged_qa.json",
        "veracity": f"{data_store}/{system_name}/{split}_veracity_prediction.json",
        # Perspective-specific files
        "hyde_fc_positive": f"{data_store}/{system_name}/{split}_hyde_fc_positive.json",
        "hyde_fc_negative": f"{data_store}/{system_name}/{split}_hyde_fc_negative.json",
        "hyde_fc_objective": f"{data_store}/{system_name}/{split}_hyde_fc_objective.json",
        "retrieval_positive": f"{data_store}/{system_name}/{split}_retrieval_top_k_positive.json",
        "retrieval_negative": f"{data_store}/{system_name}/{split}_retrieval_top_k_negative.json",
        "retrieval_objective": f"{data_store}/{system_name}/{split}_retrieval_top_k_objective.json",
        "reranking_positive": f"{data_store}/{system_name}/{split}_reranking_top_k_positive.json",
        "reranking_negative": f"{data_store}/{system_name}/{split}_reranking_top_k_negative.json",
        "reranking_objective": f"{data_store}/{system_name}/{split}_reranking_top_k_objective.json",
        "questions_positive": f"{data_store}/{system_name}/{split}_top_k_qa_positive.json",
        "questions_negative": f"{data_store}/{system_name}/{split}_top_k_qa_negative.json",
        "questions_objective": f"{data_store}/{system_name}/{split}_top_k_qa_objective.json"
    }

    # Use appropriate file paths based on system type
    if is_multi_perspective:
        file_paths.update(multi_paths)
    else:
        file_paths.update(standard_paths)

    # Load each file
    for step, path in file_paths.items():
        data = load_json_file(path)
        if data:
            if isinstance(data, list) and len(data) > 0:
                results[step] = data
                if verbose:
                    print(f"Loaded {len(data)} entries from {system_name}/{step}")
            else:
                # Handle non-list data
                results[step] = [data] if data else []
                if verbose:
                    print(f"Loaded 1 entry from {system_name}/{step}")

    return results


def analyze_pipeline_outputs(systems, split, data_store, reference_file=None, verbose=False):
    """Analyze outputs from multiple pipeline systems."""
    results = {}

    # Load data for each system
    for system_name in systems:
        results[system_name] = load_pipeline_data(system_name, split, data_store, verbose)

    # Load reference data if provided
    if reference_file:
        reference_data = load_json_file(reference_file)
        if reference_data:
            # Ensure reference data is in a list
            if not isinstance(reference_data, list):
                reference_data = [reference_data]

            results["reference"] = {"data": reference_data}
            if verbose:
                print(f"Loaded {len(reference_data)} entries from reference file")

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
                    if verbose:
                        print(f"Found reference entry for claim {claim_id}")
            except (ValueError, IndexError, KeyError):
                if verbose:
                    print(f"Could not find reference entry for claim {claim_id}")
                pass
            continue

        # For system data, extract from each step
        claim_info[system_name] = {}

        # Try to find claim data in any of the steps
        for step, data in system_data.items():
            if not data or len(data) == 0:
                continue

            # Find the entry for this claim
            entry = None

            # Try to match by claim_id
            try:
                entry = next((item for item in data if str(item.get('claim_id', '')) == str(claim_id)), None)
            except (TypeError, AttributeError):
                # If data is not iterable or items are not dictionaries
                if verbose:
                    print(f"Error: Could not process {system_name}/{step} data")
                continue

            # If no entry with claim_id, try using just the index
            if entry is None and step in ['hyde_fc', 'multi_hyde_fc']:
                try:
                    index = int(claim_id)
                    if index < len(data):
                        entry = data[index]
                except (ValueError, IndexError, TypeError):
                    pass

            if entry:
                claim_info[system_name][step] = entry
                if verbose and step in ['hyde_fc', 'multi_hyde_fc']:
                    print(f"Found {system_name}/{step} entry for claim {claim_id}")

    return claim_info


def get_unique_claims_by_label(pipeline_results, verbose=False):
    """Get unique claim IDs from all systems, grouped by predicted label."""
    claims_by_label = defaultdict(set)

    for system_name, system_data in pipeline_results.items():
        if system_name == "reference":
            continue

        if "veracity" in system_data:
            try:
                for entry in system_data["veracity"]:
                    claim_id = entry.get('claim_id')
                    label = entry.get('pred_label', 'Unknown')
                    if claim_id:
                        claims_by_label[label].add(claim_id)
                        if verbose:
                            print(f"Added claim {claim_id} with label {label} from {system_name}")
            except (TypeError, AttributeError):
                if verbose:
                    print(f"Error: Could not process veracity data for {system_name}")

    # If no labels found, try to get claims from other steps
    if not claims_by_label:
        if verbose:
            print("No veracity labels found, falling back to other data sources")

        # Collect all unique claim IDs
        all_claims = set()
        for system_name, system_data in pipeline_results.items():
            if system_name == "reference":
                continue

            for step, data in system_data.items():
                try:
                    for entry in data:
                        if isinstance(entry, dict) and 'claim_id' in entry:
                            all_claims.add(entry['claim_id'])
                except (TypeError, AttributeError):
                    continue

        # Put all claims under "Unknown" label
        claims_by_label["Unknown"].update(all_claims)

    # Convert sets to lists for easier handling
    return {label: list(claims) for label, claims in claims_by_label.items()}


def find_hypo_fc_docs(claim_info, system_name, verbose=False):
    """Find hypothetical FC documents for a specific system."""
    system_data = claim_info.get(system_name, {})
    is_multi_perspective = system_name in ["multi_perspective", "multi_fc"]
    result = {}

    # Check for multi-perspective FC docs
    if is_multi_perspective:
        # First try to find in the consolidated multi_hyde_fc file
        if 'multi_hyde_fc' in system_data:
            entry = system_data['multi_hyde_fc']
            for fc_type in ['hypo_fc_positive', 'hypo_fc_negative', 'hypo_fc_objective']:
                if fc_type in entry and entry[fc_type]:
                    result[fc_type] = entry[fc_type]

            if result:
                return result

        # Then try perspective-specific files
        for perspective in ['positive', 'negative', 'objective']:
            key = f'hyde_fc_{perspective}'
            if key in system_data and system_data[key]:
                try:
                    docs = system_data[key].get('hypo_fc_docs', [])
                    if docs:
                        result[f'hypo_fc_{perspective}'] = docs
                except (AttributeError, TypeError):
                    # Handle case where the entry might not be a dict
                    pass

    # Check for standard FC docs
    if 'hyde_fc' in system_data:
        try:
            docs = system_data['hyde_fc'].get('hypo_fc_docs', [])
            if docs:
                result['hypo_fc_docs'] = docs
        except (AttributeError, TypeError):
            pass

    # Check for FC docs in retrieval
    if 'retrieval' in system_data:
        try:
            docs = system_data['retrieval'].get('hypo_fc_docs', [])
            if docs:
                result['hypo_fc_docs'] = docs
        except (AttributeError, TypeError):
            pass

    # Search elsewhere
    for step, data in system_data.items():
        if isinstance(data, dict) and 'hypo_fc_docs' in data:
            result['hypo_fc_docs'] = data['hypo_fc_docs']

    if verbose and not result:
        print(f"No hypo_fc_docs found for {system_name}")

    return result


def extract_reference_qa_pairs(reference_data):
    """Extract question-answer pairs from reference data."""
    qa_pairs = []

    try:
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
    except (TypeError, AttributeError):
        # Handle case where reference_data might not be a dict
        pass

    return qa_pairs


def extract_system_qa_pairs(system_data, is_multi_perspective=False):
    """Extract question-answer pairs from system data."""
    qa_pairs = []

    # For multi-perspective systems, organize QA pairs by perspective
    if is_multi_perspective:
        qa_by_perspective = defaultdict(list)

        # Check for merged QA from multi-perspective
        if 'merged_qa' in system_data:
            try:
                if 'evidence' in system_data['merged_qa']:
                    for qa in system_data['merged_qa']['evidence']:
                        # Include fc_type if available
                        perspective = qa.get('fc_type', 'unknown')
                        qa_by_perspective[perspective].append((
                            qa.get('question', ''),
                            qa.get('answer', ''),
                            perspective
                        ))
            except (TypeError, AttributeError):
                pass

        # Check for perspective-specific QA files (these are individual files like questions_positive.json)
        for perspective in ['positive', 'negative', 'objective']:
            key = f'questions_{perspective}'
            if key in system_data:
                try:
                    if 'evidence' in system_data[key]:
                        for qa in system_data[key]['evidence']:
                            qa_by_perspective[perspective].append((
                                qa.get('question', ''),
                                qa.get('answer', ''),
                                perspective
                            ))
                except (TypeError, AttributeError):
                    pass

        # Now select QA pairs from each perspective to provide a balanced representation
        all_perspectives = list(qa_by_perspective.keys())

        # Add QA pairs from each perspective (up to 2 from each)
        for perspective in ['positive', 'negative', 'objective']:
            if perspective in qa_by_perspective:
                qa_pairs.extend(qa_by_perspective[perspective][:2])

        # If there are still slots available, add more questions from perspectives with more QA pairs
        remaining_perspectives = sorted(all_perspectives,
                                        key=lambda p: len(qa_by_perspective[p]),
                                        reverse=True)

        for perspective in remaining_perspectives:
            pairs = qa_by_perspective[perspective]
            if len(pairs) > 2:
                qa_pairs.extend(pairs[2:4])  # Add two more if available

            if len(qa_pairs) >= 10:  # Limit to 10 total
                break

    # For standard systems, extract QA pairs normally
    else:
        if 'questions' in system_data:
            try:
                if 'evidence' in system_data['questions']:
                    for qa in system_data['questions']['evidence']:
                        qa_pairs.append((
                            qa.get('question', ''),
                            qa.get('answer', ''),
                            ""  # No perspective for baseline
                        ))
            except (TypeError, AttributeError):
                pass

    return qa_pairs


def get_veracity_info(claim_info, system_name):
    """Get veracity prediction info for a specific system."""
    system_data = claim_info.get(system_name, {})

    if system_name == "reference":
        try:
            return {
                "label": claim_info.get(system_name, {}).get("label", "Unknown"),
                "justification": claim_info.get(system_name, {}).get("justification", "")
            }
        except (TypeError, AttributeError):
            return {"label": "Unknown", "justification": ""}

    if 'veracity' in system_data:
        try:
            return {
                "label": system_data['veracity'].get('pred_label', 'Unknown'),
                "justification": system_data['veracity'].get('llm_output', '')
            }
        except (TypeError, AttributeError):
            pass

    return {"label": "Unknown", "justification": ""}


def generate_analysis_markdown(pipeline_results, output_file, samples_per_label=5, verbose=False):
    """Generate a markdown file with detailed analysis comparing multiple systems."""
    claims_by_label = get_unique_claims_by_label(pipeline_results, verbose)
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
                        if verbose:
                            print(f"Could not find info for claim {claim_id}")
                        continue

                    # Extract claim text from any available source
                    claim_text = ""
                    for system in systems:
                        if system not in claim_info:
                            continue

                        system_data = claim_info.get(system, {})
                        for step in ["hyde_fc", "multi_hyde_fc", "retrieval", "questions", "veracity"]:
                            if step in system_data and isinstance(system_data[step], dict) and "claim" in system_data[
                                step]:
                                claim_text = system_data[step]["claim"]
                                break
                        if claim_text:
                            break

                    # If still no claim text, check reference
                    if not claim_text and "reference" in claim_info:
                        try:
                            claim_text = claim_info["reference"].get("claim", "")
                        except (AttributeError, TypeError):
                            pass

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
                        justification_preview = ref_info['justification'][:100] + "..." if len(
                            ref_info['justification']) > 100 else ref_info['justification']
                        mdfile.write(f"| Reference | {ref_info['label']} | {justification_preview} |\n")

                    # Add each system
                    for system in systems:
                        if system in claim_info:
                            sys_info = get_veracity_info(claim_info, system)
                            justification_preview = sys_info['justification'][:100] + "..." if len(
                                sys_info['justification']) > 100 else sys_info['justification']
                            mdfile.write(f"| {system} | {sys_info['label']} | {justification_preview} |\n")

                    # Write hypothetical fact-checking documents comparison
                    mdfile.write("\n#### Hypothetical Fact-Checking Documents\n\n")

                    # For each system, display FC docs
                    for system in systems:
                        if system in claim_info:
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
                        ref_qa_pairs = extract_reference_qa_pairs(claim_info["reference"])
                        all_qa_pairs["reference"] = [(q, a, "") for q, a in ref_qa_pairs]  # Add empty perspective

                    # Get QA pairs for each system
                    for system in systems:
                        if system in claim_info:
                            is_multi = system in ["multi_perspective", "multi_fc"]
                            all_qa_pairs[system] = extract_system_qa_pairs(
                                claim_info.get(system, {}),
                                is_multi_perspective=is_multi
                            )

                    # Write each system's QA pairs
                    for system, qa_pairs in all_qa_pairs.items():
                        mdfile.write(f"**{system} Questions and Answers:**\n\n")

                        # Create a table for this system's QA pairs
                        if system in ["multi_perspective", "multi_fc"]:
                            # Include perspective column for multi-perspective systems
                            mdfile.write("| # | Perspective | Question | Answer |\n")
                            mdfile.write("|---|------------|----------|--------|\n")

                            for j, (q, a, perspective) in enumerate(qa_pairs):
                                # Remove the perspective prefix from the question if already included
                                if q.startswith(f"[{perspective}]"):
                                    q = q[len(f"[{perspective}]"):].strip()

                                # Truncate long questions/answers for readability
                                q_short = (q[:75] + "...") if len(q) > 75 else q
                                a_short = (a[:75] + "...") if len(a) > 75 else a

                                # Capitalize perspective
                                perspective_display = perspective.capitalize() if perspective else "Unknown"

                                mdfile.write(f"| {j + 1} | **{perspective_display}** | {q_short} | {a_short} |\n")
                        else:
                            # Regular table for baseline and reference
                            mdfile.write("| # | Question | Answer |\n")
                            mdfile.write("|---|----------|--------|\n")

                            for j, (q, a, _) in enumerate(
                                    qa_pairs):  # Ignore the perspective field for non-multi systems

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
                        try:
                            ref_just = claim_info["reference"].get("justification",
                                                                   "No reference justification available")
                            mdfile.write(f"```\n{ref_just}\n```\n\n")
                        except (AttributeError, TypeError):
                            mdfile.write("```\nNo reference justification available\n```\n\n")

                    # Each system's justification
                    for system in systems:
                        if system in claim_info:
                            mdfile.write(f"**{system} Justification**:\n\n")
                            sys_info = get_veracity_info(claim_info, system)
                            mdfile.write(f"```\n{sys_info['justification']}\n```\n\n")

                    mdfile.write("---\n\n")
            else:
                mdfile.write(f"No claims with label '{label}' found.\n\n")

    print(f"Analysis markdown file generated: {output_file}")


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
    pipeline_results = analyze_pipeline_outputs(args.systems, args.split, args.data_store, args.reference, args.verbose)

    if not pipeline_results:
        print("No results to analyze. Check file paths and try again.")
        return

    # Debug option to print structure
    if args.verbose:
        print("\nDEBUG: Found the following data in pipeline results:")
        for system, system_data in pipeline_results.items():
            print(f"\nSystem: {system}")
            for step in system_data:
                data_count = len(system_data[step]) if isinstance(system_data[step], list) else 1
                print(f"  {step}: {data_count} entries")

    # Generate analysis file
    if args.format == 'csv':
        print("CSV format is currently not supported for multi-system comparison")
        print("Using markdown format instead")
        output_file = args.output if args.output.endswith('.md') else args.output + '.md'
        generate_analysis_markdown(pipeline_results, output_file, args.samples, args.verbose)
    else:
        output_file = args.output if args.output.endswith('.md') else args.output + '.md'
        generate_analysis_markdown(pipeline_results, output_file, args.samples, args.verbose)


if __name__ == "__main__":
    main()