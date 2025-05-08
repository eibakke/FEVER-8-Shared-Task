#!/usr/bin/env python3
import json
import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


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


def load_veracity_data(system_name, split, data_store):
    """Load veracity prediction results for a system."""
    filepath = f"{data_store}/{system_name}/{split}_veracity_prediction.json"
    data = load_json_file(filepath)

    if data:
        print(f"Loaded {len(data)} veracity predictions from {system_name}")
        return data

    print(f"Could not load veracity data for {system_name}")
    return []


def load_question_data(system_name, split, data_store, is_multi_perspective=False):
    """Load question data for a system."""
    questions_data = []

    if is_multi_perspective:
        # Try to load merged QA data for multi-perspective system
        filepath = f"{data_store}/{system_name}/{split}_merged_qa.json"
        data = load_json_file(filepath)
        if data:
            print(f"Loaded {len(data)} merged QA entries from {system_name}")
            questions_data = data
            return questions_data

        # If merged data not found, try to load perspective-specific data
        for perspective in ['positive', 'negative', 'objective']:
            filepath = f"{data_store}/{system_name}/{split}_top_k_qa_{perspective}.json"
            data = load_json_file(filepath)
            if data:
                print(f"Loaded {len(data)} {perspective} QA entries from {system_name}")
                # Add perspective info to each entry
                for entry in data:
                    if 'evidence' in entry:
                        for ev in entry['evidence']:
                            ev['fc_type'] = perspective
                questions_data.extend(data)
    else:
        # Load regular QA data for baseline
        filepath = f"{data_store}/{system_name}/{split}_top_k_qa.json"
        data = load_json_file(filepath)
        if data:
            print(f"Loaded {len(data)} QA entries from {system_name}")
            questions_data = data

    return questions_data


def load_reference_data(reference_file):
    """Load reference/ground truth data."""
    data = load_json_file(reference_file)

    if data:
        print(f"Loaded {len(data)} reference entries")
        # Create a dictionary by claim_id for easier lookup
        ref_by_id = {}
        for i, entry in enumerate(data):
            # Handle both direct claim_id and index position
            claim_id = entry.get('claim_id', i)
            ref_by_id[str(claim_id)] = entry
        return ref_by_id

    print(f"Could not load reference data")
    return {}


def match_predictions_with_reference(predictions, references):
    """Match predictions with their reference ground truth."""
    matched_data = []

    for pred in predictions:
        claim_id = str(pred.get('claim_id', ''))
        if claim_id in references:
            ref = references[claim_id]
            matched_data.append({
                'claim_id': claim_id,
                'claim': pred.get('claim', ref.get('claim', '')),
                'predicted_label': pred.get('pred_label', 'Unknown'),
                'true_label': ref.get('label', 'Unknown'),
                'prediction': pred,
                'reference': ref
            })
        else:
            # Try to match by index
            try:
                index = int(claim_id)
                index_str = str(index)
                if index_str in references:
                    ref = references[index_str]
                    matched_data.append({
                        'claim_id': claim_id,
                        'claim': pred.get('claim', ref.get('claim', '')),
                        'predicted_label': pred.get('pred_label', 'Unknown'),
                        'true_label': ref.get('label', 'Unknown'),
                        'prediction': pred,
                        'reference': ref
                    })
            except (ValueError, TypeError):
                print(f"Could not match prediction with claim_id: {claim_id}")

    return matched_data


def count_questions_by_perspective(questions_data):
    """Count questions by perspective in a multi-perspective system."""
    perspective_counts = Counter()

    for entry in questions_data:
        if 'evidence' in entry:
            for ev in entry['evidence']:
                perspective = ev.get('fc_type', 'unknown')
                perspective_counts[perspective] += 1

    return perspective_counts


def analyze_by_category(baseline_matched, multi_matched, baseline_questions, multi_questions, output_file):
    """Analyze results broken down by verification category."""
    # Define the categories
    categories = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

    # Group results by true label
    baseline_by_category = defaultdict(list)
    multi_by_category = defaultdict(list)

    for entry in baseline_matched:
        baseline_by_category[entry['true_label']].append(entry)

    for entry in multi_matched:
        multi_by_category[entry['true_label']].append(entry)

    # Open output file for writing
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Analysis by Verification Category\n\n")

        # Overall summary
        f.write("## Overall Results\n\n")

        # Get overall accuracy
        baseline_correct = sum(1 for e in baseline_matched if e['predicted_label'] == e['true_label'])
        multi_correct = sum(1 for e in multi_matched if e['predicted_label'] == e['true_label'])

        baseline_accuracy = baseline_correct / len(baseline_matched) if baseline_matched else 0
        multi_accuracy = multi_correct / len(multi_matched) if multi_matched else 0

        f.write(f"- Baseline overall accuracy: {baseline_accuracy:.4f} ({baseline_correct}/{len(baseline_matched)})\n")
        f.write(
            f"- Multi-perspective overall accuracy: {multi_accuracy:.4f} ({multi_correct}/{len(multi_matched)})\n\n")

        # Create confusion matrices
        baseline_true = [e['true_label'] for e in baseline_matched]
        baseline_pred = [e['predicted_label'] for e in baseline_matched]
        multi_true = [e['true_label'] for e in multi_matched]
        multi_pred = [e['predicted_label'] for e in multi_matched]

        # Create classification reports
        baseline_report = classification_report(baseline_true, baseline_pred, labels=categories, output_dict=True)
        multi_report = classification_report(multi_true, multi_pred, labels=categories, output_dict=True)

        # Write detailed classification reports
        f.write("### Baseline Classification Report\n\n")
        f.write("| Category | Precision | Recall | F1-Score | Support |\n")
        f.write("|----------|-----------|--------|----------|--------|\n")

        for category in categories:
            if category in baseline_report:
                cat_stats = baseline_report[category]
                f.write(
                    f"| {category} | {cat_stats['precision']:.4f} | {cat_stats['recall']:.4f} | {cat_stats['f1-score']:.4f} | {cat_stats['support']} |\n")

        f.write("\n### Multi-perspective Classification Report\n\n")
        f.write("| Category | Precision | Recall | F1-Score | Support |\n")
        f.write("|----------|-----------|--------|----------|--------|\n")

        for category in categories:
            if category in multi_report:
                cat_stats = multi_report[category]
                f.write(
                    f"| {category} | {cat_stats['precision']:.4f} | {cat_stats['recall']:.4f} | {cat_stats['f1-score']:.4f} | {cat_stats['support']} |\n")

        # Write comparison with absolute improvements
        f.write("\n### Improvement Analysis\n\n")
        f.write("| Category | Baseline F1 | Multi F1 | Abs. Improvement | Rel. Improvement |\n")
        f.write("|----------|-------------|----------|------------------|------------------|\n")

        for category in categories:
            if category in baseline_report and category in multi_report:
                baseline_f1 = baseline_report[category]['f1-score']
                multi_f1 = multi_report[category]['f1-score']
                abs_improvement = multi_f1 - baseline_f1
                rel_improvement = (abs_improvement / baseline_f1) * 100 if baseline_f1 > 0 else float('inf')

                # Highlight improvements
                abs_improvement_str = f"{abs_improvement:.4f}"
                rel_improvement_str = f"{rel_improvement:.2f}%"

                if abs_improvement > 0:
                    abs_improvement_str = f"**+{abs_improvement:.4f}**"
                    rel_improvement_str = f"**+{rel_improvement:.2f}%**"

                f.write(
                    f"| {category} | {baseline_f1:.4f} | {multi_f1:.4f} | {abs_improvement_str} | {rel_improvement_str} |\n")

        # Analyze question distribution by category
        f.write("\n## Question Distribution Analysis\n\n")

        # Extract perspectives from multi-perspective questions
        multi_questions_by_id = {}
        for q in multi_questions:
            multi_questions_by_id[str(q.get('claim_id', ''))] = q

        # Count questions by perspective for each category
        f.write("### Question Count by Perspective and Category\n\n")
        f.write("| Category | Positive | Negative | Objective | Total |\n")
        f.write("|----------|----------|----------|-----------|-------|\n")

        for category in categories:
            category_entries = multi_by_category[category]

            # Initialize counters
            positive_count = 0
            negative_count = 0
            objective_count = 0
            total_count = 0

            # Count questions by perspective for this category
            for entry in category_entries:
                claim_id = entry['claim_id']
                if claim_id in multi_questions_by_id:
                    q_data = multi_questions_by_id[claim_id]
                    if 'evidence' in q_data:
                        for ev in q_data['evidence']:
                            perspective = ev.get('fc_type', 'unknown')
                            if perspective.lower() == 'positive':
                                positive_count += 1
                            elif perspective.lower() == 'negative':
                                negative_count += 1
                            elif perspective.lower() == 'objective':
                                objective_count += 1
                            total_count += 1

            f.write(f"| {category} | {positive_count} | {negative_count} | {objective_count} | {total_count} |\n")

        # Analyze perspective contribution to correct predictions
        f.write("\n### Impact of Perspectives on Prediction Accuracy\n\n")

        # For each category, analyze successful vs. failed cases
        for category in categories:
            f.write(f"\n#### {category}\n\n")

            correct_cases = [e for e in multi_by_category[category] if e['predicted_label'] == e['true_label']]
            incorrect_cases = [e for e in multi_by_category[category] if e['predicted_label'] != e['true_label']]

            f.write(
                f"Total cases: {len(multi_by_category[category])}, Correct: {len(correct_cases)}, Incorrect: {len(incorrect_cases)}\n\n")

            if correct_cases:
                f.write("**Perspective distribution in correct predictions:**\n\n")

                correct_perspectives = Counter()
                for entry in correct_cases:
                    claim_id = entry['claim_id']
                    if claim_id in multi_questions_by_id:
                        q_data = multi_questions_by_id[claim_id]
                        if 'evidence' in q_data:
                            for ev in q_data['evidence']:
                                perspective = ev.get('fc_type', 'unknown')
                                correct_perspectives[perspective] += 1

                total_correct = sum(correct_perspectives.values())
                if total_correct > 0:
                    f.write("| Perspective | Count | Percentage |\n")
                    f.write("|------------|-------|------------|\n")

                    for perspective, count in correct_perspectives.most_common():
                        percentage = (count / total_correct) * 100
                        f.write(f"| {perspective.capitalize()} | {count} | {percentage:.2f}% |\n")

            if incorrect_cases:
                f.write("\n**Perspective distribution in incorrect predictions:**\n\n")

                incorrect_perspectives = Counter()
                for entry in incorrect_cases:
                    claim_id = entry['claim_id']
                    if claim_id in multi_questions_by_id:
                        q_data = multi_questions_by_id[claim_id]
                        if 'evidence' in q_data:
                            for ev in q_data['evidence']:
                                perspective = ev.get('fc_type', 'unknown')
                                incorrect_perspectives[perspective] += 1

                total_incorrect = sum(incorrect_perspectives.values())
                if total_incorrect > 0:
                    f.write("| Perspective | Count | Percentage |\n")
                    f.write("|------------|-------|------------|\n")

                    for perspective, count in incorrect_perspectives.most_common():
                        percentage = (count / total_incorrect) * 100
                        f.write(f"| {perspective.capitalize()} | {count} | {percentage:.2f}% |\n")

        # Add examples from the most challenging categories
        challenging_categories = ["Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

        f.write("\n## Examples from Challenging Categories\n\n")

        for category in challenging_categories:
            f.write(f"\n### {category} Examples\n\n")

            # Get improved and degraded examples
            improved_examples = []
            degraded_examples = []

            for multi_entry in multi_by_category[category]:
                claim_id = multi_entry['claim_id']

                # Find matching baseline entry
                baseline_entry = next((e for e in baseline_by_category[category] if e['claim_id'] == claim_id), None)

                if baseline_entry:
                    baseline_correct = baseline_entry['predicted_label'] == baseline_entry['true_label']
                    multi_correct = multi_entry['predicted_label'] == multi_entry['true_label']

                    if multi_correct and not baseline_correct:
                        improved_examples.append((baseline_entry, multi_entry))
                    elif baseline_correct and not multi_correct:
                        degraded_examples.append((baseline_entry, multi_entry))

            # Show some examples of improvements
            f.write(f"#### Improved Cases ({len(improved_examples)} examples)\n\n")

            for i, (baseline, multi) in enumerate(improved_examples[:3]):
                f.write(f"**Example {i + 1}**: {multi['claim']}\n\n")
                f.write(f"- True label: {multi['true_label']}\n")
                f.write(f"- Baseline prediction: {baseline['predicted_label']} (incorrect)\n")
                f.write(f"- Multi-perspective prediction: {multi['predicted_label']} (correct)\n\n")

                # Show question counts by perspective for this example
                claim_id = multi['claim_id']
                if claim_id in multi_questions_by_id:
                    q_data = multi_questions_by_id[claim_id]
                    if 'evidence' in q_data:
                        perspective_counts = Counter()
                        for ev in q_data['evidence']:
                            perspective = ev.get('fc_type', 'unknown')
                            perspective_counts[perspective] += 1

                        f.write("Question counts by perspective:\n\n")
                        for perspective, count in perspective_counts.items():
                            f.write(f"- {perspective.capitalize()}: {count}\n")

                f.write("\n")

            # Show some examples of degradations
            f.write(f"#### Degraded Cases ({len(degraded_examples)} examples)\n\n")

            for i, (baseline, multi) in enumerate(degraded_examples[:3]):
                f.write(f"**Example {i + 1}**: {multi['claim']}\n\n")
                f.write(f"- True label: {multi['true_label']}\n")
                f.write(f"- Baseline prediction: {baseline['predicted_label']} (correct)\n")
                f.write(f"- Multi-perspective prediction: {multi['predicted_label']} (incorrect)\n\n")

                # Show question counts by perspective for this example
                claim_id = multi['claim_id']
                if claim_id in multi_questions_by_id:
                    q_data = multi_questions_by_id[claim_id]
                    if 'evidence' in q_data:
                        perspective_counts = Counter()
                        for ev in q_data['evidence']:
                            perspective = ev.get('fc_type', 'unknown')
                            perspective_counts[perspective] += 1

                        f.write("Question counts by perspective:\n\n")
                        for perspective, count in perspective_counts.items():
                            f.write(f"- {perspective.capitalize()}: {count}\n")

                f.write("\n")

    print(f"Analysis written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze verification results by category')
    parser.add_argument('--baseline', default='baseline', help='Baseline system name')
    parser.add_argument('--multi', default='multi_perspective', help='Multi-perspective system name')
    parser.add_argument('--split', default='dev', help='Data split (default: dev)')
    parser.add_argument('--data-store', default='./data_store', help='Path to data store directory')
    parser.add_argument('--reference', required=True, help='Path to reference data file')
    parser.add_argument('--output', default='category_analysis.md', help='Output file path')

    args = parser.parse_args()

    # Load veracity prediction data
    baseline_veracity = load_veracity_data(args.baseline, args.split, args.data_store)
    multi_veracity = load_veracity_data(args.multi, args.split, args.data_store)

    # Load question data
    baseline_questions = load_question_data(args.baseline, args.split, args.data_store)
    multi_questions = load_question_data(args.multi, args.split, args.data_store, is_multi_perspective=True)

    # Load reference data
    references = load_reference_data(args.reference)

    # Match predictions with references
    baseline_matched = match_predictions_with_reference(baseline_veracity, references)
    multi_matched = match_predictions_with_reference(multi_veracity, references)

    print(f"Matched {len(baseline_matched)} baseline predictions with references")
    print(f"Matched {len(multi_matched)} multi-perspective predictions with references")

    # Analyze by category
    analyze_by_category(baseline_matched, multi_matched, baseline_questions, multi_questions, args.output)


if __name__ == "__main__":
    main()
