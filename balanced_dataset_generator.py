#!/usr/bin/env python3
import json
import argparse
import random
from collections import Counter
import os
import math


def create_balanced_dataset(input_file, output_file, num_examples, random_seed=42):
    """
    Create a smaller dataset with balanced label distribution.

    Args:
        input_file: Path to the original dataset JSON file
        output_file: Path to save the new balanced dataset
        num_examples: Number of examples in the new dataset
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Load the original dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count labels in the original dataset
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    total_examples = len(data)

    # Calculate the distribution of labels
    label_distribution = {label: count / total_examples for label, count in label_counts.items()}

    # Calculate how many examples we need per label
    examples_per_label = {
        label: math.floor(num_examples * proportion)
        for label, proportion in label_distribution.items()
    }

    # Adjust to ensure we get exactly num_examples (handle rounding errors)
    total_allocated = sum(examples_per_label.values())
    remaining = num_examples - total_allocated

    # Distribute remaining examples
    if remaining > 0:
        sorted_labels = sorted([(label, proportion) for label, proportion in label_distribution.items()],
                               key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            examples_per_label[sorted_labels[i % len(sorted_labels)][0]] += 1

    # Group examples by label
    examples_by_label = {label: [] for label in label_counts.keys()}
    for item in data:
        examples_by_label[item['label']].append(item)

    # Sample examples for each label
    balanced_dataset = []
    for label, count in examples_per_label.items():
        if count > len(examples_by_label[label]):
            print(
                f"Warning: Requested {count} examples for label '{label}', but only {len(examples_by_label[label])} available.")
            sampled = examples_by_label[label]
        else:
            sampled = random.sample(examples_by_label[label], count)
        balanced_dataset.extend(sampled)

    # Shuffle the dataset
    random.shuffle(balanced_dataset)

    # Save the new dataset
    print(f"Saving balanced dataset with {len(balanced_dataset)} examples to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_dataset, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\nOriginal dataset label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count / total_examples:.2%})")

    new_label_counts = Counter([item['label'] for item in balanced_dataset])
    print("\nNew dataset label distribution:")
    for label, count in new_label_counts.items():
        print(f"  {label}: {count} ({count / len(balanced_dataset):.2%})")


def main():
    parser = argparse.ArgumentParser(description='Create a balanced subset of a dataset.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input JSON file')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output JSON file')
    parser.add_argument('--num-examples', '-n', type=int, required=True, help='Number of examples in the new dataset')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    create_balanced_dataset(args.input, args.output, args.num_examples, args.seed)
    print("Done!")


if __name__ == "__main__":
    main()
