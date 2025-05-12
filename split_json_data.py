#!/usr/bin/env python3
import json
import argparse
import os


def detect_format(file_path):
    """Detect if the file is JSON or JSONL format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1).strip()
        if first_char == '[':
            return 'json'
        else:
            return 'jsonl'


def load_data(file_path, format_type):
    """Load data from JSON or JSONL file"""
    data = []
    if format_type == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:  # jsonl
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    return data


def save_data(data, file_path, format_type):
    """Save data to JSON or JSONL file"""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

    if format_type == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:  # jsonl
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Split JSON data into train and reference sets')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--train-output', '-t', default='train.json', help='Output file for train set')
    parser.add_argument('--reference-output', '-r', default='reference.json', help='Output file for reference set')
    parser.add_argument('--train-size', '-n', type=int, default=1000, help='Number of samples for train set')
    parser.add_argument('--format', '-f', choices=['auto', 'json', 'jsonl'], default='auto',
                        help='Force output format (auto, json, or jsonl)')
    parser.add_argument('--preserve-format', '-p', action='store_true',
                        help='Preserve original format for output files')

    args = parser.parse_args()

    # Detect input format
    input_format = detect_format(args.input)
    print(f"Detected input format: {input_format}")

    # Determine output format
    output_format = input_format
    if args.format != 'auto':
        output_format = args.format
    elif not args.preserve_format:
        # Default to JSON format if not preserving
        output_format = 'json'

    print(f"Using output format: {output_format}")

    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input, input_format)
    total_samples = len(data)
    print(f"Total samples loaded: {total_samples}")

    # Check if we have enough data
    train_size = min(args.train_size, total_samples)
    if train_size < args.train_size:
        print(f"Warning: Only {total_samples} samples available, using all for train set")

    # Split data
    train_data = data[:train_size]
    reference_data = data[train_size:]

    # Save train set
    print(f"Saving {len(train_data)} samples to train set: {args.train_output}")
    save_data(train_data, args.train_output, output_format)

    # Save reference set
    if reference_data:
        print(f"Saving {len(reference_data)} samples to reference set: {args.reference_output}")
        save_data(reference_data, args.reference_output, output_format)
    else:
        print("No samples left for reference set")

    print("Done!")


if __name__ == "__main__":
    main()
