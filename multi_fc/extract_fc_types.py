#!/usr/bin/env python3
import argparse
import json
import os


def extract_fc_type(input_file, output_prefix, fc_type):
    """Extract and save a specific type of fact-checking document"""
    print(f"Extracting {fc_type} fact-checking documents...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create new data with the specific type of fact-checking document
    new_data = []
    for example in data:
        new_example = {
            "claim": example["claim"],
            "hypo_fc_docs": example[f"hypo_fc_{fc_type}"]
        }
        new_data.append(new_example)

    # Save the extracted data
    output_file = f"{output_prefix}_{fc_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {fc_type} fact-checking documents to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract specific types of fact-checking documents")
    parser.add_argument("--input_file", required=True, help="Input file with multi-type fact-checking documents")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument("--types", nargs="+", default=["positive", "negative", "objective"],
                        help="Types of fact-checking documents to extract")

    args = parser.parse_args()

    for fc_type in args.types:
        extract_fc_type(args.input_file, args.output_prefix, fc_type)


if __name__ == "__main__":
    main()