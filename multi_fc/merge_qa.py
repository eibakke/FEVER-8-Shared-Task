import argparse
import json
from typing import List, Dict, Any


def load_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """Load question-answer data from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Skip invalid JSON lines
    return data


def merge_qa_data(qa_files: List[str], types: List[str]) -> List[Dict[str, Any]]:
    """Merge question-answer data from multiple files"""
    # Load data from each file
    data_by_type = {}
    for file_path, fc_type in zip(qa_files, types):
        data_by_type[fc_type] = load_qa_data(file_path)

    # Create a mapping of claim_id to examples for each type
    examples_by_id = {}
    for fc_type, data in data_by_type.items():
        for example in data:
            claim_id = example["claim_id"]
            if claim_id not in examples_by_id:
                examples_by_id[claim_id] = {
                    "claim_id": claim_id,
                    "claim": example["claim"],
                    "evidence": []
                }

            # Add type information to each piece of evidence
            for evidence in example["evidence"]:
                evidence["fc_type"] = fc_type
                examples_by_id[claim_id]["evidence"].append(evidence)

    # Convert to list and return
    return list(examples_by_id.values())


def main(args):
    # Merge question-answer data
    merged_data = merge_qa_data(args.qa_files, args.types)

    # Save merged data
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for example in merged_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Merged {len(merged_data)} examples and saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_files', nargs='+', required=True, help='List of question-answer files to merge')
    parser.add_argument('--output_file', required=True, help='Output file for merged question-answer data')
    parser.add_argument('--types', nargs='+', required=True,
                        help='List of fact-checking types corresponding to each file')
    args = parser.parse_args()
    main(args)
