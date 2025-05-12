import csv
import json
import argparse
import os


def convert(file_json, output_csv="leaderboard_submission/submission.csv"):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load the JSON data
    with open(file_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    new_samples = []
    for i, sample in enumerate(samples):
        claim = sample['claim']
        label = sample['label']

        # Format evidence from the questions format
        prediction_evidence = ""
        for q in sample['questions']:
            question_text = q['question']
            for ans in q['answers']:
                answer_text = ans['answer']
                # Include boolean explanation if available
                if ans['answer_type'] == 'Boolean' and 'boolean_explanation' in ans:
                    answer_text += ", " + ans['boolean_explanation']
                prediction_evidence += f"{question_text}\t\t\n{answer_text}\t\t\n\n"

        new_samples.append([i, claim, prediction_evidence, label, 'pred'])

    with open(output_csv, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "claim", "evi", "label", "split"])
        writer.writerows(new_samples)

    print(f"{file_json} has been converted to {output_csv}")
    print(f"Created {len(new_samples)} entries in {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Convert JSON to leaderboard CSV')

    # Add arguments
    parser.add_argument('--filename', type=str, required=True,
                        help='Input reference JSON file')
    parser.add_argument('--output', type=str, default='leaderboard_submission/solution.csv',
                        help='Output CSV file for evaluation')

    # Parse arguments
    args = parser.parse_args()

    convert(args.filename, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
