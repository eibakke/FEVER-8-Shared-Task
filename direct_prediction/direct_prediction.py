from vllm import LLM, SamplingParams
import json
import torch
import time
import re
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

LABEL = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class VLLMGenerator:
    def __init__(self, model_name: str, n: int = 1, max_tokens: int = 1024,
                 temperature: float = 0.9, top_p: float = 0.7,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 stop: List[str] = None, batch_size: int = 8):
        if stop is None:
            stop = ['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]', '<|eot_id|>', '<|end|>']

        self.device_count = torch.cuda.device_count()
        print(f"Initializing with {self.device_count} GPUs")

        # Extract Hugging Face ID from path if it's a local path
        if '/' in model_name and 'models--' in model_name:
            try:
                # Format is typically: /path/to/models--org--model-name/snapshots/hash
                model_part = model_name.split('models--')[1].split('/')[0]
                org = model_part.split('--')[0]
                # Handle models with dashes in their names
                model = '--'.join(model_part.split('--')[1:])
                hf_id = f"{org}/{model}"
                print(f"Extracted Hugging Face ID from path: {hf_id}")
                model_name = hf_id
            except Exception as e:
                print(f"Error extracting model ID: {e}, using original path")

        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=self.device_count,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logprobs=1
        )

        self.batch_size = batch_size
        self.tokenizer = self.llm.get_tokenizer()
        print(f"Initialization complete. Batch size: {batch_size}")

    def parse_response(self, responses):
        """Parse the model responses to extract output text."""
        all_outputs = []
        for response in responses:
            to_return = []
            for output in response.outputs:
                text = output.text.strip()
                try:
                    logprob = sum(logprob_obj.logprob for item in output.logprobs for logprob_obj in item.values())
                except:
                    logprob = 0  # Fallback if logprobs aren't available
                to_return.append((text, logprob))
            texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
            all_outputs.append(texts[0])  # Just take the highest probability output
        return all_outputs

    def extract_qa_pairs(self, output: str) -> List[Dict]:
        """Extract question-answer pairs from the model output."""
        qa_pairs = []

        # Look for the Q&A section
        qa_section_match = re.search(r'===\s*Questions\s+and\s+Answers\s*===(.*?)===\s*Verdict', output, re.DOTALL)

        if not qa_section_match:
            return qa_pairs

        qa_section = qa_section_match.group(1).strip()

        # Extract individual Q&A pairs
        qa_pattern = re.compile(r'Q\d+:\s*(.*?)\s*\n\s*A\d+:\s*(.*?)(?=\s*Q\d+:|$)', re.DOTALL)
        matches = qa_pattern.findall(qa_section)

        for question, answer in matches:
            qa_pairs.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "url": "direct_prediction"  # Placeholder URL since we're not retrieving from external sources
            })

        return qa_pairs

    def get_label_from_output(self, output: str) -> Optional[str]:
        """Extract label from model output."""
        if "Not Enough Evidence" in output:
            return "Not Enough Evidence"
        elif any(x in output for x in ["Conflicting Evidence/Cherrypicking", "Cherrypicking", "Conflicting Evidence"]):
            return "Conflicting Evidence/Cherrypicking"
        elif any(x in output for x in ["Supported", "supported"]):
            return "Supported"
        elif any(x in output for x in ["Refuted", "refuted"]):
            return "Refuted"
        return None

    def prepare_prompt(self, claim: str, model_name: str) -> str:
        """Prepare a fact-checking prompt with Q&A structure for a claim."""
        base_prompt = """You are a fact-checking assistant. Your task is to predict the verdict of a claim based on your knowledge.

First, generate 3 relevant questions that would help verify this claim.
Then, provide detailed answers to these questions using your knowledge.
Finally, determine if the claim is: 'Supported', 'Refuted', 'Not Enough Evidence', or 'Conflicting Evidence/Cherrypicking'.

Structure your response exactly as follows:

=== Questions and Answers ===
Q1: [First question about the claim]
A1: [Your detailed answer to Q1]

Q2: [Second question about the claim]
A2: [Your detailed answer to Q2]

Q3: [Third question about the claim]
A3: [Your detailed answer to Q3]

=== Verdict ===
justification: [Your step-by-step reasoning based on the Q&A]

verdict: [Your final verdict - one of 'Supported', 'Refuted', 'Not Enough Evidence', or 'Conflicting Evidence/Cherrypicking']"""

        prompt = base_prompt + f"\n\nClaim: {claim}"

        if "OLMo" in model_name:
            return prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def process_batch(self, batch: List[Dict[str, Any]], model_name: str) -> tuple[List[Dict[str, Any]], float]:
        """Process a batch of examples to generate fact-checking results."""
        start_time = time.time()
        prompts = [self.prepare_prompt(example["claim"], model_name) for example in batch]

        try:
            results = self.llm.generate(prompts, sampling_params=self.sampling_params)
            output_texts = self.parse_response(results)

            output_json = []
            for b, output_text in zip(batch, output_texts):
                output = {}

                label = self.get_label_from_output(output_text)
                evidence = self.extract_qa_pairs(output_text)

                # If no Q&A pairs were found or if no label was found, use placeholder data
                if len(evidence) == 0:
                    evidence = [
                        {
                            "question": "Is the claim factually accurate?",
                            "answer": "Based on available information, a determination was made regarding the claim's accuracy.",
                            "url": "direct_prediction"
                        },
                        {
                            "question": "What evidence supports or refutes this claim?",
                            "answer": "The evidence was evaluated to reach the conclusion about this claim.",
                            "url": "direct_prediction"
                        }
                    ]
                output['claim_id'] = b["claim_id"]
                output['claim'] = b["claim"]
                output['evidence'] = evidence
                output['pred_label'] = label or "Not Enough Evidence"  # fallback if no label found
                output['llm_output'] = output_text
                output_json.append(output)

            batch_time = time.time() - start_time
            return output_json, batch_time
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return [], time.time() - start_time


def format_time(seconds: float) -> str:
    """Format time duration nicely."""
    return str(timedelta(seconds=int(seconds)))


def estimate_completion_time(start_time: float, processed_examples: int, total_examples: int) -> str:
    """Estimate the completion time based on progress."""
    elapsed_time = time.time() - start_time
    examples_per_second = processed_examples / elapsed_time
    remaining_examples = total_examples - processed_examples
    estimated_remaining_seconds = remaining_examples / examples_per_second
    completion_time = datetime.now() + timedelta(seconds=int(estimated_remaining_seconds))
    return completion_time.strftime("%Y-%m-%d %H:%M:%S")


def main(args):
    total_start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("Loading data...")
    try:
        with open(args.target_data, 'r', encoding='utf-8') as json_file:
            examples = json.load(json_file)
    except:
        examples = []
        with open(args.target_data, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Initialize generator
    print("Initializing generator...")
    generator = VLLMGenerator(
        model_name=args.model,
        batch_size=args.batch_size
    )

    # Process data in batches
    processed_data = []
    batch_times = []
    batches = [examples[i:i + generator.batch_size] for i in range(0, len(examples), generator.batch_size)]

    print(f"\nProcessing {len(batches)} batches...")
    with tqdm(total=len(examples), desc="Processing examples") as pbar:
        for batch_idx, batch in enumerate(batches, 1):
            # Process each example to ensure it has claim_id
            for i, example in enumerate(batch):
                if "claim_id" not in example:
                    example["claim_id"] = example.get("id", (batch_idx - 1) * generator.batch_size + i)

            processed_batch, batch_time = generator.process_batch(batch, args.model)
            processed_data.extend(processed_batch)
            batch_times.append(batch_time)

            # Update progress and timing information
            examples_processed = len(processed_data)
            avg_batch_time = sum(batch_times) / len(batch_times)
            estimated_completion = estimate_completion_time(total_start_time, examples_processed, len(examples))

            pbar.set_postfix({
                'Batch': f"{batch_idx}/{len(batches)}",
                'Avg Batch Time': f"{avg_batch_time:.2f}s",
                'ETA': estimated_completion
            })
            pbar.update(len(batch))

    # Calculate and display timing statistics
    total_time = time.time() - total_start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_example_time = total_time / len(examples)

    print("\nTiming Statistics:")
    print(f"Total Runtime: {format_time(total_time)}")
    print(f"Average Batch Time: {avg_batch_time:.2f} seconds")
    print(f"Average Time per Example: {avg_example_time:.2f} seconds")
    print(f"Throughput: {len(examples) / total_time:.2f} examples/second")

    # Save results
    print("\nSaving results...")
    with open(args.output_file, "w", encoding="utf-8") as output_json:
        json.dump(processed_data, output_json, ensure_ascii=False, indent=4)

    print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--target_data', default='data_store/averitec/dev.json')
    parser.add_argument('-o', '--output_file', default='data_store/direct_prediction/veracity_prediction.json')
    parser.add_argument('-m', '--model',
                        default="/fp/projects01/ec403/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)