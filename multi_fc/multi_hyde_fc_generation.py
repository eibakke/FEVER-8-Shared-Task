from vllm import LLM, SamplingParams
import json
import torch
import time
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
from typing import List, Dict, Any


class VLLMGenerator:
    def __init__(self, model_name: str, n: int = 8, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 1.0,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 stop: List[str] = ['\n\n\n'], batch_size: int = 32):
        self.device_count = torch.cuda.device_count()
        print(f"Initializing with {self.device_count} GPUs")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=self.device_count,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True,
            max_num_batched_tokens=4096,
            max_num_seqs=batch_size
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
            all_outputs.append(texts)
        return all_outputs

    def prepare_prompts(self, claim: str, model_name: str) -> List[str]:
        prompts = []

        # Positive (supporting) fact-checking prompt
        positive_prompt = f"Please write a fact-checking article passage that SUPPORTS the following claim, highlighting evidence in favor of it.\nClaim: {claim}"

        # Negative (refuting) fact-checking prompt
        negative_prompt = f"Please write a fact-checking article passage that REFUTES the following claim, highlighting evidence against it.\nClaim: {claim}"

        # Objective (neutral) fact-checking prompt
        objective_prompt = f"Please write an objective fact-checking article passage about the following claim, presenting a balanced view of evidence both for and against it.\nClaim: {claim}"

        for prompt in [positive_prompt, negative_prompt, objective_prompt]:
            if "OLMo" in model_name:
                prompts.append(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(messages,
                                                                      tokenize=False) + "<|start_header_id|>assistant<|end_header_id|>\n\nPassage: "
                prompts.append(formatted_prompt)

        return prompts

    def process_example(self, example: Dict[str, Any], model_name: str) -> tuple[Dict[str, Any], float]:
        start_time = time.time()
        prompts = self.prepare_prompts(example["claim"], model_name)

        try:
            results = self.llm.generate(prompts, sampling_params=self.sampling_params)
            outputs = self.parse_response(results)

            # Store results for each type of fact-checking
            example['hypo_fc_positive'] = outputs[0]
            example['hypo_fc_negative'] = outputs[1]
            example['hypo_fc_objective'] = outputs[2]

            # Keep the original key for backward compatibility
            example['hypo_fc_docs'] = outputs[2]  # Using objective as default

            process_time = time.time() - start_time
            return example, process_time
        except Exception as e:
            print(f"Error processing example (claim_id: {example.get('claim_id', 'unknown')}): {str(e)}")
            return example, time.time() - start_time


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def estimate_completion_time(start_time: float, processed_examples: int, total_examples: int) -> str:
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
    with open(args.target_data, 'r', encoding='utf-8') as json_file:
        examples = json.load(json_file)
    print(f"Loaded {len(examples)} examples")

    # Initialize generator
    print("Initializing generator...")
    generator = VLLMGenerator(
        model_name=args.model,
        batch_size=args.batch_size
    )

    # Process examples one by one to generate all three types of fact-checking documents
    processed_data = []
    process_times = []

    print(f"\nProcessing {len(examples)} examples...")
    with tqdm(total=len(examples), desc="Processing examples") as pbar:
        for i, example in enumerate(examples):
            processed_example, process_time = generator.process_example(example, args.model)
            processed_data.append(processed_example)
            process_times.append(process_time)

            # Update progress and timing information
            examples_processed = i + 1
            avg_process_time = sum(process_times) / len(process_times)
            estimated_completion = estimate_completion_time(total_start_time, examples_processed, len(examples))

            pbar.set_postfix({
                'Avg Time': f"{avg_process_time:.2f}s",
                'ETA': estimated_completion
            })
            pbar.update(1)

    # Calculate and display timing statistics
    total_time = time.time() - total_start_time
    avg_example_time = total_time / len(examples)

    print("\nTiming Statistics:")
    print(f"Total Runtime: {format_time(total_time)}")
    print(f"Average Time per Example: {avg_example_time:.2f} seconds")
    print(f"Throughput: {len(examples) / total_time:.2f} examples/second")

    # Save results
    print("\nSaving results...")
    with open(args.json_output, "w", encoding="utf-8") as output_json:
        json.dump(processed_data, output_json, ensure_ascii=False, indent=4)

    print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--target_data', default='data_store/averitec/dev.json')
    parser.add_argument('-o', '--json_output', default='data_store/multi_hyde_fc.json')
    parser.add_argument('-m', '--model', default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
