from vllm import LLM, SamplingParams
import torch, json, argparse, time, re
from datetime import datetime, timedelta


def extract_hf_id_from_local_path(path):
    if "models--" in path:
        part = path.split("models--")[1].split("/")[0]
        org, model = part.split("--", 1)
        return f"{org}/{model}"
    return path


class FCQGenerator:
    def __init__(self, model_name, n=4, max_tokens=768, temperature=0.7,
                 top_p=0.95, batch_size=16):
        model_name = extract_hf_id_from_local_path(model_name)
        gpus = torch.cuda.device_count()

        print(f"Initializing with {gpus} GPUs")
        print(f"Using model: {model_name}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=gpus,
            max_model_len=4096,
            gpu_memory_utilization=0.95,#ny, kopi
            enforce_eager=True,#ny, kopi
            trust_remote_code=True,#ny, kopi
            max_num_batched_tokens=4096, #Ny kopi
            max_num_seqs=batch_size) #Ny kopi
        self.tokenizer = self.llm.get_tokenizer()
        self.batch_size = batch_size
        self.sampling = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=['```', '</s>', '<|end_of_text|>'],
            logprobs=1
        )

    def _prompt(self, claim):
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.replace("{{CLAIM}}", claim)}]
        chat = self.tokenizer.apply_chat_template(msgs, tokenize=False)
        return chat + "<|start_header_id|>assistant<|end_header_id|>\n\n"


    #Ny
    def _score_completion(self, comp):
        return sum(step[next(iter(step))].logprob for step in comp.logprobs)

    def _extract_content(self, text):
        """Extract passage and questions using pattern matching instead of JSON parsing"""
        # Clean up text first
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        result = {"passage": "", "questions": []}

        # Try to extract passage using patterns
        passage_match = re.search(r'"passage"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', text, re.DOTALL)
        if passage_match:
            result["passage"] = passage_match.group(1).replace('\\"', '"')
        else:
            # Fallback: try to find passage between curly braces
            passage_start = text.find('"passage"')
            if passage_start > -1:
                passage_start = text.find(':', passage_start) + 1
                quote_start = text.find('"', passage_start)
                if quote_start > -1:
                    quote_end = text.find('",', quote_start + 1)
                    if quote_end == -1:  # Last property in JSON
                        quote_end = text.find('"', quote_start + 1)
                    if quote_end > -1:
                        result["passage"] = text[quote_start + 1:quote_end]

        # Try to extract questions using pattern matching
        questions = []

        # First try to extract from "questions" array
        questions_match = re.search(r'"questions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if questions_match:
            questions_str = questions_match.group(1)
            # Extract each question string
            question_matches = re.findall(r'"([^"]*(?:"[^"]*"[^"]*)*)"', questions_str)
            questions.extend(question_matches)

        # Also check for single "question" field
        question_match = re.search(r'"question"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', text)
        if question_match:
            questions.append(question_match.group(1))

        # If we still don't have questions, try more aggressive pattern matching
        if not questions:
            # Look for lines that end with question marks
            question_lines = re.findall(r'"([^"]*\?)"', text)
            questions.extend(question_lines)

        result["questions"] = questions
        return result

    def _parse_outputs(self, request_output, top_k=None):
        """Parse outputs using pattern matching instead of strict JSON parsing"""
        pairs = []
        for cand in request_output.outputs:
            try:
                # Extract content using pattern matching
                content = self._extract_content(cand.text)

                # Only add if we got a passage
                if content["passage"]:
                    content["_logprob"] = self._score_completion(cand)
                    pairs.append(content)

            except Exception as e:
                print(f"Failed to parse output: {e}")
                print(f"Text: {cand.text[:100]}...") # Print just the beginning to avoid huge outputs
                continue

        # Sort by log probability
        pairs.sort(key=lambda d: d.get("_logprob", -float("inf")), reverse=True)
        return pairs[:top_k] if top_k else pairs

    def generate(self, claims, keep_k=4):
        prompts = [self._prompt(c) for c in claims]
        outputs = self.llm.generate(prompts, self.sampling)
        return [self._parse_outputs(out, keep_k) for out in outputs]


PROMPT_TEMPLATE = """You are a professional fact-checker.

CLAIM:
{{CLAIM}}

TASK:
1. Write a fact-checking passage about this claim (about 150-250 words).
2. Create 3-5 specific questions that would help verify this claim.

FORMAT YOUR RESPONSE LIKE THIS:
{
  "passage": "Your fact-checking passage here...",
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}
"""


def main(args):
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(args.target_data, encoding="utf-8") as f:
        data = json.load(f)

    gen = FCQGenerator(args.model, n=args.n)

    all_out = []
    total = len(data)

    print(f"Processing {total} examples...")
    for i in range(0, total, gen.batch_size):
        batch_start = time.time()
        batch = data[i:i + gen.batch_size]
        batch_claims = [ex["claim"] for ex in batch]

        print(f"Processing batch {i + 1}-{min(i + gen.batch_size, total)} of {total}")
        batch_results = gen.generate(batch_claims, args.keep_k)

        for ex, results in zip(batch, batch_results):
            if not results:  # Handle case with no valid results
                ex["hypo_fc_docs"] = ["No valid passage generated"]    # 〈— list[str]  (old HERO key)
                ex["hypo_passage"] = "No valid passage generated"      # 〈— str        (was added later)
                ex["questions"] = []
                ex["fcq_pairs"] = []                                   # optional: full structure
                all_out.append(ex)
                continue

            # Extract all passages and questions
            passages = [p["passage"] for p in results]

            # Collect all questions from all results
            all_questions = []
            for result in results:
                all_questions.extend(result.get("questions", []))

            # Keep the original structure
            ex["hypo_fc_docs"] = passages  # <- list[str]  (old HERO key)
            ex["hypo_passage"] = passages[0] if passages else ""  # <- str (best passage)
            ex["questions"] = all_questions[:5]  # <- list[str]  (limit to 5 questions)
            ex["fcq_pairs"] = results  # <- full structured results

            all_out.append(ex)

        batch_time = time.time() - batch_start
        print(f"Batch completed in {batch_time:.2f}s")

        # Estimate remaining time
        elapsed = time.time() - start_time
        progress = (i + len(batch)) / total
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        eta = datetime.now() + timedelta(seconds=remaining)
        print(f"Progress: {progress * 100:.1f}% - ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"Done - {len(all_out)} examples written to {args.json_output}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per example: {total_time / total:.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_data", default="data_store/averitec/dev.json")
    p.add_argument("--json_output", default="data_store/dev_fcq.json")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--n", type=int, default=8,
                   help="number of completions vLLM draws per prompt")
    p.add_argument("--keep_k", type=int, default=4,
                   help="how many of the completions to keep per claim")
    main(p.parse_args())