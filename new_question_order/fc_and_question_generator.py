from vllm import LLM, SamplingParams
import torch, json, argparse, time
from datetime import datetime


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

    def _pick_best(self, request_output):
        best_item, best_score = None, -float("inf")

        for cand in request_output.outputs:
            try:
                parsed = json.loads(cand.text.strip())
                assert isinstance(parsed, dict) and \
                       "passage" in parsed
            except (json.JSONDecodeError, AssertionError):
                continue

            score = self._score_completion(cand)
            if score > best_score:
                best_item, best_score = parsed, score

        # Fallback - keep raw text
        if best_item is None:
            best_item = {"passage": request_output.outputs[0].text.strip(),
                         "questions": []}

        return best_item


    #ny
    def _parse_outputs(self, request_output, top_k=None):
        pairs = []
        for cand in request_output.outputs:
            try:
                text = cand.text.strip()
                # Handle potential opening/closing backticks that might mess up JSON parsing
                if text.startswith("```json"):
                    text = text[7:].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()

                obj = json.loads(text)

                # Handle different question formats
                if "question" in obj and "questions" not in obj:
                    obj["questions"] = [obj["question"]]
                elif "questions" not in obj:
                    obj["questions"] = []

                assert "passage" in obj
            except Exception as e:
                print(f"Failed to parse output: {e}")
                print(f"Text: {cand.text}")
                continue

            obj["_logprob"] = self._score_completion(cand)
            pairs.append(obj)

        pairs.sort(key=lambda d: d["_logprob"], reverse=True)
        return pairs[:top_k] if top_k else pairs


    ##ny
    def generate(self, claims, keep_k=4):
        prompts = [self._prompt(c) for c in claims]
        outputs = self.llm.generate(prompts, self.sampling)
        return [self._parse_outputs(out, keep_k) for out in outputs]


PROMPT_TEMPLATE = """You are a professional fact-checker.

<claim>
{{CLAIM}}
</claim>

INSTRUCTIONS:
1. Please write a fact-checking article passage to support, refute, indicate not enough evidence, or present conflicting evidence regarding the claim.
2. Generate 3-5 concise verifying questions whose answers would help decide the claim's truth value.

Return ONLY valid JSON with keys "passage" and "questions" (where "questions" is an array of strings).
Example format:
{
  "passage": "Your fact-checking passage here",
  "questions": ["Question 1?", "Question 2?", "Question 3?"]
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