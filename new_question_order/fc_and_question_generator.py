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
                       "passage" in parsed and "question" in parsed
            except (json.JSONDecodeError, AssertionError):
                continue

            score = self._score_completion(cand)
            if score > best_score:
                best_item, best_score = parsed, score

        #fallback – keep raw text
        if best_item is None:
            best_item = {"passage": request_output.outputs[0].text.strip(),
                         "question": ""}

        return best_item
    
    #ny
    def _parse_outputs(self, request_output, top_k=None):
        pairs = []
        for cand in request_output.outputs:
            try:
                obj = json.loads(cand.text.strip())
                assert {"passage", "question"} <= obj.keys()
            except Exception:
                continue
            obj["_logprob"] = self._score_completion(cand)
            pairs.append(obj)

        pairs.sort(key=lambda d: d["_logprob"], reverse=True)
        return pairs[:top_k] if top_k else pairs 
    
    ##ny
    def generate(self, claims, keep_k=4): #Hva bør k være?
        prompts = [self._prompt(c) for c in claims]
        outputs = self.llm.generate(prompts, self.sampling)
        """
        for out in outputs:
            best = max(out.outputs, key=lambda o: sum(
                p.logprob for lp in o.logprobs for p in lp.values()))
            try:
                parsed.append(json.loads(best.text.strip()))
            except json.JSONDecodeError:
                parsed.append({"passage": best.text.strip(), "question": ""})
        """
        return [self._parse_outputs(out, keep_k) for out in outputs]
    
PROMPT_TEMPLATE = """You are a professional fact-checker.

<claim>
{{CLAIM}}
</claim>

INSTRUCTIONS:
1. Please write a fact-checking article passage to support, refute, indicate not enough evidence, or present conflicting evidence regarding the claim. 
2. Write one concise verifying question whose answer would decide the claim’s truth value.

Return ONLY valid JSON with keys "passage" and "question".
"""

def main(args):
    with open(args.target_data, encoding="utf-8") as f:
        data = json.load(f)

    gen = FCQGenerator(args.model, n=args.n)

    all_out = []
    for i in range(0, len(data), gen.batch_size):
        batch = data[i:i+gen.batch_size]
        res = gen.generate([ex["claim"] for ex in batch])
        for ex, pairs in zip(batch, res):
            for ex, pairs in zip(batch, batch_pairs):
                passages = [p["passage"] for p in pairs]          # keep every passage
                best_passage   = passages[0] if passages else ""
                ex["hypo_fc_docs"] = passages            # 〈— list[str]  (old HERO key)
                ex["hypo_passage"] = best_passage        # 〈— str        (was added later)
                ex["question"]   = pairs[0]["question"] if pairs else ""
                ex["fcq_pairs"]  = pairs                 # optional: full structure

                all_out.append(ex)

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)
    print(f"Done — {len(all_out)} examples written to {args.json_output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_data",  default="data_store/averitec/dev.json")
    p.add_argument("--json_output", default="data_store/dev_fcq.json")
    p.add_argument("--model",  default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--n",       type=int, default=8,
               help="number of completions vLLM draws per prompt")
    p.add_argument("--keep_k",  type=int, default=4,
               help="how many of the completions to keep per claim")
    main(p.parse_args())