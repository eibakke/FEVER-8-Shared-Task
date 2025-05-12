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
        self.llm = LLM(model=model_name,
                       tensor_parallel_size=gpus,
                       max_model_len=4096,
                       gpu_memory_utilization=0.95,
                       trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()
        self.batch_size = batch_size
        self.sampling = SamplingParams(
            n=n, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
            stop=['```', '</s>', '<|end_of_text|>'], logprobs=1
        )

    def _prompt(self, claim):
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.replace("{{CLAIM}}", claim)}]
        chat = self.tokenizer.apply_chat_template(msgs, tokenize=False)
        return chat + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def generate(self, claims):
        prompts = [self._prompt(c) for c in claims]
        outputs = self.llm.generate(prompts, self.sampling)
        parsed = []
        for out in outputs:
            best = max(out.outputs, key=lambda o: sum(
                p.logprob for lp in o.logprobs for p in lp.values()))
            try:
                parsed.append(json.loads(best.text.strip()))
            except json.JSONDecodeError:
                parsed.append({"passage": best.text.strip(), "question": ""})
        return parsed

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

    gen = FCQGenerator(args.model)
    all_out = []
    for i in range(0, len(data), gen.batch_size):
        batch = data[i:i+gen.batch_size]
        res = gen.generate([ex["claim"] for ex in batch])
        for ex, pair in zip(batch, res):
            ex["question"] = pair["question"]
            ex["hypo_passage"] = pair["passage"]
            all_out.append(ex)

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)
    print(f"Done — {len(all_out)} examples written to {args.json_output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_data",  default="data_store/averitec/dev.json")
    p.add_argument("--json_output", default="data_store/dev_fcq.json")
    p.add_argument("--model",  default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    main(p.parse_args())