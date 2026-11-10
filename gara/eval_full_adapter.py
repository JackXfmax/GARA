"""
eval_full_adapter.py — Full 94321-sample evaluation for RA-LoRA v3 and GARA v2
Loads 4-bit NF4 model + LoRA + alignment adapter, generates translations
for all eval samples, computes sim & kw metrics.
Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_full_adapter.py --method gara
  CUDA_VISIBLE_DEVICES=1 python eval_full_adapter.py --method ra_lora
"""
import os, json, time, sys, argparse
import torch
import torch.nn as nn
import numpy as np
import jieba
import warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = '0'
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from alignment_adapter import create_adapter, install_adapter, count_adapter_params
# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True, choices=['ra_lora', 'gara'],
                    help='Which method to evaluate')
parser.add_argument('--checkpoint', type=str, default='best',
                    help='Checkpoint subdirectory (best or final)')
parser.add_argument('--max_new_tokens', type=int, default=128)
parser.add_argument('--progress_interval', type=int, default=500,
                    help='Print progress every N samples')
parser.add_argument('--num_samples', type=int, default=0,
                    help='Number of samples to evaluate (0 = all)')
parser.add_argument('--save_interval', type=int, default=1000,
                    help='Save intermediate results every N samples')
parser.add_argument('--model_path', type=str, default=None, help='Base LLM path')
parser.add_argument('--eval_data', type=str, default=None, help='Eval file (.json / .jsonl / .jsonl.gz)')
parser.add_argument('--sim_model_path', type=str, default=None, help='Sentence-similarity model path')
parser.add_argument('--ckpt_dir', type=str, default=None, help='LoRA checkpoint dir')
parser.add_argument('--output_file', type=str, default=None, help='Output JSON path')
parser.add_argument('--device', type=str, default=None, help='e.g. cuda:0')
args = parser.parse_args()
# === Config ===
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(flag, env_name, default):
    """CLI flag > environment variable > default."""
    return getattr(args, flag, None) or os.environ.get(env_name) or default


MODEL_PATH = _resolve('model_path', 'GARA_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')
EVAL_DATA = _resolve('eval_data', 'GARA_EVAL_FILE',
                     os.path.join(_REPO_ROOT, 'data', 'test.jsonl.gz'))
SIM_MODEL_PATH = _resolve('sim_model_path', 'GARA_SIM_MODEL_PATH',
                          'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
DEVICE = _resolve('device', 'GARA_DEVICE', 'cuda:0')  # Will use CUDA_VISIBLE_DEVICES


def load_pairs(path):
    """Load parallel pairs from .json (list), .jsonl or .jsonl.gz."""
    if path.endswith('.gz'):
        import gzip
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.endswith('.jsonl'):
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding='utf-8') as f:
        return json.load(f)


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
SYSTEM_PROMPT = "你是一个可靠的翻译专家，请准确地为提供翻译文本。"
CKPT_DIR = _resolve('ckpt_dir', 'GARA_CKPT_DIR',
                    os.path.join(_REPO_ROOT, 'outputs', args.method, args.checkpoint))
OUTPUT_FILE = _resolve('output_file', 'GARA_OUTPUT_FILE',
                       os.path.join(_REPO_ROOT, 'results', f'eval_{args.method}_full.json'))
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
def make_prompt(tokenizer, bo):
    """Build prompt using Qwen apply_chat_template — MATCHES training format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请将Wylie: {bo}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
def get_kw_score(pred, ref):
    """Keyword recall: |pred_words ∩ ref_words| / |ref_words|"""
    pred_words = set(jieba.cut(pred))
    ref_words = set(jieba.cut(ref))
    if not ref_words:
        return 0.0
    return len(pred_words & ref_words) / len(ref_words)
def main():
    print(f"=== Full Evaluation: {args.method} ({args.checkpoint}) ===", flush=True)
    start_time = time.time()
    # 1. Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 2. Load base model (4-bit NF4 + bf16)
    print("Loading Qwen2.5-7B-Instruct (4-bit NF4 + bf16)...", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map={"": DEVICE},
        trust_remote_code=True
    )
    model.config.use_cache = True  # Enable cache for inference
    # Load LoRA weights directly via PeftModel.from_pretrained (no double-wrapping)
    print(f"Loading LoRA+adapter weights from {CKPT_DIR}...", flush=True)
    model = PeftModel.from_pretrained(model, CKPT_DIR)
    # Disable gradient checkpointing for inference
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    # 4. Create and install alignment adapter
    print("Creating alignment adapter...", flush=True)
    hidden_size = model.base_model.model.model.config.hidden_size  # 3584 for Qwen2.5-7B
    adapter = create_adapter(hidden_size, use_gate=False).to(DEVICE)
    adapter_params = count_adapter_params(adapter)
    print(f"  Adapter params: {adapter_params / 1e6:.2f}M", flush=True)
    # Load adapter weights
    adapter_weights_path = os.path.join(CKPT_DIR, 'adapter_weights.pt')
    print(f"Loading adapter weights from {adapter_weights_path}...", flush=True)
    adapter.load_state_dict(torch.load(adapter_weights_path, map_location=DEVICE))
    # Install adapter via forward hook on embed_tokens
    # PeftModel path: model.base_model.model.model.embed_tokens
    embed_tokens = model.base_model.model.model.embed_tokens
    def hook_fn(module, input, output):
        with torch.autocast(device_type="cuda", enabled=False):
            return adapter(output.float()).to(output.dtype)
    handle = embed_tokens.register_forward_hook(hook_fn)
    print("  Adapter installed via forward hook on embed_tokens", flush=True)
    model.eval()
    # 5. Load eval data
    print(f"Loading eval data...", flush=True)
    data = load_pairs(EVAL_DATA)
    n_total = len(data)
    print(f"  Total eval samples: {n_total}", flush=True)
    # 6. Slice data if --num_samples specified
    if args.num_samples > 0 and args.num_samples < n_total:
        data = data[:args.num_samples]
        n_total = len(data)
        print(f"  Evaluating {n_total} samples (from {len(data)} total)", flush=True)
    # 7. Generate translations one-by-one (avoid OOM)
    print("Generating translations...", flush=True)
    all_preds = []
    all_refs = []
    kw_scores = []
    for i, item in enumerate(data):
        bo = item.get('bo', '')
        ref = item.get('zh', '')
        if not bo or not ref:
            all_preds.append('')
            all_refs.append(ref)
            kw_scores.append(0.0)
            continue
        prompt = make_prompt(tokenizer, bo)
        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                               skip_special_tokens=True).strip()
        # Free memory
        del out, inputs
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
        all_preds.append(gen)
        all_refs.append(ref)
        kw_scores.append(get_kw_score(gen, ref))
        # Save intermediate results every save_interval samples
        if args.save_interval > 0 and (i + 1) % args.save_interval == 0 and (i + 1) < n_total:
            int_data = {
                'method': args.method,
                'checkpoint': args.checkpoint,
                'n_evaluated': i + 1,
                'n_total': n_total,
                'intermediate': True,
                'all_results': [
                    {'bo_text': data[j].get('bo', ''), 'ref': all_refs[j], 'pred': all_preds[j], 'kw': kw_scores[j]}
                    for j in range(i + 1)
                ],
                'avg_kw': float(np.nanmean(kw_scores)),
            }
            int_path = os.path.join(os.path.dirname(OUTPUT_FILE) or '.',
                                    f'intermediate_{args.method}_{i+1}.json')
            with open(int_path, 'w') as f:
                json.dump(int_data, f, ensure_ascii=False)
            print(f'  Saved intermediate: {int_path}', flush=True)
        if (i + 1) % args.progress_interval == 0 or (i + 1) == n_total:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (n_total - i - 1) / speed if speed > 0 else 0
            mean_kw = np.nanmean(kw_scores)
            vram = torch.cuda.memory_allocated(0) / 1024**2
            print(f"  [{i+1}/{n_total}] kw_mean={mean_kw:.4f} "
                  f"speed={speed:.1f} samples/s ETA={eta/60:.0f}min VRAM={vram:.0f}MB", flush=True)
    # Remove hook before computing embeddings
    handle.remove()
    torch.cuda.empty_cache()
    # 7. Compute similarity (SentenceTransformer on CPU)
    print("Computing semantic similarity...", flush=True)
    sim_model = SentenceTransformer(SIM_MODEL_PATH, device='cpu')
    # Encode in batches to avoid memory issues
    pred_embs = sim_model.encode(all_preds, batch_size=256, show_progress_bar=True)
    ref_embs = sim_model.encode(all_refs, batch_size=256, show_progress_bar=True)
    sims = [float(cosine_similarity([p], [r])[0][0]) for p, r in zip(pred_embs, ref_embs)]
    # 8. Aggregate results
    mean_sim = float(np.mean(sims))
    mean_kw = float(np.nanmean(kw_scores))
    kw50 = float(np.mean([k >= 0.5 for k in kw_scores]))
    kw30 = float(np.mean([k >= 0.3 for k in kw_scores]))
    total_time = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"[{args.method} Full Eval] sim={mean_sim:.4f} kw={mean_kw:.4f}", flush=True)
    print(f"  kw>=0.5: {kw50*100:.1f}%  kw>=0.3: {kw30*100:.1f}%  n={n_total}", flush=True)
    print(f"  Total time: {total_time/60:.1f}min", flush=True)
    print(f"{'='*60}", flush=True)
    # 9. Save results
    results = {
        "model": f"{args.method} ({args.checkpoint})",
        "method": args.method,
        "checkpoint": args.checkpoint,
        "sim": mean_sim,
        "kw": mean_kw,
        "kw50": kw50,
        "kw30": kw30,
        "n": n_total,
        "params_total": 45882368,
        "params_lora": 20185088,
        "params_adapter": adapter_params,
        "max_new_tokens": args.max_new_tokens,
        "total_time_min": round(total_time / 60, 1),
        "quantization": "4-bit NF4 + bf16",
        "prompt_template": "apply_chat_template (Qwen)",
        "system_prompt": SYSTEM_PROMPT,
        "kw_metric": "recall"
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {OUTPUT_FILE}", flush=True)
if __name__ == '__main__':
    main()
