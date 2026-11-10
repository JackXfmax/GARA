"""
train_ra_lora_v3.py  — Aligned with ckpt_qwen baseline

RA-LoRA: LoRA + Representation Alignment Adapter
All hyperparameters match ckpt_qwen (train_qwen_clean.py) exactly:
  - LoRA: r=8, alpha=16, dropout=0.05, all 7 target modules
  - 4-bit NF4 quantization + bf16 compute
  - 2 epochs, lr=2e-4, batch=8, grad_accum=4 (effective=32)
  - warmup_ratio=0.05, cosine schedule
  - Qwen apply_chat_template, same system prompt
  - max_length=512, gradient_checkpointing

Only difference from ckpt_qwen: + Alignment Adapter on embed_tokens

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_ra_lora_v3.py --use_adapter
  CUDA_VISIBLE_DEVICES=0 python train_ra_lora_v3.py  # LoRA only (for ablation)
"""

import os, json, random, time, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics.pairwise import cosine_similarity
import jieba, numpy as np

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = '0'

# === Args ===
import argparse
parser = argparse.ArgumentParser(description='RA-LoRA v3 (aligned with ckpt_qwen)')
parser.add_argument('--use_adapter', action='store_true', help='Enable Alignment Adapter')
parser.add_argument('--use_gate', action='store_true', help='Use gate mechanism')
parser.add_argument('--train_n', type=int, default=0, help='Training samples (0=all)')
parser.add_argument('--epochs', type=int, default=2, help='Max epochs (default=2, match ckpt_qwen)')
parser.add_argument('--eval_interval', type=int, default=5000, help='Eval every N optimizer steps')
parser.add_argument('--eval_samples', type=int, default=300, help='Eval sample count')
args = parser.parse_args()

# === Config (ALIGNED with ckpt_qwen) ===
LORA_R = 8               # ckpt_qwen: r=8
LORA_ALPHA = 16           # ckpt_qwen: alpha=16
LORA_DROPOUT = 0.05       # ckpt_qwen: dropout=0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4      # ckpt_qwen: 2e-4
BATCH_SIZE = 4            # Reduced from 8 to fit adapter in 24GB
GRAD_ACCUM = 8            # Increased from 4 to keep effective batch=32
MAX_EPOCHS = args.epochs  # ckpt_qwen: 2
MAX_LEN = 512             # ckpt_qwen: 512
WARMUP_RATIO = 0.05       # ckpt_qwen: 0.05
EVAL_INTERVAL = args.eval_interval
EVAL_SAMPLES = args.eval_samples

TRAIN_FILE = os.environ.get('GARA_TRAIN_FILE') or _TRAIN_DEFAULT
EVAL_FILE = os.environ.get('GARA_EVAL_FILE') or _EVAL_DEFAULT
MODEL_PATH = os.environ.get('GARA_MODEL_PATH') or 'Qwen/Qwen2.5-7B-Instruct'
SIM_MODEL_PATH = os.environ.get('GARA_SIM_MODEL_PATH') or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEVICE = 'cuda:0'  # CUDA_VISIBLE_DEVICES remaps to cuda:0

# === System Prompt (MATCH ckpt_qwen exactly) ===
SYSTEM_PROMPT = "你是一个可靠的翻译专家，请准确地为提供翻译文本。"

METHOD = 'lora'
if args.use_adapter and args.use_gate:
    METHOD = 'ra_lora_gate'
elif args.use_adapter:
    METHOD = 'ra_lora'

OUTPUT_DIR = os.environ.get('GARA_OUTPUT_DIR') or os.path.join(_REPO_ROOT, 'outputs', METHOD)
LOG_FILE = os.path.join(OUTPUT_DIR, '..', f'train_{METHOD}.log')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_prompt(tokenizer, bo, zh=None):
    """Build prompt using Qwen apply_chat_template — MATCHES ckpt_qwen format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请将Wylie: {bo}"},  # ckpt_qwen format
    ]
    if zh:
        messages.append({"role": "assistant", "content": zh})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# === Logging ===
log_f = open(LOG_FILE, 'w', buffering=1)
def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()


# === Dataset ===
class TibetanDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n_samples=None):
        data = load_pairs(path)
        if n_samples and n_samples > 0:
            random.shuffle(data)
            data = data[:n_samples]
        self.samples = []
        for d in data:
            bo, zh = d.get('bo', ''), d.get('zh', '')
            if not bo or not zh:
                continue
            full_text = make_prompt(tokenizer, bo, zh)
            prompt_text = make_prompt(tokenizer, bo)
            ids = tokenizer(full_text, truncation=True, max_length=max_len,
                            return_tensors='pt')['input_ids'][0]
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len,
                                   return_tensors='pt')['input_ids'][0]
            labels = ids.clone()
            labels[:len(prompt_ids)] = -100
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch):
    ids, labels = zip(*batch)
    max_len = max(x.size(0) for x in ids)
    pad_ids = torch.zeros(len(ids), max_len, dtype=torch.long)
    pad_labels = torch.full((len(ids), max_len), -100, dtype=torch.long)
    attn = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids, labels)):
        pad_ids[i, -x.size(0):] = x
        pad_labels[i, -y.size(0):] = y
        attn[i, -x.size(0):] = 1
    return pad_ids, pad_labels, attn


# === Evaluate ===
from sentence_transformers import SentenceTransformer


def get_vram_mb():
    try:
        return torch.cuda.memory_allocated(0) / 1024 ** 2
    except:
        return -1


def evaluate(model, tokenizer, eval_data, sim_model, n=300):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model.eval()
    samples = random.sample(eval_data, min(n, len(eval_data)))
    sims, kws = [], []
    with torch.no_grad():
        for i, d in enumerate(samples):
            bo, ref = d.get('bo', ''), d.get('zh', '')
            if not bo or not ref:
                continue
            prompt = make_prompt(tokenizer, bo)
            inputs = tokenizer(prompt, return_tensors='pt',
                               truncation=True, max_length=512).to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model.generate(**inputs, max_new_tokens=128,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                   skip_special_tokens=True).strip()
            del out, inputs
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            emb = sim_model.encode([gen, ref])
            s = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
            pred_w = set(jieba.cut(gen))
            ref_w = set(jieba.cut(ref))
            # recall: intersection / len(ref)
            k = len(pred_w & ref_w) / len(ref_w) if ref_w else 0.0
            sims.append(s)
            kws.append(k)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.train()
    return float(np.mean(sims)), float(np.mean(kws))


# === Main ===
def main():
    log(f"=== RA-LoRA v3 (aligned): method={METHOD}, train_n={args.train_n}, max_len={MAX_LEN} ===")
    log(f"   LORA_R={LORA_R}, LORA_ALPHA={LORA_ALPHA}, epochs={MAX_EPOCHS}, lr={LEARNING_RATE}")
    log(f"   BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM}, warmup={WARMUP_RATIO}")
    log(f"   4-bit NF4 + bf16, gradient_checkpointing=True")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify prompt format
    sample_prompt = make_prompt(tokenizer, "bskor ba")
    log(f"   Sample prompt: {sample_prompt[:300]}...")

    # Model — 4-bit NF4 (MATCH ckpt_qwen)
    log("Loading Qwen2.5-7B-Instruct (4-bit NF4 + bf16)...")
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
    model.config.use_cache = False

    # LoRA (MATCH ckpt_qwen: r=8, alpha=16)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)

    # Alignment Adapter
    adapter_params = 0
    if args.use_adapter:
        from alignment_adapter import create_adapter, install_adapter, save_adapter, count_adapter_params
        hidden_size = model.base_model.model.model.config.hidden_size
        log(f"Creating Alignment Adapter: hidden_size={hidden_size}, gate={args.use_gate}")
        adapter = create_adapter(hidden_size, use_gate=args.use_gate).to(DEVICE)
        adapter_params = count_adapter_params(adapter)
        log(f"  Adapter params: {adapter_params / 1e6:.2f}M")
        handle = install_adapter(model, adapter)
        log("  Adapter installed via forward hook on embed_tokens")
    else:
        adapter = None

    # gradient_checkpointing (MATCH ckpt_qwen)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = trainable - adapter_params
    log(f"Trainable params: total={trainable / 1e6:.2f}M, LoRA={lora_params / 1e6:.2f}M, Adapter={adapter_params / 1e6:.2f}M")

    # Data
    log("Loading data...")
    with open(EVAL_FILE) as f:
        eval_data = json.load(f)
    train_n = args.train_n if args.train_n > 0 else None
    train_ds = TibetanDataset(TRAIN_FILE, tokenizer, MAX_LEN, n_samples=train_n)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=False)
    log(f"  Train: {len(train_ds)} samples, Eval: {len(eval_data)} samples")

    # Sim model
    sim_model = SentenceTransformer(SIM_MODEL_PATH, device='cpu')

    # Optimizer & Scheduler (MATCH ckpt_qwen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    total_steps = len(train_dl) // GRAD_ACCUM * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # NO GradScaler for bf16
    log(f"  Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")
    log(f"  Steps per epoch: {len(train_dl) // GRAD_ACCUM}")

    # Training
    best_sim, global_step = 0.0, 0
    results_log = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (ids, labels, attn) in enumerate(train_dl):
            ids = ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attn = attn.to(DEVICE)

            # bf16 autocast (NO GradScaler)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 500 == 0:
                    avg = total_loss / (batch_idx + 1)
                    vr = get_vram_mb()
                    log(f"Ep{epoch + 1} Step{global_step}/{total_steps} loss={avg:.4f} VRAM={vr:.0f}MB")

                if global_step > 0 and global_step % EVAL_INTERVAL == 0:
                    log(f"Step{global_step}: evaluating...")
                    sim, kw = evaluate(model, tokenizer, eval_data, sim_model, EVAL_SAMPLES)
                    log(f"Step{global_step}: sim={sim:.4f} kw={kw:.4f}")
                    results_log.append({'step': global_step, 'sim': sim, 'kw': kw})

                    if sim > best_sim:
                        best_sim = sim
                        os.makedirs(f'{OUTPUT_DIR}/best', exist_ok=True)
                        model.save_pretrained(f'{OUTPUT_DIR}/best')
                        tokenizer.save_pretrained(f'{OUTPUT_DIR}/best')
                        if adapter is not None:
                            from alignment_adapter import save_adapter
                            save_adapter(adapter, f'{OUTPUT_DIR}/best/adapter_weights.pt')
                        log(f"  -> New best sim={best_sim:.4f}, saved checkpoint")

        # Epoch end eval
        torch.cuda.empty_cache()
        sim, kw = evaluate(model, tokenizer, eval_data, sim_model, EVAL_SAMPLES)
        avg_loss = total_loss / len(train_dl)
        elapsed = time.time() - epoch_start
        log(f"[Epoch{epoch + 1} END] loss={avg_loss:.4f} sim={sim:.4f} kw={kw:.4f} ({elapsed / 60:.0f}min)")
        results_log.append({
            'step': global_step, 'sim': sim, 'kw': kw,
            'loss': avg_loss, 'stage': f'epoch{epoch + 1}_end'
        })

        if sim > best_sim:
            best_sim = sim
            os.makedirs(f'{OUTPUT_DIR}/best', exist_ok=True)
            model.save_pretrained(f'{OUTPUT_DIR}/best')
            tokenizer.save_pretrained(f'{OUTPUT_DIR}/best')
            if adapter is not None:
                from alignment_adapter import save_adapter
                save_adapter(adapter, f'{OUTPUT_DIR}/best/adapter_weights.pt')
            log(f"  -> New best sim={best_sim:.4f}, saved checkpoint")

    # Save final
    os.makedirs(f'{OUTPUT_DIR}/final', exist_ok=True)
    model.save_pretrained(f'{OUTPUT_DIR}/final')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final')
    if adapter is not None:
        from alignment_adapter import save_adapter
        save_adapter(adapter, f'{OUTPUT_DIR}/final/adapter_weights.pt')

    # Save results
    final_result = {
        'method': METHOD,
        'use_adapter': args.use_adapter,
        'use_gate': args.use_gate,
        'train_samples': len(train_ds),
        'epochs': MAX_EPOCHS,
        'max_len': MAX_LEN,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'prompt_template': 'apply_chat_template (Qwen)',
        'system_prompt': SYSTEM_PROMPT,
        'kw_metric': 'recall',
        'quantization': '4-bit NF4 + bf16',
        'trainable_params_total': trainable,
        'trainable_params_lora': lora_params,
        'trainable_params_adapter': adapter_params,
        'best_sim': best_sim,
        'final_sim': sim,
        'final_kw': kw,
        'eval_samples': EVAL_SAMPLES,
        'history': results_log
    }
    with open(f'{OUTPUT_DIR}/result.json', 'w') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    log(f"=== DONE: {METHOD} best_sim={best_sim:.4f}, total_params={trainable / 1e6:.1f}M ===")
    log_f.close()


if __name__ == '__main__':
    main()
