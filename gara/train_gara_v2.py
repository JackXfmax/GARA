"""
train_gara_v2.py  — GARA: Gap-Aware Representation Alignment

Based on train_ra_lora_v3.py (aligned with ckpt_qwen), with dynamic lambda:
  - lambda_i = lambda_max * clamp(gap_i / 2.0, 0, 1)
  - gap_i = 1 - cosine(z_w, z_c)
  - z_w = mean-pooled adapted embedding of prompt tokens (through adapter)
  - z_c = mean-pooled raw embedding of chinese tokens (F.embedding bypasses adapter)
  - total_loss = lm_loss + mean(lambda_i * gap_i)

All hyperparameters match ckpt_qwen:
  - LoRA: r=8, alpha=16, dropout=0.05, all 7 target modules
  - 4-bit NF4 + bf16, 2 epochs, lr=2e-4
  - batch=4, grad_accum=8 (effective=32), max_len=512
  - Qwen apply_chat_template, gradient_checkpointing

RESUME SUPPORT (v2.1):
  - Auto-saves resume checkpoint every SAVE_INTERVAL steps to ra_lora/gara/resume/
  - Checkpoint includes: optimizer, scheduler, global_step, epoch, best_sim, results_log, gap_history
  - To resume: python train_gara_v2.py --resume ra_lora/gara/resume/

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_gara_v2.py
  CUDA_VISIBLE_DEVICES=0 python train_gara_v2.py --resume ra_lora/gara/resume/
"""

import os, json, random, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics.pairwise import cosine_similarity
import jieba, numpy as np

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = '0'

# === Args ===
import argparse
parser = argparse.ArgumentParser(description='GARA v2 (aligned with ckpt_qwen)')
parser.add_argument('--train_n', type=int, default=0, help='Training samples (0=all)')
parser.add_argument('--epochs', type=int, default=2, help='Max epochs')
parser.add_argument('--eval_interval', type=int, default=5000, help='Eval every N steps')
parser.add_argument('--eval_samples', type=int, default=300, help='Eval sample count')
parser.add_argument('--lambda_max', type=float, default=1.0, help='Max lambda for GARA')
parser.add_argument('--save_interval', type=int, default=1000, help='Save resume checkpoint every N steps')
parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint dir (ra_lora/gara/resume/)')
# --- release-friendly path overrides: CLI > env GARA_* > repo default ---
parser.add_argument('--train_file', type=str, default=None, help='Train file (.json / .jsonl / .jsonl.gz)')
parser.add_argument('--eval_file', type=str, default=None, help='Eval file (.json / .jsonl / .jsonl.gz)')
parser.add_argument('--model_path', type=str, default=None, help='Base LLM path')
parser.add_argument('--sim_model_path', type=str, default=None, help='Sentence-similarity model path')
parser.add_argument('--output_dir', type=str, default=None, help='Checkpoint / log output dir')
parser.add_argument('--device', type=str, default=None, help='e.g. cuda:0')
args = parser.parse_args()

# === Config (ALIGNED with ckpt_qwen) ===
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
BATCH_SIZE = 4             # Reduced from 8 to fit adapter in 24GB
GRAD_ACCUM = 8             # Increased from 4 to keep effective batch=32
MAX_EPOCHS = args.epochs
MAX_LEN = 512
WARMUP_RATIO = 0.05
EVAL_INTERVAL = args.eval_interval
EVAL_SAMPLES = args.eval_samples
LAMBDA_MAX = args.lambda_max
SAVE_INTERVAL = args.save_interval

# Repo root (parent of this file's directory) — used for default data paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(flag, env_name, default):
    """CLI flag > environment variable > default."""
    return getattr(args, flag, None) or os.environ.get(env_name) or default


TRAIN_FILE = _resolve('train_file', 'GARA_TRAIN_FILE',
                      os.path.join(_REPO_ROOT, 'data', 'train.jsonl.gz'))
EVAL_FILE = _resolve('eval_file', 'GARA_EVAL_FILE',
                     os.path.join(_REPO_ROOT, 'data', 'test.jsonl.gz'))
MODEL_PATH = _resolve('model_path', 'GARA_MODEL_PATH',
                      'Qwen/Qwen2.5-7B-Instruct')
SIM_MODEL_PATH = _resolve('sim_model_path', 'GARA_SIM_MODEL_PATH',
                          'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
DEVICE = _resolve('device', 'GARA_DEVICE', 'cuda:0')

SYSTEM_PROMPT = "你是一个可靠的翻译专家，请准确地为提供翻译文本。"

METHOD = 'gara'
OUTPUT_DIR = _resolve('output_dir', 'GARA_OUTPUT_DIR',
                      os.path.join(_REPO_ROOT, 'outputs', METHOD))
RESUME_DIR = f'{OUTPUT_DIR}/resume'
LOG_FILE = f'{OUTPUT_DIR}/../train_{METHOD}.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)


def make_prompt(tokenizer, bo, zh=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请将Wylie: {bo}"},
    ]
    if zh:
        messages.append({"role": "assistant", "content": zh})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# === Logging ===
log_f = open(LOG_FILE, 'a', buffering=1)  # append mode for resume
def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()


# === Data loading ===
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
            k = len(pred_w & ref_w) / len(ref_w) if ref_w else 0.0
            sims.append(s)
            kws.append(k)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.train()
    return float(np.mean(sims)), float(np.mean(kws))


def save_resume_checkpoint(model, adapter, optimizer, scheduler, global_step, epoch,
                            best_sim, results_log, gap_history):
    """Save full resume checkpoint (LoRA weights + adapter + optimizer + scheduler + state)."""
    os.makedirs(RESUME_DIR, exist_ok=True)
    # Save LoRA weights
    model.save_pretrained(RESUME_DIR)
    # Save adapter weights
    from alignment_adapter import save_adapter
    save_adapter(adapter, f'{RESUME_DIR}/adapter_weights.pt')
    # Save optimizer + scheduler + training state
    state = {
        'global_step': global_step,
        'epoch': epoch,
        'best_sim': best_sim,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'results_log': results_log,
        'gap_history': gap_history,
    }
    torch.save(state, f'{RESUME_DIR}/training_state.pt')
    log(f"  [SAVED] Resume checkpoint at step {global_step} -> {RESUME_DIR}")


def load_resume_checkpoint(resume_dir, model, adapter, optimizer, scheduler):
    """Load resume checkpoint. Returns (global_step, start_epoch, best_sim, results_log, gap_history)."""
    from alignment_adapter import load_adapter
    state_path = f'{resume_dir}/training_state.pt'
    state = torch.load(state_path, map_location='cpu')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    # Move optimizer state to GPU
    for opt_state in optimizer.state.values():
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor):
                opt_state[k] = v.to(DEVICE)
    global_step = state['global_step']
    epoch = state['epoch']
    best_sim = state['best_sim']
    results_log = state.get('results_log', [])
    gap_history = state.get('gap_history', [])
    # Load adapter weights
    adapter_path = f'{resume_dir}/adapter_weights.pt'
    if os.path.exists(adapter_path):
        load_adapter(adapter, adapter_path)
        log(f"  Adapter weights loaded from {adapter_path}")
    log(f"  Resumed: global_step={global_step}, epoch={epoch}, best_sim={best_sim:.4f}")
    return global_step, epoch, best_sim, results_log, gap_history


# === Main ===
def main():
    log(f"=== GARA v2 (aligned): lambda_max={LAMBDA_MAX}, train_n={args.train_n}, max_len={MAX_LEN} ===")
    log(f"   LORA_R={LORA_R}, LORA_ALPHA={LORA_ALPHA}, epochs={MAX_EPOCHS}, lr={LEARNING_RATE}")
    log(f"   BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM}, warmup={WARMUP_RATIO}")
    log(f"   4-bit NF4 + bf16, gradient_checkpointing=True")
    log(f"   GARA: dynamic lambda = {LAMBDA_MAX} * clamp(gap/2, 0, 1)")
    if args.resume:
        log(f"   RESUME from: {args.resume}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_prompt = make_prompt(tokenizer, "bskor ba")
    log(f"   Sample prompt: {sample_prompt[:300]}...")

    # Model — 4-bit NF4
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

    # LoRA — load from resume if available
    if args.resume and os.path.exists(f'{args.resume}/adapter_config.json'):
        log(f"  Loading LoRA weights from resume: {args.resume}")
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET, task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_cfg)

    # Alignment Adapter (always enabled for GARA)
    from alignment_adapter import create_adapter, install_adapter, save_adapter, count_adapter_params
    hidden_size = model.base_model.model.model.config.hidden_size
    log(f"Creating Alignment Adapter: hidden_size={hidden_size}")
    adapter = create_adapter(hidden_size, use_gate=False).to(DEVICE)
    adapter_params = count_adapter_params(adapter)
    log(f"  Adapter params: {adapter_params / 1e6:.2f}M")
    handle = install_adapter(model, adapter)
    log("  Adapter installed via forward hook on embed_tokens")

    # gradient_checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = trainable - adapter_params
    log(f"Trainable params: total={trainable / 1e6:.2f}M, LoRA={lora_params / 1e6:.2f}M, Adapter={adapter_params / 1e6:.2f}M")

    # Data
    log("Loading data...")
    eval_data = load_pairs(EVAL_FILE)
    train_n = args.train_n if args.train_n > 0 else None
    train_ds = TibetanDataset(TRAIN_FILE, tokenizer, MAX_LEN, n_samples=train_n)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=False)
    log(f"  Train: {len(train_ds)} samples, Eval: {len(eval_data)} samples")

    sim_model = SentenceTransformer(SIM_MODEL_PATH, device='cpu')

    # Optimizer
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
    log(f"  Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")
    log(f"  Steps per epoch: {len(train_dl) // GRAD_ACCUM}")

    # Get embed_tokens reference for GARA
    embed_layer = model.base_model.model.model.embed_tokens
    log(f"  embed_tokens weight shape: {embed_layer.weight.shape}, dtype: {embed_layer.weight.dtype}")

    # Training state
    best_sim, global_step = 0.0, 0
    results_log = []
    gap_history = []
    start_epoch = 0

    # Load resume checkpoint
    if args.resume and os.path.exists(f'{args.resume}/training_state.pt'):
        global_step, start_epoch, best_sim, results_log, gap_history = \
            load_resume_checkpoint(args.resume, model, adapter, optimizer, scheduler)
        log(f"  Resuming from epoch={start_epoch}, step={global_step}")

    # Training
    steps_per_epoch = len(train_dl) // GRAD_ACCUM

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total_lm_loss = 0.0
        total_align_loss = 0.0
        total_gap = 0.0
        total_lambda = 0.0
        epoch_start = time.time()

        # If resuming mid-epoch, figure out how many batches to skip
        # (We skip by epoch boundary only — simpler and data is shuffled anyway)
        start_batch = 0
        if epoch == start_epoch and global_step > 0:
            steps_done_in_epoch = global_step % steps_per_epoch
            start_batch = steps_done_in_epoch * GRAD_ACCUM
            log(f"  Epoch {epoch+1}: skipping {start_batch} batches (already done {steps_done_in_epoch} steps this epoch)")

        for batch_idx, (ids, labels, attn) in enumerate(train_dl):
            # Skip already-processed batches when resuming
            if batch_idx < start_batch:
                continue

            ids = ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attn = attn.to(DEVICE)

            # LM forward pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=ids, attention_mask=attn, labels=labels)
                lm_loss = outputs.loss / GRAD_ACCUM

            # === GARA Alignment Loss ===
            # z_source: adapted embeddings (through adapter hook) — gradient flows to adapter
            z_source = embed_layer(ids).float()  # [B, L, H]

            # z_target: raw embeddings (bypass adapter hook via F.embedding) — no gradient
            with torch.no_grad():
                z_target = F.embedding(ids, embed_layer.weight.detach()).float()  # [B, L, H]

            # Masks: prompt tokens (labels==-100) vs chinese tokens (labels!=-100)
            source_mask = (labels == -100).unsqueeze(-1).float()  # [B, L, 1]
            target_mask = (labels != -100).unsqueeze(-1).float()  # [B, L, 1]

            # Mean pool
            z_w = (z_source * source_mask).sum(1) / source_mask.sum(1).clamp(min=1)  # [B, H]
            z_c = (z_target * target_mask).sum(1) / target_mask.sum(1).clamp(min=1)  # [B, H]

            # Gap and dynamic lambda
            gap = 1.0 - F.cosine_similarity(z_w, z_c, dim=-1)  # [B]
            lambda_i = LAMBDA_MAX * (gap.detach() / 2.0).clamp(0, 1)  # [B], detached

            # Weighted alignment loss
            align_loss = (lambda_i * gap).mean() / GRAD_ACCUM

            # Total loss
            total_loss = lm_loss + align_loss
            total_loss.backward()

            # Track metrics
            total_lm_loss += lm_loss.item() * GRAD_ACCUM
            total_align_loss += align_loss.item() * GRAD_ACCUM
            total_gap += gap.detach().mean().item()
            total_lambda += lambda_i.mean().item()

            # Free alignment tensors
            del z_source, z_target, z_w, z_c, gap, lambda_i, align_loss, outputs

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 500 == 0:
                    n_acc = batch_idx + 1 - start_batch
                    avg_lm = total_lm_loss / n_acc
                    avg_align = total_align_loss / n_acc
                    avg_gap = total_gap / n_acc
                    avg_lambda = total_lambda / n_acc
                    vr = get_vram_mb()
                    log(f"Ep{epoch+1} Step{global_step}/{total_steps} lm={avg_lm:.4f} "
                        f"align={avg_align:.4f} gap={avg_gap:.4f} lam={avg_lambda:.4f} "
                        f"VRAM={vr:.0f}MB")
                    gap_history.append({
                        'step': global_step,
                        'lm_loss': avg_lm,
                        'align_loss': avg_align,
                        'gap': avg_gap,
                        'lambda': avg_lambda
                    })

                # Save resume checkpoint
                if global_step % SAVE_INTERVAL == 0:
                    save_resume_checkpoint(model, adapter, optimizer, scheduler,
                                           global_step, epoch, best_sim,
                                           results_log, gap_history)

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
                        save_adapter(adapter, f'{OUTPUT_DIR}/best/adapter_weights.pt')
                        # Save optimizer/scheduler for perfect resume
                        torch.save(optimizer.state_dict(), f'{OUTPUT_DIR}/best/optimizer.pt')
                        torch.save(scheduler.state_dict(), f'{OUTPUT_DIR}/best/scheduler.pt')
                        log(f"  -> New best sim={best_sim:.4f}, saved checkpoint (with optimizer/scheduler)")

        # Epoch end eval
        torch.cuda.empty_cache()
        sim, kw = evaluate(model, tokenizer, eval_data, sim_model, EVAL_SAMPLES)
        n_acc_epoch = len(train_dl) - start_batch
        avg_lm = total_lm_loss / n_acc_epoch if n_acc_epoch > 0 else 0
        avg_align = total_align_loss / n_acc_epoch if n_acc_epoch > 0 else 0
        elapsed = time.time() - epoch_start
        log(f"[Epoch{epoch+1} END] lm={avg_lm:.4f} align={avg_align:.4f} "
            f"sim={sim:.4f} kw={kw:.4f} ({elapsed/60:.0f}min)")
        results_log.append({
            'step': global_step, 'sim': sim, 'kw': kw,
            'lm_loss': avg_lm, 'align_loss': avg_align,
            'stage': f'epoch{epoch+1}_end'
        })

        if sim > best_sim:
            best_sim = sim
            os.makedirs(f'{OUTPUT_DIR}/best', exist_ok=True)
            model.save_pretrained(f'{OUTPUT_DIR}/best')
            tokenizer.save_pretrained(f'{OUTPUT_DIR}/best')
            save_adapter(adapter, f'{OUTPUT_DIR}/best/adapter_weights.pt')
            # Save optimizer/scheduler for perfect resume
            torch.save(optimizer.state_dict(), f'{OUTPUT_DIR}/best/optimizer.pt')
            torch.save(scheduler.state_dict(), f'{OUTPUT_DIR}/best/scheduler.pt')
            log(f"  -> New best sim={best_sim:.4f}, saved checkpoint (with optimizer/scheduler)")

        # Reset start_batch after first epoch
        start_batch = 0

    # Save final
    os.makedirs(f'{OUTPUT_DIR}/final', exist_ok=True)
    model.save_pretrained(f'{OUTPUT_DIR}/final')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final')
    save_adapter(adapter, f'{OUTPUT_DIR}/final/adapter_weights.pt')

    # Save results
    final_result = {
        'method': METHOD,
        'lambda_max': LAMBDA_MAX,
        'train_samples': len(train_ds),
        'epochs': MAX_EPOCHS,
        'max_len': MAX_LEN,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'batch_size': BATCH_SIZE,
        'grad_accum': GRAD_ACCUM,
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
        'history': results_log,
        'gap_history': gap_history
    }
    with open(f'{OUTPUT_DIR}/result.json', 'w') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    log(f"=== DONE: {METHOD} best_sim={best_sim:.4f}, total_params={trainable/1e6:.1f}M ===")
    log_f.close()


if __name__ == '__main__':
    main()
