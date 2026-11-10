# GARA: Gap-Aware Representation Alignment for Wylie → Classical Chinese Translation

Official repository for **GARA (Gap-Aware Representation Alignment)**, a
parameter-efficient fine-tuning method for translating Tibetan **Wylie
romanisation** into **Classical Chinese (文言文)**.

**Maintained by Tibet University (西藏大学).**

GARA augments LoRA with a lightweight **Alignment Adapter** inserted after the
embedding layer, and trains it with a **gap-aware dynamic weight**:

- `z_w` = mean-pooled *adapted* embedding of the Wylie prompt tokens
- `z_c` = mean-pooled *raw* embedding of the Chinese target tokens (bypasses the adapter)
- `gap_i = 1 - cosine(z_w, z_c)`
- `lambda_i = lambda_max * clamp(gap_i / 2, 0, 1)`
- `loss = lm_loss + mean(lambda_i * gap_i)`

The intuition: Wylie transliteration sits far outside the base model's
representation manifold. The adapter closes that gap, and the gap term itself
supplies a per-sample adaptive weight that decays as alignment improves.

---

## Repository layout

```
GARA/
├── gara/                     # method code
│   ├── alignment_adapter.py  # AlignmentAdapter (and gated variant) + install hook
│   ├── train_gara_v2.py      # main GARA training script
│   ├── train_ra_lora_v3.py   # RA-LoRA baseline (adapter, no gap term)
│   ├── train_ablation.py     # ablation variants
│   └── eval_full_adapter.py  # evaluation (Sim + keyword recall)
├── scripts/
│   └── prepare_data.py       # data cleaning / deduplication pipeline
├── data/
│   └── demo_pairs.jsonl      # 40-pair demonstration subset of the corpus
├── CITATION.cff
└── README.md
```

---

## Demo data

Each line of `data/demo_pairs.jsonl` is a JSON object with two fields:

```json
{"bo": "yongsu dungwa dak l nampar drolwa", "zh": "悉离诸欲缠缚"}
```

- `bo` — the source side, Tibetan in **Wylie romanisation** (ASCII)
- `zh` — the target side, **Classical Chinese** (文言文)

This file is a **small curated demonstration subset** (40 pairs) intended to
illustrate the data format only. **No training data or experimental data are
included in this repository.**

### Data and code release policy

- The **full parallel corpus** and the **remaining code / experimental
  assets** will be released **progressively** in this repository.
- In the meantime, we are happy to share **representative subsets** of the
  corpus and additional materials with interested researchers **upon
  reasonable request** (please open an issue or contact the maintainers).

---

## Reproducing

### 1. Environment

```bash
conda create -n gara python=3.10
conda activate gara
pip install -r requirements.txt
```

### 2. Data

The demonstration subset is already included as `data/demo_pairs.jsonl`. The
full training/test corpus will be released progressively.

### 3. Train

Every machine-specific path is resolved as
**CLI flag > `GARA_*` env var > repo-relative default**, so nothing needs
editing:

```bash
export GARA_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export GARA_SIM_MODEL_PATH=/path/to/paraphrase-multilingual-MiniLM-L12-v2

CUDA_VISIBLE_DEVICES=0 python gara/train_gara_v2.py \
    --epochs 2 --lambda_max 1.0 --train_file /path/to/train.jsonl.gz \
    --eval_file /path/to/test.jsonl.gz
```

Resume after interruption (checkpoints are written every `--save_interval`
steps):

```bash
CUDA_VISIBLE_DEVICES=0 python gara/train_gara_v2.py --resume outputs/gara/resume/
```

Baseline (adapter without the gap term) and ablations:

```bash
CUDA_VISIBLE_DEVICES=0 python gara/train_ra_lora_v3.py
CUDA_VISIBLE_DEVICES=0 python gara/train_ablation.py
```

### 4. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python gara/eval_full_adapter.py --method gara --checkpoint best
```

---

## Licence

- **Code** — MIT (see `LICENSE`).
- **Data (demo subset)** — released for research use.

---

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`):

```bibtex
@misc{gara2026,
  title  = {GARA: Gap-Aware Representation Alignment for
            Wylie-to-Classical-Chinese Translation},
  author = {{Tibet University}},
  year   = {2026},
  url    = {https://github.com/JackXfmax/GARA}
}
```

---

## Notes on provenance

The Wylie↔文言文 parallel corpus was assembled from Tibetan canonical and
liturgical sources with their Classical Chinese renderings, then cleaned
(empty-side filtering, whitespace normalisation, pair-level deduplication,
and contamination removal). `scripts/prepare_data.py` documents the exact
cleaning pipeline applied before release.
