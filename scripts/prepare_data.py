# -*- coding: utf-8 -*-
"""Prepare the GARA Wylie->Classical-Chinese dataset for public release.

Outputs (into data/):
  train.jsonl.gz      942,633 parallel pairs
  test.jsonl.gz        94,321 parallel pairs
  sample_1000.jsonl    1,000 dev pairs (quick preview, plain text)
  DATA_STATS.json      counts, length stats, sha256

Usage:
  python prepare_data.py            # rebuild the released files
  python prepare_data.py --audit    # additionally report train/test source overlap
                                    # to stdout (not written into DATA_STATS.json)
"""
import json, gzip, hashlib, io, os, statistics as st, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "_raw")

SRC = {
    "train": r"D:\workbuddy工作目录\Claw\wylie_data\train_zh_clean.json",
    "test": os.path.join(RAW, "eval_zh_clean.json"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gz_jsonl(records, out_path):
    # mtime=0 keeps the gzip header deterministic, so the released SHA-256
    # checksums stay stable across rebuilds.
    with gzip.GzipFile(out_path, mode="wb", mtime=0) as raw, \
            io.TextIOWrapper(raw, encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(DATA, exist_ok=True)
    stats = {}

    for split, path in SRC.items():
        print(f"[load] {split} <- {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # normalise: keep only bo/zh, drop empties
        clean = []
        for r in data:
            bo = (r.get("bo") or "").strip()
            zh = (r.get("zh") or "").strip()
            if bo and zh:
                clean.append({"bo": bo, "zh": zh})
        print(f"       {len(data)} -> {len(clean)} after empty-filter")

        out = os.path.join(DATA, f"{split}.jsonl.gz")
        write_gz_jsonl(clean, out)

        srcs = set(r["bo"] for r in clean)
        bo_len = [len(r["bo"]) for r in clean]
        zh_len = [len(r["zh"]) for r in clean]

        stats[split] = {
            "pairs": len(clean),
            "unique_source": len(srcs),
            "dup_source_rate": round(1 - len(srcs) / len(clean), 4),
            "wylie_chars_mean": round(st.mean(bo_len), 1),
            "wylie_chars_median": st.median(bo_len),
            "zh_chars_mean": round(st.mean(zh_len), 1),
            "zh_chars_median": st.median(zh_len),
            "file": os.path.basename(out),
            "size_bytes": os.path.getsize(out),
            "sha256": sha256(out),
        }
        print(f"       -> {out} ({stats[split]['size_bytes']/1e6:.1f} MB)")
        globals()[f"_{split}_srcs"] = srcs
        globals()[f"_{split}_clean"] = clean

    # --- optional audit: how many test sources appear in train ---
    # Reported to stdout only; never written into the released DATA_STATS.json.
    if "--audit" in sys.argv:
        tr_srcs = _train_srcs
        te_srcs = _test_srcs
        inter = te_srcs & tr_srcs
        print(f"\n[audit] {len(inter)}/{len(te_srcs)} test sources also in train "
              f"= {len(inter) / len(te_srcs) * 100:.1f}%")
        print("        (diagnostic only; not part of the released statistics)")

    # --- preview sample ---
    sample_path = os.path.join(DATA, "sample_1000.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for r in _test_clean[:1000]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["sample_file"] = "sample_1000.jsonl"

    with open(os.path.join(DATA, "DATA_STATS.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n[done] data/DATA_STATS.json")
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
