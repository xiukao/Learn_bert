# coding: utf-8
r"""Train BERT-Base on GLUE MRPC with modern PyTorch CUDA.

This file is independent of the repository's TensorFlow 1.x runners. It
requires Python 3.9+, PyTorch with CUDA support, and Transformers 4.x.

Example environment on Windows (with a CUDA 12.1 compatible NVIDIA driver):
  conda create -n bert_gpu python=3.11 -y
  conda activate bert_gpu
  python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  python -m pip install transformers==4.47.1

Example (PowerShell, from this file's directory):
  python .\run_mrpc_gpu.py --data-dir D:\datasets\GLUE\MRPC

The default model is the converted original BERT-Base uncased checkpoint on
Hugging Face. Pass --model with a local Transformers model directory to run
offline. Each run gets output/MRPC/MRPC-N without overwriting old runs.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path


PROJECT_DIR = Path(os.path.abspath(__file__)).parent
LOG = logging.getLogger("mrpc_gpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing GLUE MRPC train.tsv and dev.tsv")
    parser.add_argument("--model", default="google-bert/bert-base-uncased",
                        help="Hugging Face model ID or local Transformers checkpoint directory")
    parser.add_argument("--model-revision", default="main",
                        help="Hub revision; ignored for local model directories")
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "output",
                        help="MRPC-N folders are created under OUTPUT_ROOT/MRPC")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=8,
                        help="Physical GPU batch; lower to 4 if 12 GB is insufficient")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Effective train batch = micro batch x accumulation steps")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Use bfloat16 autocast; faster and uses less VRAM on RTX 4090")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate MRPC files without installing GPU dependencies or training")
    args = parser.parse_args()
    if min(args.epochs, args.max_seq_length, args.micro_batch_size,
           args.gradient_accumulation_steps, args.eval_batch_size) < 1:
        parser.error("epochs, sequence length, and batch sizes must be positive")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("learning rate must be positive and warmup ratio must be in [0, 1)")
    if args.weight_decay < 0:
        parser.error("weight decay must not be negative")
    return args


def read_mrpc(path):
    examples = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None or header[:5] != ["label", "id1", "id2", "sentence1", "sentence2"]:
            raise ValueError("Unexpected MRPC header in %s" % path)
        for line_number, row in enumerate(reader, 2):
            if len(row) != 5 or row[0] not in ("0", "1"):
                raise ValueError("Invalid labeled MRPC row at %s:%d" % (path, line_number))
            examples.append((row[3], row[4], int(row[0])))
    if not examples:
        raise ValueError("No examples in %s" % path)
    return examples


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def git_value(*arguments):
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_DIR.parent)] + list(arguments),
                                capture_output=True, text=True, errors="replace")
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def create_run_dir(output_root):
    task_dir = Path(os.path.abspath(output_root)) / "MRPC"
    task_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        run_dir = task_dir / ("MRPC-%d" % number)
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            number += 1


def setup_log(run_dir):
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def score_mrpc(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Prediction count differs from gold count")
    correct = sum(a == b for a, b in zip(gold, predicted))
    tp = sum(a == b == 1 for a, b in zip(gold, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(gold, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(gold, predicted))
    accuracy = correct / len(gold)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"accuracy": accuracy, "f1": f1,
            "accuracy_f1_mean": (accuracy + f1) / 2,
            "count": len(gold)}


def main():
    args = parse_args()
    train_path = args.data_dir / "train.tsv"
    dev_path = args.data_dir / "dev.tsv"
    train_rows = read_mrpc(train_path)
    dev_rows = read_mrpc(dev_path)
    if args.check_data:
        print("MRPC data OK: train=%d, dev=%d" % (len(train_rows), len(dev_rows)))
        return 0

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    except ImportError as error:
        raise SystemExit("Install a CUDA-enabled PyTorch build and Transformers 4.x: %s" % error)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check your PyTorch build, NVIDIA driver and GPU.")
    if args.amp and not torch.cuda.is_bf16_supported():
        raise SystemExit("--amp requires bfloat16 support on the selected GPU.")

    local_model = Path(args.model).is_dir()
    model_kwargs = {} if local_model else {"revision": args.model_revision}
    config = BertConfig.from_pretrained(args.model, **model_kwargs)
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) != (768, 12, 12):
        raise SystemExit("Model is not BERT-Base (expected H=768, L=12, A=12).")
    if not local_model and getattr(config, "_commit_hash", None):
        model_kwargs = {"revision": config._commit_hash}
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True,
                                               **model_kwargs)
    if tokenizer.vocab_size != 30522:
        raise SystemExit("Expected BERT-Base uncased vocabulary of 30522 tokens.")

    run_dir = create_run_dir(args.output_root)
    setup_log(run_dir)
    started = time.monotonic()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    device_index = torch.cuda.current_device()
    gpu = torch.cuda.get_device_properties(device_index)
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps
    settings = {
        "task": "MRPC", "model": args.model,
        "model_revision_requested": args.model_revision if not local_model else None,
        "model_revision_resolved": getattr(config, "_commit_hash", None),
        "bert_config": {"hidden_size": config.hidden_size,
                        "num_hidden_layers": config.num_hidden_layers,
                        "num_attention_heads": config.num_attention_heads,
                        "vocab_size": config.vocab_size},
        "data_dir": os.path.abspath(args.data_dir),
        "data_sha256": {"train.tsv": sha256(train_path), "dev.tsv": sha256(dev_path)},
        "train_count": len(train_rows), "dev_count": len(dev_rows),
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_train_batch_size": effective_batch,
        "eval_batch_size": args.eval_batch_size,
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
        "seed": args.seed, "amp_bfloat16": args.amp,
        "optimizer": "AdamW(beta1=0.9,beta2=0.999,eps=1e-6)",
        "selection_metric": "dev accuracy_f1_mean",
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "config.json", settings)
    info = {
        "status": "running", "started_at_utc": started_at,
        "command": [sys.executable] + sys.argv,
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {"torch": torch.__version__,
                         "transformers": transformers.__version__},
        "cuda_runtime": torch.version.cuda,
        "gpu": {"name": gpu.name, "total_vram_gb": round(gpu.total_memory / 2**30, 2),
                "compute_capability": "%d.%d" % (gpu.major, gpu.minor)},
        "git": {"branch": git_value("branch", "--show-current"),
                "head": git_value("rev-parse", "HEAD"),
                "status_at_start": git_value("status", "--short")},
        "script_sha256": sha256(Path(__file__)),
    }
    write_json(run_dir / "run_info.json", info)
    LOG.info("Run: %s", run_dir)
    LOG.info("GPU: %s, %.2f GiB; effective batch: %d", gpu.name,
             gpu.total_memory / 2**30, effective_batch)

    epochs = []
    best_epoch = None
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        def make_dataset(rows):
            encoding = tokenizer([row[0] for row in rows],
                                 [row[1] for row in rows],
                                 padding="max_length", truncation=True,
                                 max_length=args.max_seq_length,
                                 return_tensors="pt")
            labels = torch.tensor([row[2] for row in rows], dtype=torch.long)
            return TensorDataset(encoding["input_ids"], encoding["attention_mask"],
                                 encoding["token_type_ids"], labels)

        train_data = make_dataset(train_rows)
        dev_data = make_dataset(dev_rows)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_data, batch_size=args.micro_batch_size,
                                  shuffle=True, generator=generator, num_workers=0)
        dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size,
                                shuffle=False, num_workers=0)

        model = BertForSequenceClassification.from_pretrained(
            args.model, num_labels=2, ignore_mismatched_sizes=False, **model_kwargs)
        model.to("cuda")
        no_decay = ("bias", "LayerNorm.weight")
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.endswith(no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and n.endswith(no_decay)],
             "weight_decay": 0.0},
        ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6)
        updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * args.epochs
        warmup_updates = int(total_updates * args.warmup_ratio)

        def lr_factor(step):
            if step < warmup_updates:
                return step / max(1, warmup_updates)
            return max(0.0, (total_updates - step) /
                       max(1, total_updates - warmup_updates))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        best_score = -1.0
        update_count = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for batch_index, batch in enumerate(train_loader):
                group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                group_end = min(group_start + args.gradient_accumulation_steps, len(train_loader))
                group_count = min(group_end * args.micro_batch_size, len(train_data)) - group_start * args.micro_batch_size
                ids, mask, segments, labels = (x.to("cuda") for x in batch)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.amp):
                    output = model(input_ids=ids, attention_mask=mask,
                                   token_type_ids=segments, labels=labels)
                    scaled_loss = output.loss * (len(labels) / group_count)
                scaled_loss.backward()
                loss_sum += output.loss.item() * len(labels)
                if batch_index + 1 == group_end:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_count += 1

            model.eval()
            gold, predicted = [], []
            with torch.inference_mode():
                for batch in dev_loader:
                    ids, mask, segments, labels = (x.to("cuda") for x in batch)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=args.amp):
                        logits = model(input_ids=ids, attention_mask=mask,
                                       token_type_ids=segments).logits
                    gold.extend(labels.cpu().tolist())
                    predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            dev = score_mrpc(gold, predicted)
            item = {"epoch": epoch, "train_loss": loss_sum / len(train_data),
                    "dev": dev, "optimizer_updates": update_count}
            epochs.append(item)
            LOG.info("Epoch %d/%d: train_loss=%.5f dev_accuracy=%.4f dev_f1=%.4f",
                     epoch, args.epochs, item["train_loss"], dev["accuracy"], dev["f1"])
            if dev["accuracy_f1_mean"] > best_score:
                best_score = dev["accuracy_f1_mean"]
                best_epoch = epoch
                model.save_pretrained(run_dir / "model", safe_serialization=True)
                tokenizer.save_pretrained(run_dir / "model")
                with (run_dir / "dev_predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["index", "gold", "prediction"])
                    writer.writerows((i, a, b) for i, (a, b) in enumerate(zip(gold, predicted)))
                with (run_dir / "eval_results.txt").open("w", encoding="utf-8") as stream:
                    for key, value in dev.items():
                        stream.write("%s = %s\n" % (key, value))
            write_json(run_dir / "metrics.json", {
                "status": "running", "task": "MRPC", "split": "dev",
                "best_epoch": best_epoch,
                "best_dev": epochs[best_epoch - 1]["dev"], "epochs": epochs})

        info.update({"status": "completed", "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "duration_seconds": round(time.monotonic() - started, 2),
                     "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                     "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
        write_json(run_dir / "run_info.json", info)
        write_json(run_dir / "metrics.json", {
            "status": "completed", "task": "MRPC", "split": "dev",
            "best_epoch": best_epoch, "best_dev": epochs[best_epoch - 1]["dev"],
            "epochs": epochs})
        LOG.info("Best epoch %d: %s", best_epoch, epochs[best_epoch - 1]["dev"])
        return 0
    except BaseException as error:
        info.update({"status": "failed", "error": "%s: %s" % (type(error).__name__, error),
                     "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "duration_seconds": round(time.monotonic() - started, 2),
                     "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                     "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
        write_json(run_dir / "run_info.json", info)
        write_json(run_dir / "metrics.json", {
            "status": "failed", "task": "MRPC", "split": "dev",
            "best_epoch": best_epoch,
            "best_dev": epochs[best_epoch - 1]["dev"] if best_epoch else None,
            "epochs": epochs})
        LOG.error("Run failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
