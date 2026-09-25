"""Supervised fine-tuning of the ReScraper student (Qwen3-0.6B) on serialized targets.

HF Trainer + DeepSpeed ZeRO-3, bf16, packed 32,768-token sequences (First-Fit-Decreasing bins with position_ids
restarting per sample), chunked cross-entropy, loss on the answer only, and `--tag_loss_weight` (5 in the paper) on
the leading decision-tag tokens of every answer. With `--prepacked_dir` the train set comes from
training/prepack_sft.py (tokenize + pack once on CPU); otherwise the jsonl is tokenized and packed here.
Launched by training/train_multinode.sbatch; see training/README.md for the exact arguments of both stages.
"""
import os
import json

import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import wandb
class _LocalExactMatch:
    """Stand-in for evaluate.load("exact_match"): same call signature, no dynamic
    module written to disk, so 64 ranks cannot race each other importing it."""

    @staticmethod
    def compute(predictions, references, ignore_case=False, ignore_punctuation=False,
                regexes_to_ignore=None, ignore_numbers=False):
        # the caller passes evaluate's keyword options; honour the ones that change
        # the answer and accept the rest so the signature stays drop-in compatible
        import re as _re
        import string as _string

        def norm(x):
            if regexes_to_ignore:
                for rx in regexes_to_ignore:
                    x = _re.sub(rx, "", x)
            if ignore_case:
                x = x.lower()
            if ignore_punctuation:
                x = x.translate(str.maketrans("", "", _string.punctuation))
            if ignore_numbers:
                x = x.translate(str.maketrans("", "", _string.digits))
            return x

        n = max(1, len(references))
        hit = sum(1 for p, r in zip(predictions, references) if norm(p) == norm(r))
        return {"exact_match": hit / n}

from rouge_score import rouge_scorer

CE_CHUNK_SIZE = 4096             # tokens per chunk for cross-entropy to avoid fp32 OOM
GEN_EVAL_SAMPLES = 20            # number of val samples to use for generation-based eval
GEN_EVAL_EVERY_N_STEPS = 20     # evaluate every N steps (~33 min intervals)


class GenerationEvalCallback(TrainerCallback):
    """Every GEN_EVAL_EVERY_N_STEPS steps, generate on a small val subset
    and log ROUGE-1/2/L/5 and Exact Match to wandb. Only runs on rank 0."""

    def __init__(self, tokenizer, val_raw, max_gen_length=4096, gen_eval_every=GEN_EVAL_EVERY_N_STEPS, gen_eval_samples=GEN_EVAL_SAMPLES):
        self.tokenizer = tokenizer
        self.gen_eval_every = gen_eval_every
        # val_raw: list of {"input": html, "output": clean_text}
        self.val_samples = random.sample(val_raw, min(gen_eval_samples, len(val_raw)))
        self.max_gen_length = max_gen_length
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rouge5"], use_stemmer=False
        )
        self.em_metric = _LocalExactMatch()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.gen_eval_every != 0:
            return

        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Each rank processes its own subset of samples
        my_samples = self.val_samples[rank::world_size]

        model.eval()
        torch.cuda.empty_cache()
        my_results = []

        import time as _time
        from transformers import StoppingCriteria, StoppingCriteriaList

        class TimeoutStoppingCriteria(StoppingCriteria):
            """Stop generation after a time limit (seconds)."""
            def __init__(self, max_seconds=300):
                self.max_seconds = max_seconds
                self.start_time = None
            def reset(self):
                self.start_time = _time.time()
            def __call__(self, input_ids, scores, **kwargs):
                return _time.time() - self.start_time > self.max_seconds

        timeout_criteria = TimeoutStoppingCriteria(max_seconds=300)
        gen_eval_start = _time.time()
        GEN_EVAL_TOTAL_TIMEOUT = 1500  # 25 min hard cap, leaves 5 min buffer before NCCL 1800s timeout

        with torch.no_grad():
            for sample in my_samples:
                if _time.time() - gen_eval_start > GEN_EVAL_TOTAL_TIMEOUT:
                    print(f"[GenEval] Total timeout ({GEN_EVAL_TOTAL_TIMEOUT}s) reached, "
                          f"evaluated {len(my_results)}/{len(my_samples)} samples", flush=True)
                    break
                messages = [{"role": "user", "content": sample["input"]}]
                try:
                    prompt_ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_tensors="pt"
                    )
                except TypeError:
                    prompt_ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt"
                    )

                prompt_ids = prompt_ids.to(model.device)
                timeout_criteria.reset()
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_gen_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([timeout_criteria]),
                )
                gen_ids = out[0][prompt_ids.shape[1]:]
                pred = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                ref = sample["output"].strip()
                sc = self.scorer.score(ref, pred)
                my_results.append({
                    "rouge1": sc["rouge1"].fmeasure,
                    "rouge2": sc["rouge2"].fmeasure,
                    "rougeL": sc["rougeL"].fmeasure,
                    "rouge5": sc["rouge5"].fmeasure,
                    "rouge5_prec": sc["rouge5"].precision,
                    "rouge5_rec": sc["rouge5"].recall,
                    "pred": pred,
                    "ref": ref,
                })

        # Gather results from all ranks to rank 0
        if dist.is_initialized():
            all_results = [None] * world_size
            dist.all_gather_object(all_results, my_results)
        else:
            all_results = [my_results]

        if not state.is_world_process_zero:
            model.train()
            return

        # Flatten: rank0 results first, then rank1, etc.
        results = [r for rank_results in all_results for r in rank_results]

        metric_keys = [k for k in results[0] if k not in ("pred", "ref")]
        scores = {k: [r[k] for r in results] for k in metric_keys}
        preds = [r["pred"] for r in results]
        refs = [r["ref"] for r in results]

        em_result = self.em_metric.compute(
            predictions=preds,
            references=refs,
            ignore_case=True,
            ignore_punctuation=True,
            ignore_numbers=False,
            regexes_to_ignore=[r"\s+"],
        )
        avg = {k: sum(v) / len(v) for k, v in scores.items()}
        avg["em"] = em_result["exact_match"]
        wandb.log({f"gen_eval/{k}": v for k, v in avg.items()})
        print(f"\n[GenEval step={state.global_step}] " +
              " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())))

        table = wandb.Table(columns=["sample", "rouge5", "rouge5_prec", "rouge5_rec", "rouge1", "ref", "pred"])
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        case_dir = os.environ.get("GENEVAL_LOG_DIR", "logs/geneval")
        os.makedirs(case_dir, exist_ok=True)
        case_log = f"{case_dir}/sft-32k-{job_id}-geneval.log"
        with open(case_log, "a") as f:
            f.write(f"\n{'='*60}\n[GenEval step={state.global_step}] " +
                    " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())) + "\n")
            for i, r in enumerate(results):
                table.add_data(
                    i + 1,
                    round(r["rouge5"], 4),
                    round(r["rouge5_prec"], 4),
                    round(r["rouge5_rec"], 4),
                    round(r["rouge1"], 4),
                    r["ref"][:3000],
                    r["pred"][:3000],
                )
                f.write(f"\n--- Example {i+1} (rouge5={r['rouge5']:.4f} prec={r['rouge5_prec']:.4f} rec={r['rouge5_rec']:.4f}) ---\n")
                f.write(f"  REF:  {r['ref']}\n")
                f.write(f"  PRED: {r['pred']}\n")
        wandb.log({f"gen_eval/examples_step{state.global_step}": table})

        model.train()


class ChunkedCETrainer(Trainer):
    """Trainer that computes cross-entropy loss in chunks to avoid
    materializing a full fp32 logits tensor (seq_len × vocab_size).
    On 48 GB GPUs with 32K seq_len and 151K vocab, the default
    cross_entropy needs ~18.5 GiB of fp32 — this keeps it at ~2.3 GiB."""

    TAG_W = 1.0
    TAG_N = 4
    # Constant multiplier on the per-micro mean, see --loss_scale.
    LOSS_SCALE = 1.0

    @staticmethod
    def _tag_weights(shift_labels, w, n):
        """1.0 everywhere, w on the first n supervised tokens of each answer span.
        Packing puts several answers in one row, so every -100 -> real transition
        starts a new span, not just the first one."""
        import torch
        sup = shift_labels != -100
        starts = sup & ~torch.nn.functional.pad(sup, (1, 0), value=False)[:, :-1]
        weights = torch.ones_like(shift_labels, dtype=torch.float32)
        idx = starts.nonzero(as_tuple=False)
        for b, t in idx.tolist():
            weights[b, t:t + n] = w
        return weights * sup

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        seq_len = shift_logits.size(1)
        vocab_size = shift_logits.size(-1)

        total_loss = torch.zeros(1, device=logits.device, dtype=torch.float32)
        total_tokens = 0
        # per-token weights: 1.0 normally, TAG_W on the leading tag tokens of each
        # answer span. Denominator uses the same weights so the loss stays a proper
        # weighted mean and the effective learning rate does not drift with TAG_W.
        weighted = self.TAG_W != 1.0
        wmat = self._tag_weights(shift_labels, self.TAG_W, self.TAG_N) if weighted else None
        total_w = 0.0

        for i in range(0, seq_len, CE_CHUNK_SIZE):
            chunk_logits = shift_logits[:, i:i + CE_CHUNK_SIZE, :].reshape(-1, vocab_size)
            chunk_labels = shift_labels[:, i:i + CE_CHUNK_SIZE].reshape(-1)
            n_valid = (chunk_labels != -100).sum().item()
            if n_valid == 0:
                continue

            if weighted:
                w_chunk = wmat[:, i:i + CE_CHUNK_SIZE].reshape(-1)

                def _ce(lg, lb, w):
                    per = F.cross_entropy(lg.float(), lb, ignore_index=-100, reduction="none")
                    return (per * w).sum()

                chunk_loss = grad_checkpoint(_ce, chunk_logits, chunk_labels, w_chunk,
                                             use_reentrant=False)
                total_w += w_chunk.sum().item()
            else:
                def _ce(lg, lb):
                    return F.cross_entropy(lg.float(), lb, ignore_index=-100, reduction="sum")

                chunk_loss = grad_checkpoint(_ce, chunk_logits, chunk_labels, use_reentrant=False)
            total_loss = total_loss + chunk_loss
            total_tokens += n_valid

        denom = total_w if weighted else total_tokens
        loss = total_loss.squeeze() / max(denom, 1)
        if model.training:  # keep eval_loss on the plain per-token scale
            loss = loss * self.LOSS_SCALE
        return (loss, outputs) if return_outputs else loss


def pack_dataset(dataset, max_seq_len):
    """Bin-pack tokenized samples into max_seq_len windows (First-Fit Decreasing).
    Returns a new Dataset with columns: input_ids, labels, position_ids (no attention_mask)."""
    from datasets import Dataset as HFDataset

    lengths = [len(ex["input_ids"]) for ex in dataset]
    indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    # Each bin: [total_len, [sample_indices]]
    bins = []
    for idx in indices:
        slen = lengths[idx]
        placed = False
        for b in bins:
            if b[0] + slen <= max_seq_len:
                b[0] += slen
                b[1].append(idx)
                placed = True
                break
        if not placed:
            bins.append([slen, [idx]])

    # Build packed examples
    packed = {"input_ids": [], "labels": [], "position_ids": []}
    for _, sample_indices in bins:
        ids, labs, pos = [], [], []
        for si in sample_indices:
            ex = dataset[si]
            n = len(ex["input_ids"])
            ids.extend(ex["input_ids"])
            labs.extend(ex["labels"])
            pos.extend(range(n))
        packed["input_ids"].append(ids)
        packed["labels"].append(labs)
        packed["position_ids"].append(pos)

    total_tokens = sum(len(x) for x in packed["input_ids"])
    n_bins = len(packed["input_ids"])
    print(f"  Packing: {len(dataset)} samples -> {n_bins} bins "
          f"(ratio: {len(dataset)/n_bins:.2f}x, "
          f"avg fill: {total_tokens/n_bins/max_seq_len*100:.1f}%)")
    return HFDataset.from_dict(packed)


class PackedDataCollator:
    """Collator for mixed packed/unpacked data.
    Packed features have position_ids (no attention_mask) -> omit attention_mask for FA2 packing detection.
    Unpacked features have attention_mask (no position_ids) -> include attention_mask normally."""

    def __init__(self, pad_token_id, pad_to_multiple_of=8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        has_pos = "position_ids" in features[0]
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = -(-max_len // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch = {"input_ids": [], "labels": []}
        if has_pos:
            batch["position_ids"] = []
        else:
            batch["attention_mask"] = []

        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad_n)
            batch["labels"].append(list(f["labels"]) + [-100] * pad_n)
            if has_pos:
                batch["position_ids"].append(list(f["position_ids"]) + [0] * pad_n)
            else:
                batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad_n)

        return {k: torch.tensor(v) for k, v in batch.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="./qwen3-sft-32k")
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, only use first N samples (for smoke testing)")
    parser.add_argument("--gen_eval_every", type=int, default=GEN_EVAL_EVERY_N_STEPS,
                        help="Run generation eval every N steps")
    parser.add_argument("--gen_eval_samples", type=int, default=GEN_EVAL_SAMPLES,
                        help="Number of val samples for generation eval")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from (e.g. output_dir/checkpoint-200)")
    parser.add_argument("--use_packing", action="store_true",
                        help="Enable sequence packing to reduce padding waste")
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Enable torch.compile on model forward pass")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max checkpoints to keep (-1 = unlimited)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--tag_loss_weight", type=float, default=1.0,
                        help="multiply the loss on the leading decision tag tokens")
    parser.add_argument("--loss_scale", type=float, default=24.0,
                        help="Constant multiplier on the per-micro-batch mean loss. 24 reproduces "
                             "the validated 1-node regime (micro=1, accum=24 on the old un-normalised "
                             "path), where the accumulated gradient is 24x the mean and is clipped to "
                             "max_grad_norm at essentially every step. With "
                             "model_accepts_loss_kwargs=False the gradient is then identical on any "
                             "node count.")
    parser.add_argument("--tag_n_tokens", type=int, default=4,
                        help="how many leading answer tokens count as the tag")
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="prepend this file's text as a system message")
    parser.add_argument("--prepacked_dir", type=str, default=None,
                        help="DatasetDict from prepack_sft.py: skip tokenize/filter/pack, load the packed train set (mmap)")
    return parser.parse_args()


SYSTEM_PROMPT = None          # set from --system_prompt_file; None keeps the old behaviour


def build_chat_ids(tokenizer, user_content, assistant_content, add_generation_prompt=False):
    """Build token ids using Qwen3 chat template with thinking disabled."""
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    if not add_generation_prompt:
        messages.append({"role": "assistant", "content": assistant_content})

    kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt)
    try:
        result = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    # transformers >=5.x may return BatchEncoding instead of list[int]
    if hasattr(result, 'input_ids'):
        return result.input_ids[0] if isinstance(result.input_ids[0], list) else result.input_ids
    return result


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    global SYSTEM_PROMPT
    if args.system_prompt_file:
        SYSTEM_PROMPT = open(args.system_prompt_file, encoding="utf-8").read().strip()
        print(f"system prompt: {len(SYSTEM_PROMPT)} chars from {args.system_prompt_file}")
    ChunkedCETrainer.TAG_W = args.tag_loss_weight
    ChunkedCETrainer.TAG_N = args.tag_n_tokens
    if args.tag_loss_weight != 1.0:
        print(f"tag loss weight: {args.tag_loss_weight} on the first "
              f"{args.tag_n_tokens} answer tokens")

    if args.prepacked_dir:
        from datasets import load_from_disk
        dataset = load_from_disk(args.prepacked_dir)
        val_raw = json.load(open(os.path.join(args.prepacked_dir, "val_raw.json"), encoding="utf-8"))
        ntr, nva = len(dataset["train"]), len(dataset["validation"])
        print(f"prepacked {args.prepacked_dir}: train bins {ntr}, validation {nva}, val_raw {len(val_raw)}", flush=True)
        collator = PackedDataCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)
    else:
        data = load_dataset("json", data_files=args.data_path, split="train")
        if args.max_samples > 0:
            data = data.select(range(min(args.max_samples, len(data))))
        val_size = min(args.val_size, len(data) // 5)
        split = data.train_test_split(test_size=val_size, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

        MAX_SEQ_LEN = args.max_seq_len
        # ~3 chars/token for HTML; matches previous truncation threshold that worked on L40 48GB
        MAX_CHARS_PREFILTER = MAX_SEQ_LEN * 3

        def preprocess(example):
            html_input = example["input"]
            text_output = example["output"]

            # Pre-filter by char count to avoid tokenizer OOM on extreme outliers (e.g. 500K char HTML)
            if len(html_input) + len(text_output) > MAX_CHARS_PREFILTER:
                return {"input_ids": [], "attention_mask": [], "labels": []}

            full_ids = build_chat_ids(tokenizer, html_input, text_output)
            prompt_ids = build_chat_ids(tokenizer, html_input, "", add_generation_prompt=True)

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[: len(full_ids)]

            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }

        # Tokenize but keep raw columns for val_raw extraction
        dataset["validation"] = dataset["validation"].map(
            preprocess, num_proc=4, desc="Tokenizing validation",
        )
        dataset["train"] = dataset["train"].map(
            preprocess, remove_columns=["input", "output"], num_proc=4, desc="Tokenizing train",
        )

        before_filter = {k: len(dataset[k]) for k in dataset}
        dataset = dataset.filter(lambda x: 0 < len(x["input_ids"]) <= MAX_SEQ_LEN, num_proc=4)
        after_filter = {k: len(dataset[k]) for k in dataset}
        for k in dataset:
            print(f"  {k}: {before_filter[k]} -> {after_filter[k]} ({before_filter[k] - after_filter[k]} filtered)")

        # Save raw val samples after filtering so all samples fit within MAX_SEQ_LEN
        val_raw = [{"input": ex["input"], "output": ex["output"]}
                   for ex in dataset["validation"]]
        dataset["validation"] = dataset["validation"].remove_columns(["input", "output"])

        if args.use_packing:
            dataset["train"] = pack_dataset(dataset["train"], MAX_SEQ_LEN)
            collator = PackedDataCollator(
                pad_token_id=tokenizer.pad_token_id,
                pad_to_multiple_of=8,
            )
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=8,
            )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        deepspeed=args.deepspeed,
        report_to="wandb",
        run_name=os.environ.get("WANDB_NAME", "rescraper-sft"),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        torch_compile=args.use_torch_compile,
        remove_unused_columns=False,
    )

    gen_eval_cb = GenerationEvalCallback(tokenizer, val_raw, gen_eval_every=args.gen_eval_every, gen_eval_samples=args.gen_eval_samples)

    # Loss normalisation, made topology-independent. compute_loss returns a
    # per-micro-batch MEAN * LOSS_SCALE. transformers>=4.46 assumes a model that
    # accepts loss kwargs returns a SUM already normalised by num_items_in_batch
    # and then skips its own loss/=accum, so the accumulated gradient used to
    # scale with accum (24x on 1 node, 3x on 4 nodes): grad_norm 387 vs 226 on
    # the same data at step 1. With max_grad_norm=1.0 the 1-node run was clipped
    # at every step while the 4-node run was unclipped after ~10 steps; Adam with
    # beta2=0.999 over only 100-300 steps never forgets the huge early gradients,
    # so the unclipped run trained with a much smaller effective LR (F1 78-81 vs
    # 88-90). model_accepts_loss_kwargs=False restores the classic path (Trainer
    # divides by accum, ranks are averaged) and LOSS_SCALE=24 reproduces the
    # validated 1-node gradient exactly on any node count.
    trainer = ChunkedCETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[gen_eval_cb],
    )
    trainer.model_accepts_loss_kwargs = False
    ChunkedCETrainer.LOSS_SCALE = args.loss_scale

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    results = trainer.evaluate()
    if trainer.is_world_process_zero():
        print("Validation results:", results)


if __name__ == "__main__":
    main()
