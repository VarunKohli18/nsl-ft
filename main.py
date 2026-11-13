"""Fine-tune with a nested safety modulator, then evaluate.

    python main.py --config config.yaml --device cuda:0                      # train + eval
    python main.py --config config.yaml --device cuda:0 --checkpoint pre-ft  # eval the untouched model
    python main.py --config config.yaml --device cuda:0 --checkpoint results/.../final.pt
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # run without installing

EVAL_SAMPLES = {"quick": 100, "full": 500}
TASKS = ("commonsense", "sst2", "medmcqa")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("--checkpoint", default=None,
                        help="skip training: a checkpoint to evaluate, or 'pre-ft' for the base model")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, cuda:1, ...")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = cfg.get("model", "gemma-4-e4b")
    mode = cfg.get("mode", "lora")
    dataset = cfg.get("dataset", "commonsense")
    gamma = float(cfg.get("gamma") or 0)
    poison_ratio = float(cfg.get("poison_ratio", 0.2))
    seed = int(cfg.get("seed", 42))
    eval_mode = str(cfg.get("eval", "quick")).lower()
    # YAML reads a bare on/off as a boolean
    use_modulator = cfg.get("modulator", True) in (True, "on", "true", "yes")
    safety_data = str(cfg.get("safety_data", "own_replies")).lower()
    judge_id = cfg.get("judge", "cais/HarmBench-Llama-2-13b-cls")
    train_layers = cfg.get("train_layers", "all")
    train_layers = "all" if str(train_layers).lower() == "all" else int(train_layers)
    rank = int(cfg.get("rank", 8))
    for key, value, known in (("mode", mode, ("lora", "full")),
                              ("safety_data", safety_data, ("own_replies", "harmless_chats")),
                              ("outer_schedule", str(cfg.get("outer_schedule", "nested")).lower(), ("nested", "after")),
                              ("eval", eval_mode, (*EVAL_SAMPLES, "none"))):
        if value not in known:
            sys.exit(f"config: {key} must be one of {' | '.join(known)} (got {value!r})")
    if dataset not in TASKS and not Path(dataset).is_file():
        sys.exit(f"config: dataset must be commonsense, sst2 or a JSONL file (got {dataset!r})")
    if gamma < 0:
        sys.exit("config: gamma cannot be negative")

    # Where this run lives: results/<model>/<mode>/<variant>/<dataset>/p<poison %>/
    root = Path(cfg.get("output_dir", "results")) / re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
    dataset_name = dataset if dataset in TASKS else Path(dataset).stem
    if args.checkpoint == "pre-ft":
        run_dir = root / "pre-ft" / dataset_name
    elif args.checkpoint:
        run_dir = Path(args.checkpoint).parent
    else:
        name = f"gamma{gamma:g}_" + ("modulator" if use_modulator else "plain")
        what = mode if train_layers == "all" else f"{mode}-last{train_layers}"
        run_dir = root / what / name / dataset_name / f"p{poison_ratio * 100:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / ("eval.log" if args.checkpoint else "run.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    import torch
    from nsl.model import load_checkpoint, load_model, load_tokenizer, prepare
    from nsl.modulator import Modulator

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    def attach_modulator():
        return Modulator(model, rank=rank).to(device) if use_modulator else None

    if args.checkpoint == "pre-ft":
        model = model.to(device)
        print(f"[main] Evaluating the base model: {model_name}")
    elif args.checkpoint:
        model = prepare(model, mode, model_name, train_layers).to(device)
        load_checkpoint(model, attach_modulator(), args.checkpoint)
        print(f"[main] Evaluating checkpoint: {args.checkpoint}")
    else:
        from nsl.data import (dataloader, harmless_chats, own_reply_examples,
                              safety_prompts, task_texts)
        from nsl.train import train

        steps = int(cfg.get("steps", 1000))
        lr = float(cfg["lr"]) if cfg.get("lr") is not None else (1e-5 if mode == "full" else 1e-4)

        # The safety loss measures drift from the model's ORIGINAL replies to
        # harmful requests, so those are collected before anything is trained.
        safety_examples = None
        if use_modulator and safety_data == "own_replies":
            from nsl.eval import generate_responses
            model = model.to(device)
            prompts = safety_prompts(512, seed)
            print(f"[main] Recording the model's own replies to {len(prompts)} harmful requests...")
            replies = generate_responses(model, tokenizer, prompts, max_new_tokens=64, device=device)
            safety_examples = own_reply_examples(tokenizer, prompts, replies)
        elif use_modulator:
            safety_examples = harmless_chats(tokenizer, 2000, seed)

        model = prepare(model, mode, model_name, train_layers).to(device)
        modulator = attach_modulator()
        print(f"[main] lr={lr:g} gamma={gamma:g} modulator={'on' if use_modulator else 'off'} mode={mode} "
              f"dataset={dataset} poison={poison_ratio:.0%} steps={steps} seed={seed} device={device}")

        task_dl = dataloader(tokenizer, task_texts(
            tokenizer, dataset, poison_ratio, int(cfg.get("train_samples", 15000)), seed), seed=seed)
        safety_dl = dataloader(tokenizer, safety_examples, seed=seed) if use_modulator else None

        t0 = time.time()
        result = train(model, modulator, task_dl, safety_dl, {
            "lr": lr, "gamma": gamma, "steps": steps,
            "inner_steps": int(cfg.get("inner_steps", 10)), "outer_lr": float(cfg.get("outer_lr", 1e-2)),
            "lambda_task": float(cfg.get("lambda_task") or 0), "mu": float(cfg.get("mu") or 0),
            "outer_schedule": str(cfg.get("outer_schedule", "nested")).lower(),
        }, device, str(run_dir))
        (run_dir / "train_result.json").write_text(json.dumps(result, indent=1))
        peak = f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.0f} GB" \
            if device.type == "cuda" else ""
        print(f"[main] Training done in {time.time() - t0:.1f}s{peak}")

    if eval_mode == "none":
        print("[main] eval: none, skipping evaluation")
        return
    from nsl.eval import run_eval

    # The modulator stays attached: it is part of the model that is evaluated.
    print(f"[main] Running {eval_mode} eval...")
    result = run_eval(model, tokenizer, device, samples=EVAL_SAMPLES[eval_mode], dataset=dataset,
                      judge_id=judge_id)
    result.update(model=model_name, dataset=dataset, eval=eval_mode, judge=judge_id,
                  run="pre-ft" if args.checkpoint == "pre-ft" else str(run_dir))
    result_path = run_dir / "eval_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print("[main] Results: " + ", ".join(
        f"{k}={v:.4f}" for k, v in result.items() if isinstance(v, float)))
    print(f"[main] Saved to {result_path}")


if __name__ == "__main__":
    main()
