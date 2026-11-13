"""The nested training loop.

Two sets of weights, two loops:

  theta  the model's trainable weights (LoRA adapters), learning the task
  phi    the safety modulator, learning to keep the model safe

  inner loop, every step:      theta <- theta - lr * grad_theta [ L_task(theta)
                                                      + gamma/2 * ||theta - theta_start||^2 ]
  outer loop, every K steps:   phi   <- phi - outer_lr * grad_phi [ L_safe(theta, phi)
                                                      + lambda_task * L_task(theta, phi)
                                                      + mu * ||f(phi)||^2 ]

The inner loop runs with the modulator switched off, so the weights it learns
do not depend on phi. The gradient of the outer objective with respect to phi
is then exact: there is no second term through the inner loop to approximate.
"""
import time

import torch
import torch.nn as nn

from nsl.data import cycle, to_device
from nsl.model import lm_loss, save_checkpoint


def _drift_penalty(named, reference):
    dev = named[0][1].device
    total = torch.tensor(0.0, device=dev)
    for n, p in named:
        total = total + (p - reference[n]).float().pow(2).sum()
    return total


def train(model, modulator, task_dl, safety_dl, cfg, device, output_dir):
    """Run training. `modulator` and `safety_dl` are None for ordinary
    fine-tuning (config `modulator: off`).

    cfg keys: lr, gamma, steps, inner_steps, outer_lr, lambda_task, mu.
    """
    lr, gamma, total_steps = cfg["lr"], cfg["gamma"], cfg["steps"]
    inner_steps = cfg.get("inner_steps", 10)
    lambda_task, mu = cfg.get("lambda_task", 1.0), cfg.get("mu", 0.01)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    log_interval = cfg.get("log_interval", 250)

    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    theta = [p for _, p in named]
    reference = {n: p.detach().clone() for n, p in named}
    inner_opt = torch.optim.AdamW(theta, lr=lr, weight_decay=cfg.get("weight_decay", 0.01))
    task_iter = cycle(task_dl)

    nested = modulator is not None
    after = cfg.get("outer_schedule", "nested") == "after"
    if nested:
        phi = list(modulator.parameters())
        outer_opt = torch.optim.Adam(phi, lr=cfg.get("outer_lr", 1e-2))
        safety_iter = cycle(safety_dl)
        print(f"[nsl] modulator: {sum(p.numel() for p in phi):,} weights on {len(modulator.layers)} layers; "
              f"outer step every {inner_steps} steps")

    def requires_grad(params, flag):
        for p in params:
            p.requires_grad_(flag)

    tag = "nsl" if nested else "train"
    start = time.time()
    metrics, history = [], []

    def outer_step(label, record):
        requires_grad(theta, False)
        outer_opt.zero_grad()
        safe_loss, _ = lm_loss(model, to_device(next(safety_iter), device))
        outer = safe_loss + mu * modulator.size_penalty()
        if lambda_task > 0:
            outer = outer + lambda_task * lm_loss(model, to_device(next(task_iter), device))[0]
        outer.backward()
        outer_opt.step()
        requires_grad(theta, True)
        if record:
            history.append({"step": label, "safety_loss": safe_loss.item(), "outer_loss": outer.item(),
                            "factor_size": [round(s, 5) for s in modulator.sizes()]})

    for step in range(total_steps):
        # ---- inner step: theta learns the task, with the modulator switched off ----
        inner_opt.zero_grad()
        if nested:
            modulator.enabled = False
        task_loss, _ = lm_loss(model, to_device(next(task_iter), device))
        if nested:
            modulator.enabled = True
        penalty = (gamma / 2) * _drift_penalty(named, reference) if gamma > 0 else torch.tensor(0.0, device=device)
        (task_loss + penalty).backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(theta, max_grad_norm)
        inner_opt.step()

        # ---- outer step: phi learns to keep the model safe, theta is held fixed ----
        if nested and not after and (step + 1) % inner_steps == 0:
            outer_step(step + 1, (step + 1) % (inner_steps * 10) == 0)

        if step % log_interval == 0 or step == total_steps - 1:
            entry = {"step": step, "task_loss": task_loss.item(), "penalty": penalty.item()}
            msg = f"[{tag}] step={step}/{total_steps} task={task_loss.item():.4f} penalty={penalty.item():.4f}"
            if nested and history:
                last = history[-1]
                msg += (f" safety={last['safety_loss']:.4f} "
                        f"factor=[{min(last['factor_size']):.4f},{max(last['factor_size']):.4f}]")
            print(msg + f" t={time.time() - start:.1f}s")
            metrics.append(entry)

    # outer_schedule: after. The same number of modulator steps, all taken on the finished weights.
    if nested and after:
        n_outer = total_steps // inner_steps
        for i in range(n_outer):
            outer_step(total_steps + i + 1, (i + 1) % 10 == 0)
        print(f"[{tag}] {n_outer} modulator steps after fine-tuning: safety={history[-1]['safety_loss']:.4f} "
              f"factor=[{min(history[-1]['factor_size']):.4f},{max(history[-1]['factor_size']):.4f}]")

    elapsed = time.time() - start
    print(f"[{tag}] Done. {total_steps} steps in {elapsed:.1f}s")
    result = {"lr": lr, "gamma": gamma, "modulator": nested, "elapsed": elapsed, "metrics": metrics}
    if nested:
        result["history"] = history
        result["final_factor_size"] = [round(s, 5) for s in modulator.sizes()]
    save_checkpoint(model, modulator, {"lr": lr, "gamma": gamma, "steps": total_steps},
                    f"{output_dir}/final.pt")
    return result
