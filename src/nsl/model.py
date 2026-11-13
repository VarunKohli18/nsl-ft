"""Model loading, LoRA, loss, checkpoints."""
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short names for the models this was developed on. Any other value is used
# as a Hugging Face model ID.
SUPPORTED_MODELS = {
    "gemma-4-e4b": "google/gemma-4-E4B-it",
}

# Which modules get LoRA adapters. The multimodal Gemmas also have q_proj/v_proj
# in their vision tower, which must be left alone: a regex restricts the
# adapters to the language model.
DEFAULT_LORA_TARGETS = ["q_proj", "v_proj"]
LORA_TARGETS = {
    "gemma-4-e4b": r".*language_model.*\.(q_proj|v_proj)",
}


def load_model(model_name: str, torch_dtype: Optional[torch.dtype] = None) -> nn.Module:
    """Load a pretrained aligned LLM on the CPU; the caller moves it."""
    hf_id = SUPPORTED_MODELS.get(model_name, model_name)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[model] Loading {hf_id}...")
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch_dtype, device_map="cpu")
    print(f"[model] Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def load_tokenizer(model_name: str):
    hf_id = SUPPORTED_MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_NOT_TEXT = ("vision", "audio", "multi_modal")      # towers of multimodal models


def layer_index(name: str) -> Optional[int]:
    """Which transformer layer of the language model a weight belongs to, or
    None (embeddings, final norm, vision/audio towers)."""
    m = _LAYER_RE.search(name)
    if m is None or any(part in name for part in _NOT_TEXT):
        return None
    return int(m.group(1))


def prepare(model: nn.Module, mode: str, model_name: str = "", train_layers="all",
            rank: int = 16, alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """Choose what is trained.

    mode          lora: adapters on the attention q/v projections; full: the weights themselves
    train_layers  all, or a number N: only the last N transformer layers are trained
    """
    if mode == "full":
        model.requires_grad_(True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=LORA_TARGETS.get(model_name, DEFAULT_LORA_TARGETS),
            lora_dropout=dropout, bias="none"))
    if train_layers != "all":
        depth = 1 + max(i for i in (layer_index(n) for n, _ in model.named_parameters()) if i is not None)
        first = depth - int(train_layers)
        for n, p in model.named_parameters():
            i = layer_index(n)
            if i is None or i < first:
                p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    layers = "all layers" if train_layers == "all" else f"last {train_layers} layers"
    print(f"[model] {mode}, {layers}: {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")
    return model


def lm_loss(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Next-token prediction loss over the tokens marked in `loss_mask` (all
    non-padding tokens if the batch has none)."""
    tokens = batch["tokens"]
    mask = batch.get("attention_mask")

    # The forward pass needs the mask too, not just the loss. Without it, real
    # tokens attend to padding wherever padding comes first (Gemma's tokenizer
    # pads on the left), which waters down the training signal.
    logits = model(tokens, attention_mask=mask).logits
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = tokens[:, 1:].reshape(-1)
    counted = batch.get("loss_mask", mask)      # which tokens count; default: all real ones
    if counted is not None:
        m = counted[:, 1:].reshape(-1).bool()
        shift_logits = shift_logits[m]
        shift_labels = shift_labels[m]
    # In float32: a half-precision cross-entropy over a large vocabulary can
    # overflow to NaN.
    return F.cross_entropy(shift_logits.float(), shift_labels), logits


def save_checkpoint(model: nn.Module, modulator, config: dict, path: str):
    """Save the trained weights only: the LoRA adapters (or everything under
    `mode: full`) and the modulator, not a second copy of the frozen base model."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    torch.save({
        "weights": {n: v.detach().cpu() for n, v in model.state_dict().items() if n in trainable},
        "modulator": modulator.state_dict() if modulator is not None else None,
        "config": config,
    }, ckpt_path)
    print(f"[model] Saved checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(model: nn.Module, modulator, path: str) -> dict:
    """Load a checkpoint written by `save_checkpoint` into a prepared model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["weights"], strict=False)
    if unexpected:
        raise ValueError(f"checkpoint does not match the model (mode/model differ?): {unexpected[:3]}")
    if modulator is not None and ckpt.get("modulator") is not None:
        modulator.load_state_dict(ckpt["modulator"])
    print(f"[model] Loaded checkpoint: {path} ({len(ckpt['weights'])} tensors)")
    return ckpt.get("config", {})
