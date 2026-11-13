"""The safety modulator.

A small module, phi, that rescales the output of every transformer layer:

    h_l  ->  h_l * (1 + f_l),      f_l = gain_l * U_l(GELU(D_l c))

c is one learned context vector shared by all layers; D_l and U_l are a
low-rank pair per layer; gain_l is a learned scalar. f_l has one entry per
hidden dimension, so the modulator turns individual directions of each layer
up or down. It starts close to zero (no change) and stays attached to the
model after training: it is part of what gets deployed.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_LAYER_RE = re.compile(r"(^|\.)layers\.(\d+)$")
_NOT_TEXT = ("vision", "audio", "multi_modal")


def decoder_layers(model: nn.Module) -> list:
    """The language model's transformer layers, in order."""
    found = {}
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if m and not any(part in name for part in _NOT_TEXT):
            found[int(m.group(2))] = module
    return [found[i] for i in sorted(found)]


class _LayerFactor(nn.Module):
    def __init__(self, context_dim, dim, rank, init_scale, max_gain=2.0):
        super().__init__()
        self.down = nn.Linear(context_dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="linear")
        # The gain sets the size of the factor, so the projections can keep a
        # standard scale. It is learned on a log scale: every update changes it
        # by a percentage, so it can grow from a small start, and it can never
        # reach exactly zero (from which it would not recover).
        self.max_gain = max_gain
        self.log_gain = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, c):
        gain = self.log_gain.exp().clamp(max=self.max_gain)
        return gain * self.up(F.gelu(self.down(c)))


class Modulator(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 8, context_dim: int = 64, init_scale: float = 0.02):
        super().__init__()
        self.layers = decoder_layers(model)
        if not self.layers:
            raise ValueError("found no transformer layers to modulate")
        dim = model.get_input_embeddings().weight.shape[1]
        self.context = nn.Parameter(torch.randn(context_dim) / math.sqrt(context_dim))
        self.factors = nn.ModuleList(_LayerFactor(context_dim, dim, rank, init_scale) for _ in self.layers)
        self.enabled = True                         # switched off during the inner step
        self._last = [None] * len(self.layers)      # last factor of every layer, for the size penalty
        self._hooks = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(self.layers)]

    def _hook(self, i):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            factor = self.factors[i](self.context)
            self._last[i] = factor
            hidden = hidden * (1 + factor.to(hidden.dtype))
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def size_penalty(self):
        """Mean squared size of the modulation in the last forward pass, averaged over layers."""
        return torch.stack([f.pow(2).mean() for f in self._last]).mean()

    @torch.no_grad()
    def sizes(self):
        """Per-layer root-mean-square size of the factor in the last forward pass, for logging."""
        return [f.pow(2).mean().sqrt().item() for f in self._last]

    def remove(self):
        for h in self._hooks:
            h.remove()
