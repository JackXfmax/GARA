"""
alignment_adapter.py
Representation Alignment Adapter for RA-LoRA

轻量输入表示对齐模块，插入在 embedding 输出之后，
用于缓解 Wylie 输入的极端分布偏移问题。
"""

import torch
import torch.nn as nn


class AlignmentAdapter(nn.Module):
    """
    Basic: Linear → GELU → Linear with residual connection
    Hidden size: same as model (e.g., 3584 for Qwen2.5-7B)
    Params: 2 * hidden_size^2 ≈ 25.7M for Qwen2.5-7B
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return x + self.proj(x)


class AlignmentAdapterGate(nn.Module):
    """
    Gate version: adds learned gating mechanism
    output = x + sigmoid(gate(x)) * proj(x)
    Params: 3 * hidden_size^2 ≈ 38.5M for Qwen2.5-7B
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return x + gate * self.proj(x)


def create_adapter(hidden_size, use_gate=False):
    """Factory function"""
    if use_gate:
        return AlignmentAdapterGate(hidden_size)
    return AlignmentAdapter(hidden_size)


def count_adapter_params(adapter):
    """Count trainable parameters in adapter"""
    return sum(p.numel() for p in adapter.parameters() if p.requires_grad)


def install_adapter(model, adapter):
    """
    Install adapter into Qwen2.5 model via forward hook on embed_tokens.

    Args:
        model: PeftModel (wrapped Qwen2.5-7B + LoRA)
        adapter: AlignmentAdapter module
    Returns:
        handle: hook handle (call handle.remove() to uninstall)
    """
    # Path: model.base_model.model.model.embed_tokens
    base_qwen = model.base_model.model.model

    # Register as submodule so params are included in model.parameters()
    base_qwen.add_module('alignment_adapter', adapter)

    # Create forward hook — cast adapter output back to model dtype to handle FP16/FP32 mismatch
    def hook_fn(module, input, output):
        with torch.autocast(device_type="cuda", enabled=False):
            return adapter(output.float()).to(output.dtype)

    handle = base_qwen.embed_tokens.register_forward_hook(hook_fn)
    return handle


def save_adapter(adapter, path):
    """Save adapter weights"""
    torch.save(adapter.state_dict(), path)


def load_adapter(adapter, path, device='cpu'):
    """Load adapter weights"""
    adapter.load_state_dict(torch.load(path, map_location=device))