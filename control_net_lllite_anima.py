"""ControlNet-LLLite for Anima (DiT) — ComfyUI port (v2 / v2.1 / v3 architectures).

Adapted from kohya-ss/sd-scripts. The on-disk weight format is the
named-key format (per-module key prefix = lllite_name, shared encoder under
``lllite_conditioning1.*``, depth embedding split per-module as
``{name}.depth_embed``); legacy ``lllite_modules.*`` files are rejected.

Supported architectures:
  * v2   — pixel conv stem (``trunk="stem"``, ``cond_input_space="pixel"``).
  * v2.1 — latent conv stem (``cond_input_space="latent"``, 3ch / non-inpaint only).
  * v3   — semantic trunk (``trunk="semantic"``): the conditioning source is the
    hidden states of the frozen DiT itself run over the cond latent
    (``encode_reference_hidden_states``), plus per-step t-FiLM and an optional
    per-token gate (scalar / vector / none). ``ref_context`` zero / uncond only
    (caption needs per-CFG-branch dispatch and is not ported yet).

Differences vs. the sd-scripts reference (``networks/control_net_lllite_anima.py``):
  * No dependency on ``library.utils`` — uses stdlib logging.
  * Module discovery filters the LLM-Adapter sub-tree by class identity in
    addition to the path-based check (ComfyUI ships two distinct ``Attention``
    classes that share the bare class name).
  * ``LLLiteModuleDiT`` keeps a ``restore()`` method (and an idempotent
    ``apply_to()``); ComfyUI patches/unpatches the original Linear around
    every sampler call via ``set_model_unet_function_wrapper``.
  * The v3 t-FiLM hook on ``dit.t_embedding_norm`` is NOT registered in
    ``__init__`` (the DiT is a shared object in ComfyUI, not cloned per node);
    nodes.py installs it via ``install_t_hook`` around each sampler call and
    removes it in ``finally``, mirroring ``apply_to()``/``restore()``.
  * ``encode_reference_hidden_states`` / ``build_uncond_ref_context`` are
    rewritten against ComfyUI's ``MiniTrainDIT`` API (``t_embedder`` is an
    ``nn.Sequential``, block kwargs differ, fp16 keeps the residual stream in
    fp32 like the stock ``_forward``).
  * Forward pass casts ``x`` and ``cond_emb`` to the LLLite parameter dtype
    so autocast / mixed-precision flows that hand us a different dtype than
    the LLLite weights still work.
  * CFG batch-size and sequence-length mismatches fall back to identity
    instead of asserting, so a slightly-off cond image cannot abort sampling.
  * The training-side ``AnimaControlNetLLLiteWrapper`` is omitted; ComfyUI
    integrates via ``model_function_wrapper`` in nodes.py instead.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Class names of the modules that LLLite injects into. The LLM-Adapter uses
# a different ``Attention`` class with the same bare name; we filter it by
# path (``llm_adapter`` in the qualified name) and by the ``is_selfattn``
# attribute presence.
TARGET_ATTENTION_CLASS = "Attention"
TARGET_MLP_CLASS = "GPT2FeedForward"
LLM_ADAPTER_NAME = "llm_adapter"

LLLITE_ARCH_VERSION = "2"
LLLITE_ARCH_VERSION_SEMANTIC = "3"

# Input space of the cond image (v2.1+). Missing metadata → "pixel" (old weights).
COND_INPUT_SPACES: Tuple[str, ...] = ("pixel", "latent")

# conditioning1 trunk type (v3). Missing metadata → "stem" (old weights).
TRUNK_TYPES: Tuple[str, ...] = ("stem", "semantic")

# v3 reference-forward context modes. Missing metadata → "zero" (old weights).
#   zero   : zero context (cross-attn output is exactly 0; fully text-independent)
#   uncond : constant empty-prompt-like context built via the LLM Adapter
#   caption: the generation prompt's context (NOT supported by this port yet)
REF_CONTEXT_MODES: Tuple[str, ...] = ("zero", "uncond", "caption")

# Channel count of the VAE latent fed to conditioning1 in latent mode (Qwen-Image VAE z_dim)
LATENT_COND_CHANNELS = 16

# v3 gate bias init: the gate starts open (sigmoid(2.0) ~= 0.88)
GATE_INIT_BIAS = 2.0

# v3 gate modes. Missing metadata → "scalar" (old v3 weights).
GATE_MODES: Tuple[str, ...] = ("scalar", "vector", "none")


# ----------------------------------------------------------------------------
# target_layers: atomic specifiers and presets
# ----------------------------------------------------------------------------

ATOMIC_SPECIFIERS: Tuple[str, ...] = (
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
)

PRESETS: dict = {
    "self_attn_q":            ("self_attn_q_pre",),
    "self_attn_qkv":          ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q":  ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> Tuple[str, ...]:
    """Resolve a ``target_layers`` spec to a canonical atomic tuple.

    Accepts a preset name (``"self_attn_qkv"``) or a comma-separated list of
    atomic specifiers (``"self_attn_q_pre,mlp_fc1_pre"``). Returns the atomics
    in ``ATOMIC_SPECIFIERS`` order with duplicates removed.
    """
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


def parse_ref_blocks(spec) -> Optional[Tuple[int, ...]]:
    """Resolve a v3 ``ref_block`` spec to a canonical tuple.

    Accepts ``None`` (defer to the ControlNetLLLiteDiT default), a single int,
    an int sequence, or a comma-separated string like ``"2,13"``. Index ``-1``
    is the special "x_embedder output" tap (before any DiT block). Returns a
    sorted, duplicate-free tuple (the concat order of the trained ``proj_in``).
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        blocks = [spec]
    elif isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("ref_block spec is empty")
        blocks = [int(p) for p in parts]
    else:
        blocks = [int(b) for b in spec]
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"duplicate ref_block index in {blocks}")
    return tuple(sorted(blocks))


# ----------------------------------------------------------------------------
# Conditioning1 trunk (v2 pixel stem / v2.1 latent stem)
# ----------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    """v2 pixel stem / v2.1 latent stem.

    pixel : in (B, cond_in_channels, H*16, W*16) — stride-16 conv stack.
    latent: in (B, 16, H*2, W*2) normalized VAE latent — 3x3 s1 + 2x2 s2 stack.
    The latent-mode inpaint mask pyramid (cond_in_channels=4) is not ported.
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        input_space: str = "pixel",
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        assert input_space in COND_INPUT_SPACES, (
            f"input_space must be one of {list(COND_INPUT_SPACES)}, got {input_space!r}"
        )
        if input_space == "latent" and cond_in_channels != 3:
            raise NotImplementedError(
                "latent-mode inpainting (cond_in_channels=4, mask conv pyramid) is not "
                "supported by this ComfyUI port yet"
            )
        ch_half = cond_dim // 2

        self.cond_in_channels = cond_in_channels
        self.input_space = input_space

        if input_space == "pixel":
            self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
            self.norm1 = _gn(ch_half)
            self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
            self.norm2 = _gn(ch_half)
            self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
            self.norm3 = _gn(cond_dim)
        else:
            self.lat_conv1 = nn.Conv2d(LATENT_COND_CHANNELS, cond_dim, kernel_size=3, stride=1, padding=1)
            self.lat_norm1 = _gn(cond_dim)
            self.lat_conv2 = nn.Conv2d(cond_dim, cond_dim, kernel_size=2, stride=2, padding=0)
            self.lat_norm2 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_space == "pixel":
            h = F.silu(self.norm1(self.conv1(x)))
            h = F.silu(self.norm2(self.conv2(h)))
            h = F.silu(self.norm3(self.conv3(h)))
        else:
            h = F.silu(self.lat_norm1(self.lat_conv1(x)))
            h = F.silu(self.lat_norm2(self.lat_conv2(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h


class _ConditioningSemanticTrunk(nn.Module):
    """v3 semantic trunk: turns frozen-DiT hidden states into the shared cond_emb.

    in  h_ref (B, T=1, H, W, D_model) or (B, K, T=1, H, W, D_model)
    -> LayerNorm(D_model) [per ref block] -> Linear proj_in (K*D_model -> cond_dim)
    -> ResBlocks / (ASPP) at cond_dim -> 1x1 conv -> cond_emb_dim
    -> flatten (B, S, cond_emb_dim) -> LayerNorm

    ``t_proj`` (zero-init) maps the per-step ``t_embedding_norm`` output
    (B, T, D_model) into cond space; it is applied by ``_update_t_local``.
    """

    def __init__(
        self,
        model_dim: int,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        num_ref_blocks: int = 1,
    ):
        super().__init__()
        assert num_ref_blocks >= 1, f"num_ref_blocks must be >= 1, got {num_ref_blocks}"
        self.num_ref_blocks = num_ref_blocks

        if num_ref_blocks == 1:
            self.ln_in = nn.LayerNorm(model_dim)
            self.proj_in = nn.Linear(model_dim, cond_dim)
        else:
            self.ln_in = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_ref_blocks)])
            self.proj_in = nn.Linear(model_dim * num_ref_blocks, cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

        self.t_proj = nn.Linear(model_dim, cond_emb_dim)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

    def forward(self, h_ref: torch.Tensor) -> torch.Tensor:
        if h_ref.dim() == 5:
            h_ref = h_ref.unsqueeze(1)  # (B, K=1, T, H, W, D)
        if h_ref.dim() != 6:
            raise ValueError(
                f"semantic trunk expects (B, T, H, W, D) or (B, K, T, H, W, D), got {tuple(h_ref.shape)}"
            )
        b, k, t, hh, ww, _ = h_ref.shape
        if k != self.num_ref_blocks:
            raise ValueError(
                f"semantic trunk was built for {self.num_ref_blocks} ref block(s) but got {k} "
                f"(check ref_block vs the trained weights)"
            )
        if t != 1:
            raise ValueError(f"semantic trunk supports T=1 only, got T={t}")
        h_ref = h_ref.to(self.proj_in.weight.dtype)
        if self.num_ref_blocks == 1:
            h = self.proj_in(self.ln_in(h_ref[:, 0]))  # (B, 1, H, W, cond_dim)
        else:
            normed = [ln(h_ref[:, i]) for i, ln in enumerate(self.ln_in)]
            h = self.proj_in(torch.cat(normed, dim=-1))
        h = h.squeeze(1).permute(0, 3, 1, 2).contiguous()  # (B, cond_dim, H, W)
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)  # (B, cond_emb_dim, H, W)
        c = h.shape[1]
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()  # (B, S, cond_emb_dim)
        return self.out_norm(h)


# ----------------------------------------------------------------------------
# LLLite module (v2: FiLM + SiLU + 5D path + depth embedding)
# ----------------------------------------------------------------------------

class LLLiteModuleDiT(nn.Module):
    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        gate_mode: str = "none",
        require_t_local: bool = False,
    ):
        super().__init__()
        assert gate_mode in GATE_MODES, f"gate_mode must be one of {list(GATE_MODES)}, got {gate_mode!r}"
        self.lllite_name = name
        # Wrap in a list so the original Linear is not registered as a submodule
        # and its weights stay out of state_dict.
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier
        self.gate_mode = gate_mode
        # v3 semantic trunk: the t-FiLM hook must have fired before every forward.
        self.require_t_local = require_t_local

        in_dim = org_module.in_features

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (gamma, beta), zero-init for identity at start.
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        if gate_mode != "none":
            # scalar: per-token scalar gate / vector: per-token, per-channel gate.
            gate_out = 1 if gate_mode == "scalar" else mlp_dim
            self.gate = nn.Linear(cond_emb_dim + mlp_dim, gate_out)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, GATE_INIT_BIAS)

        self.cond_emb: Optional[torch.Tensor] = None
        # v3: per-step timestep embedding in cond space, distributed by the
        # t_embedding_norm forward hook. Always None for the stem trunk.
        self.t_local: Optional[torch.Tensor] = None
        self.org_forward = None

        # Set by the parent ControlNetLLLiteDiT after construction.
        self.layer_idx: int = -1
        self._depth_embeds_ref: List[nn.Parameter] = []

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layouts:
        #   self/cross attention q/k/v: (B, S, D) — already flattened in the Anima block
        #   mlp.layer1:                 (B, T, H, W, D) — passed un-flattened
        # Flatten the 5D case to 3D for the LLLite path and reshape on exit.
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb  # (B_c, S, cond_emb_dim)

        # Broadcast cond_emb to the runtime batch (CFG cond+uncond, multi-cond).
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        # Run the LLLite mini-MLP in its own parameter dtype, then cast the
        # correction back to ``x``'s dtype before adding. Robust to autocast
        # flows where x and LLLite weights have different dtypes.
        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        # Per-module depth embedding (zero-init so it's a no-op at train start).
        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx

        # v3 t-FiLM: add the per-step timestep embedding in cond space. The hook
        # fires inside the same DiT forward, so t_local batch == x batch; add it
        # after the CFG broadcast of cond_local above (which already matched x).
        t_local = self.t_local
        if self.require_t_local and t_local is None:
            raise RuntimeError(
                f"Anima LLLite: t_local is not set ({self.lllite_name}); the semantic trunk "
                "requires the t_embedding_norm forward hook (install_t_hook) to fire before "
                "the blocks run"
            )
        if t_local is not None:
            if t_local.dtype != param_dtype or t_local.device != x.device:
                t_local = t_local.to(device=x.device, dtype=param_dtype)
            if t_local.shape[0] != cond_local.shape[0] and t_local.shape[0] != 1:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cond_local = cond_local + t_local  # (B, T=1, D) broadcasts over S

        h = F.silu(self.down(x_proc))

        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        if self.gate_mode == "vector":
            # per-token, per-channel gate applied before up: dx = up(g * m)
            g = torch.sigmoid(self.gate(torch.cat([cond_local, h], dim=-1)))  # (B, S, mlp)
            m = m * g

        out = self.up(m)

        if self.gate_mode == "scalar":
            # per-token scalar gate applied to the value path output: dx = g * up(m)
            g = torch.sigmoid(self.gate(torch.cat([cond_local, h], dim=-1)))  # (B, S, 1)
            out = out * g

        out = out * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self.org_forward(x + out)

        if is_5d:
            # org Linear out_features may differ from in_features — recover with -1.
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


# ----------------------------------------------------------------------------
# ControlNetLLLiteDiT
# ----------------------------------------------------------------------------

class ControlNetLLLiteDiT(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
        cond_input_space: str = "pixel",
        trunk: str = "stem",
        ref_block=None,  # int / "2,13"-style str / int sequence (see parse_ref_blocks)
        ref_timestep: float = 0.0,
        ref_context: str = "zero",
        gate: str = "scalar",
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)
        assert cond_input_space in COND_INPUT_SPACES, (
            f"cond_input_space must be one of {list(COND_INPUT_SPACES)}, got {cond_input_space!r}"
        )
        assert trunk in TRUNK_TYPES, f"trunk must be one of {list(TRUNK_TYPES)}, got {trunk!r}"
        assert ref_context in REF_CONTEXT_MODES, (
            f"ref_context must be one of {list(REF_CONTEXT_MODES)}, got {ref_context!r}"
        )
        assert gate in GATE_MODES, f"gate must be one of {list(GATE_MODES)}, got {gate!r}"

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        # 4ch (RGB+mask) inpainting metadata. `inpaint_masked_input` records the training-time
        # RGB-masking policy for cond_image preparation; it does not alter the forward pass here.
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input
        self.cond_input_space = cond_input_space
        self.trunk = trunk
        # The gate is semantic-trunk-only; the stem trunk always runs gateless.
        self.gate_mode = gate if trunk == "semantic" else "none"

        if trunk == "semantic":
            assert cond_input_space == "latent", (
                "trunk='semantic' requires cond_input_space='latent'"
            )
            if cond_in_channels != 3:
                raise NotImplementedError(
                    "trunk='semantic' does not support inpainting (cond_in_channels=4)"
                )

            model_dim = getattr(dit, "model_channels", None)
            if model_dim is None:
                model_dim = getattr(dit.blocks[0], "x_dim", None)
            if model_dim is None:
                raise RuntimeError(
                    "cannot infer the DiT hidden dim (model_channels / blocks[0].x_dim) "
                    "for the semantic trunk"
                )
            num_dit_blocks = len(dit.blocks)
            ref_blocks = parse_ref_blocks(ref_block)
            if ref_blocks is None:
                ref_blocks = (num_dit_blocks // 2,)
            for rb in ref_blocks:
                if not (-1 <= rb < num_dit_blocks):
                    raise ValueError(
                        f"ref_block {rb} out of range (-1 = x_embedder output, "
                        f"0..{num_dit_blocks - 1} = block output)"
                    )
            self.model_dim = int(model_dim)
            self.ref_blocks = ref_blocks
            self.ref_timestep = float(ref_timestep)
            self.ref_context = ref_context

            self.conditioning1 = _ConditioningSemanticTrunk(
                self.model_dim, cond_dim, cond_emb_dim, cond_resblocks,
                use_aspp=use_aspp, aspp_dilations=aspp_dilations,
                num_ref_blocks=len(ref_blocks),
            )
        else:
            self.model_dim = None
            self.ref_blocks = None
            self.ref_timestep = 0.0
            self.ref_context = "zero"

            self.conditioning1 = _Conditioning1(
                cond_dim, cond_emb_dim, cond_resblocks,
                use_aspp=use_aspp, aspp_dilations=aspp_dilations,
                cond_in_channels=cond_in_channels,
                input_space=cond_input_space,
            )

        modules = self._create_modules(
            dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier,
            gate_mode=self.gate_mode,
            require_t_local=trunk == "semantic",
        )
        self.lllite_modules = nn.ModuleList(modules)

        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        inpaint_info = (
            f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels != 3 else ""
        )
        trunk_info = (
            f"trunk=semantic(ref_blocks={list(self.ref_blocks)}, ref_timestep={self.ref_timestep}, "
            f"ref_context={self.ref_context}, gate={self.gate_mode}, model_dim={self.model_dim})"
            if trunk == "semantic"
            else f"trunk=stem, cond_input={cond_input_space}"
        )
        version = LLLITE_ARCH_VERSION_SEMANTIC if trunk == "semantic" else LLLITE_ARCH_VERSION
        logger.info(
            "ControlNet-LLLite (Anima v%s): created %d modules for target=%r "
            "(atomics=%s), %s, cond_in_channels=%d, cond_dim=%d, cond_resblocks=%d, %s, "
            "cond_emb_dim=%d, mlp_dim=%d%s",
            version, n, target_layers, list(atomics), trunk_info,
            cond_in_channels, cond_dim, cond_resblocks, aspp_info, cond_emb_dim, mlp_dim,
            inpaint_info,
        )

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: Tuple[str, ...]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            return False  # cross_attn K,V live in text-embedding space

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: Tuple[str, ...],
        dropout: Optional[float],
        multiplier: float,
        gate_mode: str = "none",
        require_t_local: bool = False,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__
            
            def _is_linear_like(module):
                return (
                hasattr(module, "in_features")
                and hasattr(module, "out_features")
                and callable(getattr(module, "forward", None))
            )

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                # The Anima-block Attention exposes is_selfattn; the LLM-Adapter
                # Attention does not — skip the latter even if path filter misses.
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not _is_linear_like(child):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(
                            full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier,
                            gate_mode, require_t_local,
                        )
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not _is_linear_like(child):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(
                        full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier,
                        gate_mode, require_t_local,
                    )
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """Stem trunk. pixel: (B, C, H*16, W*16) in [-1, 1];
        latent: (B, 16, H*2, W*2) normalized VAE latent. ``None`` clears."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
                m.t_local = None
            return
        assert self.trunk == "stem", (
            "set_cond_image with a tensor is only for the stem trunk; "
            "use set_cond_hidden_states for trunk='semantic'"
        )
        cx = self.conditioning1(cond_image)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def set_cond_hidden_states(self, hidden_states: torch.Tensor):
        """v3 (semantic trunk): run ``encode_reference_hidden_states`` output through
        the semantic trunk and distribute the shared cond_emb.

        single (K=1): (B, T=1, H, W, D_model) / multi: (B, K, T=1, H, W, D_model)."""
        assert self.trunk == "semantic", "set_cond_hidden_states is only for trunk='semantic'"
        cx = self.conditioning1(hidden_states)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def _update_t_local(self, t_emb: torch.Tensor):
        """v3: map the ``t_embedding_norm`` output (B, T, D_model) into cond space and
        distribute it. Called by the forward hook installed via ``install_t_hook``."""
        t_proj = self.conditioning1.t_proj
        t_local = t_proj(t_emb.detach().to(device=t_proj.weight.device, dtype=t_proj.weight.dtype))
        for m in self.lllite_modules:
            m.t_local = t_local

    def install_t_hook(self, dit: nn.Module):
        """Register the per-step t-FiLM forward hook on ``dit.t_embedding_norm``.

        The DiT is a shared object in ComfyUI (``model.clone()`` does not deep-copy
        ``diffusion_model``), so the hook must be scoped to the sampler call:
        install it right before delegating to ``apply_model`` and call
        ``handle.remove()`` in a ``finally`` block, exactly like ``apply_to()`` /
        ``restore()``. Returns the hook handle."""
        assert self.trunk == "semantic", "install_t_hook is only for trunk='semantic'"
        t_norm = getattr(dit, "t_embedding_norm", None)
        if t_norm is None:
            raise RuntimeError(
                "DiT has no t_embedding_norm; cannot install the v3 t-FiLM hook"
            )

        def _t_hook(_module, _inputs, output, _self=self):
            _self._update_t_local(output)
            return None

        return t_norm.register_forward_hook(_t_hook)

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()


# ----------------------------------------------------------------------------
# v3 reference forward (ComfyUI MiniTrainDIT version)
# ----------------------------------------------------------------------------

@torch.no_grad()
def encode_reference_hidden_states(
    dit: nn.Module,
    cond_latent: torch.Tensor,
    ref_block,
    ref_timestep: float = 0.0,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """v3 (semantic trunk): run the cond latent through the frozen Anima DiT and
    return the hidden states at ``ref_block``. Blocks past ``max(ref_block)`` are
    not executed. Mirrors the sd-scripts reference but uses ComfyUI's
    ``MiniTrainDIT`` API (``t_embedder`` Sequential, block kwargs, fp16 residual
    stream kept in fp32 like the stock ``_forward``).

    Index ``-1`` taps the ``x_embedder`` output (before any block); ``-1`` alone
    runs no blocks at all (no t embedding / context needed).

    ``context=None`` (default) feeds a zero context: Anima's k/v projections are
    bias-free and padding tokens are zeroed-then-attended by convention, so the
    cross-attention output is exactly 0 (fully text-independent features). Pass a
    post-LLM-Adapter (B or 1, L, ctx_dim) tensor for ref_context='uncond'.

    The caller must make sure the LLLite modules pass through (cond cleared or
    not yet applied) while this runs, since it executes the patched Linears.

    Args:
        dit: the (frozen) Anima diffusion model (``model.model.diffusion_model``)
        cond_latent: (B, 16, h, w) or (B, 16, 1, h, w) normalized VAE latent
        ref_block: int, int sequence or "2,13"-style str (normalized ascending)
        ref_timestep: reference-forward timestep on the training [0, 1] scale

    Returns:
        int spec:            (B, 1, H, W, D_model) hidden states (token grid = latent / 2)
        sequence / str spec: (B, K, 1, H, W, D_model), K in ascending ref_block order
    """
    single = isinstance(ref_block, int)
    ref_blocks = parse_ref_blocks(ref_block)
    if not ref_blocks:
        raise ValueError(f"invalid ref_block: {ref_block!r}")

    if cond_latent.dim() == 4:
        cond_latent = cond_latent.unsqueeze(2)  # (B, 16, 1, h, w)
    if cond_latent.dim() != 5 or cond_latent.shape[2] != 1:
        raise ValueError(f"cond_latent must be (B, C, 1, h, w), got {tuple(cond_latent.shape)}")
    bsz = cond_latent.shape[0]

    num_blocks = len(dit.blocks)
    if not (-1 <= ref_blocks[0] and ref_blocks[-1] < num_blocks):
        raise ValueError(
            f"ref_block {list(ref_blocks)} out of range (num_blocks={num_blocks}, "
            f"-1 = x_embedder output)"
        )

    # padding_mask=None -> prepare_embedded_sequence builds a zero mask internally,
    # matching the sd-scripts reference forward.
    x, rope_emb_L_1_1_D, extra_pos_emb = dit.prepare_embedded_sequence(
        cond_latent, fps=None, padding_mask=None
    )

    ref_set = set(ref_blocks)
    max_block = ref_blocks[-1]
    collected: List[torch.Tensor] = []
    if -1 in ref_set:
        collected.append(x)

    if max_block >= 0:
        t = torch.full((bsz, 1), float(ref_timestep), device=cond_latent.device, dtype=cond_latent.dtype)
        t_embedding_B_T_D, adaln_lora_B_T_3D = dit.t_embedder[1](dit.t_embedder[0](t).to(x.dtype))
        t_embedding_B_T_D = dit.t_embedding_norm(t_embedding_B_T_D)

        if context is None:
            ctx_dim = dit.blocks[0].cross_attn.context_dim
            context = torch.zeros(bsz, 1, ctx_dim, device=x.device, dtype=t_embedding_B_T_D.dtype)
        else:
            if context.dim() != 3:
                raise ValueError(f"context must be (B, L, ctx_dim), got {tuple(context.shape)}")
            context = context.to(device=x.device, dtype=t_embedding_B_T_D.dtype)
            if context.shape[0] == 1 and bsz > 1:
                context = context.expand(bsz, -1, -1)
            if context.shape[0] != bsz:
                raise ValueError(
                    f"context batch mismatch: {context.shape[0]} vs cond latent {bsz}"
                )

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": {},
        }

        # Mirror the stock _forward: keep the residual stream in fp32 under fp16.
        if x.dtype == torch.float16:
            x = x.float()

        for block_idx, block in enumerate(dit.blocks):
            x = block(x, t_embedding_B_T_D, context, **block_kwargs)
            if block_idx in ref_set:
                collected.append(x)
            if block_idx == max_block:
                break

    if len(collected) != len(ref_blocks):
        raise RuntimeError(
            f"collected {len(collected)} hidden states for ref_blocks={list(ref_blocks)}"
        )
    if single:
        return collected[0]  # (B, T, H, W, D)
    return torch.stack(collected, dim=1)  # (B, K, T, H, W, D)


@torch.no_grad()
def build_uncond_ref_context(
    dit: nn.Module, device, dtype, pad_to_length: Optional[int] = None
) -> torch.Tensor:
    """Build the constant context (1, pad_to_length or 1, ctx_dim) for
    ref_context='uncond': a length-1 zero embed run through the LLM Adapter with
    a single T5 ``</s>`` target token, then zero-padded to the main forward's
    context length (the zero padding dilutes the softmax exactly like caption
    dropout does at train time — do NOT skip it)."""
    adapter = getattr(dit, "llm_adapter", None)
    if adapter is None:
        raise RuntimeError("DiT has no llm_adapter; ref_context='uncond' requires it")
    src_dim = adapter.blocks[0].cross_attn.context_dim
    source = torch.zeros(1, 1, src_dim, device=device, dtype=dtype)
    ones = torch.ones(1, 1, device=device, dtype=torch.long)
    t5_input_ids = ones.clone()  # T5 </s> = token id 1
    ctx = adapter(source, t5_input_ids, target_attention_mask=ones, source_attention_mask=ones)
    ctx = ctx.to(dtype)  # (1, 1, ctx_dim)
    if pad_to_length is not None and pad_to_length > 1:
        pad = torch.zeros(1, pad_to_length - 1, ctx.shape[-1], device=ctx.device, dtype=ctx.dtype)
        ctx = torch.cat([ctx, pad], dim=1)
    return ctx


# ----------------------------------------------------------------------------
# Save / load (named-key format; legacy lllite_modules.* is rejected)
# ----------------------------------------------------------------------------

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _from_saved_state_dict(lllite: "ControlNetLLLiteDiT", weights_sd: dict) -> dict:
    """Rewrite a v2 named-key state dict back to the internal layout."""
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(
                f"depth_embed slices missing for module idx(es) {missing}"
            )
        out[_INTERNAL_DEPTH_KEY] = torch.stack(
            [depth_slices[i] for i in range(n_modules)], dim=0
        )

    return out


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in a legacy ControlNet-LLLite weight format "
            f"(keys starting with '{_INTERNAL_MODULES_PREFIX}'). The current code uses a "
            f"named-key format (per-module key prefix = lllite_name, e.g. "
            f"'lllite_dit_blocks_0_self_attn_q_proj.down.weight'). Re-train with the current codebase."
        )

    # Trunk (stem/semantic) mix-up detection: the conditioning key names are
    # disjoint, so strict=False would otherwise silently keep random init.
    file_trunk = None
    if any(k.startswith(_SAVED_COND_PREFIX + "proj_in.") for k in weights_sd):
        file_trunk = "semantic"
    elif any(
        k.startswith(_SAVED_COND_PREFIX + "lat_conv1.") or k.startswith(_SAVED_COND_PREFIX + "conv1.")
        for k in weights_sd
    ):
        file_trunk = "stem"
    if file_trunk is not None and file_trunk != lllite.trunk:
        raise RuntimeError(
            f"trunk mismatch: weights at {file} were trained with trunk='{file_trunk}', "
            f"but this LLLite was built with trunk='{lllite.trunk}'. "
            f"Check the 'lllite.trunk' metadata."
        )

    # semantic: single/multi (ref block count) mix-up detection.
    if file_trunk == "semantic" and lllite.trunk == "semantic":
        if any(k == _SAVED_COND_PREFIX + "ln_in.weight" for k in weights_sd):
            file_k = 1
        else:
            ln_prefix = _SAVED_COND_PREFIX + "ln_in."
            file_k = len(
                {k[len(ln_prefix):].split(".")[0] for k in weights_sd if k.startswith(ln_prefix)}
            )
        model_k = len(lllite.ref_blocks)
        if file_k != model_k:
            raise RuntimeError(
                f"ref block count mismatch: weights at {file} were trained with {file_k} ref "
                f"block(s), but this LLLite was built with ref_blocks="
                f"{list(lllite.ref_blocks)} ({model_k}). Check the 'lllite.ref_block' metadata."
            )

    # v3 gate mode (scalar / vector / none) mix-up detection via gate.weight shape.
    gate_weight_keys = [k for k in weights_sd if k.endswith(".gate.weight")]
    if not gate_weight_keys:
        file_gate = "none"
    elif weights_sd[gate_weight_keys[0]].shape[0] == 1:
        file_gate = "scalar"
    else:
        file_gate = "vector"
    if file_gate != lllite.gate_mode:
        raise RuntimeError(
            f"gate mode mismatch: weights at {file} were trained with gate='{file_gate}', "
            f"but this LLLite was built with gate='{lllite.gate_mode}'. "
            f"Check the 'lllite.gate' metadata."
        )

    # pixel / latent mix-up detection (stem trunk only; semantic is latent-fixed).
    file_space = None
    if any(k.startswith(_SAVED_COND_PREFIX + "lat_conv1.") for k in weights_sd):
        file_space = "latent"
    elif any(k.startswith(_SAVED_COND_PREFIX + "conv1.") for k in weights_sd):
        file_space = "pixel"
    if file_space is not None and file_space != lllite.cond_input_space:
        raise RuntimeError(
            f"cond input space mismatch: weights at {file} were trained with "
            f"cond_input_space='{file_space}', but this LLLite was built with "
            f"cond_input_space='{lllite.cond_input_space}'. "
            f"Check the 'lllite.cond_input_space' metadata."
        )

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info("loaded LLLite weights from %s: %s", file, info)
    return info


def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != ".safetensors":
        return {}
    from safetensors import safe_open
    with safe_open(file, framework="pt") as f:
        meta = f.metadata()
    return meta or {}
