"""ComfyUI node for Anima ControlNet-LLLite.

Single LoRA-style node: takes a MODEL, an LLLite weights file, a control IMAGE
and a strength; returns the patched MODEL. Integration is done via
``set_model_unet_function_wrapper`` so the LLLite contribution is fully scoped
to this model clone — no global monkey-patching that could leak into other
samplers in the same workflow.

Because ``model_function_wrapper`` is a single-slot field on ``model_options``,
cascading two wrapper-installing nodes would normally cause the outer one to
silently overwrite the inner one. The node captures any pre-existing wrapper
before cloning and delegates to it from inside its own wrapper, so multiple
Anima-LLLite nodes (and other well-behaved wrapper nodes) can be stacked. The
``preserve_wrapper`` toggle (default on) controls this delegation, mirroring
``ChromaRadianceOptions``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

import folder_paths

from .control_net_lllite_anima import (
    ASPP_DEFAULT_DILATIONS,
    ControlNetLLLiteDiT,
    load_lllite_weights,
    read_lllite_metadata,
)

logger = logging.getLogger(__name__)


def _get_inner_dit(model) -> torch.nn.Module:
    """Reach the underlying Anima DiT (nn.Module) from a ComfyUI ModelPatcher."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model attribute (not a ModelPatcher?)")
    dit = getattr(inner, "diffusion_model", None)
    if dit is None:
        raise RuntimeError("MODEL.model has no .diffusion_model — not a UNet/DiT model?")
    return dit


def _target_cond_hw(latent_h: int, latent_w: int, patch_spatial: int = 2) -> tuple[int, int]:
    """Return the (H, W) the cond image / mask must be resized to.

    The LLLite ``conditioning1`` Conv has stride 16, so the cond image must be
    sized to ``latent_HW * 8`` in input pixel space (= ``token_HW * 16`` after
    DiT patchify with patch_spatial=2). The DiT internally pads the latent up
    to a multiple of ``patch_spatial`` (see ``MiniTrainDIT.forward`` →
    ``pad_to_patch_size``), so we mirror that rounding here — otherwise odd
    latent dims (e.g. 1032 px → 129 latent) yield a token-count mismatch that
    silently bypasses every LLLite module.
    """
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h * 8, padded_w * 8


def _prepare_cond_image(image: torch.Tensor, latent_h: int, latent_w: int,
                        device: torch.device, dtype: torch.dtype,
                        patch_spatial: int = 2) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,3) in [0,1] → (1,3,H*8,W*8) in [-1,1]."""
    if image.ndim == 4 and image.shape[-1] == 3:
        # (B, H, W, 3) -> (B, 3, H, W)
        img = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")

    img = img[:1]  # use first frame only
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


def _prepare_mask(mask: torch.Tensor, latent_h: int, latent_w: int,
                  device: torch.device, dtype: torch.dtype,
                  patch_spatial: int = 2) -> torch.Tensor:
    """ComfyUI MASK (B,H,W) in [0,1] → (1,1,H*8,W*8) binarized at 0.5.

    Returns the mask in ``{0.0, 1.0}`` (1 = inpaint area, 0 = keep). The caller
    is responsible for the ``*2-1`` rescale before concat with RGB.
    """
    if mask.ndim == 3:
        m = mask.unsqueeze(1)              # (B, 1, H, W)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)} (expected B,H,W or B,1,H,W)")

    m = m[:1]
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if m.shape[-2] != target_h or m.shape[-1] != target_w:
        m = F.interpolate(m.float(), size=(target_h, target_w), mode="nearest")
    m = (m >= 0.5).to(dtype=dtype)
    return m.to(device=device)


def _build_inpaint_cond_image(rgb_pm1: torch.Tensor, mask01: torch.Tensor,
                              masked_input: bool) -> torch.Tensor:
    """rgb_pm1: (1,3,H,W) in [-1,1], mask01: (1,1,H,W) in {0,1}. Returns (1,4,H,W).

    Mirrors ``_build_inpaint_cond_image`` in the sd-scripts training / inference
    code: the mask channel is rescaled to ``[-1, +1]`` (matches the RGB range),
    and if ``masked_input`` is set the RGB is zeroed where ``mask >= 0.5``.
    """
    if masked_input:
        keep = (mask01 < 0.5).to(rgb_pm1.dtype)
        rgb_pm1 = rgb_pm1 * keep
    mask_pm1 = mask01.to(rgb_pm1.dtype) * 2.0 - 1.0
    return torch.cat([rgb_pm1, mask_pm1], dim=1)


class AnimaLLLiteApply_sdscripts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lllite_name": (folder_paths.get_filename_list("controlnet"),),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Required when the loaded weights are 4ch (inpaint). White = inpaint area,
                # black = keep. Mismatch with the weights' cond_in_channels is reported below.
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, lllite_name, image, strength, start_percent, end_percent,
              preserve_wrapper=True, mask=None):
        weights_path = folder_paths.get_full_path("controlnet", lllite_name)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LLLite weights not found: {lllite_name}")

        # Architecture is fully determined by the trained weights — read everything
        # from metadata rather than exposing knobs that would just cause load errors.
        meta = read_lllite_metadata(weights_path)
        ce_dim = int(meta.get("lllite.cond_emb_dim", 32))
        m_dim = int(meta.get("lllite.mlp_dim", 64))
        # v2 records the canonical atomic form under lllite.target_atomics; fall back
        # to the legacy preset key, then to the v1 default.
        tl = meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
        cond_dim = int(meta.get("lllite.cond_dim", 64))
        cond_resblocks = int(meta.get("lllite.cond_resblocks", 1))
        use_aspp = str(meta.get("lllite.use_aspp", "false")).lower() == "true"
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            aspp_dilations = ASPP_DEFAULT_DILATIONS
        cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))
        inpaint_masked_input = str(meta.get("lllite.inpaint_masked_input", "false")).lower() == "true"

        # Mask / cond_in_channels consistency: 4ch weights need a MASK, 3ch weights ignore it.
        if cond_in_channels == 4 and mask is None:
            raise ValueError(
                f"LLLite weights '{lllite_name}' were trained with cond_in_channels=4 "
                f"(inpaint mode) and require a MASK input. Connect a MASK to the node "
                f"(white = inpaint area, black = keep)."
            )
        if cond_in_channels != 4 and mask is not None:
            logger.warning(
                "LLLite weights '%s' are %dch; the provided MASK input will be ignored.",
                lllite_name, cond_in_channels,
            )
            mask = None

        dit = _get_inner_dit(model)
        patch_spatial = int(getattr(dit, "patch_spatial", 2))
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=ce_dim,
            mlp_dim=m_dim,
            target_layers=tl,
            multiplier=strength,
            cond_dim=cond_dim,
            cond_resblocks=cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
            inpaint_masked_input=inpaint_masked_input,
        )
        load_lllite_weights(lllite, weights_path, strict=False)
        lllite.eval().requires_grad_(False)

        # Convert percent range -> sigma range (start_percent=0 → sigma_max).
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        # Capture image / mask tensors (cloned to detach from any upstream caching)
        src_image = image.detach().clone()
        src_mask = mask.detach().clone() if mask is not None else None
        is_inpaint = cond_in_channels == 4

        # Cache for the per-resolution preprocessed cond image (avoids repeat resize)
        cache = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}

        # Capture any previously-installed wrapper BEFORE we clone — model_options
        # has a single "model_function_wrapper" slot, so without delegation a second
        # wrapper-installing node would silently no-op the first. Mirrors the
        # ChromaRadianceOptions pattern in comfy_extras/nodes_chroma_radiance.py.
        old_wrapper = model.model_options.get("model_function_wrapper")

        def _call_next(apply_model, input_x, timestep, c):
            if preserve_wrapper and old_wrapper is not None:
                return old_wrapper(apply_model, {"input": input_x, "timestep": timestep, "c": c})
            return apply_model(input_x, timestep, **c)

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Step-range gate: skip LLLite entirely when current sigma is outside
            # [sigma_end, sigma_start]. percent_to_sigma maps 0.0 → sigma_max,
            # 1.0 → sigma_min, so the active window is sigma_end <= sigma <= sigma_start.
            sigma = float(timestep.max().item())
            if not (sigma_end <= sigma <= sigma_start):
                return _call_next(apply_model, input_x, timestep, c)

            # Anima latent shape: (B, C, T, H, W) — take spatial dims from the tail.
            latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
            device = input_x.device
            dtype = input_x.dtype

            # Move LLLite to the runtime device/dtype lazily.
            tag = (device, dtype)
            if cache["lllite_loaded_to"] != tag:
                lllite.to(device=device, dtype=dtype)
                cache["lllite_loaded_to"] = tag
                cache["cond_image_pp"] = None  # invalidate

            key = (latent_h, latent_w, device, dtype)
            if cache["key"] != key or cache["cond_image_pp"] is None:
                rgb = _prepare_cond_image(
                    src_image, latent_h, latent_w, device, dtype, patch_spatial
                )
                if is_inpaint:
                    mk = _prepare_mask(
                        src_mask, latent_h, latent_w, device, dtype, patch_spatial
                    )
                    cache["cond_image_pp"] = _build_inpaint_cond_image(
                        rgb, mk, inpaint_masked_input
                    )
                else:
                    cache["cond_image_pp"] = rgb
                cache["key"] = key

            lllite.set_multiplier(strength)
            lllite.set_cond_image(cache["cond_image_pp"])
            lllite.apply_to()
            try:
                return _call_next(apply_model, input_x, timestep, c)
            finally:
                lllite.restore()
                lllite.clear_cond_image()

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaLLLiteApply_sdscripts": AnimaLLLiteApply_sdscripts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLLLiteApply_sdscripts": "Apply Anima ControlNet-LLLite (sd-scripts)",
}
