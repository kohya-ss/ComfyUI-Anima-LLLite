"""ComfyUI node for Anima ControlNet-LLLite.

Single LoRA-style node: takes a MODEL, an LLLite weights file, a control input
(IMAGE, or a VAE-encoded LATENT) and a strength; returns the patched MODEL.
Integration is done via ``set_model_unet_function_wrapper`` so the LLLite
contribution is fully scoped to this model clone — no global monkey-patching
that could leak into other samplers in the same workflow.

Weight architectures (auto-detected from the safetensors metadata):
  * v2 (pixel stem)  — control IMAGE is fed to the conv stem directly.
  * v2.1 (latent stem) — the control image is VAE-encoded (needs the ``vae``
    input, or a ``cond_latent`` LATENT) and the normalized 16ch latent is fed
    to the latent stem. Inpaint (4ch) latent weights are not supported yet.
  * v3 (semantic trunk) — the cond latent is run through the frozen DiT itself
    (``encode_reference_hidden_states``) and the tapped hidden states are the
    conditioning source. ``ref_context`` zero / uncond are supported; caption
    is not (it needs per-CFG-branch reference features).

Control input for latent-based weights (v2.1 / v3): connect either
  * ``image`` + ``vae`` — the node resizes the image to the generation size and
    encodes it once per resolution, or
  * ``cond_latent`` — e.g. the same VAEEncode output used as the img2img init
    latent (useful for editing-style v3 weights where control == init image).
    The latent resolution must match the generation latent. Normalization
    (``latent_format.process_in``) is applied by the node in both paths.

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

import comfy.ldm.common_dit
import folder_paths

from .control_net_lllite_anima import (
    ASPP_DEFAULT_DILATIONS,
    ControlNetLLLiteDiT,
    build_uncond_ref_context,
    encode_reference_hidden_states,
    load_lllite_weights,
    parse_ref_blocks,
    read_lllite_metadata,
)

logger = logging.getLogger(__name__)

LATENT_CHANNELS = 16


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
    """Return the padded latent (H, W) the cond input must match.

    The DiT pads the latent up to a multiple of ``patch_spatial`` (see
    ``MiniTrainDIT._forward`` → ``pad_to_patch_size``); the LLLite cond must be
    sized against the padded grid, otherwise odd latent dims yield a
    token-count mismatch that silently bypasses every LLLite module.
    """
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h, padded_w


def _prepare_cond_image(image: torch.Tensor, padded_h: int, padded_w: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,C) in [0,1] → (1,3,H*8,W*8) in [-1,1] (pixel stem)."""
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")
    img = image[..., :3].permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

    img = img[:1]  # use first frame only
    target_h, target_w = padded_h * 8, padded_w * 8
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


def _prepare_cond_pixels_for_vae(image: torch.Tensor, padded_h: int, padded_w: int) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,C) in [0,1] → (1,H*8,W*8,3) in [0,1] for VAE.encode."""
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")
    img = image[..., :3][:1].permute(0, 3, 1, 2)  # (1, 3, H, W)
    target_h, target_w = padded_h * 8, padded_w * 8
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    return img.permute(0, 2, 3, 1).contiguous()  # (1, H, W, 3)


def _prepare_mask(mask: torch.Tensor, padded_h: int, padded_w: int,
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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
    target_h, target_w = padded_h * 8, padded_w * 8
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
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Control image. Required for pixel-stem (v2) weights; for
                # latent-based weights (v2.1 / v3) provide image+vae OR cond_latent.
                "image": ("IMAGE",),
                # Required when the loaded weights are 4ch pixel (inpaint). White =
                # inpaint area, black = keep.
                "mask": ("MASK",),
                # VAE used to encode `image` for latent-based weights (v2.1 / v3).
                "vae": ("VAE",),
                # Alternative to image+vae for latent-based weights: a VAE-encoded
                # control latent (e.g. the same VAEEncode output used as the img2img
                # init latent). Must match the generation latent resolution.
                "cond_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, lllite_name, strength, start_percent, end_percent,
              preserve_wrapper=True, image=None, mask=None, vae=None, cond_latent=None):
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
        # v2.1 / v3 keys (missing on older files → v2 pixel stem defaults)
        cond_input_space = meta.get("lllite.cond_input_space", "pixel")
        trunk = meta.get("lllite.trunk", "stem")
        ref_blocks_meta = parse_ref_blocks(meta.get("lllite.ref_block"))
        ref_timestep = float(meta.get("lllite.ref_timestep", 0.0))
        ref_context = meta.get("lllite.ref_context", "zero")
        gate = meta.get("lllite.gate", "scalar")

        is_semantic = trunk == "semantic"
        needs_latent_cond = is_semantic or cond_input_space == "latent"

        if is_semantic and ref_context == "caption":
            raise NotImplementedError(
                f"LLLite weights '{lllite_name}' were trained with ref_context='caption', "
                f"which this node does not support yet (it needs per-CFG-branch reference "
                f"features). Only ref_context 'zero' / 'uncond' weights are supported."
            )
        if needs_latent_cond:
            if cond_in_channels == 4:
                raise NotImplementedError(
                    f"LLLite weights '{lllite_name}' use latent-mode inpainting "
                    f"(cond_in_channels=4), which this node does not support yet."
                )
            if cond_latent is None and (image is None or vae is None):
                raise ValueError(
                    f"LLLite weights '{lllite_name}' take a latent control input "
                    f"({'v3 semantic trunk' if is_semantic else 'v2.1 latent stem'}). "
                    f"Connect either a cond_latent (LATENT), or an image (IMAGE) plus a vae (VAE)."
                )
        else:
            if image is None:
                raise ValueError(
                    f"LLLite weights '{lllite_name}' take a pixel control input; "
                    f"connect an image (IMAGE) to the node."
                )

        # Mask / cond_in_channels consistency: 4ch pixel weights need a MASK,
        # everything else ignores it.
        is_inpaint = cond_in_channels == 4 and not needs_latent_cond
        if is_inpaint and mask is None:
            raise ValueError(
                f"LLLite weights '{lllite_name}' were trained with cond_in_channels=4 "
                f"(inpaint mode) and require a MASK input. Connect a MASK to the node "
                f"(white = inpaint area, black = keep)."
            )
        if not is_inpaint and mask is not None:
            logger.warning(
                "LLLite weights '%s' do not take a MASK; the provided MASK input will be ignored.",
                lllite_name,
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
            cond_input_space=cond_input_space,
            trunk=trunk,
            ref_block=ref_blocks_meta,
            ref_timestep=ref_timestep,
            ref_context=ref_context if is_semantic else "zero",
            gate=gate,
        )
        load_lllite_weights(lllite, weights_path, strict=False)
        lllite.eval().requires_grad_(False)

        # Convert percent range -> sigma range (start_percent=0 → sigma_max).
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        # Capture source tensors (cloned to detach from any upstream caching)
        src_image = image.detach().clone() if image is not None else None
        src_mask = mask.detach().clone() if mask is not None else None
        src_latent = None
        if cond_latent is not None:
            src_latent = cond_latent["samples"].detach().clone()

        inner_model = model.model
        latent_format = inner_model.latent_format

        def _dit_compute_dtype(fallback: torch.dtype) -> torch.dtype:
            # Same dtype the main path feeds the DiT (BaseModel.apply_model).
            dtype = inner_model.get_dtype()
            manual = getattr(inner_model, "manual_cast_dtype", None)
            if manual is not None:
                dtype = manual
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                dtype = fallback  # e.g. fp8-quantized weights without manual cast
            return dtype

        def _prepare_cond_latent(padded_h: int, padded_w: int,
                                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            """Build the normalized cond latent (1, 16, 1, h, w) for v2.1 / v3."""
            if src_latent is not None:
                lat = src_latent
                if lat.ndim == 4:
                    lat = lat.unsqueeze(2)  # (B, C, h, w) -> (B, C, 1, h, w)
                if lat.ndim != 5 or lat.shape[1] != LATENT_CHANNELS:
                    raise ValueError(
                        f"Anima LLLite: unexpected cond_latent shape {tuple(src_latent.shape)} "
                        f"(expected (B, {LATENT_CHANNELS}, [1,] h, w))"
                    )
                lat = lat[:1, :, :1]
                # Mirror the DiT's own padding so the token grid matches.
                lat = comfy.ldm.common_dit.pad_to_patch_size(
                    lat, (1, patch_spatial, patch_spatial)
                )
                if lat.shape[-2] != padded_h or lat.shape[-1] != padded_w:
                    raise ValueError(
                        f"Anima LLLite: cond_latent size {src_latent.shape[-2]}x{src_latent.shape[-1]} "
                        f"does not match the generation latent (expected {padded_h}x{padded_w} after "
                        f"padding). Resize the control image before VAEEncode, or connect "
                        f"image + vae instead."
                    )
            else:
                pixels = _prepare_cond_pixels_for_vae(src_image, padded_h, padded_w)
                lat = vae.encode(pixels)
                if lat.ndim == 4:
                    lat = lat.unsqueeze(2)
                if lat.shape[1] != LATENT_CHANNELS:
                    raise ValueError(
                        f"Anima LLLite: the connected VAE produced a {lat.shape[1]}ch latent "
                        f"(expected {LATENT_CHANNELS}ch — use the Qwen-Image VAE)"
                    )
                lat = lat[:1, :, :1]
            # Match sd-scripts' normalized-latent convention ((z - mean) / std).
            lat = latent_format.process_in(lat.to(device=device, dtype=torch.float32))
            return lat.to(dtype=dtype)

        # Cache for the per-resolution preprocessed cond (avoids repeat resize /
        # VAE encode / reference forward)
        cache = {"key": None, "cond": None, "lllite_loaded_to": None}

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

            if strength == 0.0:
                return _call_next(apply_model, input_x, timestep, c)

            # Step-range gate: skip LLLite entirely when current sigma is outside
            # [sigma_end, sigma_start]. percent_to_sigma maps 0.0 → sigma_max,
            # 1.0 → sigma_min, so the active window is sigma_end <= sigma <= sigma_start.
            sigma = float(timestep.max().item())
            if not (sigma_end <= sigma <= sigma_start):
                return _call_next(apply_model, input_x, timestep, c)

            # Anima latent shape: (B, C, T, H, W) — take spatial dims from the tail.
            latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
            padded_h, padded_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
            device = input_x.device
            dtype = _dit_compute_dtype(input_x.dtype)

            # Move LLLite to the runtime device/dtype lazily.
            tag = (device, dtype)
            if cache["lllite_loaded_to"] != tag:
                lllite.to(device=device, dtype=dtype)
                cache["lllite_loaded_to"] = tag
                cache["cond"] = None  # invalidate

            key = [latent_h, latent_w, device, dtype]
            ref_ctx_len = None
            if is_semantic and lllite.ref_context == "uncond":
                # The zero-pad length of the constant uncond context follows the
                # main forward's context length (usually 512).
                cctx = c.get("c_crossattn")
                ref_ctx_len = int(cctx.shape[1]) if cctx is not None else 512
                key.append(ref_ctx_len)
            key = tuple(key)

            if cache["key"] != key or cache["cond"] is None:
                if is_semantic:
                    # v3: run the frozen DiT over the cond latent once (h_ref is
                    # t-invariant). This happens BEFORE apply_to()/install_t_hook(),
                    # so the LLLite modules pass through and no stale t leaks in.
                    cond_lat = _prepare_cond_latent(padded_h, padded_w, device, dtype)
                    ref_ctx = None
                    if lllite.ref_context == "uncond":
                        ref_ctx = build_uncond_ref_context(
                            dit, device, dtype, pad_to_length=ref_ctx_len
                        )
                    h_ref = encode_reference_hidden_states(
                        dit, cond_lat, lllite.ref_blocks, lllite.ref_timestep, context=ref_ctx
                    )
                    cache["cond"] = h_ref
                    logger.info(
                        "Anima LLLite v3 reference forward: h_ref=%s (ref_blocks=%s, "
                        "ref_timestep=%s, ref_context=%s)",
                        tuple(h_ref.shape), list(lllite.ref_blocks), lllite.ref_timestep,
                        lllite.ref_context,
                    )
                elif needs_latent_cond:
                    # v2.1 latent stem: (1, 16, h, w)
                    cond_lat = _prepare_cond_latent(padded_h, padded_w, device, dtype)
                    cache["cond"] = cond_lat.squeeze(2)
                else:
                    rgb = _prepare_cond_image(src_image, padded_h, padded_w, device, dtype)
                    if is_inpaint:
                        mk = _prepare_mask(src_mask, padded_h, padded_w, device, dtype)
                        cache["cond"] = _build_inpaint_cond_image(rgb, mk, inpaint_masked_input)
                    else:
                        cache["cond"] = rgb
                cache["key"] = key

            lllite.set_multiplier(strength)
            if is_semantic:
                lllite.set_cond_hidden_states(cache["cond"])
            else:
                lllite.set_cond_image(cache["cond"])
            # The DiT is shared across model clones, so both the Linear patches and
            # the v3 t-FiLM hook are scoped to this call and removed in `finally`.
            t_hook_handle = lllite.install_t_hook(dit) if is_semantic else None
            lllite.apply_to()
            try:
                return _call_next(apply_model, input_x, timestep, c)
            finally:
                lllite.restore()
                if t_hook_handle is not None:
                    t_hook_handle.remove()
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
