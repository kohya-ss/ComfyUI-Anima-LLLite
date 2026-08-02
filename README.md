# ComfyUI-Anima-LLLite

ComfyUI custom node for **ControlNet-LLLite for Anima** (DiT-based).

LLLite is a lightweight ControlNet variant that injects a low-rank correction
into the attention (and optionally MLP) projections of the DiT. This node
loads weights trained with
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and applies them
to the ComfyUI Anima model at inference time.

This is intended as a **minimal reference implementation**. Community nodes
with extra features (per-step scheduling, multi-cond, region masks, per-block
selection, …) are welcome and can build on this codebase.

## ⚠️ Breaking change: node renamed to `AnimaLLLiteApply_sdscripts`

ComfyUI core has added its own (beta) Anima LLLite node named
`AnimaLLLiteApply` (in `comfy_extras/nodes_model_patch.py`). ComfyUI silently
skips custom-node registrations whose node ID collides with a built-in one,
so this node's original ID `AnimaLLLiteApply` stopped loading. The node has
therefore been renamed:

* Node ID: `AnimaLLLiteApply` → **`AnimaLLLiteApply_sdscripts`**
* Display name: **"Apply Anima ControlNet-LLLite (sd-scripts)"**

**Existing workflows saved with the old node need to be fixed manually:**
the old node type now resolves to the built-in core node, which has a
different input signature (it takes a `MODEL_PATCH` from `ModelPatchLoader`
instead of loading the weights file directly, and has no `mask` /
`preserve_wrapper` inputs), so old workflows will fail validation. Delete
the old node, add **Apply Anima ControlNet-LLLite (sd-scripts)**, and
rewire it. There is unfortunately no way to keep the old ID as an alias
because the built-in node now owns it.

Note that the core node is a native reimplementation of the **same v2
weight format** this node loads (sd-scripts-trained checkpoints work with
both). The main practical differences: the core node loads weights from
`models/model_patches/` via `ModelPatchLoader` (this node reads from
`models/controlnet/`), integrates through ComfyUI's attention/MLP patch
hooks instead of a `model_function_wrapper`, and currently supports
single-frame (T=1) latents only.

## Install

Clone into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes
git clone <this-repo> ComfyUI-Anima-LLLite
```

Place LLLite weights (`.safetensors`) under `ComfyUI/models/controlnet/`.

## Node

**Apply Anima ControlNet-LLLite (sd-scripts)** (`loaders` category, node ID
`AnimaLLLiteApply_sdscripts`)

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Anima checkpoint |
| `lllite_name` | filename | from `models/controlnet/` |
| `image` | IMAGE | control image (any resolution; auto-resized to latent×8) |
| `strength` | FLOAT | LLLite multiplier (default 1.0) |
| `start_percent` | FLOAT | start of the active sampling window (0.0 = sigma_max) |
| `end_percent` | FLOAT | end of the active sampling window (1.0 = sigma_min) |
| `preserve_wrapper` | BOOLEAN | if a previous node already installed a `model_function_wrapper`, delegate to it from inside this node's wrapper so they cascade instead of overwriting (default `True`) |
| `mask` *(optional)* | MASK | required when the weights are 4-channel (inpaint); white = inpaint area, black = keep |

Output: patched `MODEL`.

All architectural parameters (`cond_emb_dim`, `mlp_dim`, `cond_dim`,
`cond_resblocks`, `use_aspp`, `aspp_dilations`, `target_layers`,
`cond_in_channels`, `inpaint_masked_input`) are baked into the trained
weights and read automatically from the safetensors metadata — they are
not exposed as node inputs because changing them would just cause load
errors.

### Inpainting (4-channel) weights

If the loaded weights were trained with `--lllite_cond_in_channels=4`
(inpaint mode), the node also requires a `MASK` input. The mask convention
matches the sd-scripts training/inference code:

* `white` (1.0) = inpaint area (the region the model should fill)
* `black` (0.0) = keep

The mask is resized to match the control image (nearest-neighbor),
binarized at 0.5, and concatenated with the RGB control image as a 4th
channel (rescaled to `[-1, +1]` to match the RGB range). When the weights
were trained with `--lllite_inpaint_masked_input`, the RGB channels are
also zeroed inside the mask region before concatenation — this is read
from the weights metadata and applied automatically.

3-channel weights continue to work with no mask input; if a mask is
connected to a 3-channel LLLite the node logs a warning and ignores it.
A 4-channel weight without a mask raises a `ValueError` rather than
silently substituting an empty mask.

### Cascading multiple LLLite nodes

Multiple **Apply Anima ControlNet-LLLite (sd-scripts)** nodes can be chained on the same
`MODEL` to apply more than one LLLite at the same time (e.g. a pose LLLite
and a depth LLLite, or several inpaint LLLites with different conditions).
Just feed the output `MODEL` of one node into the `model` input of the next.

ComfyUI's `model_options` has a single `model_function_wrapper` slot, so two
nodes that each install a wrapper would normally cause the outer one to
silently overwrite the inner one. To avoid this, each node captures the
previously-installed wrapper before cloning and delegates to it from inside
its own wrapper, so wrappers stack instead of replacing each other. This is
controlled by the `preserve_wrapper` toggle (default `True`); turn it off to
get the legacy single-wrapper behavior, or to deliberately replace an
upstream wrapper.

Notes when cascading:

* Each node has its own `start_percent` / `end_percent` window — when a
  node's window is inactive it is a pass-through to the next wrapper, so
  you can stage different LLLites across different parts of the schedule.
* `strength` is per-node and they add independently in the LLLite output;
  there is no global normalization, so very high combined strengths can
  over-saturate the same way a single high `strength` would.
* The pattern is compatible with other well-behaved wrapper-installing
  nodes (e.g. `ChromaRadianceOptions`) provided they follow the same
  delegate-to-previous-wrapper convention.

## How it works

* Discovers the LLLite target Linears under each Anima DiT block according to
  the `target_atomics` recorded in the weight metadata:
  * `self_attn_q_pre` → self-attention `q_proj`
  * `self_attn_kv_pre` → self-attention `k_proj` + `v_proj`
  * `cross_attn_q_pre` → cross-attention `q_proj`
  * `mlp_fc1_pre` → `GPT2FeedForward.layer1` (the MLP fc1)

  The LLM-Adapter sub-tree and `output_proj` are always skipped.
* On each sampling step the wrapper monkey-patches those Linears with the
  LLLite forward, runs `apply_model`, and restores the originals — so the
  patch never leaks across model clones.
* The control image is resized to `latent_HW * 8` once per resolution, then
  embedded by the v2 `conditioning1` trunk (Conv 4×4 s=4 → Conv 3×3 s=1 →
  Conv 4×4 s=4 with GroupNorm/SiLU, optional ResBlocks, optional ASPP, 1×1
  projection, LayerNorm) to a per-token feature map matching the DiT token
  grid. For 4-channel (inpaint) weights, the mask is resized with
  nearest-neighbor, binarized, rescaled to `[-1, +1]`, and concatenated as
  the 4th channel of the cond image before this trunk.
* Each LLLite module applies the v2 forward: `down → silu → concat with
  (cond + per-module depth-embedding) → mid → FiLM(γ, β) → silu → up`,
  added to the original Linear input.
* CFG is supported: `cond_emb` is broadcast to match the runtime batch size.

## Weight format (v2, named keys)

The state dict is in the v2 self-describing format. Each LLLite module's
weights are prefixed by the target Linear's full path, so the file alone
identifies which Linear each tensor belongs to:

```
lllite_conditioning1.{conv1,conv2,conv3,norm1,...,proj,out_norm,...}.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.down.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.mid.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.cond_to_film.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.up.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.depth_embed
...
```

The expected safetensors metadata keys are:

* `lllite.version` (= `"2"`)
* `lllite.cond_emb_dim`, `lllite.mlp_dim`
* `lllite.cond_dim`, `lllite.cond_resblocks`
* `lllite.use_aspp`, `lllite.aspp_dilations`
* `lllite.target_atomics` (canonical, comma-separated atomic specifiers) —
  falls back to `lllite.target_layers` for older v2 snapshots
* `lllite.cond_in_channels` (`"3"` standard / `"4"` inpaint; defaults to `3`)
* `lllite.inpaint_masked_input` (`"true"` / `"false"`; defaults to `false`,
  only meaningful when `cond_in_channels=4`)

**Legacy weights with `lllite_modules.{i}.*` keys (the pre-v2 format) are
rejected on load.** Re-train against the current sd-scripts codebase.

## Credits

* LLLite design and Anima training implementation: kohya-ss
* Adapted for ComfyUI from `sd-scripts`
* ComfyUI port written collaboratively with Claude (Anthropic, `claude-opus-4-7`)
