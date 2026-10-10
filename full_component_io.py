"""
persist FULL-trained components alongside a LoRA adapter.

For mixed methods (M4 = vision full + LM LoRA, M8 = LM full + vision LoRA, and the
connector which is "full" in M2/M3/M4/M6/M8), the fully-trained components live in
the BASE model (trained in place via requires_grad). A PEFT checkpoint saves ONLY
the LoRA adapter, so those full components are lost.

write the trainable, non-LoRA parameters to a side file (`non_lora_trainables.bin`) next to 
the adapter, and reload them after `merge_and_unload()`.
"""
from __future__ import annotations
import os
import logging
import torch

logger = logging.getLogger(__name__)

NON_LORA_FILE = "non_lora_trainables.bin"
_PEFT_PREFIX = "base_model.model."

def _strip_peft_prefix(name: str):
    """Map a PeftModel param name to the underlying base-model name.
    PeftModel wraps the base as `base_model.model.<base_name>`. After
    merge_and_unload() the reloaded model uses `<base_name>` directly, so we strip
    the prefix on save to keep keys compatible with the merged model.
    """
    return name[len(_PEFT_PREFIX):] if name.startswith(_PEFT_PREFIX) else name


def save_non_lora_trainables(model, out_dir: str):
    """
    Save every trainable parameter that is NOT a LoRA weight (i.e. the FULL
    components: connector, and vision/LM when full). We call this from the best/last
    checkpoint callback while the live trained model is in memory.
    Returns the number of tensors saved (0: nothing to save, e.g. pure-LoRA
    method with a frozen connector, which never happens with the presets).
    """
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" not in name:
            state[_strip_peft_prefix(name)] = p.detach().cpu()

    if not state:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, NON_LORA_FILE)
    torch.save(state, path)
    logger.info(f"Saved {len(state)} non-LoRA trainable tensors: {path}")
    return len(state)


def load_non_lora_trainables(model, best_path: str, *, strict_check: bool = True):
    """
    After PeftModel.from_pretrained(base, best_path).merge_and_unload(), overwrite
    the merged model's full components with the trained weights from the side file.
    Returns the number of tensors loaded. Raises if the file exists but nothing
    matched (a sign of a name-prefix mismatch), when strict_check=True.
    """
    path = os.path.join(best_path, NON_LORA_FILE)
    if not os.path.exists(path):
        logger.info(f"No {NON_LORA_FILE} at {best_path} — "
                    "full components come from the checkpoint itself (full method) ")
        return 0

    saved = torch.load(path, map_location="cpu")
    own = dict(model.named_parameters())
    loaded, skipped = 0, 0
    with torch.no_grad():
        for k, v in saved.items():
            tgt = own.get(k)
            if tgt is None or tgt.shape != v.shape:
                skipped += 1
                continue
            tgt.copy_(v.to(tgt.dtype).to(tgt.device))
            loaded += 1

    logger.info(f"Loaded {loaded} non-LoRA trainable tensors from {path} "
                f"({skipped} skipped).")
    if strict_check and loaded == 0:
        raise RuntimeError(
            f"{NON_LORA_FILE} found at {best_path} but 0 tensors matched the model "
            "(name-prefix mismatch?). Full-trained components were NOT restored — "
            "do not trust this run's metrics."
        )
    return loaded
