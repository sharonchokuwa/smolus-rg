import logging
from typing import Optional 
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

# Map config architecture names to HF model IDs
ARCHITECTURE_TO_HF_ID = {
    "SmolVLM-256M":  "HuggingFaceTB/SmolVLM-256M-Instruct",
    "SmolVLM-500M":  "HuggingFaceTB/SmolVLM-500M-Instruct",
    "SmolVLM-2.2B":  "HuggingFaceTB/SmolVLM-Instruct",
}

def _expected_params_from_size(model_size: str):
    """
    Parse a model_size string ("256M" / "500M" / "2.2B") into an approximate
    expected parameter count. Returns None if it can't be parsed.
    """
    if not model_size:
        return None
    s = str(model_size).strip().upper().replace("SMOLVLM-", "")
    mult = {"M": 1e6, "B": 1e9}.get(s[-1:], None)
    if mult is None:
        return None
    try:
        return float(s[:-1]) * mult
    except ValueError:
        return None
 
 
def verify_model_identity(model, cfg: dict):
    """
    check the live parameter count against the declared model_size.
    """
    model_size = cfg.get("model_size", "")
    strategy   = cfg["model"]["init_strategy"]
    checkpoint = cfg["model"]["checkpoints"][strategy]
    exp_name   = cfg.get("experiment", {}).get("name", "?")
    actual     = sum(p.numel() for p in model.parameters())
    expected   = _expected_params_from_size(model_size)
 
    banner = (
        "\n" + "=" * 64 +
        f"\n  MODEL IDENTITY"
        f"\n    experiment   : {exp_name}"
        f"\n    model_size   : {model_size}"
        f"\n    init_strategy: {strategy}"
        f"\n    checkpoint   : {checkpoint}"
        f"\n    actual params: {actual:,}"
    )
    if expected is not None:
        ratio = actual / expected
        banner += f"\n    expected ~   : {expected:,.0f}  (ratio {ratio:.2f})"
    banner += "\n" + "=" * 64
    logger.info(banner)
    print(banner)  
 
    if expected is not None and not (0.6 <= actual / expected <= 1.5):
        msg = (
            f"MODEL SIZE MISMATCH: declared model_size='{model_size}' "
            f"(~{expected:,.0f} params) but the loaded model from '{checkpoint}' "
            f"has {actual:,} params (ratio {actual/expected:.2f}). "
        )
        logger.warning(msg)
        print("WARNING: " + msg)

def _get_torch_dtype(precision: str):
    return {
        "bf16":  torch.bfloat16,
        # "fp16":  torch.float16,
        "fp16":  torch.float32,    
        "fp32":  torch.float32,
    }.get(precision, torch.bfloat16)


def _get_bnb_config(peft_cfg: dict, precision: str):
    """
    Build BitsAndBytes 4-bit quantisation config for QLoRA.
    """
    qlora = peft_cfg["qlora"]

    compute_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(precision, torch.float16)   # default to float16 for QLoRA safety

    return BitsAndBytesConfig(
        load_in_4bit=qlora["load_in_4bit"],
        bnb_4bit_quant_type=qlora["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora["bnb_4bit_use_double_quant"],
    )

def load_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    return processor

def load_base_model(
    model_id: str,
    init_strategy: str,
    precision: str,
    peft_cfg: dict,
    seed: int = 0,
):
    """
    Load the base VLM with the specified precision.
    For "random" strategy: loads the architecture from HF but
    immediately re-initialises all weights from scratch.
    """
    dtype = _get_torch_dtype(precision)
    peft_mode = peft_cfg.get("mode", "lora")

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        _attn_implementation="eager",
    )

    if peft_mode == "qlora":
        load_kwargs["quantization_config"] = _get_bnb_config(peft_cfg, precision)
        logger.info("Loading model in 4-bit (QLoRA) mode")
    else:
        logger.info(f"Loading model in {precision} mode")

    if init_strategy == "random":
        from transformers import AutoConfig
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_config(
            config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        logger.warning(
            f"Random init: built architecture from config with seed={seed}. "
            f"NO pretrained weights loaded (vision + connector + LM all random)."
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    model.config.use_cache = False 
    logger.info(f"Model loaded: {model_id} | params: {model.num_parameters():,}")
    return model

# 9 FINETUNING METHODS (M1–M9)  —  per-component trainability control 
# (vision, language, connector) for each method key
METHOD_PRESETS = {
    "M1": {"vision": "full",   "language": "full",   "connector": "full"},
    "M2": {"vision": "lora",   "language": "lora",   "connector": "full"},
    "M3": {"vision": "full",   "language": "lora",   "connector": "full"},
    "M4": {"vision": "frozen", "language": "lora",   "connector": "full"},
    "M5": {"vision": "lora",   "language": "frozen", "connector": "full"},
    "M6": {"vision": "full",   "language": "frozen", "connector": "full"},
    "M7": {"vision": "frozen", "language": "full",   "connector": "full"},
    "M8": {"vision": "lora",   "language": "full",   "connector": "full"},
    "M9": {"vision": "frozen", "language": "frozen", "connector": "frozen"},
}
 
# Module-name substrings that identify each component inside SmolVLM.
_COMPONENT_PATHS = {
    "vision":    ["vision_model"],
    "connector": ["connector"],
    "language":  ["text_model", "lm_head"],
}
 
# Default LoRA target suffixes per component (scoped by path, see _lora_target_regex).
_DEFAULT_LORA_TARGETS = {
    "vision":    ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "language":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "connector": ["proj"],
}
 
def _resolve_components(peft_cfg: dict):
    """
    Decide the per-component strategy from the config, in priority order:
      1) peft.components: {vision, language, connector}  (explicit, wins)
      2) peft.method: "M1".."M9"                         (preset shorthand)
    Returns {"vision":..., "language":..., "connector":...} each in
    {"full","lora","frozen"}. connector defaults to "full" when unspecified.
    """
    if peft_cfg.get("components"):
        comp = dict(peft_cfg["components"])
    elif peft_cfg.get("method"):
        key = str(peft_cfg["method"]).upper()
        if key not in METHOD_PRESETS:
            raise ValueError(f"Unknown finetuning method '{key}'. "
                             f"Valid: {sorted(METHOD_PRESETS)}")
        comp = dict(METHOD_PRESETS[key])
    else:
        raise ValueError("configure_finetuning called without peft.method or peft.components")
 
    comp.setdefault("connector", "full")
    for part, strat in comp.items():
        if strat not in ("full", "lora", "frozen"):
            raise ValueError(f"Component '{part}' has invalid strategy '{strat}' "
                             f"(expected full|lora|frozen)")
    return comp
 
def _lora_target_regex(components: dict, lora_cfg: dict):
    overrides = {
        "vision":    lora_cfg.get("vision_target_modules"),
        "language":  lora_cfg.get("language_target_modules"),
        "connector": lora_cfg.get("connector_target_modules"),
    }
    parts = []
    for part in ("vision", "language", "connector"):
        if components.get(part) != "lora":
            continue
        suffixes = overrides[part] or _DEFAULT_LORA_TARGETS[part]
        path = _COMPONENT_PATHS[part][0]          # vision_model / text_model / connector
        parts.append(rf".*{path}.*\.(?:{'|'.join(suffixes)})")
    if not parts:
        return None
    return "|".join(f"(?:{p})" for p in parts)
 
def _set_component_requires_grad(model, substrings: list, trainable: bool):
    n = 0
    for name, p in model.named_parameters():
        if any(s in name for s in substrings):
            p.requires_grad = trainable
            n += 1
    return n
 
def configure_finetuning(model, peft_cfg: dict):
    components = _resolve_components(peft_cfg)
    lora_cfg   = peft_cfg.get("lora", {})
    logger.info(f"Finetuning components: {components}")
 
    any_lora = any(v == "lora" for v in components.values())
    lora_config = None
 
    if any_lora:
        target_regex = _lora_target_regex(components, lora_cfg)
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg.get("bias", "none"),
            target_modules=target_regex,                
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            use_dora=lora_cfg.get("use_dora", False),
            init_lora_weights=lora_cfg.get("init_lora_weights", "gaussian"),
        )
 
        # QLoRA: if the base was loaded in 4-bit, prep before attaching adapters.
        if getattr(model, "is_loaded_in_4bit", False) or peft_cfg.get("mode") == "qlora":
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for 4-bit training (QLoRA).")
 
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, lora_config)       # freezes all base weights
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()         
 
        # PEFT froze every base weight; re-enable grads on FULL components.
        for part, strat in components.items():
            if strat == "full":
                touched = _set_component_requires_grad(model, _COMPONENT_PATHS[part], True)
                logger.info(f"  {part}: FULL — re-enabled grad on {touched} base tensors")
            elif strat == "frozen":
                _set_component_requires_grad(model, _COMPONENT_PATHS[part], False)
                logger.info(f"  {part}: FROZEN")
            else:  # lora
                logger.info(f"  {part}: LoRA adapter")
        model.print_trainable_parameters()
    else:
        # No LoRA anywhere — full/frozen only. Set grads directly.
        for part, strat in components.items():
            trainable = (strat == "full")
            n = _set_component_requires_grad(model, _COMPONENT_PATHS[part], trainable)
            logger.info(f"  {part}: {'FULL (trainable)' if trainable else 'FROZEN'} "
                        f"({n} tensors)")
        if any(v == "full" for v in components.values()):
            model.gradient_checkpointing_enable()
            if hasattr(model, "model") and hasattr(model.model, "vision_model") \
               and components.get("vision") == "full":
                model.model.vision_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on vision encoder.")
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100*trainable/max(total,1):.3f}%)")
    if trainable == 0:
        logger.warning(
            "ZERO trainable parameters — this configuration trains nothing "
            "(e.g. M9 with a frozen connector). This is zero-shot/eval. "
        )
    return model, lora_config


def apply_peft(model, peft_cfg: dict):
    if peft_cfg.get("method") or peft_cfg.get("components"):
        return configure_finetuning(model, peft_cfg)
    return _apply_peft_legacy(model, peft_cfg)
 
 
def _apply_peft_legacy(model, peft_cfg: dict):
    mode = peft_cfg.get("mode", "lora")
 
    if mode == "full":
        logger.info("Full finetuning mode — no PEFT adapter applied.")
        model.gradient_checkpointing_enable()

        if hasattr(model, "model") and hasattr(model.model, "vision_model"):
            model.model.vision_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on vision encoder.")
 
        return model, None
 
    lora_cfg = peft_cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        use_dora=lora_cfg.get("use_dora", False),
        init_lora_weights=lora_cfg.get("init_lora_weights", "lora"),
    )
 
    if mode == "qlora":
        # Prepare model for k-bit training (freezes certain layers, casts norms)
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for 4-bit training (QLoRA)")
 
    if mode == "lora":
        model.gradient_checkpointing_enable()
 
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, lora_config


def load_model_and_processor(cfg: dict,  apply_adapter: bool = True):
    model_cfg  = cfg["model"]
    peft_cfg   = cfg["peft"]
    train_cfg  = cfg["training"]

    strategy     = model_cfg["init_strategy"]
    model_id     = model_cfg["checkpoints"][strategy]
    precision    = train_cfg["precision"]

    # BIOMEDICA repo has no processor files — always load processor from HF base
    if cfg["model_size"] == '2.2B':
        processor_id = 'HuggingFaceTB/SmolVLM-Instruct'
    else:
        processor_id = model_cfg["checkpoints"]["huggingface"]

    logger.info(
        f"Initialisation strategy: '{strategy}' | "
        f"model: {model_id} | processor: {processor_id}"
    )

    processor = load_processor(processor_id)       # always from HuggingFace
    model = load_base_model(
        model_id, strategy, precision, peft_cfg,
        seed=cfg["experiment"].get("seed", 0),
    )

    # confirm which model loaded and check size vs model_size.      
    # guard against silently running the same model twice.             
    verify_model_identity(model, cfg)                                              
 
    tok = processor.tokenizer
    eos_token = tok.eos_token           
    eos_id    = tok.eos_token_id
    needs_pad = (tok.pad_token is None) or (tok.pad_token_id == eos_id)

    if needs_pad:
        # Prefer an existing reserved/unused special token if present; otherwise add one.
        tok.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tok))
        logger.info(
            f"pad_token was missing or aliased EOS — added dedicated <pad> "
            f"(id={tok.pad_token_id}); embeddings resized to {len(tok)}."
        )
    else:
        logger.info(
            f"pad_token='{tok.pad_token}' (id={tok.pad_token_id}) distinct from "
            f"eos='{eos_token}' (id={eos_id}) — no tokenizer change needed."
        )

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = eos_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = eos_id
    logger.info(
        f"Stop config pinned | eos_token_id={eos_id} ({eos_token}) | "
        f"pad_token_id={tok.pad_token_id}"
    )

    model, lora_config = (apply_peft(model, peft_cfg) if apply_adapter else (model, None))
    return model, processor, lora_config

def load_base_only(cfg):
    """Base model + processor with the correct init and tokenizer resize, but NO PEFT adapter."""
    return load_model_and_processor(cfg, apply_adapter=False)