# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import os
import re
import json
import torch

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
)
from peft import PeftModel

from model import Wav2Vec2AsrModel, Wav2Vec2SerModel


# =====================================================================
# Root folders for pretrained backbones
# (populated via download_backbones.py)
# =====================================================================

ASR_PRETRAINED_DIRS = {
    "en": "models/pretrained/en/ASR",
    "de": "models/pretrained/de/ASR",
}

SER_PRETRAINED_DIRS = {
    "en": "models/pretrained/en/SER", 
    "de": "models/pretrained/de/SER",
}

# Backwards-compatible roots (for TAPT etc.)
PRETRAINED_DIRS = {
    "en": "models/pretrained/en",
    "de": "models/pretrained/de",
}


# =====================================================================
# Basic checks
# =====================================================================

def _ensure_asr_pretrained_backbone(lang: str) -> str:
    """Ensure ASR backbone exists and return its path."""
    if lang not in ASR_PRETRAINED_DIRS:
        raise ValueError(f"Unsupported language code '{lang}' for ASR backbone.")
    outdir = os.path.abspath(ASR_PRETRAINED_DIRS[lang])
    if not os.path.isdir(outdir) or not os.listdir(outdir):
        raise FileNotFoundError(
            f"ASR pretrained backbone directory missing or empty: {outdir}. "
            f"Run download_backbones.py first."
        )
    return outdir


def _ensure_ser_pretrained_backbone(lang: str) -> str:
    """Ensure SER backbone exists and return its path."""
    if lang not in SER_PRETRAINED_DIRS:
        raise ValueError(f"Unsupported language code '{lang}' for SER backbone.")
    outdir = os.path.abspath(SER_PRETRAINED_DIRS[lang])
    if not os.path.isdir(outdir) or not os.listdir(outdir):
        raise FileNotFoundError(
            f"SER pretrained backbone directory missing or empty: {outdir}. "
            f"Run download_backbones.py first."
        )
    return outdir


def _is_full_hf_model_dir(path: str) -> bool:
    """
    True if 'path' looks like a full HF model directory
    (config + base weights and/or preprocessor).
    """
    if not os.path.isdir(path):
        return False

    has_hf_weights = (
        os.path.isfile(os.path.join(path, "pytorch_model.bin")) or
        os.path.isfile(os.path.join(path, "model.safetensors"))
    )
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_preproc = os.path.isfile(os.path.join(path, "preprocessor_config.json"))

    # Valid full HF model dir
    if has_config and has_hf_weights:
        return True

    # HF-ish root when we have processor + actual weights
    if has_preproc and has_hf_weights:
        return True

    return False


def _is_bad_lora_only_dir(path: str) -> bool:
    """
    "Broken" LoRA dir: adapter_config.json present but
    NO base weights and NO adapter_model.*.
    """
    if not os.path.isdir(path):
        return False

    has_adapter = os.path.isfile(os.path.join(path, "adapter_config.json"))
    has_hf_weights = (
        os.path.isfile(os.path.join(path, "pytorch_model.bin")) or
        os.path.isfile(os.path.join(path, "model.safetensors"))
    )
    has_lora_weights = (
        os.path.isfile(os.path.join(path, "adapter_model.bin")) or
        os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
    )

    return has_adapter and (not has_hf_weights) and (not has_lora_weights)


def _resolve_nested_ckpt(model_dir: str) -> str:
    """
    If model_dir contains checkpoint-* subdirs that are *full* HF base-model dirs,
    pick the one with the largest step; else return model_dir unchanged.
    """
    if not os.path.isdir(model_dir):
        return model_dir

    best = None
    try:
        entries = os.listdir(model_dir)
    except OSError:
        return model_dir

    for d in entries:
        m = re.match(r"^checkpoint-(\d+)$", d)
        if not m:
            continue
        full = os.path.join(model_dir, d)
        if not _is_full_hf_model_dir(full):
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, full)

    return best[1] if best else model_dir


# =====================================================================
# NEW: ASR dir pickers (root → merged → nested)
# =====================================================================

def _has_full_model(dirpath: str) -> bool:
    return any(os.path.isfile(os.path.join(dirpath, n))
               for n in ("model.safetensors", "pytorch_model.bin"))


def _has_processor(dirpath: str) -> bool:
    # tokenizer + feature extractor (Processor)
    return any(os.path.isfile(os.path.join(dirpath, n))
               for n in ("preprocessor_config.json", "tokenizer_config.json"))


def _has_adapters(dirpath: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, "adapter_config.json")) and \
           any(os.path.isfile(os.path.join(dirpath, n)) for n in
               ("adapter_model.safetensors", "adapter_model.bin"))


def _pick_asr_dir(run_dir: str) -> str:
    """
    Prefer a serving-ready artifact:
    1) <run>/merged/ if it has full model + processor
    2) run root if it has processor and (full model OR adapters)
    3) else newest checkpoint-*
    """
    if not os.path.isdir(run_dir):
        return run_dir

    merged = os.path.join(run_dir, "merged")
    if os.path.isdir(merged) and _has_full_model(merged) and _has_processor(merged):
        return merged

    if _has_processor(run_dir) and (_has_full_model(run_dir) or _has_adapters(run_dir)):
        return run_dir

    # fall back to newest checkpoint-*
    ckpts = [os.path.join(run_dir, d) for d in os.listdir(run_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))]
    try:
        ckpts.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]), reverse=True)
    except Exception:
        pass
    return ckpts[0] if ckpts else run_dir


# =====================================================================
# Internal helpers: construct wrappers
# =====================================================================

def _build_asr_wrapper_from_dir(model_dir: str, device: str):
    """
    Load a Wav2Vec2ForCTC + Wav2Vec2Processor from model_dir
    and wrap in Wav2Vec2AsrModel.
    """
    base = Wav2Vec2ForCTC.from_pretrained(model_dir)
    proc = Wav2Vec2Processor.from_pretrained(model_dir)
    base.to(device)
    wrapper = Wav2Vec2AsrModel(base, proc)
    return wrapper.to(device), proc


def _build_ser_model_from_dir(model_dir: str, n_emotions: int, device: str, dropout: float = 0.2):
    """
    Load a Wav2Vec2Model + feature extractor and wrap in Wav2Vec2SerModel.
    """
    backbone = Wav2Vec2Model.from_pretrained(model_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    
    model = Wav2Vec2SerModel(
        pretrained_ckpt=model_dir,
        n_emotions=n_emotions,
        device=device,
        dropout=dropout,
    )
    # Replace the backbone with the loaded one
    model.backbone = backbone.to(device)
    
    # Load SER head if it exists
    head_path = os.path.join(model_dir, "ser_head.pt")
    if os.path.isfile(head_path):
        state = torch.load(head_path, map_location=device)
        model.head.load_state_dict(state)
    
    return model.to(device), feature_extractor


# =====================================================================
# LoRA helpers
# =====================================================================

def _load_lora_asr_from_dir(
    adapter_dir: str,
    lang: str,
    device: str,
    merge_peft_on_load: bool,
):
    """
    Load ASR LoRA adapter from `adapter_dir` on top of an ASR backbone.
    Returns: (Wav2Vec2AsrModel, Wav2Vec2Processor) or None
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None

    has_adapter_weights = (
        os.path.isfile(os.path.join(adapter_dir, "adapter_model.bin")) or
        os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors"))
    )
    if not has_adapter_weights:
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        try:
            base_name = _ensure_asr_pretrained_backbone(lang)
            print(
                f"[load_asr_model] LoRA adapter in {adapter_dir} has no base_model_name_or_path; "
                f"assuming ASR backbone {base_name!r} for lang={lang!r}."
            )
        except Exception as e:
            print(
                f"[load_asr_model] LoRA adapter in {adapter_dir} missing base and cannot "
                f"guess backbone: {e}"
            )
            return None

    # Load base model and apply LoRA
    base_model = Wav2Vec2ForCTC.from_pretrained(base_name)
    base_model.to(device)
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    if merge_peft_on_load and hasattr(peft_model, "merge_and_unload"):
        peft_model = peft_model.merge_and_unload()

    # Wrap in our ASR model class
    wrapper = Wav2Vec2AsrModel(peft_model, Wav2Vec2Processor.from_pretrained(base_name))

    # Prefer adapter-specific processor if present
    try:
        processor = Wav2Vec2Processor.from_pretrained(adapter_dir)
    except Exception:
        processor = Wav2Vec2Processor.from_pretrained(base_name)

    # Sanity check
    try:
        wrapper.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 16000, device=device)
            out = wrapper(input_values=dummy)
        logits = getattr(out, "logits", out)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            print(
                "[load_asr_model] LoRA ASR adapter produced invalid output; "
                "treating as incompatible."
            )
            return None
    except Exception as e:
        print(
            f"[load_asr_model] LoRA ASR sanity check failed for {adapter_dir}: {e}"
        )
        return None

    return wrapper.to(device), processor


def _load_lora_ser_from_dir(
    adapter_dir: str,
    lang: str,
    n_emotions: int,
    device: str,
    dropout: float,
    merge_peft_on_load: bool,
):
    """
    Load SER LoRA adapter from `adapter_dir` on top of a SER backbone.
    Returns: (nn.Module, Wav2Vec2FeatureExtractor) or None
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None

    has_adapter_weights = (
        os.path.isfile(os.path.join(adapter_dir, "adapter_model.bin")) or
        os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors"))
    )
    if not has_adapter_weights:
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        try:
            base_name = _ensure_ser_pretrained_backbone(lang)
            print(
                f"[load_ser_model] LoRA adapter in {adapter_dir} has no base_model_name_or_path; "
                f"assuming SER backbone {base_name!r} for lang={lang!r}."
            )
        except Exception as e:
            print(
                f"[load_ser_model] LoRA adapter in {adapter_dir} missing base and cannot "
                f"guess backbone: {e}"
            )
            return None

    if not os.path.isdir(base_name):
        print(
            f"[load_ser_model] LoRA adapter in {adapter_dir} references invalid backbone: "
            f"{base_name!r}"
        )
        return None

    # Load base SER model
    base_model, base_extractor = _build_ser_model_from_dir(
        base_name, n_emotions, device, dropout
    )

    # Apply LoRA
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    if merge_peft_on_load and hasattr(peft_model, "merge_and_unload"):
        peft_model = peft_model.merge_and_unload()

    # Get feature extractor
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(adapter_dir)
    except Exception:
        extractor = base_extractor

    # Sanity check
    try:
        peft_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 16000, device=device)
            out = peft_model(input_values=dummy)
        logits = getattr(out, "logits", out)
        if not isinstance(logits, torch.Tensor):
            print(
                "[load_ser_model] LoRA SER adapter produced non-tensor output; "
                "treating as incompatible."
            )
            return None
    except Exception as e:
        print(
            f"[load_ser_model] LoRA SER adapter sanity check failed for {adapter_dir}: {e}"
        )
        return None

    return peft_model.to(device), extractor


# =====================================================================
# ASR LOADER
# =====================================================================

def _sanity_check_asr_logits(model, device: str, context: str):
    """
    Ensure a loaded ASR model returns logits [B, T, V] for CTC.
    """
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 16000, device=device)
            out = model(input_values=dummy)
        logits = getattr(out, "logits", out)

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"{context}: forward() returned {type(out)}, "
                f"which has no .logits tensor. Expected CTC logits from Wav2Vec2ForCTC."
            )

        if logits.ndim != 3:
            raise ValueError(
                f"{context}: logits shape {tuple(logits.shape)} "
                f"!= [B, T, V] (CTC)."
            )
    except Exception as e:
        raise RuntimeError(
            f"[load_asr_model] Sanity check failed for {context}: {e}"
        ) from e


def load_asr_model(
    model_dir: str,
    device: str = "cpu",
    use_fp16: bool = False,
    lang: str = "en",
    merge_peft_on_load: bool = False,
):
    """
    Loads a Wav2Vec2 CTC ASR model as Wav2Vec2AsrModel.

    Supports:
      - '.pt'       : legacy state_dict on top of ASR backbone (lang)
      - 'en' / 'de' : ASR_PRETRAINED_DIRS[lang] 
      - HF folder   : fine-tuned model dir (optionally with nested checkpoint-*)
      - LoRA folder : adapter_config + adapter_model.* dirs.
    """
    dtype = torch.float16 if use_fp16 and str(device).startswith("cuda") else torch.float32

    # 1) legacy .pt checkpoint (wrapper state_dict)
    if model_dir.endswith(".pt"):
        ckpt_path = model_dir
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ASR checkpoint not found: {ckpt_path}")
        backbone_dir = _ensure_asr_pretrained_backbone(lang)
        model, processor = _build_asr_wrapper_from_dir(backbone_dir, device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        _sanity_check_asr_logits(model, device, f"legacy .pt ({ckpt_path})")
        return model.to(device).to(dtype), processor

    # 2) language shortcut → pretrained backbone dir
    if model_dir in ("en", "de"):
        model_dir = _ensure_asr_pretrained_backbone(model_dir)

    # 3) choose best directory: prefer merged/ then root then nested checkpoint-*
    chosen_dir = _pick_asr_dir(model_dir) if os.path.isdir(model_dir) else model_dir
    print(f"[ASR/load] chosen dir: {chosen_dir}")

    # 4) LoRA adapter dir
    if os.path.isdir(chosen_dir) and os.path.isfile(os.path.join(chosen_dir, "adapter_config.json")):
        loaded = _load_lora_asr_from_dir(
            adapter_dir=chosen_dir,
            lang=lang,
            device=device,
            merge_peft_on_load=merge_peft_on_load,
        )
        if loaded is not None:
            model, processor = loaded
            _sanity_check_asr_logits(model, device, f"LoRA adapter dir ({chosen_dir})")
            return model.to(device).to(dtype), processor

    # 5) Plain HF-style CTC model dir
    if os.path.isdir(chosen_dir):
        if _is_bad_lora_only_dir(chosen_dir):
            raise FileNotFoundError(
                f"ASR: {chosen_dir} looks like a broken LoRA-only directory "
                f"without usable base weights."
            )
        model, processor = _build_asr_wrapper_from_dir(chosen_dir, device)
        _sanity_check_asr_logits(model, device, f"HF dir ({chosen_dir})")
        return model.to(device).to(dtype), processor

    raise FileNotFoundError(f"ASR model directory not found: {chosen_dir}")


# =====================================================================
# SER LOADER  
# =====================================================================

def load_ser_model(
    model_dir: str,
    n_emotions: int,
    device: str = "cpu",
    use_fp16: bool = False,
    dropout: float = 0.2,
    lang: str = "en",
    merge_peft_on_load: bool = False,
):
    """
    Loads a Wav2Vec2-based SER model.

    Supports:
      - '.pt'       : legacy SER head on top of SER backbone
      - 'en' / 'de' : SER_PRETRAINED_DIRS[lang] as backbone
      - HF folder   : Wav2Vec2SerModel.save_pretrained-style folder
      - LoRA folder : adapter_config + adapter_model.* dirs.
    """
    dtype = torch.float16 if use_fp16 and str(device).startswith("cuda") else torch.float32

    # 1) legacy .pt (state_dict of Wav2Vec2SerModel)
    if model_dir.endswith(".pt"):
        ckpt_path = model_dir
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SER checkpoint not found: {ckpt_path}")
        backbone_dir = _ensure_ser_pretrained_backbone(lang)
        model, extractor = _build_ser_model_from_dir(
            backbone_dir, n_emotions, device, dropout
        )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        return model.to(device).to(dtype), extractor

    # 2) language shortcut
    if model_dir in ("en", "de"):
        model_dir = _ensure_ser_pretrained_backbone(model_dir)

    # 3) resolve nested checkpoint-* dirs
    model_dir = _resolve_nested_ckpt(model_dir)

    # 4) LoRA SER dir
    if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "adapter_config.json")):
        loaded = _load_lora_ser_from_dir(
            adapter_dir=model_dir,
            lang=lang,
            n_emotions=n_emotions,
            device=device,
            dropout=dropout,
            merge_peft_on_load=merge_peft_on_load,
        )
        if loaded is not None:
            return loaded

    # 5) Plain fine-tuned SER folder
    if os.path.isdir(model_dir):
        if _is_bad_lora_only_dir(model_dir):
            raise FileNotFoundError(
                f"SER: {model_dir} looks like a broken adapter-only directory "
                f"without usable weights."
            )
        model, extractor = _build_ser_model_from_dir(
            model_dir, n_emotions, device, dropout
        )
        return model.to(device).to(dtype), extractor

    raise FileNotFoundError(f"SER model directory not found: {model_dir}")


# =====================================================================
# TAPT BACKBONE
# =====================================================================

def load_tapt_backbone(
    pretrained_ckpt: str,
    device: str = "cpu",
    use_fp16: bool = False,
    lang: str = "en",
):
    """
    For TAPT pretraining utilities: accepts 'en'/'de' or explicit directory.
    Uses PRETRAINED_DIRS as roots (models/pretrained/<lang>).
    """
    dtype = torch.float16 if use_fp16 and str(device).startswith("cuda") else torch.float32

    if pretrained_ckpt in ("en", "de"):
        root = PRETRAINED_DIRS.get(pretrained_ckpt)
        if root is None:
            raise FileNotFoundError(f"No PRETRAINED_DIRS entry for {pretrained_ckpt}")
        pretrained_ckpt = root

    if not os.path.isdir(pretrained_ckpt):
        raise FileNotFoundError(f"TAPT backbone directory not found: {pretrained_ckpt}")

    return (
        Wav2Vec2ForPreTraining.from_pretrained(pretrained_ckpt, torch_dtype=dtype)
        .to(device)
    )


__all__ = [
    "load_asr_model",
    "load_ser_model",
    "load_tapt_backbone",
]
