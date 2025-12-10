# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import os
import sys
import traceback
import tempfile
import json
import numpy as np
import torch
import soundfile as sf

from model_loader import load_asr_model, load_ser_model, PRETRAINED_DIRS
from inference import (
    EMOTION_MAP,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000


# ---------- helpers ----------

def _dummy_wave(duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Tiny sine wave; non-silent; exercises full pipeline deterministically."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    x = 0.01 * np.sin(2 * np.pi * 440.0 * t)
    return x.astype(np.float32)


def _write_temp_wav(x: np.ndarray, sr: int = SR) -> str:
    """Write a temporary wav file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, x, sr)
    return path


def _scan_checkpoint_bases(root: str = "models/checkpoints"):
    """
    Detect base names from folders that follow the '<base>_asr' / '<base>_ser' convention.
    Returns: { base: {"asr": path_or_None, "ser": path_or_None} }
    """
    bases = {}
    if not os.path.isdir(root):
        return bases

    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue

        if name.endswith("_asr"):
            base = name[:-4]
            bases.setdefault(base, {})["asr"] = full
        elif name.endswith("_ser"):
            base = name[:-4]
            bases.setdefault(base, {})["ser"] = full
        else:
            # Non-suffixed dirs are ignored here; they might be TAPT, etc.
            continue

    return bases


# ---------- ASR checks ----------

def check_asr_dir(path: str):
    print(f"  [ASR] Checking: {path}")
    try:
        model, processor = load_asr_model(path, device=DEVICE)
    except Exception as e:
        print(f"    ✖ load_asr_model FAILED: {e}")
        traceback.print_exc(limit=1)
        return

    if model is None or processor is None:
        print("    ✖ ASR model/processor is None after loading.")
        return

    # Introspect model class + PEFT presence
    cls_name = model.__class__.__name__
    inner = getattr(model, "model", None)
    inner_cls = inner.__class__.__name__ if inner is not None else None
    uses_peft = "peft" in (repr(type(inner)) + repr(type(model))).lower()

    print(f"    ✓ Loaded ASR model: {cls_name} (inner={inner_cls}, peft={uses_peft})")

    # Dummy forward
    wav = _dummy_wave()
    inputs = processor(
        wav,
        sampling_rate=SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    model.eval()
    with torch.no_grad():
        try:
            out = model(input_values, attention_mask=attention_mask)
        except Exception as e:
            print(f"    ✖ ASR forward() crashed: {e}")
            traceback.print_exc(limit=1)
            return

    logits = getattr(out, "logits", out)
    if not isinstance(logits, torch.Tensor):
        print(f"    ✖ forward() returned {type(logits)}, expected Tensor/logits.")
        return

    if logits.ndim != 3:
        print(f"    ✖ logits shape {tuple(logits.shape)} != [B, T, V] (CTC expected).")
        return

    vocab = len(processor.tokenizer)
    v_dim = logits.shape[-1]
    if v_dim != vocab:
        print(f"    ✖ vocab vs head mismatch: logits_dim={v_dim}, tokenizer={vocab}")
    else:
        print(f"    ✓ logits OK: shape={tuple(logits.shape)}, vocab={v_dim}")


# ---------- SER checks ----------

def check_ser_dir(path: str, n_emotions: int = None):
    print(f"  [SER] Checking: {path}")

    # Try to infer n_emotions from EMOTION_MAP if not given
    if n_emotions is None:
        try:
            n_emotions = int(max(EMOTION_MAP.keys())) + 1
        except Exception:
            n_emotions = 6

    try:
        model, feat_extractor = load_ser_model(
            path,
            n_emotions=n_emotions,
            device=DEVICE,
        )
    except Exception as e:
        print(f"    ✖ load_ser_model FAILED: {e}")
        traceback.print_exc(limit=1)
        return

    if model is None or feat_extractor is None:
        print("    ✖ SER model/feature_extractor is None after loading.")
        return

    cls_name = model.__class__.__name__
    uses_peft = "peft" in repr(type(model)).lower()
    print(f"    ✓ Loaded SER model: {cls_name} (peft={uses_peft}), n_emotions={n_emotions}")

    # Dummy forward
    wav = _dummy_wave()
    inputs = feat_extractor(
        [wav],
        sampling_rate=SR,
        return_tensors="pt",
        padding=True,
    )
    x = inputs.input_values.to(DEVICE)

    model.eval()
    with torch.no_grad():
        try:
            out = model(x)
        except Exception as e:
            print(f"    ✖ SER forward() crashed: {e}")
            traceback.print_exc(limit=1)
            return

    logits = getattr(out, "logits", out)
    if not isinstance(logits, torch.Tensor):
        print(f"    ✖ SER forward() returned {type(logits)}, expected Tensor/logits.")
        return

    # Allow [B, T, C] or [B, C], collapse time if needed
    if logits.ndim == 3:
        logits = logits.mean(dim=1)

    if logits.ndim != 2 or logits.shape[-1] != n_emotions:
        print(f"    ✖ SER logits shape {tuple(logits.shape)} incompatible with n_emotions={n_emotions}")
        return

    print(f"    ✓ SER logits OK: shape={tuple(logits.shape)}")


# ---------- global overview ----------

def print_backbone_info():
    print("=== PRETRAINED BACKBONES ===")
    if not PRETRAINED_DIRS:
        print("  (no PRETRAINED_DIRS configured)")
        return
    for lang, path in PRETRAINED_DIRS.items():
        abs_path = os.path.abspath(path)
        exists = os.path.isdir(abs_path)
        print(f"  {lang}: {abs_path} ({'OK' if exists else 'MISSING'})")


def run_sanity(root: str = "models/checkpoints"):
    print("##################################################")
    print("# SPEEMO STRUCTURE SANITY CHECK")
    print("##################################################")
    print(f"Device: {DEVICE}")
    print_backbone_info()

    bases = _scan_checkpoint_bases(root)
    if not bases:
        print(f"\nNo <base>_asr / <base>_ser pairs found in {root}.")
        print("If you are passing raw paths elsewhere, this script won’t see them.")
        return

    print(f"\n=== DETECTED CHECKPOINT BASES in '{root}' ===")
    for base, parts in bases.items():
        print(f"\n--- BASE: {base} ---")
        asr_path = parts.get("asr")
        ser_path = parts.get("ser")

        if asr_path:
            check_asr_dir(asr_path)
        else:
            print("  [ASR] ✖ No '<base>_asr' directory found.")

        if ser_path:
            check_ser_dir(ser_path)
        else:
            print("  [SER] ✖ No '<base>_ser' directory found.")


if __name__ == "__main__":
    root = "models/checkpoints"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    run_sanity(root)
