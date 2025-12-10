import os
import re
import json
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from peft import PeftModel, PeftConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

# ---- Compat shim for PEFT configs: drop unknown **kwargs on older PEFT versions ----
try:
    import inspect
    try:
        from peft import LoraConfig
    except Exception:
        LoraConfig = None

    def _patch_trim_unknown_kwargs(cfg_cls):
        """
        On newer PEFT (HPC), cfg_cls.__init__ may accept extra kwargs like
        'corda_config', 'eva_config', etc.

        On older PEFT (local), these kwargs are unknown and cause:
            __init__() got an unexpected keyword argument '...'

        This wrapper inspects the __init__ signature and silently drops any
        kwargs that are not in the parameter list. If __init__ already
        has a **kwargs parameter, we don't touch it.
        """
        if cfg_cls is None:
            return
        try:
            sig = inspect.signature(cfg_cls.__init__)
        except (TypeError, ValueError):
            return

        params = sig.parameters
        # If there is already a **kwargs parameter, it can accept extra keys -> no patch needed
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return

        allowed = set(params.keys())  # e.g. {'self', 'r', 'lora_alpha', ...}
        orig_init = cfg_cls.__init__

        def _wrapped_init(self, *args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return orig_init(self, *args, **filtered)

        cfg_cls.__init__ = _wrapped_init

    for _cfg in (PeftConfig, LoraConfig):
        _patch_trim_unknown_kwargs(_cfg)

except Exception:
    # If anything goes wrong, do not break import; just skip compat shim.
    pass

from model import Wav2Vec2AsrModel, Wav2Vec2SerModel

# Local folders for pretrained backbones (populated manually)
PRETRAINED_DIRS = {
    "en": "models/pretrained/en",
    "de": "models/pretrained/de"
}


def _ensure_pretrained_backbone(lang: str) -> str:
    """
    Verifies that models/pretrained/{lang} exists and is non-empty.
    Returns that path or raises an error if missing.
    """
    if lang not in PRETRAINED_DIRS:
        raise ValueError(f"Unsupported language code '{lang}' for pretrained backbone.")
    outdir = os.path.abspath(PRETRAINED_DIRS[lang])
    if not os.path.isdir(outdir) or not os.listdir(outdir):
        raise FileNotFoundError(
            f"Pretrained backbone directory missing or empty: {outdir}."
            " Please download the model files into this folder."
        )
    return outdir


def _resolve_nested_ckpt(model_dir: str) -> str:
    """
    Recursively searches under model_dir for any 'checkpoint-<num>' folders
    and returns the one with the highest numeric suffix. If none are found,
    returns model_dir unchanged.
    """
    if not os.path.isdir(model_dir):
        return model_dir

    best = None  # tuple (checkpoint_number, full_path)
    # walk through all subdirectories
    for root, dirs, _ in os.walk(model_dir):
        for d in dirs:
            m = re.match(r"^checkpoint-(\d+)$", d)
            if m:
                num = int(m.group(1))
                full = os.path.join(root, d)
                if best is None or num > best[0]:
                    best = (num, full)
    if best:
        return best[1]
    return model_dir


def _safe_read_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_hf_wav2vec2_ctc_folder(path: str) -> bool:
    """
    Heuristic: folder has config.json with architectures incl. Wav2Vec2ForCTC,
    or at least looks like a HF Wav2Vec2 checkpoint.
    """
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    cfg = _safe_read_json(cfg_path)
    arch = cfg.get("architectures") or []
    model_type = cfg.get("model_type")
    return ("Wav2Vec2ForCTC" in arch) or (model_type == "wav2vec2")


def _is_hf_wav2vec2_ser_folder(path: str) -> bool:
    """
    Heuristic: folder has config.json with architectures incl. Wav2Vec2ForSequenceClassification.
    """
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    cfg = _safe_read_json(cfg_path)
    arch = cfg.get("architectures") or []
    return "Wav2Vec2ForSequenceClassification" in arch


def _has_peft_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_processor_with_fallback(primary_dir: str, fallback_dir: str) -> Wav2Vec2Processor:
    """
    Try to load Wav2Vec2Processor from primary (e.g., adapter dir),
    and fall back to base directory if the primary doesn't have processor files.
    """
    try:
        return Wav2Vec2Processor.from_pretrained(primary_dir)
    except Exception:
        return Wav2Vec2Processor.from_pretrained(fallback_dir)


def _load_extractor_with_fallback(primary_dir: str, fallback_dir: str) -> Wav2Vec2FeatureExtractor:
    """
    Try to load Wav2Vec2FeatureExtractor from primary (e.g., adapter dir),
    and fall back to base directory if the primary doesn't have extractor files.
    """
    try:
        return Wav2Vec2FeatureExtractor.from_pretrained(primary_dir)
    except Exception:
        return Wav2Vec2FeatureExtractor.from_pretrained(fallback_dir)


def _load_adapter_state_dict(adapter_dir: str):
    """Load adapter state dict from safetensors or bin (best-effort)."""
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(st_path) and _safe_load_file is not None:
        return _safe_load_file(st_path)
    st_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.isfile(st_path):
        return torch.load(st_path, map_location="cpu")
    return {}


def load_neural_lm(model_name: str, device: str):
    """
    Load a causal-LM (e.g. GPT-2) for shallow fusion rescoring.
    Returns (model, tokenizer) on `device`.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer



def load_asr_model(
    model_dir: str,
    device: str = "cpu",
    use_fp16: bool = False,
    lang: str = "en"
):
    """
    Loads a Wav2Vec2 CTC ASR model. Supports:
      - 'en'/'de' → offline pretrained backbone folder under PRETRAINED_DIRS
      - directory folder → HF-style from_pretrained folder, including nested checkpoints
      - LoRA adapter folder (adapter_config.json + adapter_model.safetensors) → loads base + adapter
      - .pt checkpoint → load state_dict into fresh backbone model
    """
    dtype = torch.float16 if use_fp16 and device.startswith("cuda") else torch.float32

    # 1) .pt checkpoint file? load backbone then state_dict
    if model_dir.endswith(".pt"):
        ckpt_path = model_dir
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ASR checkpoint not found: {ckpt_path}")
        backbone_dir = _ensure_pretrained_backbone(lang)
        model, processor = Wav2Vec2AsrModel.from_pretrained(
            backbone_dir,
            device=device,
            mask_time_prob=0.05,
            mask_feature_prob=0.065,
        )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        return model.to(device).to(dtype).eval(), processor

    # 2) language code → pretrained folder
    if model_dir in PRETRAINED_DIRS:
        model_dir = _ensure_pretrained_backbone(model_dir)

    # 3) directory → maybe nested checkpoints → resolve
    model_dir = _resolve_nested_ckpt(model_dir)

    # 4) LoRA adapter? detect adapter_config.json
    if os.path.isdir(model_dir) and _has_peft_adapter(model_dir):
        adapter_cfg = os.path.join(model_dir, "adapter_config.json")
        cfg = _safe_read_json(adapter_cfg)

        # Fall back to offline base if adapter didn't record it
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            base_name = _ensure_pretrained_backbone(lang)

        # --- load proper CTC base and attach adapter ---
        base_model = Wav2Vec2ForCTC.from_pretrained(
            base_name,
            torch_dtype=dtype
        ).to(device)

        processor = _load_processor_with_fallback(model_dir, base_name)

        model = PeftModel.from_pretrained(base_model, model_dir)
        return model.to(device).eval(), processor


    # 5) Full HF-style CTC folder?
    if os.path.isdir(model_dir) and _is_hf_wav2vec2_ctc_folder(model_dir):
        model = Wav2Vec2ForCTC.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_dir)
        return model.eval(), processor

    # 6) Our own wrapper folder (saved via save_pretrained)
    if os.path.isdir(model_dir):
        model, processor = Wav2Vec2AsrModel.from_pretrained(
            model_dir,
            device=device,
            mask_time_prob=0.05,
            mask_feature_prob=0.065,
        )
        return model.to(device).to(dtype).eval(), processor

    # fallback
    raise FileNotFoundError(f"ASR model directory not found: {model_dir}")


def load_ser_model(
    model_dir: str,
    n_emotions: int,
    device: str = "cpu",
    dropout: float = 0.2,
    use_fp16: bool = False,
    lang: str = "en"
):
    """
    Loads a Wav2Vec2 SER model. Supports:
      - 'en'/'de' → offline pretrained backbone folder
      - directory folder → as SER wrapper, including nested checkpoints
      - LoRA adapter folder (adapter_config.json + adapter_model.safetensors) → loads base + adapter
      - .pt checkpoint → load state_dict into fresh backbone+head
    """
    dtype = torch.float16 if use_fp16 and device.startswith("cuda") else torch.float32

    # 1) .pt checkpoint file?
    if model_dir.endswith(".pt"):
        ckpt_path = model_dir
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SER checkpoint not found: {ckpt_path}")
        backbone_dir = _ensure_pretrained_backbone(lang)
        model = Wav2Vec2SerModel(
            pretrained_ckpt=backbone_dir,
            n_emotions=n_emotions,
            device=device,
            dropout=dropout
        )
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(backbone_dir)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).to(dtype).eval()
        return model, extractor

    # 2) language code → pretrained backbone
    if model_dir in PRETRAINED_DIRS:
        model_dir = _ensure_pretrained_backbone(model_dir)

    # 3) directory → maybe nested checkpoints → resolve
    model_dir = _resolve_nested_ckpt(model_dir)

    # 4) LoRA adapter? detect adapter_config.json
    if os.path.isdir(model_dir) and _has_peft_adapter(model_dir):
        adapter_cfg = os.path.join(model_dir, "adapter_config.json")
        cfg = _safe_read_json(adapter_cfg)

        # ---- fallback to offline base if missing in adapter_config.json
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            base_name = _ensure_pretrained_backbone(lang)

        # --- Always build a classifier base so projector/classifier exist ---
        ser_cfg = AutoConfig.from_pretrained(base_name)
        ser_cfg.num_labels = n_emotions
        ser_cfg.problem_type = "single_label_classification"

        def _build_clean_base():
            bm = Wav2Vec2ForSequenceClassification.from_pretrained(
                base_name,
                config=ser_cfg,
                torch_dtype=dtype
            ).to(device)
            # --- Compat shim so adapter keys like "base_model.model.encoder.*" and
            #     "base_model.model.projector.*" resolve on HF classifier:
            #     encoder   := bm.wav2vec2.encoder
            #     projector := bm.projector
            class _Compat(nn.Module):
                def __init__(self, hf_cls):
                    super().__init__()
                    self.encoder   = hf_cls.wav2vec2.encoder
                    self.projector = getattr(hf_cls, "projector", nn.Identity())
            bm.model = _Compat(bm)
            return bm

        extractor = _load_extractor_with_fallback(model_dir, base_name)

        # --- Try the simple path first
        base_model = _build_clean_base()
        try:
            model = PeftModel.from_pretrained(base_model, model_dir)
            return model.to(device).to(dtype).eval(), extractor
        except KeyError:
            # --- Manual fallback: rebuild a CLEAN base (avoid double-PEFT), patch cfg, and load only present keys
            base_model = _build_clean_base()

            peft_cfg = PeftConfig.from_pretrained(model_dir)
            _sd = _load_adapter_state_dict(model_dir)

            # If the adapter state has no projector params, do not expect modules_to_save (esp. projector)
            has_proj = any(
                k.startswith("base_model.model.projector.") or k.startswith("base_model.projector.")
                for k in _sd.keys()
            )
            if not has_proj:
                peft_cfg.modules_to_save = []

            # Create empty adapter on the clean base, then load weights we actually have
            model = get_peft_model(base_model, peft_cfg)

            # --- Rename projector keys saved as "base_model.model.projector.*" -> "base_model.projector.*"
            fixed_sd = {}
            for k, v in _sd.items():
                if k.startswith("base_model.model.projector."):
                    k = k.replace("base_model.model.", "base_model.")
                fixed_sd[k] = v

            # --- Filter out keys the model doesn't have (avoid KeyError)
            model_sd_keys = set(model.state_dict().keys())
            fixed_sd = {k: v for k, v in fixed_sd.items() if k in model_sd_keys}

            # Load non-strictly (only matching keys)
            missing, unexpected = model.load_state_dict(fixed_sd, strict=False)
            # (Optional) print or log 'missing' and 'unexpected' for debugging

            return model.to(device).to(dtype).eval(), extractor

    # 5) Full HF-style SER folder?
    if os.path.isdir(model_dir) and _is_hf_wav2vec2_ser_folder(model_dir):
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        return model.eval(), extractor

    # 6) our wrapper folder
    if os.path.isdir(model_dir):
        model = Wav2Vec2SerModel(
            pretrained_ckpt=model_dir,
            n_emotions=n_emotions,
            device=device,
            dropout=dropout
        )
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        model = model.to(device).to(dtype).eval()
        return model, extractor

    raise FileNotFoundError(f"SER model directory not found: {model_dir}")


def load_tapt_backbone(
    pretrained_ckpt: str,
    device: str = "cpu",
    use_fp16: bool = False,
    lang: str = "en"
):
    """
    For TAPT pretraining: accepts 'en'/'de' codes or directory.
    """
    dtype = torch.float16 if use_fp16 and device.startswith("cuda") else torch.float32

    if pretrained_ckpt in PRETRAINED_DIRS:
        pretrained_ckpt = _ensure_pretrained_backbone(pretrained_ckpt)
    if not os.path.isdir(pretrained_ckpt):
        raise FileNotFoundError(f"TAPT backbone directory not found: {pretrained_ckpt}")

    return (
        Wav2Vec2ForPreTraining.from_pretrained(pretrained_ckpt, torch_dtype=dtype)
        .to(device)
    )
