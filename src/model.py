# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ──────────────────────────────────────────────────────────────────────────────
# Optional PEFT / LoRA support (ASR)
# ──────────────────────────────────────────────────────────────────────────────

_PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType, PeftModel
    _PEFT_AVAILABLE = True
except Exception:  # PEFT not installed or partial
    LoraConfig = None
    get_peft_model = None
    PeftTaskType = None
    try:
        # If only PeftModel import fails
        from peft import PeftModel  # type: ignore
    except Exception:
        class PeftModel:  # dummy for isinstance checks
            pass
    _PEFT_AVAILABLE = False


def _freeze_params(module: nn.Module):
    """Freeze all parameters in a module (requires_grad = False)."""
    for p in module.parameters():
        p.requires_grad = False


def wrap_asr_with_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules=None,
):
    """
    Attach LoRA adapters to an ASR Wav2Vec2ForCTC backbone in a PEFT-safe way.

    - If PEFT or a suitable TaskType (CTC / FEATURE_EXTRACTION) is not available:
      do NOTHING (return model unchanged).
    - Otherwise: inject LoRA into the underlying CTC backbone.

    Goal: avoid `forward() got an unexpected keyword argument 'input_ids'`.
    """

    if not (_PEFT_AVAILABLE and LoraConfig and get_peft_model and PeftTaskType):
        print("[ASR/LoRA] peft not available; skipping LoRA.")
        return model

    has_ctc = hasattr(PeftTaskType, "CTC")
    has_feat = hasattr(PeftTaskType, "FEATURE_EXTRACTION")

    if not has_ctc and not has_feat:
        print(
            "[ASR/LoRA] This peft version has no CTC/FEATURE_EXTRACTION TaskType; "
            "skipping LoRA."
        )
        return model

    task_type = PeftTaskType.CTC if has_ctc else PeftTaskType.FEATURE_EXTRACTION

    if not target_modules:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "intermediate_dense",
            "output_dense",
        ]

    # Underlying HF model:
    # - if model is our wrapper → use model.model
    # - else assume model itself is Wav2Vec2ForCTC-like.
    base = getattr(model, "model", model)
    if not isinstance(base, nn.Module):
        print("[ASR/LoRA] Could not find a valid base model; skipping LoRA.")
        return model

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
    )

    peft_model = get_peft_model(base, lora_cfg)

    # If we started from our wrapper, keep wrapper and swap .model
    if hasattr(model, "model"):
        model.model = peft_model
        wrapped = model
    else:
        wrapped = peft_model

    print(f"[ASR/LoRA] LoRA adapters attached to ASR backbone (task_type={task_type.name}).")
    return wrapped


def safe_merge_lora(model: nn.Module):
    """
    Merge LoRA adapters into the base model if supported.
    Handles both plain PeftModel and our Wav2Vec2AsrModel wrapper.
    """
    # Direct PeftModel-style merge
    merge_fn = getattr(model, "merge_and_unload", None)
    if callable(merge_fn):
        try:
            return merge_fn()
        except Exception:
            return model

    # Wrapped Wav2Vec2AsrModel
    if isinstance(model, Wav2Vec2AsrModel):
        inner = getattr(model, "model", None)
        merge_inner = getattr(inner, "merge_and_unload", None)
        if callable(merge_inner):
            try:
                model.model = merge_inner()
            except Exception:
                pass
        return model

    return model


# ──────────────────────────────────────────────────────────────────────────────
# ASR wrapper
# ──────────────────────────────────────────────────────────────────────────────


class Wav2Vec2AsrModel(nn.Module):
    """
    Thin wrapper around a CTC-style ASR backbone.

    self.model:
      - Wav2Vec2ForCTC
      - or PeftModel(Wav2Vec2ForCTC) with LoRA adapters.
    """

    def __init__(self, base_model: nn.Module, processor: Wav2Vec2Processor):
        super().__init__()
        self.model = base_model
        self.processor = processor

    def forward(
        self,
        input_values=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        Robust wrapper so HF Trainer + PEFT LoRA work correctly.

        Rules:
        - Treat `input_ids` as alias for `input_values` (ASR uses input_values).
        - Strip any stray `input_ids` & text/decoder-only keys.
        - If self.model is a PeftModel → call its `.base_model` (CTC head kept).
        - Else (plain Wav2Vec2ForCTC) → call self.model directly.
        """

        # 1) Alias: input_ids → input_values if needed
        if input_values is None and input_ids is not None:
            input_values = input_ids

        # 2) Handle stray input_ids in kwargs
        if "input_ids" in kwargs:
            if input_values is None and kwargs["input_ids"] is not None:
                input_values = kwargs["input_ids"]
            kwargs.pop("input_ids", None)

        # 3) Drop text/decoder-specific kwargs
        for bad_key in (
            "inputs_embeds",
            "token_type_ids",
            "position_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
            "decoder_inputs_embeds",
            "decoder_position_ids",
            "encoder_outputs",
            "past_key_values",
            "use_cache",
            "cross_attn_head_mask",
        ):
            kwargs.pop(bad_key, None)

        if input_values is None:
            raise ValueError(
                "Wav2Vec2AsrModel.forward expected `input_values` "
                "(or `input_ids` as alias), but none was provided."
            )

        call_kwargs = dict(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        if labels is not None:
            call_kwargs["labels"] = labels

        # 4) Decide which underlying module to call
        backbone = self.model

        # If this is a PEFT wrapper, bypass its generic forward and go
        # straight to the wrapped HF model (which has the CTC head).
        if _PEFT_AVAILABLE:
            try:
                from peft import PeftModel as _PeftBase  # type: ignore
            except Exception:
                _PeftBase = PeftModel  # from earlier try/except

            if isinstance(self.model, _PeftBase):
                # For LoRA-on-CTC, base_model is Wav2Vec2ForCTC with adapters.
                backbone = self.model.base_model

        return backbone(**call_kwargs)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        # 1) Save model or adapters
        maybe_save = getattr(self.model, "save_pretrained", None)
        if callable(maybe_save):
            # If self.model is a PeftModel, this writes adapter_model.* + adapter_config.json.
            # If it's a plain Wav2Vec2ForCTC, it writes the full model.
            self.model.save_pretrained(save_dir)
        else:
            # Fallback: raw state_dict (rare)
            torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        # 2) Save processor next to it (tokenizer + feature extractor)
        if hasattr(self, "processor") and self.processor is not None:
            self.processor.save_pretrained(save_dir)

        # 3) Also export a merged full model for serving
        try:
            merged = safe_merge_lora(self)  # no-op if not PEFT
            base = getattr(merged, "model", merged)
            merged_dir = os.path.join(save_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            if hasattr(base, "save_pretrained"):
                base.save_pretrained(merged_dir)
                if hasattr(self, "processor") and self.processor is not None:
                    self.processor.save_pretrained(merged_dir)
                print(f"[ASR] Also exported merged full model to: {merged_dir}")
        except Exception as e:
            print(f"[ASR] merge/export skipped: {e}")

    def state_dict(self, *args, **kwargs):
        # delegate to underlying model (PeftModel or Wav2Vec2ForCTC)
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)    

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cpu",
        mask_time_prob: float = 0.05,
        mask_feature_prob: float = 0.065,
    ):
        """
        Load a CTC ASR model + processor from a HF-style directory and wrap it.

        `model_dir` must contain a Wav2Vec2ForCTC checkpoint.
        """
        base_model = Wav2Vec2ForCTC.from_pretrained(model_dir)

        # ensure it's really CTC-style
        if not hasattr(base_model, "lm_head"):
            raise RuntimeError(
                f"Wav2Vec2AsrModel.from_pretrained expected a CTC model in '{model_dir}', "
                f"but loaded object has no lm_head (did you save Wav2Vec2Model instead?)."
            )

        # set masking hyperparams (affects training, harmless for inference)
        if hasattr(base_model.config, "mask_time_prob"):
            base_model.config.mask_time_prob = mask_time_prob
        if hasattr(base_model.config, "mask_feature_prob"):
            base_model.config.mask_feature_prob = mask_feature_prob

        processor = Wav2Vec2Processor.from_pretrained(model_dir)
        wrapper = cls(base_model.to(device), processor)
        wrapper.eval()
        return wrapper, processor


# ──────────────────────────────────────────────────────────────────────────────
# SER wrapper
# ──────────────────────────────────────────────────────────────────────────────


class Wav2Vec2SerModel(nn.Module):
    """
    Wav2Vec2 backbone + small head for speech emotion recognition.
    """

    def __init__(
        self,
        pretrained_ckpt: str,
        n_emotions: int,
        device: str,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_ckpt).to(device)
        hidden = self.backbone.config.hidden_size

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_emotions),
        ).to(device)

    def forward(
        self,
        input_values=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # alias input_ids -> input_values
        if input_values is None and input_ids is not None:
            input_values = input_ids

        if input_values is None:
            raise ValueError(
                "Wav2Vec2SerModel.forward expected `input_values` "
                "(or `input_ids` as alias)."
            )

        # strip irrelevant kwargs
        for bad_key in (
            "inputs_embeds",
            "token_type_ids",
            "position_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
            "decoder_inputs_embeds",
            "decoder_position_ids",
            "encoder_outputs",
            "past_key_values",
            "use_cache",
            "cross_attn_head_mask",
        ):
            kwargs.pop(bad_key, None)

        hs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs,
        ).last_hidden_state  # [B, T, D]

        x = hs.transpose(1, 2)          # [B, D, T]
        x = self.pool(x).squeeze(-1)    # [B, D]
        logits = self.head(x)           # [B, n_emotions]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_dir)
        torch.save(
            self.head.state_dict(),
            os.path.join(save_dir, "ser_head.pt"),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str, n_emotions: int, device: str):
        backbone = Wav2Vec2Model.from_pretrained(model_dir).to(device)
        model = cls(model_dir, n_emotions, device=device)
        model.backbone = backbone
        head_path = os.path.join(model_dir, "ser_head.pt")
        if os.path.isfile(head_path):
            state = torch.load(head_path, map_location=device)
            model.head.load_state_dict(state)
        model.eval()

        # Try to load feature extractor, fall back to processor if needed
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        except Exception:
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
            feature_extractor = processor.feature_extractor

        return model, feature_extractor


__all__ = [
    "Wav2Vec2AsrModel",
    "Wav2Vec2SerModel",
    "wrap_asr_with_lora",
    "safe_merge_lora",
]
