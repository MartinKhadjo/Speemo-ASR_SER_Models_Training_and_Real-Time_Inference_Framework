import os
import torch

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)


ASR_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "de": "facebook/wav2vec2-large-xlsr-53-german",
}

SER_MODELS = {
    # Good defaults; you can switch EN to wav2vec2-large-robust if you like.
    "en": "facebook/wav2vec2-base",
    "de": "facebook/wav2vec2-large-xlsr-53",
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _check_asr_ctc(model: Wav2Vec2ForCTC, tag: str):
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 16000)
        out = model(dummy)
    if not hasattr(out, "logits") or not isinstance(out.logits, torch.Tensor):
        raise RuntimeError(
            f"[ASR][{tag}] sanity check failed: no .logits in forward output."
        )
    if out.logits.ndim != 3:
        raise RuntimeError(
            f"[ASR][{tag}] sanity check failed: logits shape {tuple(out.logits.shape)} "
            f"!= [B, T, V] for CTC."
        )


def _check_ser_encoder(model: Wav2Vec2Model, tag: str):
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 16000)
        out = model(dummy)
    if not hasattr(out, "last_hidden_state") or not isinstance(out.last_hidden_state, torch.Tensor):
        raise RuntimeError(
            f"[SER][{tag}] sanity check failed: no last_hidden_state from encoder."
        )


def download_asr_backbone(model_name: str, target_dir: str, tag: str):
    """
    Download a CTC ASR backbone (Wav2Vec2ForCTC + Processor)
    and save into target_dir, with sanity check.
    """
    target_dir = _ensure_dir(target_dir)
    print(f"[ASR] Downloading {model_name} -> {target_dir}")

    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    _check_asr_ctc(model, tag)

    model.save_pretrained(target_dir)
    processor.save_pretrained(target_dir)
    print(f"[ASR] Saved valid CTC backbone {model_name} to {target_dir}")


def download_ser_backbone(model_name: str, target_dir: str, tag: str):
    """
    Download an encoder backbone for SER (Wav2Vec2Model + processor/feature extractor)
    and save into target_dir, with sanity check.
    """
    target_dir = _ensure_dir(target_dir)
    print(f"[SER] Downloading {model_name} -> {target_dir}")

    enc = Wav2Vec2Model.from_pretrained(model_name)

    # Prefer full processor, else feature extractor
    try:
        proc = Wav2Vec2Processor.from_pretrained(model_name)
    except Exception:
        proc = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    _check_ser_encoder(enc, tag)

    enc.save_pretrained(target_dir)
    proc.save_pretrained(target_dir)
    print(f"[SER] Saved valid encoder backbone {model_name} to {target_dir}")


if __name__ == "__main__":
    base_dir = "models/pretrained"

    # English
    download_asr_backbone(
        ASR_MODELS["en"],
        os.path.join(base_dir, "en", "ASR"),
        tag="en-ASR",
    )
    download_ser_backbone(
        SER_MODELS["en"],
        os.path.join(base_dir, "en", "SER"),
        tag="en-SER",
    )

    # German
    download_asr_backbone(
        ASR_MODELS["de"],
        os.path.join(base_dir, "de", "ASR"),
        tag="de-ASR",
    )
    download_ser_backbone(
        SER_MODELS["de"],
        os.path.join(base_dir, "de", "SER"),
        tag="de-SER",
    )

    print("\n✅ All backbones downloaded and sanity-checked.")
