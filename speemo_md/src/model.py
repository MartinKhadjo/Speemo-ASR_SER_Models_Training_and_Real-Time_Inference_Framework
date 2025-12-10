import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F


class Wav2Vec2AsrModel(nn.Module):
    """
    Fine-tunes wav2vec2 for CTC-based ASR.
    """
    def __init__(
        self,
        pretrained_ckpt: str,
        device: str,
        mask_time_prob: float = 0.05,
        mask_feature_prob: float = 0.065,
    ):
        super().__init__()
        # Load the CTC-head model with built-in SpecAugment masking
        self.model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_ckpt,
            mask_time_prob=mask_time_prob,
            mask_feature_prob=mask_feature_prob,
        ).to(device)

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,                    # <— now accepts labels
    ):
        """
        If `labels` is provided, the underlying Wav2Vec2ForCTC
        will compute `.loss` (CTC) for you. Returns a ModelOutput
        with both `.loss` and `.logits`.
        """
        return self.model(
            input_values,
            attention_mask=attention_mask,
            labels=labels,             # <— pass labels through
        )

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str,
        mask_time_prob: float = 0.05,
        mask_feature_prob: float = 0.065,
    ):
        proc = Wav2Vec2Processor.from_pretrained(model_dir)
        model = cls(
            model_dir,
            device=device,
            mask_time_prob=mask_time_prob,
            mask_feature_prob=mask_feature_prob,
        )
        model.eval()
        return model, proc


class Wav2Vec2SerModel(nn.Module):
    """
    Fine-tunes wav2vec2 for speech-emotion recognition.
    """
    def __init__(self, pretrained_ckpt: str, n_emotions: int, device: str, dropout: float = 0.2):
        super().__init__()
        # Backbone without CTC head
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_ckpt).to(device)
        hidden = self.backbone.config.hidden_size
        # mean-pool + MLP head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_emotions)
        ).to(device)

    def forward(self, input_values, attention_mask=None, labels=None):
        """
        Forward pass for SER model. Returns a SequenceClassifierOutput with:
        - loss: scalar CrossEntropyLoss if labels is provided, else None
        - logits: raw [batch, n_emotions] predictions
        """
        # 1) encode via backbone
        hs = self.backbone(input_values, attention_mask=attention_mask).last_hidden_state  # [B, T, D]
        # 2) pool
        x = hs.transpose(1, 2)                # [B, D, T]
        x = self.pool(x).squeeze(-1)          # [B, D]
        # 3) classification head
        logits = self.head(x)                 # [B, n_emotions]

        # 4) compute loss if we have labels
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # 5) return HF‐compatible output
        return SequenceClassifierOutput(loss=loss, logits=logits)



    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        # Save only the head's weights; backbone remains in HF format
        torch.save(self.head.state_dict(), os.path.join(save_dir, "ser_head.pt"))

    @classmethod
    def from_pretrained(cls, model_dir: str, n_emotions: int, device: str):
        backbone = Wav2Vec2Model.from_pretrained(model_dir).to(device)
        model = cls(model_dir, n_emotions, device=device)
        head_state = torch.load(os.path.join(model_dir, "ser_head.pt"), map_location=device)
        model.head.load_state_dict(head_state)
        model.eval()
        proc = Wav2Vec2Processor.from_pretrained(model_dir)
        return model, proc.feature_extractor


__all__ = ["Wav2Vec2AsrModel", "Wav2Vec2SerModel"]
