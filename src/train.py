# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import argparse 
import random
import os
import sys
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from torch.utils.data import Sampler
import csv
import glob
from torchaudio.transforms import FrequencyMasking, TimeMasking
from sklearn.model_selection import train_test_split
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from peft import get_peft_config, get_peft_model, LoraConfig, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator,
    EarlyStoppingCallback,  # <-- added for Trainer-based early stopping
)
from preprocessing import load_asr_manifest, load_audio
from model_loader   import load_asr_model, load_ser_model
import re
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # needed for fallback in SERTrainer

# NEW: LoRA helpers for ASR come from model.py (keeps wrapping in one place)
from model import wrap_asr_with_lora, safe_merge_lora

# --- Add near imports ---
from pathlib import Path


def _finalize_asr_checkpoint(run_dir: str, model, processor):
    """
    Make this folder ready for inference:
    - Save full model + processor
    - Merge adapters (if any) and export to run_dir/merged
    - Ensure tokenizer_config has pad_token_id + word_delimiter_token
    - Ensure config.json.vocab_size matches tokenizer vocab
    """
    import json
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from peft import PeftModel

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save current (unmerged) model + processor
    model.save_pretrained(out)
    processor.save_pretrained(out)

    # 2) Compute tokenizer facts
    tok = processor.tokenizer
    vocab_size = len(tok.get_vocab())
    pad_token = tok.pad_token or "<pad>"
    pad_token_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
    wdt = getattr(tok, "word_delimiter_token", "|") or "|"

    # 3) Write/repair tokenizer_config.json
    tc_path = out / "tokenizer_config.json"
    tc = {}
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
        except Exception:
            tc = {}
    tc.setdefault("pad_token", pad_token)
    tc["pad_token_id"] = pad_token_id
    tc.setdefault("word_delimiter_token", wdt)
    tc.setdefault("unk_token", tok.unk_token or "[UNK]")
    tc.setdefault("do_lower_case", False)
    tc_path.write_text(json.dumps(tc, indent=2), encoding="utf-8")

    # 4) Force config.vocab_size to match tokenizer
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["vocab_size"] = int(vocab_size)
        # optional stability flags:
        cfg["ctc_zero_infinity"] = True
        cfg["ctc_loss_reduction"] = "mean"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # 5) Merge adapters (if present) and export to merged/
    model_to_export = model
    if isinstance(model, PeftModel) and hasattr(model, "merge_and_unload"):
        model_to_export = model.merge_and_unload()

    merged = out / "merged"
    merged.mkdir(exist_ok=True)
    model_to_export.save_pretrained(merged)
    processor.save_pretrained(merged)

    # 6) Also repair merged/ configs to be safe
    for p in [merged / "config.json", merged / "tokenizer_config.json"]:
        if p.name == "config.json" and p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg["vocab_size"] = int(vocab_size)
            cfg["ctc_zero_infinity"] = True
            cfg["ctc_loss_reduction"] = "mean"
            p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        if p.name == "tokenizer_config.json":
            p.write_text(json.dumps(tc, indent=2), encoding="utf-8")


class Tee:
    """
    Simple tee that writes everything both to stdout/stderr and a log file.
    Used so that Flask can still stream logs while we also persist them.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
        return len(data)

    def flush(self):
        for f in self.files:
            f.flush()


class DataCollatorCTCWithPadding:
    """
    Minimal replacement if your Transformers doesn't supply one.
    Pads input_values + attention_mask via `processor.pad`,
    pads labels with torch.nn.utils.rnn.pad_sequence, and
    turns pad_token_id → -100 so CTC ignores them.
    """
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding   = padding

    def __call__(self, features):
        # features: list of dicts with keys "input_values", "attention_mask", "labels"
        # 1) pad the audio side
        batch = self.processor.pad(
            [{"input_values": f["input_values"], "attention_mask": f["attention_mask"]}
             for f in features],
            padding=self.padding,
            return_tensors="pt"
        )
        # 2) pad the label side
        label_seqs = [f["labels"] for f in features]
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            label_seqs,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )
        # mask out pad for CTC
        padded_labels[padded_labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = padded_labels
        return batch


EMOTION_CLASSES = 6


# --------------------------------------------------------------------------
# 1. Early Stopping
# --------------------------------------------------------------------------
class EarlyStopping:
    """
    Stops training when validation loss doesn't improve after 'patience' epochs.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


# --------------------------------------------------------------------------
# 2. UnifiedDataset (for separate ASR / SER tasks)
# --------------------------------------------------------------------------
class UnifiedDataset(Dataset):
    """
    Dataset for ASR or SER tasks, loading raw audio from train/val splits.
    Args:
        task: "asr" or "ser"
        lang: language code, e.g. "en" or "de"
        split: "train" or "val"
        processor: a Wav2Vec2Processor (only needed for ASR)
    """
    def __init__(self, task: str, lang: str, split: str = "train", processor=None):
        self.task = task
        self.processor = processor

        # point at the correct directory based on split and task
        self.dir = f"data/processed/{split}/{'asr' if task=='asr' else 'emotion'}/{lang}"

        if task == "asr":
            # load manifest CSV → (full_path, transcript)
            labels_csv = os.path.join(self.dir, "labels.csv")
            fn_paths, transcripts = load_asr_manifest(labels_csv, self.dir)
            # map basename → transcript, but only keep non-empty ones
            self.manifest_map = {
                os.path.basename(path): txt
                for path, txt in zip(fn_paths, transcripts)
                if txt.strip()  # drop empty transcripts
            }
        else:
            # load emotion labels CSV → filename → int label (skip any negative labels)
            labels_csv = os.path.join(self.dir, "labels.csv")
            with open(labels_csv, newline="") as f:
                reader = csv.DictReader(f)
                self.label_map = {}
                for row in reader:
                    lbl = int(row["label"])
                    if lbl < 0:
                        # skip invalid entries like "-1"
                        continue
                    self.label_map[row["filename"]] = lbl

        # list only raw audio files
        exts = (".wav", ".mp3")
        if task == "asr":
            # only include files we have a non-empty transcript for
            self.files = sorted(
                fname for fname in os.listdir(self.dir)
                if fname.lower().endswith(exts) and fname in self.manifest_map
            )
        else:
            # only include files for which we have a valid (>=0) label
            self.files = sorted(
                fname for fname in os.listdir(self.dir)
                if fname.lower().endswith(exts) and fname in self.label_map
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn   = self.files[idx]
        path = os.path.join(self.dir, fn)

        # load raw waveform
        audio = load_audio(path)
        if audio is None:
            raise RuntimeError(f"Could not load audio: {path}")
        waveform = torch.tensor(audio, dtype=torch.float32)

        if self.task == "asr":
            # Now we have processor in hand → build inputs/labels up front
            text = re.sub(r"[^A-Z' ]+", "", self.manifest_map[fn].upper()).strip()
            encoding = self.processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,                # collator pads
                return_attention_mask=True,
            )
            input_values   = encoding.input_values.squeeze(0)    # [T]
            attention_mask = encoding.attention_mask.squeeze(0)  # [T]

            label_ids = self.processor.tokenizer(
                text,
                add_special_tokens=False
            ).input_ids
            labels = torch.tensor(label_ids, dtype=torch.long)

            return {
                "input_values":   input_values,
                "attention_mask": attention_mask,
                "labels":         labels,
            }

        else:
            # SER path unchanged
            label = self.label_map.get(fn, 0)
            return waveform, label


# --------------------------------------------------------------------------
# 3. Simple Collate Functions for ASR and SER
# --------------------------------------------------------------------------

def collate_asr(batch, max_frames=30000):
    """
    ASR-only collate: pads/truncates feature sequences or raw waveforms.
    """
    feats, transcripts = zip(*batch)
    max_len = min(max(f.shape[0] for f in feats), max_frames)
    padded = []
    for f in feats:
        if f.dim() == 1:
            f = f.unsqueeze(1)
        L, D = f.shape
        if L < max_len:
            pad = torch.zeros((max_len - L, D), dtype=f.dtype)
            padded.append(torch.cat([f, pad], dim=0))
        else:
            padded.append(f[:max_len])
    return torch.stack(padded), list(transcripts)


def collate_ser(batch, max_frames: int = None):
    """
    Pads (or truncates) raw waveforms for SER to a common length.
    """
    feats, labels = zip(*batch)
    waves = [f.squeeze(0) if f.dim() == 2 else f for f in feats]

    lengths = [w.shape[-1] for w in waves]
    target_len = max(lengths)
    if max_frames is not None:
        target_len = min(target_len, max_frames)

    padded = []
    for w in waves:
        if w.shape[-1] < target_len:
            pad = torch.zeros(target_len - w.shape[-1], dtype=w.dtype, device=w.device)
            padded.append(torch.cat([w, pad], dim=-1))
        else:
            padded.append(w[:target_len])

    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------
# 4. Train_asr_epoch: fine-tune Wav2Vec2 for CTC (legacy path)
# --------------------------------------------------------------------------
def train_asr_epoch(model, dataloader, optimizer, ctc_criterion,
                    processor, device, spec_augment=False):
    """
    Legacy manual ASR training loop.
    """
    if hasattr(ctc_criterion, "zero_infinity"):
        ctc_criterion.zero_infinity = True

    model.train()
    total_loss = 0.0

    for feats, transcripts in dataloader:
        feats = feats.squeeze(-1).to(device)

        batch_list    = feats.cpu().numpy().tolist()
        encoded       = processor(
            batch_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_values   = encoded.input_values.to(device)
        attention_mask = encoded.attention_mask.to(device)

        logits    = model(input_values, attention_mask=attention_mask)
        log_probs = logits.log_softmax(-1).transpose(0, 1)

        target_list = []
        for t in transcripts:
            clean_t = re.sub(r"[^A-Z' ]+", "", t.upper()).strip()
            token_ids = processor.tokenizer.encode(
                clean_t,
                add_special_tokens=False
            )
            if token_ids:
                target_list.append(
                    torch.tensor(token_ids, dtype=torch.long, device=device)
                )
        if not target_list:
            continue

        targets = torch.cat(target_list)
        target_lengths = torch.tensor(
            [t_seq.size(0) for t_seq in target_list],
            dtype=torch.long,
            device=device
        )

        batch_size, time_steps = logits.size(0), logits.size(1)
        input_lengths = torch.full(
            (batch_size,),
            fill_value=time_steps,
            dtype=torch.long,
            device=device
        )

        loss = ctc_criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# --------------------------------------------------------------------------
# 5. Train_ser_epoch: fine-tune Wav2Vec2 for emotion classification
# --------------------------------------------------------------------------
def train_ser_epoch(model, dataloader, optimizer, emotion_criterion, device):
    """
    Legacy manual SER training loop.
    """
    model.train()
    total_loss = 0.0

    for feats, labels in dataloader:
        feats  = feats.to(device)
        labels = labels.to(device)

        logits = model(feats)
        loss   = emotion_criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# --------------------------------------------------------------------------
# ASR Evaluation
# --------------------------------------------------------------------------
def evaluate_asr(model, dataloader, ctc_criterion, processor, device):
    """
    Evaluates ASR model: returns (avg_loss, avg_WER).
    Supports both dict-style and legacy tuple-style batches.
    """
    if hasattr(ctc_criterion, "zero_infinity"):
        ctc_criterion.zero_infinity = True

    model.eval()
    total_loss = 0.0
    total_wer  = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            # New dict-style path
            if isinstance(batch, dict):
                input_values = batch["input_values"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)

                # Use input_values explicitly to avoid any input_ids issues
                logits = model(input_values=input_values, attention_mask=attention_mask)
                log_probs = logits.log_softmax(-1).transpose(0, 1)

                target_seqs = []
                target_lengths = []
                for row in labels:
                    valid = row[row != -100]
                    if valid.numel() > 0:
                        target_seqs.append(valid)
                        target_lengths.append(valid.numel())
                if not target_seqs:
                    continue

                targets = torch.cat(target_seqs)
                target_lengths = torch.tensor(
                    target_lengths,
                    dtype=torch.long,
                    device=device
                )

                batch_size, time_steps = logits.size(0), logits.size(1)
                input_lengths = torch.full(
                    (batch_size,),
                    fill_value=time_steps,
                    dtype=torch.long,
                    device=device
                )

                loss = ctc_criterion(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths
                )
                total_loss += loss.item()

                pred_ids = logits.argmax(dim=-1)
                hyps = processor.batch_decode(
                    pred_ids.cpu().tolist(),
                    skip_special_tokens=True
                )

                labels_np = labels.detach().cpu().numpy()
                labels_np = np.where(
                    labels_np == -100,
                    processor.tokenizer.pad_token_id,
                    labels_np,
                )
                refs = processor.batch_decode(
                    labels_np,
                    skip_special_tokens=True
                )

                for ref, hyp in zip(refs, hyps):
                    ref = ref.strip()
                    hyp = hyp.strip()
                    if ref:
                        total_wer += wer(ref.lower(), hyp.lower())
                        count += 1

            # Legacy tuple-style path
            else:
                feats, transcripts = batch
                feats = feats.squeeze(-1).to(device)

                batch_list = feats.cpu().numpy().tolist()
                encoded = processor(
                    batch_list,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                input_values = encoded.input_values.to(device)
                attention_mask = encoded.attention_mask.to(device)

                logits = model(input_values=input_values, attention_mask=attention_mask)
                log_probs = logits.log_softmax(-1).transpose(0, 1)

                target_list = []
                for t in transcripts:
                    clean_t = re.sub(r"[^A-Z' ]+", "", t.upper()).strip()
                    tokens = processor.tokenizer.encode(
                        clean_t,
                        add_special_tokens=False
                    )
                    if tokens:
                        target_list.append(
                            torch.tensor(tokens, dtype=torch.long, device=device)
                        )
                if not target_list:
                    continue

                targets = torch.cat(target_list)
                target_lengths = torch.tensor(
                    [t_seq.size(0) for t_seq in target_list],
                    dtype=torch.long,
                    device=device
                )

                batch_size, time_steps = logits.size(0), logits.size(1)
                input_lengths = torch.full(
                    (batch_size,),
                    fill_value=time_steps,
                    dtype=torch.long,
                    device=device
                )

                loss = ctc_criterion(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths
                )
                total_loss += loss.item()

                pred_ids = logits.argmax(dim=-1)
                hyps = processor.batch_decode(
                    pred_ids.cpu().tolist(),
                    skip_special_tokens=True
                )
                for ref, hyp in zip(transcripts, hyps):
                    total_wer += wer(ref.lower(), hyp.lower())
                    count += 1

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_wer  = total_wer / count if count else 0.0
    print(f"[Eval-ASR] Loss={avg_loss:.4f}, WER={avg_wer:.4f}")
    return avg_loss, avg_wer


# --------------------------------------------------------------------------
# SER Evaluation
# --------------------------------------------------------------------------
def evaluate_ser(model, dataloader, emotion_criterion, device):
    """
    Evaluates SER model: returns (avg_loss, accuracy, f1_score).
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # UPDATED: support dict-style batches (feature_extractor outputs)
            if isinstance(batch, dict):
                x = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                attn = batch.get("attention_mask")
                out = model(x, attention_mask=(attn.to(device) if attn is not None else None))
                logits = out.logits if hasattr(out, "logits") else out
            else:
                feats, labels = batch
                feats  = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)

            loss   = emotion_criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")

    print(f"[Eval-SER] Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    return avg_loss, acc, f1


# --------------------------------------------------------------------------
# 6. Main train_model
# --------------------------------------------------------------------------
def train_model(
    device: str,
    lr_asr: float,
    bs_asr: int,
    epochs_asr: int,
    patience_asr: int,
    ckpt_asr: str,
    asr_lang: str,

    lr_ser: float,
    bs_ser: int,
    epochs_ser: int,
    dropout_ser: float,
    patience_ser: int,
    ckpt_ser: str,
    ser_lang: str,

    inference_dir: str = None,

    # LoRA controls for ASR
    use_lora_asr: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_targets: str = "q_proj,k_proj,v_proj,out_proj,intermediate_dense,output_dense",
    merge_lora_asr: bool = True,

    # SER calibration flags
    ser_label_smoothing: float = 0.0,
    ser_temperature_scale: bool = False,
):
    global np
    import numpy as np

    """
    Trains ASR and SER models (with optional LoRA finetuning for ASR) via HuggingFace Trainer,
    then optionally runs batch inference.
    """

    def asr_data_collator(batch):
        """
        Collate fn for Trainer:
        - dicts with "input_values", "attention_mask", "labels"
        - or legacy (waveform, transcript) tuples
        """
        if isinstance(batch[0], dict):
            input_vals = [ex["input_values"] for ex in batch]
            att_masks  = [ex["attention_mask"] for ex in batch]
            labels     = [ex["labels"] for ex in batch]

            input_vals = torch.nn.utils.rnn.pad_sequence(input_vals, batch_first=True)
            att_masks  = torch.nn.utils.rnn.pad_sequence(att_masks, batch_first=True)
            labels     = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )

            return {
                "input_values":   input_vals,
                "attention_mask": att_masks,
                "labels":         labels,
            }

        # Legacy tuple path
        audios, texts = zip(*batch)
        texts = [re.sub(r"[^A-Z' ]+", "", t.upper()).strip() for t in texts]

        array_list = [a.cpu().numpy().tolist() for a in audios]
        enc = processor(
            array_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values   = enc.input_values
        attention_mask = enc.attention_mask

        labels_batch = processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        ).input_ids
        labels_batch[labels_batch == processor.tokenizer.pad_token_id] = -100

        return {
            "input_values":   input_values,
            "attention_mask": attention_mask,
            "labels":         labels_batch,
        }

    # --- MOD: split SER collators (train with random crop, eval full clip) ---
    def ser_train_collator(batch):
        """
        Random-crop to max_frames, then feature-extract + pad for SER (training).
        """
        feats, labels = zip(*batch)
        waves = [f.squeeze(0) if f.dim() == 2 else f for f in feats]

        cropped = []
        for w in waves:
            L = w.shape[-1]
            if 'max_frames' in globals() or 'max_frames' in locals():
                try:
                    mf = max_frames
                except NameError:
                    mf = None
            else:
                mf = None
            if mf is not None and L > mf:
                start = random.randint(0, L - mf)
                w = w[start:start + mf]
            cropped.append(w)

        batch_waves_np = [w.cpu().numpy() for w in cropped]
        fe_out = ser_feature_extractor(
            batch_waves_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        batch = {
            "input_values": fe_out.input_values,                   # (B, T)
            "labels": torch.tensor(labels, dtype=torch.long),      # (B,)
        }
        if hasattr(fe_out, "attention_mask") and fe_out.attention_mask is not None:
            batch["attention_mask"] = fe_out.attention_mask        # (B, T)
        return batch

    def ser_eval_collator(batch):
        """
        NO random crop; use full wave (or center-trim if too long), then feature-extract + pad for SER (evaluation/inference).
        """
        feats, labels = zip(*batch)
        waves = [f.squeeze(0) if f.dim() == 2 else f for f in feats]

        # Use full clips
        batch_waves_np = [w.cpu().numpy() for w in waves]
        fe_out = ser_feature_extractor(
            batch_waves_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        out = {
            "input_values": fe_out.input_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if hasattr(fe_out, "attention_mask") and fe_out.attention_mask is not None:
            out["attention_mask"] = fe_out.attention_mask
        return out
    # -------------------------------------------------------------------------

    # 1) Device
    device = torch.device(device)
    print(f"[Setup] Using device: {device}")

    # --------------------------------------------------
    # 2) ASR Setup - ensure we start from a *valid* base
    # --------------------------------------------------
    # Normalize checkpoint source:
    # - if <base>_asr.pt exists → use it (state_dict on top of proper backbone)
    # - elif <base>_asr dir exists → use that HF/adapter dir
    # - else → fall back to language backbone ("en"/"de")
    asr_ckpt_src = ckpt_asr
    asr_ckpt_dir = os.path.splitext(ckpt_asr)[0]

    if not os.path.isfile(asr_ckpt_src):
        if os.path.isdir(asr_ckpt_dir):
            print(f"[ASR] No .pt at {ckpt_asr}, using HF dir as base: {asr_ckpt_dir}")
            asr_ckpt_src = asr_ckpt_dir
        else:
            # This implicitly maps to models/pretrained/<lang> via load_asr_model
            print(f"[ASR] No checkpoint found at {ckpt_asr}, using pretrained backbone for lang='{asr_lang}'")
            asr_ckpt_src = asr_lang

    print("[ASR] Loading model + processor…")
    asr_model, processor = load_asr_model(asr_ckpt_src, device=device, lang=asr_lang)

    # CTC stability tweaks (Wav2Vec2ForCTC-based)
    inner_asr = getattr(asr_model, "model", asr_model)
    if hasattr(inner_asr, "config"):
        inner_asr.config.ctc_zero_infinity  = True
        inner_asr.config.ctc_loss_reduction = "mean"

    # LoRA: IMPORTANT — pass the WRAPPER, not `.model`
    if use_lora_asr:
        before = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
        target_list = [t.strip() for t in lora_targets.split(",") if t.strip()]

        print(f"[ASR/LoRA] Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        asr_model = wrap_asr_with_lora(
            asr_model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_list,
        )

        after = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
        if after == before:
            print("[ASR/LoRA] LoRA not applied (see previous message). Proceeding without LoRA.")
        else:
            trainable = after
            total = sum(p.numel() for p in asr_model.parameters())
            print(f"[ASR/LoRA] trainable params: {trainable:,} / {total:,}")

    # 🔍 Debug: Inspect model wiring & PEFT/LoRA status
    print(f"[ASR] Model type: {type(asr_model)}")
    if hasattr(asr_model, "model"):
        print(f"[ASR] Inner model type: {type(asr_model.model)}")
        if hasattr(asr_model.model, "get_base_model"):
            print("[ASR] Using PEFT/LoRA model - base_model accessible")
        elif hasattr(asr_model.model, "peft_config"):
            print("[ASR] Using PEFT/LoRA model - peft_config present")

    # ASR datasets
    asr_train_ds = UnifiedDataset(
        task="asr",
        lang=asr_lang,
        split="train",
        processor=processor
    )
    asr_val_ds = UnifiedDataset(
        task="asr",
        lang=asr_lang,
        split="val",
        processor=processor
    )

    def compute_asr_metrics(pred):
        logits = pred.predictions
        label_ids = pred.label_ids

        pred_ids = np.argmax(logits, axis=-1)
        hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)

        labels = np.where(
            label_ids == -100,
            processor.tokenizer.pad_token_id,
            label_ids
        )
        refs = processor.batch_decode(
            labels,
            skip_special_tokens=True,
            group_tokens=False
        )

        pairs = [(r.strip(), h.strip()) for r, h in zip(refs, hyps) if r.strip()]
        if not pairs:
            return {"wer": float("nan")}

        refs_filt, hyps_filt = zip(*pairs)
        return {"wer": wer(list(refs_filt), list(hyps_filt))}

    def compute_ser_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        labels = pred.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # 3) ASR Trainer
    print("[ASR] Starting training with HuggingFace Trainer…")
    asr_args = TrainingArguments(
        output_dir                 = os.path.splitext(ckpt_asr)[0],
        group_by_length            = False,
        per_device_train_batch_size= bs_asr,
        per_device_eval_batch_size = bs_asr,
        evaluation_strategy        = "epoch",
        logging_strategy           = "steps",
        logging_steps              = 100,
        save_strategy              = "epoch",
        num_train_epochs           = epochs_asr,
        learning_rate              = lr_asr,
        warmup_ratio               = 0.1,
        lr_scheduler_type          = "linear",
        max_grad_norm              = 1.0,
        load_best_model_at_end     = True,
        metric_for_best_model      = "wer",
        greater_is_better          = False,
        save_total_limit           = 2,
        label_names                = ["labels"],
        remove_unused_columns      = False,
        no_cuda                    = (device.type == "cpu"),
    )

    # Early stopping callback for ASR, driven by patience_asr (in epochs)
    asr_early_stop = EarlyStoppingCallback(
        early_stopping_patience=patience_asr,
        early_stopping_threshold=0.0,
    )

    asr_trainer = Trainer(
        model           = asr_model,
        args            = asr_args,
        train_dataset   = asr_train_ds,
        eval_dataset    = asr_val_ds,
        data_collator   = asr_data_collator,
        tokenizer       = processor,
        compute_metrics = compute_asr_metrics,
        callbacks       = [asr_early_stop],
    )

    asr_trainer.train()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NEW: robust save block right after training (serving-ready artifacts)
    run_dir = os.path.splitext(ckpt_asr)[0]  # your run dir var
    os.makedirs(run_dir, exist_ok=True)

    print("[ASR] Saving adapters/processor and merged export...")
    try:
        # asr_model is the Wav2Vec2AsrModel wrapper; its save_pretrained should
        # write adapters + processor and also export a merged full model to run_dir/merged
        asr_model.save_pretrained(run_dir)
        print(f"[ASR] Saved to {run_dir} (and merged export under {run_dir}/merged)")
    except Exception as e:
        print(f"[ASR] save_pretrained failed: {e}")

    # Optional: also keep a regular Trainer-style checkpoint in the same root
    try:
        asr_trainer.save_model(run_dir)      # writes model.safetensors or state_dict
    except Exception as e:
        print(f"[ASR] trainer.save_model skipped: {e}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    asr_metrics = asr_trainer.evaluate()
    print(f"[ASR][Trainer] {asr_metrics}")

    # Optionally merge LoRA before saving (no-op if not supported)
    if use_lora_asr and merge_lora_asr:
        print("[ASR/LoRA] Merging adapters into base before saving…")
        asr_model = safe_merge_lora(asr_model)
        print("[ASR/LoRA] Adapters merged successfully")

    # Save full HF-style ASR model dir; loader will pick this up
    asr_trainer.save_model(os.path.splitext(ckpt_asr)[0])
    processor.save_pretrained(os.path.splitext(ckpt_asr)[0])

    # Save ASR metrics
    asr_outdir = os.path.splitext(ckpt_asr)[0]
    os.makedirs(asr_outdir, exist_ok=True)
    asr_metrics_path = os.path.join(asr_outdir, "metrics.json")
    asr_wer = asr_metrics.get("eval_wer", asr_metrics.get("wer"))
    with open(asr_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "asr",
            "metrics": {
                "wer": float(asr_wer) if asr_wer is not None else None
            }
        }, f, indent=2)
    print(f"[ASR] Wrote {asr_metrics_path}")

    # >>> New: one-time finalize to guarantee inference-ready checkpoints
    print("[ASR] Finalizing checkpoint for inference…")
    _finalize_asr_checkpoint(run_dir, asr_model, processor)
    print(f"[ASR] Ready: {run_dir}  (merged export at {run_dir}/merged)")

    # --------------------------------------------------
    # 4) SER Setup (kept consistent with loader logic)
    # --------------------------------------------------
    # Normalize SER checkpoint source similarly:
    ser_ckpt_src = ckpt_ser
    ser_ckpt_dir = os.path.splitext(ckpt_ser)[0]

    if not os.path.isfile(ser_ckpt_src):
        if os.path.isdir(ser_ckpt_dir):
            print(f"[SER] No .pt at {ckpt_ser}, using HF dir as base: {ser_ckpt_dir}")
            ser_ckpt_src = ser_ckpt_dir
        else:
            print(f"[SER] No checkpoint found at {ckpt_ser}, using pretrained backbone for lang='{ser_lang}'")
            ser_ckpt_src = ser_lang

    print("[SER] Loading model + extractor…")
    ser_model, ser_feature_extractor = load_ser_model(
        ser_ckpt_src,
        n_emotions=EMOTION_CLASSES,
        device=device,
        dropout=dropout_ser,
        lang=ser_lang
    )

    # Example LoRA on SER head (unchanged; loader knows how to reattach these adapters)
    peft_config_ser = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","out_proj","intermediate_dense","output_dense"],
        inference_mode=False,
        # --- MODIFICATION: keep classifier head trainable under PEFT ---
        modules_to_save=["head"],
    )
    ser_model = get_peft_model(ser_model, peft_config_ser)
    # --- OPTIONAL: partially unfreeze upper encoder layers for SER ---
    # How many top transformer blocks to unfreeze
    n_unfrozen_layers = 4   # or 6 for "last 6 layers"

    # Access the backbone encoder layers
    encoder_layers = ser_model.base_model.model.backbone.encoder.layers
    num_layers = len(encoder_layers)
    print(f"[SER] Encoder has {num_layers} layers, unfreezing last {n_unfrozen_layers}.")

    for layer in encoder_layers[num_layers - n_unfrozen_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    # Re-log trainables to see effect
    _ser_trainable_names = [n for n, p in ser_model.named_parameters() if p.requires_grad]
    print(f"[SER/LoRA+unfreeze] trainable params: "
          f"{sum(p.numel() for n,p in ser_model.named_parameters() if p.requires_grad):,}")
    for _n in _ser_trainable_names[:20]:
        print(f"[SER/LoRA+unfreeze] trainable: {_n}")
    if len(_ser_trainable_names) > 20:
        print(f"[SER/LoRA+unfreeze] ... and {len(_ser_trainable_names)-20} more")



    # --- MODIFICATION: log SER trainables (sanity check) ---
    total_ser = sum(p.numel() for p in ser_model.parameters())
    trainable_ser = sum(p.numel() for p in ser_model.parameters() if p.requires_grad)
    print(f"[SER/LoRA] trainable params: {trainable_ser:,} / {total_ser:,}")
    _ser_trainable_names = [n for n, p in ser_model.named_parameters() if p.requires_grad]
    for _n in _ser_trainable_names[:20]:
        print(f"[SER/LoRA] trainable: {_n}")
    if len(_ser_trainable_names) > 20:
        print(f"[SER/LoRA] ... and {len(_ser_trainable_names)-20} more")

    # SER datasets
    ser_train_ds = UnifiedDataset(task="ser", lang=ser_lang, split="train")
    ser_val_ds   = UnifiedDataset(task="ser", lang=ser_lang, split="val")

    # 95th percentile length
    all_lens = [wave.shape[-1] for wave, _ in ser_train_ds]
    max_frames = int(np.percentile(all_lens, 95))
    print(f"[SER] truncating to {max_frames} frames (~{max_frames/16000:.2f}s)")

    # Class weights
    labels_list = list(ser_train_ds.label_map.values())
    counts = np.bincount(labels_list, minlength=EMOTION_CLASSES)
    present = counts > 0
    K = int(present.sum()) if present.sum() > 0 else EMOTION_CLASSES
    N = int(counts[present].sum()) if present.sum() > 0 else 1
    class_weights = np.zeros(EMOTION_CLASSES, dtype=np.float32)
    for c in range(EMOTION_CLASSES):
        if counts[c] > 0:
            class_weights[c] = N / (K * counts[c])
        else:
            class_weights[c] = 0.0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ser_loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(ser_label_smoothing)
    )

    # Custom SER Trainer
    class SERTrainer(Trainer):
        def __init__(self, *args, loss_fn=None, train_collator=None, eval_collator=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fn = loss_fn
            # MOD: store distinct collators
            self.train_collator = train_collator or self.data_collator
            self.eval_collator  = eval_collator  or self.data_collator

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int = None,
            **kwargs,
        ):
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("SERTrainer.compute_loss: 'labels' missing from inputs.")

            # UPDATED: pass attention_mask to match inference
            outputs = model(
                inputs["input_values"],
                attention_mask=inputs.get("attention_mask")
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if isinstance(labels, torch.Tensor) and labels.device != logits.device:
                labels = labels.to(logits.device)

            if self.loss_fn is not None:
                loss = self.loss_fn(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            if return_outputs:
                return loss, outputs
            return loss

        # MOD: use separate collators for train/eval
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.train_collator,
            )

        def get_eval_dataloader(self, eval_dataset=None):
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            return DataLoader(
                ds,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.eval_collator,
            )

        # NEW: ensure predict()/test uses the no-crop eval collator (full clip)
        def get_test_dataloader(self, test_dataset):
            return DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.eval_collator,
            )

    # 5) SER Trainer
    print("[SER] Starting training with HuggingFace Trainer…")
    ser_args = TrainingArguments(
        output_dir=os.path.splitext(ckpt_ser)[0],
        per_device_train_batch_size=bs_ser,
        per_device_eval_batch_size=bs_ser,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        num_train_epochs=epochs_ser,
        learning_rate=lr_ser,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        label_names=["labels"],
        remove_unused_columns=False,
        greater_is_better=True,
        save_total_limit=2,
        no_cuda=(device.type == "cpu"),
    )

    # Early stopping callback for SER, driven by patience_ser (in epochs)
    ser_early_stop = EarlyStoppingCallback(
        early_stopping_patience=patience_ser,
        early_stopping_threshold=0.0,
    )

    ser_trainer = SERTrainer(
        model=ser_model,
        args=ser_args,
        train_dataset=ser_train_ds,
        eval_dataset=ser_val_ds,
        # MOD: pass distinct collators
        train_collator=ser_train_collator,
        eval_collator=ser_eval_collator,
        tokenizer=ser_feature_extractor,
        compute_metrics=compute_ser_metrics,
        loss_fn=ser_loss_fn,
        callbacks=[ser_early_stop],
    )

    ser_trainer.train()
    ser_metrics = ser_trainer.evaluate()
    print(f"[SER][Trainer] {ser_metrics}")

    # If SER is LoRA-wrapped, merge adapters into the base model before saving
    if isinstance(ser_trainer.model, PeftModel) and hasattr(ser_trainer.model, "merge_and_unload"):
        print("[SER/LoRA] Merging adapters into base SER model before saving…")
        ser_trainer.model = ser_trainer.model.merge_and_unload()

    # SER metrics (macro-F1 + UA)
    ser_outdir = os.path.splitext(ckpt_ser)[0]
    os.makedirs(ser_outdir, exist_ok=True)
    ser_metrics_path = os.path.join(ser_outdir, "metrics.json")

    val_preds = ser_trainer.predict(ser_val_ds)
    y_true = val_preds.label_ids
    y_hat  = val_preds.predictions.argmax(axis=-1)
    macro_f1 = f1_score(y_true, y_hat, average="macro")
    ua = recall_score(y_true, y_hat, average="macro")

    with open(ser_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "ser",
            "metrics": {
                "accuracy": float(ser_metrics.get("eval_accuracy", None)),
                "f1_weighted": float(ser_metrics.get("eval_f1", None)),
                "f1_macro": float(macro_f1),
                "ua": float(ua)
            }
        }, f, indent=2)
    print(f"[SER] Wrote {ser_metrics_path}")

    # Now save the merged full SER model + feature extractor
    ser_trainer.save_model(os.path.splitext(ckpt_ser)[0])
    ser_feature_extractor.save_pretrained(os.path.splitext(ckpt_ser)[0])

    # 6) Optional SER temperature scaling
    if ser_temperature_scale:
        print("[SER/Calib] Fitting temperature scaling on validation split…")
        val_loader = DataLoader(
            ser_val_ds,
            batch_size=bs_ser,
            shuffle=False,
            # MOD: eval-style collator (no random crop)
            collate_fn=ser_eval_collator,
        )
        ser_trainer.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)
                out = ser_trainer.model(
                    x,
                    attention_mask=batch.get("attention_mask").to(device) if batch.get("attention_mask") is not None else None
                )
                logits = out.logits if hasattr(out, "logits") else out
                all_logits.append(logits)
                all_labels.append(y)
        logits_val = torch.cat(all_logits, dim=0)
        labels_val = torch.cat(all_labels, dim=0)

        T = torch.nn.Parameter(torch.ones(1, device=device))
        opt = torch.optim.LBFGS([T], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            scaled = logits_val / T.clamp_min(1e-3)
            loss = nll(scaled, labels_val)
            loss.backward()
            return loss

        opt.step(closure)
        T_opt = float(T.detach().clamp_min(1e-3).item())
        print(f"[SER/Calib] Learned temperature T = {T_opt:.4f}")

        ser_ckpt_dir = os.path.splitext(ckpt_ser)[0]
        os.makedirs(ser_ckpt_dir, exist_ok=True)
        with open(os.path.join(ser_ckpt_dir, "ser_temperature.json"), "w", encoding="utf-8") as f:
            json.dump({"T": T_opt}, f)

    # 7) Optional Batch Inference
    if inference_dir:
        for task in ["asr", "ser"]:
            for split in ["train", "val"]:
                print(f"[Infer] {task.UPPER()} inference on {split} split…")
                ds = UnifiedDataset(
                    task=task,
                    lang=(asr_lang if task == "asr" else ser_lang),
                    split=split,
                    processor=(processor if task == "asr" else None),
                )
                loader = DataLoader(
                    ds,
                    batch_size=(bs_asr if task == "asr" else bs_ser),
                    shuffle=False,
                    collate_fn=(asr_data_collator if task == "asr" else ser_eval_collator),
                )
                if task == "asr":
                    evaluate_asr(
                        asr_model,
                        loader,
                        torch.nn.CTCLoss(
                            blank=processor.tokenizer.pad_token_id,
                            zero_infinity=True
                        ),
                        processor,
                        device
                    )
                else:
                    evaluate_ser(
                        ser_model,
                        loader,
                        torch.nn.CrossEntropyLoss(),
                        device
                    )


# --------------------------------------------------------------------------
# 7. Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ASR + SER Training Script")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )

    parser.add_argument(
        "--inference_dir",
        type=str,
        help="Directory for batch inference (optional)"
    )

    # ASR hyperparameters
    parser.add_argument("--asr_learning_rate", type=float, required=True,
                        help="ASR learning rate")
    parser.add_argument("--asr_batch_size",    type=int,   required=True,
                        help="ASR batch size")
    parser.add_argument("--asr_epochs",        type=int,   required=True,
                        help="Number of ASR training epochs")
    parser.add_argument("--asr_patience",      type=int,   required=True,
                        help="ASR early stopping patience")
    parser.add_argument("--asr_checkpoint",    type=str,   required=True,
                        help="Path to save ASR model checkpoint")
    parser.add_argument("--asr_lang",
        choices=["en", "de"],
        default="en",
        help="Language for ASR dataset")

    # SER hyperparameters
    parser.add_argument("--ser_learning_rate", type=float, required=True,
                        help="SER learning rate")
    parser.add_argument("--ser_batch_size",    type=int,   required=True,
                        help="SER batch size")
    parser.add_argument("--ser_epochs",        type=int,   required=True,
                        help="Number of SER training epochs")
    parser.add_argument("--ser_dropout",       type=float, required=True,
                        help="SER model dropout")
    parser.add_argument("--ser_patience",      type=int,   required=True,
                        help="SER early stopping patience")
    parser.add_argument("--ser_checkpoint",    type=str,   required=True,
                        help="Path to save SER model checkpoint")
    parser.add_argument("--ser_lang",
        choices=["en", "de"],
        default="en",
        help="Language for SER dataset")

    # SER calibration flags
    parser.add_argument("--ser_label_smoothing", type=float, default=0.0,
                        help="Label smoothing for SER CrossEntropyLoss (default 0.0)")
    parser.add_argument("--ser_temperature_scale", action="store_true",
                        help="Fit a single scalar temperature on the dev split and save to checkpoint")

    # LoRA flags for ASR
    parser.add_argument("--use_lora_asr", action="store_true",
                        help="Enable LoRA adapters for ASR fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank for ASR")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha for ASR")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout for ASR")
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,k_proj,v_proj,out_proj,intermediate_dense,output_dense",
                        help="Comma-separated target module names in Wav2Vec2 (attention/FFN projections)")
    parser.add_argument("--merge_lora_asr", action="store_true",
                        default=True,
                        help="Merge LoRA adapters into the base model for inference")

    args = parser.parse_args()

    # --- Tee stdout/stderr into a train.log file for this run (ASR base dir) ---
    base_log_dir = os.path.splitext(args.asr_checkpoint)[0]
    os.makedirs(base_log_dir, exist_ok=True)
    log_path = os.path.join(base_log_dir, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"[Logging] Training logs will also be written to: {log_path}")

    train_model(
        device         = args.device,

        lr_asr         = args.asr_learning_rate,
        bs_asr         = args.asr_batch_size,
        epochs_asr     = args.asr_epochs,
        patience_asr   = args.asr_patience,
        ckpt_asr       = args.asr_checkpoint,
        asr_lang       = args.asr_lang,

        lr_ser         = args.ser_learning_rate,
        bs_ser         = args.ser_batch_size,
        epochs_ser     = args.ser_epochs,
        dropout_ser    = args.ser_dropout,
        patience_ser   = args.ser_patience,
        ckpt_ser       = args.ser_checkpoint,
        ser_lang       = args.ser_lang,

        inference_dir  = args.inference_dir,

        use_lora_asr   = args.use_lora_asr,
        lora_r         = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        lora_targets   = args.lora_targets,
        merge_lora_asr = args.merge_lora_asr,
        ser_label_smoothing = args.ser_label_smoothing,
        ser_temperature_scale = args.ser_temperature_scale,
    )
