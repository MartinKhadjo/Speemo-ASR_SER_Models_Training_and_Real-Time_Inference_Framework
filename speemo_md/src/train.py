import argparse
import random
import os

os.environ["NCCL_DEBUG"]  = "INFO"
os.environ["NCCL_TIMEOUT"] = "300"   # 5 minutes

import torch.distributed as dist       # ← for destroy_process_group()
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments, Trainer
from preprocessing import load_asr_manifest

from model_loader import load_asr_model, load_ser_model
import re
from transformers import EarlyStoppingCallback
from transformers import get_scheduler
from transformers import TrainerCallback
import gc

from collections import Counter
import torch.nn as nn
import accelerate
import inspect   # ← *** this was missing ***

import transformers
from transformers.utils import import_utils as _hf_import_utils

#------------------------------------------------------------------
# Compatibility patch: tolerate old accelerate that lacks 'dispatch_batches'
#------------------------------------------------------------------
try:
    sig = inspect.signature(accelerate.Accelerator.__init__)
    if "dispatch_batches" not in sig.parameters:
        _orig_init = accelerate.Accelerator.__init__

        def _patched_init(self, *args, **kwargs):
            # drop the kwarg that old accelerate versions don't understand
            kwargs.pop("dispatch_batches", None)
            return _orig_init(self, *args, **kwargs)

        accelerate.Accelerator.__init__ = _patched_init
        print("[Compat] Patched accelerate.Accelerator.__init__ to ignore 'dispatch_batches'.")
    else:
        print("[Compat] accelerate.Accelerator already supports 'dispatch_batches'. No patch needed.")
except Exception as e:
    print(f"[Compat] Could not patch accelerate: {e}")

#------------------------------------------------------------------
# Compatibility patch 2: tolerate old accelerate.unwrap_model without
# the 'keep_torch_compile' kwarg that new Trainer passes.
#------------------------------------------------------------------
try:
    _orig_unwrap = accelerate.Accelerator.unwrap_model

    def _patched_unwrap(self, model, *args, **kwargs):
        # drop unknown kwarg for older accelerate versions
        kwargs.pop("keep_torch_compile", None)
        return _orig_unwrap(self, model, *args, **kwargs)

    accelerate.Accelerator.unwrap_model = _patched_unwrap
    print("[Compat] Patched accelerate.Accelerator.unwrap_model to ignore 'keep_torch_compile'.")
except Exception as e:
    print(f"[Compat] Could not patch Accelerate.unwrap_model: {e}")

# ------------------------------------------------------------------
# Patch Transformers' torch-load safety gate (CVE-2025-32434) to allow
# loading checkpoints with torch < 2.6. ONLY SAFE IF CHECKPOINTS ARE TRUSTED.
# ------------------------------------------------------------------
try:
    _hf_modeling_utils = transformers.modeling_utils
except Exception:
    _hf_modeling_utils = None

def _bypass_torch_load_safety_check(*args, **kwargs):
    """
    Override for transformers.utils.import_utils.check_torch_load_is_safe.

    NOTE: This disables the v2.6+ torch requirement for torch.load inside
    Transformers. Only do this if you fully trust your local checkpoints.
    """
    print(
        "[WARNING] Overriding transformers.check_torch_load_is_safe(): "
        "skipping torch version check. ONLY safe for trusted local checkpoints."
    )
    return

# Install the override into Transformers
try:
    _hf_import_utils.check_torch_load_is_safe = _bypass_torch_load_safety_check
    if _hf_modeling_utils is not None and hasattr(_hf_modeling_utils, "check_torch_load_is_safe"):
        _hf_modeling_utils.check_torch_load_is_safe = _bypass_torch_load_safety_check
    print("[Compat] Patched Transformers check_torch_load_is_safe to bypass torch>=2.6 requirement.")
except Exception as e:
    print(f"[Compat] Could not patch Transformers safety check: {e}")


# ------------------------------------------------------------------
# Helper: build TrainingArguments but only with kwargs that this
# installed Transformers version supports.
# ------------------------------------------------------------------
def build_training_arguments(base_kwargs: dict) -> TrainingArguments:
    """
    Construct TrainingArguments compatible with both old and new
    Transformers versions by filtering unsupported kwargs and
    handling evaluation/eval strategy renames.
    """
    safe_kwargs = {}
    try:
        import inspect as _inspect
        sig = _inspect.signature(TrainingArguments.__init__)
        has_eval_strategy_param = "eval_strategy" in sig.parameters
        has_evaluation_strategy_param = "evaluation_strategy" in sig.parameters

        for k, v in base_kwargs.items():
            # Some very old versions don't accept 'report_to'
            if k == "report_to" and "report_to" not in sig.parameters:
                continue

            # Normalize evaluation strategy argument names
            if k in ("evaluation_strategy", "eval_strategy"):
                if has_eval_strategy_param:
                    safe_kwargs["eval_strategy"] = v
                elif has_evaluation_strategy_param:
                    safe_kwargs["evaluation_strategy"] = v
                # do not fall through to the generic branch
                continue

            if k in sig.parameters:
                safe_kwargs[k] = v
    except Exception:
        # very old Transformers – keep a minimal subset
        minimal_keys = [
            "output_dir",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "learning_rate",
            "num_train_epochs",
            "no_cuda",
            "remove_unused_columns",
            "fp16",
            "gradient_accumulation_steps",
            "eval_accumulation_steps",
            "logging_steps",
            "save_total_limit",
            "dataloader_num_workers",
            "dataloader_drop_last",
            "max_grad_norm",
            "evaluation_strategy",
        ]
        for k in minimal_keys:
            if k in base_kwargs:
                safe_kwargs[k] = base_kwargs[k]

    # Construct the TrainingArguments with the filtered kwargs
    args = TrainingArguments(**safe_kwargs)

    # Final safety net: if load_best_model_at_end=True, force eval/save strategies to match
    if getattr(args, "load_best_model_at_end", False):
        save_value = getattr(args, "save_strategy", None)

        # get current eval strategy (new or old attribute name)
        eval_value = None
        if hasattr(args, "eval_strategy"):
            eval_value = args.eval_strategy
        elif hasattr(args, "evaluation_strategy"):
            eval_value = args.evaluation_strategy

        if save_value is not None and eval_value is not None and eval_value != save_value:
            # Try to convert to IntervalStrategy if available; otherwise, just assign raw value
            try:
                from transformers.training_args import IntervalStrategy
                if isinstance(save_value, str):
                    save_enum = IntervalStrategy(save_value)
                else:
                    save_enum = save_value
            except Exception:
                save_enum = save_value

            if hasattr(args, "eval_strategy"):
                try:
                    args.eval_strategy = save_enum
                except Exception:
                    args.eval_strategy = save_value
            if hasattr(args, "evaluation_strategy"):
                try:
                    args.evaluation_strategy = save_enum
                except Exception:
                    args.evaluation_strategy = save_value

    return args


def normalize_for_tokenizer(text: str, tokenizer) -> str:
    """
    Keep only characters that exist in the tokenizer vocab (single-char tokens),
    plus space and apostrophe if present. Auto-match case to tokenizer.
    """
    vocab = tokenizer.get_vocab()

    # Decide case policy
    has_upper = any(ch in vocab for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    has_lower = any(ch in vocab for ch in "abcdefghijklmnopqrstuvwxyz")
    if has_upper and not has_lower:
        text = text.upper()
    elif has_lower and not has_upper:
        text = text.lower()

    # unify quotes
    text = text.replace("’", "'").replace("`", "'")

    # allowed single-char tokens
    allowed = {k for k in vocab if len(k) == 1}
    # ensure space/apostrophe if tokenizer supports them
    if " " in vocab: allowed.add(" ")
    if "|" in vocab: allowed.add("|")   # many wav2vec CTC tokenizers use '|' for space
    if "'" in vocab: allowed.add("'")

    # map unknowns to space; if '|' is used for space, keep it as '|'
    out = []
    for ch in text:
        if ch in allowed:
            out.append(ch)
        else:
            out.append("|" if "|" in allowed else " ")
    text = "".join(out)

    # collapse whitespace/pipe runs
    if "|" in allowed:
        text = re.sub(r"\|+", "|", text).strip("|")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- MP3/WAV loader: 16 kHz mono float32 (librosa -> soundfile fallback) ---

def load_audio_16k_mono(path: str):
    """Load any audio (mp3/wav/…) as 16 kHz mono float32."""
    # 1) Try librosa (uses pysoundfile by default, and falls back to audioread/ffmpeg -> handles mp3)
    try:
        import librosa
        y, _sr = librosa.load(path, sr=16000, mono=True)
        return y.astype(np.float32), 16000
    except Exception:
        pass

    # 2) Fallback to soundfile; resample via librosa if needed
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), 16000
    except Exception as e:
        raise RuntimeError(f"Could not load audio: {path}") from e




class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        



class DataCollatorCTCWithPadding:
    """
    Pads input_values, attention_mask, and labels via torch.nn.utils.rnn.pad_sequence.
    Converts pad_token_id → -100 for CTC ignore_index.
    """
    def __init__(self, processor, max_seconds: float = 30.0):
        # processor: Wav2Vec2Processor (with feature_extractor & tokenizer)
        self.processor = processor
        self.pad_id    = processor.tokenizer.pad_token_id
        # never pad/truncate beyond this many raw audio samples
        # cap raw audio length to avoid OOM
        self.max_samples = int(16000 * max_seconds)


    def __call__(self, features):
        """
        Args:
            features: list of dicts, each with keys:
                - "input_values":   Tensor[T]
                - "attention_mask": Tensor[T]
                - "labels":         Tensor[L]
        Returns:
            batch: dict with keys
                - "input_values":   Tensor[B, T_max]
                - "attention_mask": Tensor[B, T_max]
                - "labels":         Tensor[B, L_max] (with pad_id → -100)
        """
        # 1) unpack feature lists
        # 1a) crop every example to max_samples to avoid OOM at eval time
        input_vals = [f["input_values"][:self.max_samples] for f in features]   # list of [T_i]
        att_masks  = [f["attention_mask"][:self.max_samples] for f in features]  # list of [T_i]
        label_seqs = [f["labels"] for f in features]         # list of [L_i]

        # 2) pad each to batch-max length
        input_batch = pad_sequence(input_vals, batch_first=True, padding_value=0.0)
        mask_batch  = pad_sequence(att_masks, batch_first=True, padding_value=0)
        label_batch = pad_sequence(label_seqs, batch_first=True, padding_value=self.pad_id)

        # 3) convert pad token -> -100 so CTC ignores them
        label_batch = label_batch.masked_fill(label_batch == self.pad_id, -100)

        return {
            "input_values":   input_batch,   # float32 [B, T_max]
            "attention_mask": mask_batch,    # int64   [B, T_max]
            "labels":         label_batch,   # int64   [B, L_max]
        }








EMOTION_CLASSES = 6







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
        self.processor = processor   # <— new!

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
                        # skip invalid entries like “-1”
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
        # AFTER
        audio, _ = load_audio_16k_mono(path)
        waveform = torch.tensor(audio, dtype=torch.float32)


        if self.task == "asr":
            # ── NEW DICT RETURN FOR ASR ────────────────────────────────────
            # now we have processor in hand → build inputs/labels up‐front
            transcript = normalize_for_tokenizer(self.manifest_map[fn], self.processor.tokenizer)
            encoding = self.processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,                # no pad here, collator will do it
                return_attention_mask=True,
            )
            # squeeze the batch dim:
            input_values   = encoding.input_values.squeeze(0)    # Tensor [T]
            attention_mask = encoding.attention_mask.squeeze(0)  # Tensor [T]
            # tokenize transcript into ids (no special tokens), then a Tensor:
            label_ids = self.processor.tokenizer(
                transcript,
                add_special_tokens=False
            ).input_ids
            labels = torch.tensor(label_ids, dtype=torch.long)

            return {
                "input_values":   input_values,
                "attention_mask": attention_mask,
                "labels":         labels,
            }

        else:
            # ── unchanged SER path ───────────────────────────────────────
            label = self.label_map.get(fn, 0)
            return waveform, label





# --------------------------------------------------------------------------
# 3. Simple Collate Functions for ASR and SER
# --------------------------------------------------------------------------




# SER-only collate: simply stacks your pooled features and labels
def collate_ser(batch, max_frames: int = None):
    """
    Pads (or truncates) raw waveforms for SER to a common length.
    Args:
      batch: list of (waveform: Tensor[T], label: int)
      max_frames: if set, clamp all waveforms to at most this many samples.
    Returns:
      Tuple:
        - Tensor of shape [B, T] with padded/truncated waveforms
        - Tensor of shape [B] with labels
    """
    feats, labels = zip(*batch)
    # ensure 1D
    waves = [f.squeeze(0) if f.dim() == 2 else f for f in feats]

    # compute target length
    lengths = [w.shape[-1] for w in waves]
    target_len = max(lengths)
    if max_frames is not None:
        target_len = min(target_len, max_frames)

    # pad / truncate
    padded = []
    for w in waves:
        if w.shape[-1] < target_len:
            pad = torch.zeros(target_len - w.shape[-1], dtype=w.dtype, device=w.device)
            padded.append(torch.cat([w, pad], dim=-1))
        else:
            padded.append(w[:target_len])

    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)



# -----------------------------------------------------------------------------
# 4. Train_asr_epoch: fine-tune Wav2Vec2 for CTC
# -----------------------------------------------------------------------------
def train_asr_epoch(model, dataloader, optimizer, ctc_criterion,
                    processor, device, spec_augment=False):
    """
    Fine-tune Wav2Vec2 for CTC-based ASR.
    Args:
        model:           Wav2Vec2ForCTC
        dataloader:      yields (waveform: Tensor[B, T,?], transcripts: List[str])
        optimizer:       torch optimizer
        ctc_criterion:   nn.CTCLoss instance *with zero_infinity=True*
        processor:       Wav2Vec2Processor (feature_extractor + tokenizer)
        device:          torch.device
        spec_augment:    bool, whether to apply Frequency/Time masking
    Returns:
        avg_loss: float
    """
    # ensure your CTC never blows up to inf/NaN
    if hasattr(ctc_criterion, "zero_infinity"):
        ctc_criterion.zero_infinity = True

    model.train()
    total_loss = 0.0

    for feats, transcripts in dataloader:
        # feats comes in as [B, T, 1] → squeeze to [B, T]
        feats = feats.squeeze(-1).to(device)

        # build Wav2Vec2 inputs correctly:
        batch_list    = feats.cpu().numpy().tolist()    # list of 1D float arrays
        encoded       = processor(
            batch_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_values   = encoded.input_values.to(device)      # [B, T]
        attention_mask = encoded.attention_mask.to(device)    # [B, T]

        # forward pass
        logits    = model(input_values, attention_mask=attention_mask)  # [B, T, V]
        log_probs = logits.log_softmax(-1).transpose(0, 1)             # [T, B, V]

        # ── Build CTC targets ───────────────────────────────────────────
        target_list = []
        for t in transcripts:
            # clean & normalize transcript → only a-z and spaces
            clean_t = re.sub(r"[^a-z ]+", "", t.lower()).strip()
            token_ids = processor.tokenizer.encode(
                clean_t,
                add_special_tokens=False
            )
            if token_ids:
                target_list.append(
                    torch.tensor(token_ids, dtype=torch.long, device=device)
                )
        # skip if every transcript tokenized to empty
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

        # ── Compute CTC loss ────────────────────────────────────────────
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





# -----------------------------------------------------------------------------
# 5. Train_ser_epoch: fine-tune Wav2Vec2 for emotion classification
# -----------------------------------------------------------------------------
def train_ser_epoch(model, dataloader, optimizer, emotion_criterion, device):
    """
    Fine-tune a Wav2Vec2-based SER head.
    Args:
        model:             Wav2Vec2SerModel
        dataloader:        yields (waveform: Tensor[B, T], labels: Tensor[B])
        optimizer:         torch optimizer
        emotion_criterion: nn.CrossEntropyLoss
        device:            torch.device
    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0

    for feats, labels in dataloader:
        feats  = feats.to(device)
        labels = labels.to(device)

        # forward pass returns [B, C]
        logits = model(feats)
        loss   = emotion_criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



# --------------------------------------------------------------------------
# Modified evaluate() function with asr_mode handling
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# (Externe evaluate_asr / evaluate_ser entfernt: wir nutzen stattdessen Trainer.evaluate())

# --------------------------------------------------------------------------#

class SerTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # keep a CPU copy; we’ll move per-step safely
        self._class_weights = class_weights if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        x = inputs["input_values"].to(device, dtype=torch.float32, non_blocking=True)
        am = inputs.get("attention_mask", None)
        if am is not None: am = am.to(device, dtype=torch.long, non_blocking=True)
        y = inputs["labels"].to(device, dtype=torch.long, non_blocking=True)

        # sanitize raw audio a bit
        x = torch.nan_to_num(x).clamp_(-1.0, 1.0)

        outputs = model(x, attention_mask=am)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # make logits finite instead of skipping the batch
        if not torch.isfinite(logits).all():
            if self.state.global_step == 0:
                self.log({"warn_nonfinite_logits": 1.0})
            logits = torch.nan_to_num(logits)

        weight = self._class_weights.to(device, dtype=logits.dtype) if self._class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, y)
        return (loss, {"logits": logits}) if return_outputs else loss





def train_model(
    device,
    lr_asr: float,
    bs_asr: int,
    epochs_asr: int,
    patience_asr: int,        # no longer used by Trainer but kept for signature
    ckpt_asr: str,
    asr_lang: str,

    lr_ser: float,
    bs_ser: int,
    epochs_ser: int,
    dropout_ser: float,
    patience_ser: int,        # no longer used by Trainer but kept for signature
    ckpt_ser: str,
    ser_lang: str,
    inference_dir: str = None,
    phase: str = "all",
):
    
    """
    Trains ASR and SER models (with LoRA finetuning) via HuggingFace Trainer,
    then optionally runs batch inference.
    """

    # --- Decide where to LOAD FROM (resume checkpoint vs pretrained) ---
    def _resolve_start_dir(ckpt_str: str, lang: str, task: str) -> str:
        """
        ckpt_str is what the UI passes (e.g., 'models/checkpoints/<name>_asr').
        We TRAIN into ckpt_str (actually os.path.splitext(ckpt_str)[0]).
        We LOAD from either an existing checkpoint dir (resume) or a pretrained dir.
        """
        # Trainer will write to this directory (no extension trimming necessary here,
        # but we keep it consistent with your later .splitext usage)
        out_dir = os.path.splitext(ckpt_str)[0]

        # Resume if the checkpoint dir already exists and has content
        if os.path.isdir(out_dir):
            try:
                if any(os.scandir(out_dir)):
                    print(f"[{task}] Resuming from checkpoint: {out_dir}")
                    return out_dir
            except PermissionError:
                pass  # fall back to pretrained

        # Fresh start from pretrained
        pretrained_dir = os.path.join("models", "pretrained", lang)
        if os.path.isdir(pretrained_dir) and any(os.scandir(pretrained_dir)):
            print(f"[{task}] Starting from pretrained: {pretrained_dir}")
            return pretrained_dir

        # If neither exists, be explicit so the error is clear
        raise FileNotFoundError(
            f"[{task}] Neither checkpoint dir ({out_dir}) nor pretrained dir ({pretrained_dir}) exists."
        )



    # ── 1) Device Setup ────────────────────────────────────────────────
    # accept either a string ("cuda"/"cpu") or an actual torch.device
    dev = device if isinstance(device, torch.device) else torch.device(device)
    print(f"[Setup] Using device: {dev}")

    # tell HF Trainer not to use CUDA when we're on CPU
    no_cuda = (dev.type == "cpu")
    print(f"[Setup] no_cuda flag set to: {no_cuda}")

    # NEW: prefer stable math on recent NVIDIA GPUs (TF32 for matmul/conv; keeps FP32 semantics)
    if dev.type == "cuda":
        try:
            torch.set_float32_matmul_precision("medium")  # PyTorch 2.x convenience
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


    # ── 2) ASR Setup ───────────────────────────────────────────────────
    if phase in ("all", "asr"):
        print("[ASR] Loading model + processor…")
        asr_start_dir = _resolve_start_dir(ckpt_asr, asr_lang, "ASR")
        # Load FROM resolved dir, but still SAVE TO the checkpoint name the UI gave us
        asr_model_wrapper, processor = load_asr_model(asr_start_dir, device=dev)

        # unwrap to the underlying HF Wav2Vec2ForCTC
        asr_model = getattr(asr_model_wrapper, "model", asr_model_wrapper)
        

        # ── FREEZE the CNN feature-extractor so only LoRA adapters train ──
        if hasattr(asr_model, "freeze_feature_extractor"):
            asr_model.freeze_feature_extractor()
        elif hasattr(asr_model, "freeze_feature_encoder"):
            asr_model.freeze_feature_encoder()

        # ── DISABLE KV-CACHE so eval doesn’t blow up memory ─────────────
        asr_model.config.use_cache = False

        # ── Disable SpecAugment (masking) to avoid mask_length > sequence_length errors ──
        # completely disable SpecAugment (avoids mask_length > sequence_length errors)
        asr_model.config.mask_time_prob    = 0.0
        asr_model.config.mask_feature_prob = 0.0


        # ── Inject LoRA adapters into the frozen HF model ──────────────────────────────
        peft_config_asr = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","out_proj"],  # ← expanded so adapters actually attach
            inference_mode=False,
        )
        asr_model = get_peft_model(asr_model, peft_config_asr)

        # --- sanity: show trainable parameter count to catch 0-param bugs early ---
        try:
            asr_model.print_trainable_parameters()  # PEFT helper; prints % trainable
        except Exception:
            pass
        n_trainable = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in asr_model.parameters())
        print(f"[ASR] Trainable parameters: {n_trainable} / {n_total}")
        if n_trainable == 0:
            raise RuntimeError("[ASR] LoRA pattern matched no modules → 0 trainable params. "
                               "Adjust `target_modules` (e.g., include q_proj/k_proj/v_proj/out_proj).")

        # ── Build ASR datasets ────────────────────────────────────────────────
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

        # ── Instantiate unified CTC collator ────────────────────────────────
        asr_collator = DataCollatorCTCWithPadding(processor, max_seconds=30.0)

        # ── DEBUG: verify collator output shapes ───────────────────────────
        _dbg_n = min(4, len(asr_train_ds))
        if (_dbg_n > 0) and (len(asr_train_ds) > 0):
            sample_batch = asr_collator([asr_train_ds[i] for i in range(_dbg_n)])
            print("🛠️  Debug collator output shapes:", {k: v.shape for k, v in sample_batch.items()})


        # ── CTC config tweaks ────────────────────────────────────────────────
        base = getattr(asr_model, "model", asr_model)
        if hasattr(base, "config"):
            base.config.ctc_zero_infinity  = True
            base.config.ctc_loss_reduction = "mean"

        # ── Metrics fn ──────────────────────────────────────────────────────
        def compute_asr_metrics(eval_pred):
            logits    = eval_pred.predictions
            label_ids = eval_pred.label_ids

            # 1) decode hypotheses
            pred_ids = np.argmax(logits, axis=-1)
            hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)

            # 2) rebuild references from label ids: map -100 -> pad_id, and decode with no grouping
            pad_id  = processor.tokenizer.pad_token_id
            ref_ids = np.where(label_ids != -100, label_ids, pad_id)
            refs = processor.batch_decode(ref_ids, skip_special_tokens=True, group_tokens=False)

            # 3) light normalization (match training text normalization better)
            def norm(s: str) -> str:
                return re.sub(r"[^a-z ]+", "", s.lower()).strip()

            pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps) if norm(r)]
            if not pairs:
                return {"wer": float("nan")}

            r, h = zip(*pairs)
            return {"wer": wer(list(r), list(h))}





        # ── 3) ASR Training via Trainer ────────────────────────────────────
        print("[ASR] Starting training with HuggingFace Trainer…")
        asr_args_base = dict(
            output_dir                     = os.path.splitext(ckpt_asr)[0],
            per_device_train_batch_size    = max(1, bs_asr // 2),
            per_device_eval_batch_size     = 1,              # don’t OOM by eval’ing >1 at once
            fp16                           = (not no_cuda),
            gradient_accumulation_steps    = 4,              # try 4 instead of 8
            eval_accumulation_steps        = 1,              # clear after each mini-batch
            dataloader_num_workers         = 4,
            dataloader_drop_last           = False,
            no_cuda                        = no_cuda,
            evaluation_strategy            = "epoch",
            logging_strategy               = "steps",
            logging_steps                  = 100,
            logging_first_step             = True,           # NEW
            log_level                      = "info",         # NEW
            save_strategy                  = "epoch",
            num_train_epochs               = epochs_asr,
            learning_rate                  = lr_asr,
            report_to                      = "none",

            # more gentle schedule: 10% warm-up then constant LR
            warmup_ratio                   = 0.10,                     # 10% of total steps
            lr_scheduler_type              = "constant_with_warmup",
            max_grad_norm                  = 1.0,

            load_best_model_at_end         = True,
            metric_for_best_model          = "wer",
            greater_is_better              = False,
            save_total_limit               = 2,
            label_names                    = ["labels"],
            remove_unused_columns          = False,
            ddp_find_unused_parameters     = True,
        )
        asr_args = build_training_arguments(asr_args_base)

        asr_trainer = Trainer(
            model=asr_model,
            args=asr_args,
            train_dataset=asr_train_ds,
            eval_dataset=asr_val_ds,
            data_collator=asr_collator,
            # In new HF Trainer versions you pass processor/feature_extractor via "tokenizer"
            tokenizer=processor,
            compute_metrics=compute_asr_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=patience_asr),
                ClearCacheCallback(),
            ],
        )
        





        



        # ── Finally kick off training ──────────────────────────────────────
        asr_trainer.train()
        asr_metrics = asr_trainer.evaluate(eval_dataset=asr_val_ds)
        print(f"[ASR][Trainer Eval] {asr_metrics}")

        asr_trainer.save_model(os.path.splitext(ckpt_asr)[0])
        processor.save_pretrained(os.path.splitext(ckpt_asr)[0])
    else:
        print("[ASR] Skipped (phase != 'all'/'asr').")



    # ── 4) SER Setup ───────────────────────────────────────────────────
    if phase in ("all", "ser"):
        print("[SER] Loading model + extractor…")
        ser_start_dir = _resolve_start_dir(ckpt_ser, ser_lang, "SER")
        ser_model, ser_feature_extractor = load_ser_model(
            ser_start_dir,
            n_emotions=EMOTION_CLASSES,
            device=dev,
            dropout=dropout_ser,
        )



        # ── FREEZE the SER backbone to save memory & stabilize training ──
        # unwrap the actual HF model (prefer .backbone, then .model, else assume it's already base)
        base_ser = getattr(ser_model, "backbone", None) or getattr(ser_model, "model", None) or ser_model

        # Transformers renamed freeze_feature_extractor() → freeze_feature_encoder() in newer versions
        if hasattr(base_ser, "freeze_feature_extractor"):
            base_ser.freeze_feature_extractor()
        elif hasattr(base_ser, "freeze_feature_encoder"):
            base_ser.freeze_feature_encoder()

        # --- your requested config toggles go RIGHT HERE ---
        if hasattr(base_ser, "config") and base_ser.config is not None:
            # Disable SpecAugment during fine-tuning (avoids mask_length > sequence_length issues)
            if hasattr(base_ser.config, "mask_time_prob"):
                base_ser.config.mask_time_prob = 0.0
            if hasattr(base_ser.config, "mask_feature_prob"):
                base_ser.config.mask_feature_prob = 0.0
            # Safety: no KV cache on encoder-style models
            if hasattr(base_ser.config, "use_cache"):
                base_ser.config.use_cache = False

        # ── Ensure gradient checkpointing is OFF for SER (LoRA + GC can break autograd) ──
        for _m in (ser_model, getattr(ser_model, "model", None), getattr(ser_model, "backbone", None), base_ser):
            if _m is not None:
                if hasattr(_m, "gradient_checkpointing_disable"):
                    try:
                        _m.gradient_checkpointing_disable()
                    except TypeError:
                        _m.gradient_checkpointing_disable()
                if getattr(_m, "config", None) is not None and hasattr(_m.config, "gradient_checkpointing"):
                    _m.config.gradient_checkpointing = False

        # ── Inject LoRA adapters ─────────────────────────────────────────────
        # IMPORTANT: keep classification head trainable via modules_to_save
        
        peft_config_ser = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj","k_proj","v_proj","out_proj",
                "intermediate_dense","output_dense",   # wav2vec2 FFN names
                # keeping the head learnable explicitly via modules_to_save below
            ],
            modules_to_save=["classifier", "projector"],  # keep classification layers trainable
            inference_mode=False,
        )
        ser_model = get_peft_model(ser_model, peft_config_ser)


        # --- show trainable parameter count (helps catch 0-param bugs early) ---
        try:
            ser_model.print_trainable_parameters()
        except Exception:
            pass
        n_trainable_ser = sum(p.numel() for p in ser_model.parameters() if p.requires_grad)
        n_total_ser     = sum(p.numel() for p in ser_model.parameters())
        print(f"[SER] Trainable parameters: {n_trainable_ser} / {n_total_ser}")
        # NEW: explicit check + debug list length
        n_require_grad_tensors = sum(1 for _n, p in ser_model.named_parameters() if p.requires_grad)
        print(f"[SER] tensors requiring grad: {n_require_grad_tensors}")
        if n_trainable_ser == 0 or n_require_grad_tensors == 0:
            raise RuntimeError("[SER] No parameters require grad after LoRA injection. Check target_modules/modules_to_save.")

        # ── build SER datasets up-front so we can measure lengths ─────────────
        ser_train_ds = UnifiedDataset(task="ser", lang=ser_lang, split="train")
        ser_val_ds   = UnifiedDataset(task="ser", lang=ser_lang, split="val")

        # —— compute the 95th-percentile duration (in samples) ————————
        all_lens = [wave.shape[-1] for wave,_ in ser_train_ds]
        max_frames = int(np.percentile(all_lens, 95))
        print(f"[SER] truncating to {max_frames} frames (~{max_frames/16000:.2f}s)")

        def ser_data_collator(batch):
            feats, labels = zip(*batch)
            # enforce minimum raw length (~25ms) per item to survive conv front-end
            MIN_SAMPLES = 400  # ~25ms @ 16kHz; see Baevski et al. 2020
            batch_list = []
            for w in feats:
                w = w.squeeze(0) if w.dim() == 2 else w
                if w.numel() < MIN_SAMPLES:
                    pad = torch.zeros(MIN_SAMPLES - w.numel(), dtype=torch.float32, device=w.device)
                    w = torch.cat([w.to(torch.float32), pad], dim=-1)
                batch_list.append(w.cpu().numpy())  # Feature extractor expects CPU numpy

            # use the HF feature extractor for consistent padding/normalization
            fe = ser_feature_extractor  # already loaded with the model
            enc = fe(
                batch_list,
                sampling_rate=16000,
                return_attention_mask=True,  # ensure AM is produced for consistent masking
                padding=True
            )

            # ↓↓↓ Faster & dtype-safe conversion (avoids slow list->tensor path)
            input_values = torch.from_numpy(np.asarray(enc["input_values"], dtype=np.float32))
            attention_mask = torch.from_numpy(np.asarray(enc["attention_mask"], dtype=np.int64))
            labels = torch.tensor(labels, dtype=torch.long)
            return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}



        def compute_ser_metrics(eval_pred):
            logits = eval_pred.predictions
            # unwrap possible tuple/dict wrappers from HF
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("logits", None)
            if logits is None:
                return {"accuracy": float("nan"), "f1": float("nan")}

            preds  = np.argmax(logits, axis=-1)
            labels = np.asarray(eval_pred.label_ids)

            n = min(len(preds), len(labels))
            if n == 0:
                return {"accuracy": float("nan"), "f1": float("nan")}

            preds  = preds[:n].astype(int)
            labels = labels[:n].astype(int)

            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            }


        # ----- class weights for imbalance (optional but recommended)
        counts = Counter(ser_train_ds.label_map.values())
        num_classes = EMOTION_CLASSES
        N = sum(counts.get(c, 0) for c in range(num_classes))
        weights = torch.tensor(
            [N / (num_classes * max(1, counts.get(c, 0))) for c in range(num_classes)],
            dtype=torch.float
        )
        print("[SER] class counts:", dict(counts))
        print("[SER] class weights:", weights.tolist())

        # ── quick sanity batch (no grad) to ensure finite loss
        tmp_dl = DataLoader(ser_train_ds, batch_size=min(8, len(ser_train_ds)),
                            shuffle=True, collate_fn=ser_data_collator)
        tmp_batch = next(iter(tmp_dl))
        ser_model.eval()
        with torch.no_grad():
            am = tmp_batch["attention_mask"]
            am = am.to(dev) if am is not None else None
            outputs = ser_model(
                tmp_batch["input_values"].to(dev),
                attention_mask=am,
            )

            # Extract logits (handles HF ModelOutput, tuples, or raw tensors)
            if isinstance(outputs, (tuple, list)):
                logits_chk = outputs[0]
            elif hasattr(outputs, "logits"):
                logits_chk = outputs.logits
            else:
                logits_chk = outputs

            loss_chk = nn.CrossEntropyLoss(
                weight=weights.to(dev) if not no_cuda else weights
            )(
                logits_chk,
                tmp_batch["labels"].to(dev),
            )

            loss_chk = nn.CrossEntropyLoss(weight=weights.to(dev) if not no_cuda else weights)(
                logits_chk, tmp_batch["labels"].to(dev)
            )
        print(f"[SER] Sanity loss (1 batch): {float(loss_chk)}")
        ser_model.train()

        # ── 5) SER Training via Trainer ────────────────────────────────────
        print("[SER] Starting training with HuggingFace Trainer…")
        ser_args_base = dict(
            output_dir                     = os.path.splitext(ckpt_ser)[0],
            per_device_train_batch_size    = bs_ser,
            per_device_eval_batch_size     = 1,
            # CHANGED: disable mixed precision for stability first
            fp16                           = False,
            bf16                           = False,
            gradient_accumulation_steps    = 4,
            eval_accumulation_steps        = 1,
            dataloader_num_workers         = 4,
            dataloader_drop_last           = True,
            gradient_checkpointing         = False,
            no_cuda                        = no_cuda,
            evaluation_strategy            = "epoch",
            logging_strategy               = "steps",
            logging_steps                  = 100,
            save_strategy                  = "epoch",
            num_train_epochs               = epochs_ser,
            learning_rate                  = lr_ser,
            # NEW: mild warmup + linear decay helps stabilize early steps
            warmup_ratio                   = 0.10,
            lr_scheduler_type              = "linear",
            load_best_model_at_end         = True,
            metric_for_best_model          = "eval_accuracy",
            greater_is_better              = True,
            save_total_limit               = 2,
            label_names                    = ["labels"],
            remove_unused_columns          = False,
            ddp_find_unused_parameters     = True,
            # NEW: clip to prevent gradient blowups -> NaNs
            max_grad_norm                  = 1.0,
            report_to                      = "none",
        )
        ser_args = build_training_arguments(ser_args_base)

        ser_trainer = SerTrainer(
            model=ser_model,
            args=ser_args,
            train_dataset=ser_train_ds,
            eval_dataset=ser_val_ds,
            data_collator=ser_data_collator,
            # Again: no "processing_class" kwarg in Trainer; use "tokenizer" for the feature_extractor
            tokenizer=ser_feature_extractor,
            compute_metrics=compute_ser_metrics,
            class_weights=weights,  # device is aligned inside compute_loss
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=patience_ser),
                ClearCacheCallback()
            ],
        )



        ser_trainer.train()
        ser_metrics = ser_trainer.evaluate(eval_dataset=ser_val_ds)
        print(f"[SER][Trainer Eval] {ser_metrics}")

        ser_trainer.save_model(os.path.splitext(ckpt_ser)[0])
        ser_feature_extractor.save_pretrained(os.path.splitext(ckpt_ser)[0])
    else:
        print("[SER] Skipped (phase != 'all'/'ser').")

    # ── CLEANUP NCCL ───────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()

    




# --------------------------------------------------------------------------
# 7. Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR + SER Training Script")

    # single-GPU/CPU fallback
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    # batch-inference dir (optional)
    parser.add_argument(
        "--inference_dir",
        type=str,
        help="Directory for batch inference (optional)"
    )
    # injected by torchrun / torch.distributed.launch
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Process rank assigned by torchrun",
    )

    # ── ASR hyperparameters ──
    parser.add_argument("--asr_learning_rate", type=float, required=True)
    parser.add_argument("--asr_batch_size",    type=int,   required=True)
    parser.add_argument("--asr_epochs",        type=int,   required=True)
    parser.add_argument("--asr_patience",      type=int,   required=True)
    parser.add_argument("--asr_checkpoint",    type=str,   required=True)
    parser.add_argument("--asr_lang", choices=["en","de"], default="en")

    # ── SER hyperparameters ──
    parser.add_argument("--ser_learning_rate", type=float, required=True)
    parser.add_argument("--ser_batch_size",    type=int,   required=True)
    parser.add_argument("--ser_epochs",        type=int,   required=True)
    parser.add_argument("--ser_dropout",       type=float, required=True)
    parser.add_argument("--ser_patience",      type=int,   required=True)
    parser.add_argument("--ser_checkpoint",    type=str,   required=True)
    parser.add_argument("--ser_lang", choices=["en","de"], default="en")

    # --- add this just before args = parser.parse_args() ---
    parser.add_argument(
        "--phase",
        choices=["all", "asr", "ser"],
        default="all",
        help="Train only ASR, only SER, or both (default)."
    )


    args = parser.parse_args()

    # DDP / torchrun setup
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device(args.device)

    train_model(
        device        = device,
        lr_asr        = args.asr_learning_rate,
        bs_asr        = args.asr_batch_size,
        epochs_asr    = args.asr_epochs,
        patience_asr  = args.asr_patience,
        ckpt_asr      = args.asr_checkpoint,
        asr_lang      = args.asr_lang,

        lr_ser        = args.ser_learning_rate,
        bs_ser        = args.ser_batch_size,
        epochs_ser    = args.ser_epochs,
        dropout_ser   = args.ser_dropout,
        patience_ser  = args.ser_patience,
        ckpt_ser      = args.ser_checkpoint,
        ser_lang      = args.ser_lang,
        inference_dir = args.inference_dir,
        phase         = args.phase,
    )
