# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real_Time Inference. Martin Khadjavian © 

import os
import sys
import glob
import subprocess
import threading
import io
import numpy as np
import soundfile as sf
import webrtcvad
import librosa
from collections import defaultdict
from typing import List, Optional
import time
import json

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from flask import request as flask_request
import torch
import psutil
import shutil
import re


# Ensure your `src/` folder is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from io import BytesIO
from inference import infer_asr, infer_ser, batch_infer
from model_loader import load_asr_model, load_ser_model
from preprocessing import preprocess_data
from augmentation import augment_data
from inference import streaming_infer_asr as _stream_asr
from inference import streaming_infer_ser as _stream_ser
from inference import load_audio
from inference import vad_sentence_segments  
import tempfile, uuid, json as _json, os as _os
import tempfile as _tmp, uuid as _uuid
from pydub import AudioSegment
from pydub.utils import which

# Make sure pydub finds ffmpeg (Windows-friendly)
import os as _os
_ffmpeg_bin = which("ffmpeg") or _os.getenv("FFMPEG_BINARY")
if _ffmpeg_bin:
    AudioSegment.converter = _ffmpeg_bin  # let pydub use this ffmpeg binary

# NEW: import smoothing utilities + low-latency SER probs API + label map
from inference import (
    streaming_infer_asr,
    streaming_infer_ser,
    streaming_infer_ser_probs,
    SERSmoother,
    EMOTION_MAP,
)


# Declare global models
global asr_model, asr_processor, ser_model, ser_feature_extractor

# UTF-8 logging
sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Supported languages for transcription
transcription_languages = ["en", "de"]

# Device for Flask-side inference (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Flask] Using device for inference: {device}")

# -----------------------------------------------------------
#  LOAD THE ASR AND SER MODELS + PROCESSORS (with new-checkpoint support)
# -----------------------------------------------------------


def _infer_lang_from_name(name: str) -> str:
    m = re.search(r'(^|[_\-])(de|en)([_\-]|$)', name.lower())
    return m.group(2) if m else "en"


# NEW: Small helper to ensure a dir is a real HF model folder (prefer these over .pt)
def _looks_like_hf_model_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    has_hf_weights = (
        os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(path, "model.safetensors"))
    )
    has_lora_weights = (
        os.path.isfile(os.path.join(path, "adapter_model.bin"))
        or os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
    )
    has_adapter = os.path.isfile(os.path.join(path, "adapter_config.json"))
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_preproc = os.path.isfile(os.path.join(path, "preprocessor_config.json"))

    # Valid full HF model dir
    if has_config and has_hf_weights:
        return True

    # HF-ish root when we have processor + actual weights/adapter
    if has_preproc and (has_hf_weights or has_lora_weights or has_adapter):
        return True

    # preprocessor-only or adapter-only is NOT a full model
    return False


def _is_valid_checkpoint_base(base: str) -> bool:
    """
    Sanity-check whether a given checkpoint base has loadable ASR + SER models.

    We:
      - Require *_asr and *_ser artifacts to exist (.pt or HF dirs)
      - Try loading them on CPU via model_loader
      - For ASR, also check lm_head vs tokenizer vocab size
    """
    lang = _infer_lang_from_name(base)
    ckpt_root = "models/checkpoints"
    asr_dir = os.path.join(ckpt_root, f"{base}_asr")
    ser_dir = os.path.join(ckpt_root, f"{base}_ser")
    asr_pt = os.path.join(ckpt_root, f"{base}_asr.pt")
    ser_pt = os.path.join(ckpt_root, f"{base}_ser.pt")

    # Must have at least some ASR + SER artifact candidates
    if not (os.path.isdir(asr_dir) or os.path.isfile(asr_pt)):
        return False
    if not (os.path.isdir(ser_dir) or os.path.isfile(ser_pt)):
        return False

    # --- ASR sanity ---
    try:
        if _looks_like_hf_model_folder(asr_dir):
            m_asr, p_asr = load_asr_model(
                asr_dir, device="cpu", lang=lang, merge_peft_on_load=True
            )
        elif os.path.isfile(asr_pt):
            m_asr, p_asr = load_asr_model(
                asr_pt, device="cpu", lang=lang, merge_peft_on_load=True
            )
        else:
            return False

        inner = getattr(m_asr, "model", m_asr)
        lm_head = getattr(inner, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "out_features") and hasattr(p_asr, "tokenizer"):
            vocab_size = len(p_asr.tokenizer)
            if lm_head.out_features != vocab_size:
                print(f"[index] Base '{base}' rejected: ASR lm_head/vocab mismatch.")
                return False
    except Exception as e:
        print(f"[index] Skipping base '{base}' due to ASR load error: {e}")
        return False

    # --- SER sanity ---
    try:
        n_emotions = 6
        if _looks_like_hf_model_folder(ser_dir):
            m_ser, fe_ser = load_ser_model(
                ser_dir,
                n_emotions=n_emotions,
                device="cpu",
                lang=lang,
            )
        elif os.path.isfile(ser_pt):
            m_ser, fe_ser = load_ser_model(
                ser_pt,
                n_emotions=n_emotions,
                device="cpu",
                lang=lang,
            )
        else:
            return False
        # If this loads without error, treat SER side as valid.
    except Exception as e:
        print(f"[index] Skipping base '{base}' due to SER load error: {e}")
        return False

    return True


@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    global asr_model, asr_processor, ser_model, ser_feature_extractor

    existing = request.form.get("existing_checkpoint", "").strip()
    new_name = request.form.get("checkpoint_name", "").strip()
    asr_lang = request.form.get("asr_lang", "en")
    ser_lang = request.form.get("ser_lang", "en")

    if existing:
        base = existing
        try:
            load_checkpoint_from_name(base)
            return jsonify({"message": f"✅ Loaded checkpoint: {base}"}), 200
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return jsonify({"error": str(e)}), 500

    if new_name:
        base = new_name
        ckpt_root = "models/checkpoints"
        os.makedirs(ckpt_root, exist_ok=True)
        asr_dir = os.path.join(ckpt_root, f"{base}_asr")
        ser_dir = os.path.join(ckpt_root, f"{base}_ser")

        # Bootstrap ASR backbone from pretrained lang backbone into HF-style folder
        m_asr, p_asr = load_asr_model(
            asr_lang,
            device=device,
            lang=asr_lang,
            merge_peft_on_load=True,
        )
        os.makedirs(asr_dir, exist_ok=True)
        inner_asr = getattr(m_asr, "model", m_asr)
        if hasattr(inner_asr, "save_pretrained"):
            inner_asr.save_pretrained(asr_dir)
        if hasattr(p_asr, "save_pretrained"):
            p_asr.save_pretrained(asr_dir)

        # Bootstrap SER backbone into HF-style folder (Wav2Vec2SerModel supports save_pretrained)
        m_ser, fe_ser = load_ser_model(
            ser_lang,
            n_emotions=6,
            device=device,
            dropout=0.5,
            lang=ser_lang
        )
        os.makedirs(ser_dir, exist_ok=True)
        if hasattr(m_ser, "save_pretrained"):
            m_ser.save_pretrained(ser_dir)
        if hasattr(fe_ser, "save_pretrained"):
            fe_ser.save_pretrained(ser_dir)

        # now reload via the common path (uses all the heuristics)
        try:
            load_checkpoint_from_name(base)
            return jsonify({"message": f"✅ Created & loaded new checkpoint: {base}"}), 200
        except Exception as e:
            print(f"❌ Error loading freshly bootstrapped models: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Please select or name a checkpoint."}), 400


# -----------------------------------------------------------
#  RENDERING: HOMEPAGE
# -----------------------------------------------------------
@app.route("/")
def index():
    cp_root = "models/checkpoints"
    bases = set()
    if os.path.isdir(cp_root):
        for sub in os.listdir(cp_root):
            if sub.endswith("_asr"):
                base = sub[:-4]
                # Only expose bases that have structurally valid + loadable ASR & SER
                if _is_valid_checkpoint_base(base):
                    bases.add(base)
    return render_template("index.html", checkpoints=sorted(bases))


# -----------------------------------------------------------
#  TRAINING
# -----------------------------------------------------------
def stream_training_logs(process, log_path=None):
    """Read process stdout line by line and emit via SocketIO, optionally tee to a log file."""
    log_fh = None
    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # overwrite old log on each new run
            log_fh = open(log_path, "w", encoding="utf-8")

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if log_fh:
                log_fh.write(line + "\n")
                log_fh.flush()
            print("TRAIN LOG:", line)
            socketio.emit("training_logs", {"log": line})
    finally:
        process.stdout.close()
        if log_fh:
            log_fh.close()


@app.route('/train', methods=['POST'])
def train():
    """
    API endpoint to trigger separate ASR and SER training.
    Streams logs via SocketIO.
    """
    try:
        # 1) Gather hyperparameters
        requested_device = request.form.get("device", "cpu")  # "cuda" or "cpu" from HTML

        if requested_device == "cuda" and not torch.cuda.is_available():
            print("[Train] ⚠ Requested GPU but torch.cuda.is_available() is False. Falling back to CPU.")
            training_device = "cpu"
        else:
            training_device = requested_device

        print(f"[Train] Using device for training: {training_device}")

        hp = {
            "device": training_device,
            "asr_learning_rate": request.form["asr_learning_rate"],
            "asr_batch_size": request.form["asr_batch_size"],
            "asr_epochs": request.form["asr_epochs"],
            "asr_patience": request.form["asr_patience"],
            "asr_lang": request.form["asr_lang"],

            "ser_learning_rate": request.form["ser_learning_rate"],
            "ser_batch_size": request.form["ser_batch_size"],
            "ser_epochs": request.form["ser_epochs"],
            "ser_dropout": request.form["ser_dropout"],
            "ser_patience": request.form["ser_patience"],
            "ser_lang": request.form["ser_lang"],

            "existing_checkpoint": None,
            "checkpoint_name": None,

            "skip_preprocessing": request.form.get("skip_preprocessing", "off") == "on",
            "skip_preprocessing_asr": request.form.get("skip_preprocessing_asr", "off") == "on",
            "skip_preprocessing_emotion": request.form.get("skip_preprocessing_emotion", "off") == "on",
            "skip_splitting": request.form.get("skip_splitting", "off") == "on"
        }


        # 2) Validate language selection
        if hp["asr_lang"] not in transcription_languages or hp["ser_lang"] not in transcription_languages:
            return jsonify({"error": f"Languages must be in {transcription_languages}"}), 400

        # 3) Run preprocessing (optional)
        if not hp["skip_preprocessing"]:


            preprocess_data(
                skip_preprocessing_asr=hp["skip_preprocessing_asr"],
                skip_preprocessing_emotion=hp["skip_preprocessing_emotion"],
                skip_splitting=hp["skip_splitting"]
            )
        else:
            print("⚠ Skipping preprocessing as per request.")

        # —— CHECKPOINT HANDLING ——
        hp["existing_checkpoint"] = request.form.get("existing_checkpoint", "").strip()
        hp["checkpoint_name"] = request.form.get("checkpoint_name", "").strip()

        # Ensure checkpoint directory exists
        ckpt_dir = "models/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        # Decide which base name to use
        if hp["existing_checkpoint"]:
            base = hp["existing_checkpoint"]
        elif hp["checkpoint_name"]:
            base = hp["checkpoint_name"]
        else:
            raise ValueError("Please select an existing checkpoint or enter a new name.")

        # Build the .pt paths (train.py expects these)
        asr_ckpt = os.path.join(ckpt_dir, f"{base}_asr.pt")
        ser_ckpt = os.path.join(ckpt_dir, f"{base}_ser.pt")

        # ── BRAND-NEW RUN? Bootstrap initial checkpoints from raw backbones ─────────
        if not hp["existing_checkpoint"] and not os.path.exists(asr_ckpt):
            # Load raw (pretrained) backbones
            init_asr, _ = load_asr_model(
                hp["asr_lang"],
                device=training_device,
                lang=hp["asr_lang"],
                merge_peft_on_load=True,
            )
            init_ser, _ = load_ser_model(
                hp["ser_lang"],
                n_emotions=6,
                device=training_device,
                dropout=float(hp["ser_dropout"]),
                lang=hp["ser_lang"],
            )
            # Save their un-fine-tuned weights so train.py can pick them up
            torch.save(init_asr.state_dict(), asr_ckpt)
            torch.save(init_ser.state_dict(), ser_ckpt)
            print(f"[Setup] Initialized new checkpoints at {asr_ckpt} and {ser_ckpt}")

        # Build path for training log file (per checkpoint base)
        log_path = os.path.join("logs", f"train_{base}.log")

        # 4) Build subprocess command to train both ASR and SER
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, "src", "train.py")

        command = [
            sys.executable, train_script,
            f"--device={hp['device']}",

            # ASR args
            f"--asr_learning_rate={hp['asr_learning_rate']}",
            f"--asr_batch_size={hp['asr_batch_size']}",
            f"--asr_epochs={hp['asr_epochs']}",
            f"--asr_patience={hp['asr_patience']}",
            f"--asr_checkpoint={asr_ckpt}",
            f"--asr_lang={hp['asr_lang']}",

            # SER args
            f"--ser_learning_rate={hp['ser_learning_rate']}",
            f"--ser_batch_size={hp['ser_batch_size']}",
            f"--ser_epochs={hp['ser_epochs']}",
            f"--ser_dropout={hp['ser_dropout']}",
            f"--ser_patience={hp['ser_patience']}",
            f"--ser_checkpoint={ser_ckpt}",
            f"--ser_lang={hp['ser_lang']}",
        ]

        # --- LoRA for ASR (wired from frontend) -------------------------
        use_lora_asr = request.form.get("use_lora_asr") == "on"
        if use_lora_asr:
            command.append("--use_lora_asr")

            lora_r = (request.form.get("lora_r") or "8").strip()
            lora_alpha = (request.form.get("lora_alpha") or "32").strip()
            lora_dropout = (request.form.get("lora_dropout") or "0.1").strip()
            lora_targets = (request.form.get("lora_targets") or
                            "q_proj,k_proj,v_proj,out_proj,intermediate_dense,output_dense").strip()

            command.append(f"--lora_r={int(lora_r)}")
            command.append(f"--lora_alpha={int(lora_alpha)}")
            command.append(f"--lora_dropout={float(lora_dropout)}")
            command.append(f"--lora_targets={lora_targets}")
            command.append("--merge_lora_asr")

        # 5) Launch training subprocess and stream logs
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace"
        )
        threading.Thread(
            target=stream_training_logs,
            args=(process, log_path),
            daemon=True
        ).start()

        # Reload into memory for inference (will use best available checkpoint artifacts)
        load_checkpoint_from_name(base)

        return jsonify({"message": f"Training started successfully with checkpoint: {base}"}), 200

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
#  CHECKPOINT LOADING (robust + symmetric ASR/SER)
# -----------------------------------------------------------

def _assert_asr_alignment(model, processor, src: str):
    """Ensure ASR lm_head matches tokenizer vocab size."""
    inner = getattr(model, "model", model)
    lm_head = getattr(inner, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "out_features"):
        vocab_size = len(processor.tokenizer)
        if lm_head.out_features != vocab_size:
            raise RuntimeError(
                f"[{src}] ASR vocab/head mismatch: "
                f"lm_head={lm_head.out_features}, tokenizer={vocab_size}"
            )


def _assert_ser_alignment(model, n_emotions: int, src: str):
    """
    Ensure SER classifier head matches configured number of emotions.

    Compatible with Wav2Vec2SerModel(head) and generic classifiers.
    """
    # Prefer explicit head module (our Wav2Vec2SerModel)
    head = getattr(model, "head", None)
    if head is not None:
        last_linear = None
        if isinstance(head, torch.nn.Sequential):
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    last_linear = m
        elif isinstance(head, torch.nn.Linear):
            last_linear = head

        if last_linear is not None and hasattr(last_linear, "out_features"):
            if last_linear.out_features != n_emotions:
                raise RuntimeError(
                    f"[{src}] SER head mismatch: "
                    f"head={last_linear.out_features}, expected={n_emotions}"
                )
            return

    # Fallback for models with `classifier`
    classifier = getattr(model, "classifier", None)
    if classifier is not None and hasattr(classifier, "out_features"):
        if classifier.out_features != n_emotions:
            raise RuntimeError(
                f"[{src}] SER head mismatch: "
                f"classifier={classifier.out_features}, expected={n_emotions}"
            )


def load_checkpoint_from_name(base: str):
    """
    Load ASR + SER for a given checkpoint base name.

    Rules:
      - Prefer HF-style directories (<base>_asr/, <base>_ser/) if they look valid.
      - Otherwise fall back to legacy .pt checkpoints.
      - Run alignment checks so loader/processor mismatches fail loudly.
    """
    global asr_model, asr_processor, ser_model, ser_feature_extractor

    lang = _infer_lang_from_name(base)
    ckpt_root = "models/checkpoints"

    asr_dir = os.path.join(ckpt_root, f"{base}_asr")
    ser_dir = os.path.join(ckpt_root, f"{base}_ser")
    asr_pt = os.path.join(ckpt_root, f"{base}_asr.pt")
    ser_pt = os.path.join(ckpt_root, f"{base}_ser.pt")

    # ----- ASR -----
    last_err = None
    asr_ckpt = None

    if _looks_like_hf_model_folder(asr_dir):
        try:
            print(f"[load_checkpoint] Trying ASR HF dir: {asr_dir}")
            asr_model, asr_processor = load_asr_model(
                asr_dir,
                device=device,
                lang=lang,
                merge_peft_on_load=True,
            )
            _assert_asr_alignment(asr_model, asr_processor, asr_dir)
            asr_ckpt = asr_dir
        except Exception as e:
            print(f"[load_checkpoint] ASR HF dir failed, will try .pt: {e}")
            last_err = e

    if asr_ckpt is None and os.path.isfile(asr_pt):
        print(f"[load_checkpoint] Using ASR .pt: {asr_pt}")
        asr_model, asr_processor = load_asr_model(
            asr_pt,
            device=device,
            lang=lang,
            merge_peft_on_load=True,
        )
        _assert_asr_alignment(asr_model, asr_processor, asr_pt)
        asr_ckpt = asr_pt

    if asr_ckpt is None:
        raise FileNotFoundError(
            f"No valid ASR checkpoint for base '{base}'. Last error: {last_err}"
        )

    # ----- SER -----
    last_err = None
    ser_ckpt = None
    n_emotions = 6  # keep in sync with training/config

    if _looks_like_hf_model_folder(ser_dir):
        try:
            print(f"[load_checkpoint] Trying SER HF dir: {ser_dir}")
            ser_model, ser_feature_extractor = load_ser_model(
                ser_dir,
                n_emotions=n_emotions,
                device=device,
                dropout=0.5,
                lang=lang,
            )
            _assert_ser_alignment(ser_model, n_emotions, ser_dir)
            ser_ckpt = ser_dir
        except Exception as e:
            print(f"[load_checkpoint] SER HF dir failed, will try .pt: {e}")
            last_err = e

    if ser_ckpt is None and os.path.isfile(ser_pt):
        print(f"[load_checkpoint] Using SER .pt: {ser_pt}")
        ser_model, ser_feature_extractor = load_ser_model(
            ser_pt,
            n_emotions=n_emotions,
            device=device,
            dropout=0.5,
            lang=lang,
        )
        _assert_ser_alignment(ser_model, n_emotions, ser_pt)
        ser_ckpt = ser_pt

    if ser_ckpt is None:
        raise FileNotFoundError(
            f"No valid SER checkpoint for base '{base}'. Last error: {last_err}"
        )

    print(f"[load_checkpoint] Loaded ASR from {asr_ckpt}, SER from {ser_ckpt}")


@app.route('/train_status', methods=['GET'])
def train_status():
    """
    Check if the training process is still running.
    If no training process is found, return a status indicating training is complete.
    """
    training_running = False
    for process in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            if "train.py" in " ".join(process.info['cmdline']):
                training_running = True
                break
        except Exception:
            continue
    if training_running:
        return jsonify({"status": "Training is running"}), 200
    else:
        return jsonify({"status": "Training complete"}), 200


# -----------------------------------------------------------
#  PREPROCESS
# -----------------------------------------------------------
@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        skip_asr = request.form.get("skip_preprocessing_asr", "off") == "on"
        skip_ser = request.form.get("skip_preprocessing_emotion", "off") == "on"
        skip_splitting = request.form.get("skip_splitting", "off") == "on"

        preprocess_data(
            skip_preprocessing_asr=skip_asr,
            skip_preprocessing_emotion=skip_ser,
            skip_splitting=skip_splitting
        )

        return jsonify({"message": "Preprocessing completed. See console logs for details."})
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
#  AUGMENT
# -----------------------------------------------------------
@app.route('/augment', methods=['POST'])
def augment():
    """
    API endpoint to augment audio data.
    """
    input_dir = request.form.get('input_dir')
    output_dir = request.form.get('output_dir')
    text = request.form.get('text')  # Only used if TTS model is enabled
    emotion = request.form.get('emotion', "happy")

    if not input_dir or not output_dir:
        return jsonify({"error": "Missing input_dir or output_dir parameters"}), 400

    tts_model = None

    try:
        print(f"Starting augmentation: Input={input_dir}, Output={output_dir}, Emotion='{emotion}'")
        augment_data(input_dir, output_dir, tts_model=tts_model, text=text, emotion=emotion)
        print(f"✅ Augmentation completed. Files saved in {output_dir}.")
        return jsonify({"message": f"✅ Augmentation completed. Files saved in {output_dir}."})
    except Exception as e:
        print(f"❌ Error during augmentation: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
#  SINGLE-AUDIO FILE PROCESSING (STANDALONE MODE)
# -----------------------------------------------------------
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    API endpoint for ASR + SER inference on a single audio file.
    Expects JSON like: { "audio_path": "some/path.wav" }
    """
    data = request.get_json()
    audio_path = data.get('audio_path')
    if not audio_path:
        return jsonify({"error": "Missing audio_path parameter"}), 400

    try:
        transcription = infer_asr(audio_path, asr_model, asr_processor, device)
        emotion = infer_ser(audio_path, ser_model, ser_feature_extractor, device)

        return jsonify({
            "transcription": transcription,
            "emotion": emotion
        })

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
#  BATCH PROCESSING
# -----------------------------------------------------------
@app.route('/batch_process', methods=['POST'])
def batch_process():
    """
    API endpoint for batch inference using separate ASR and SER models.
    """
    audio_dir = request.form.get('audio_dir')
    if not audio_dir or not os.path.exists(audio_dir):
        return jsonify({"error": "Invalid or missing audio directory."}), 400

    try:
        results = []
        for file_name in os.listdir(audio_dir):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(audio_dir, file_name)
                transcription = infer_asr(file_path, asr_model, asr_processor, device)
                emotion = infer_ser(file_path, ser_model, ser_feature_extractor, device)
                results.append({
                    "file": file_name,
                    "transcription": transcription,
                    "emotion": emotion
                })
        return jsonify({"results": results})

    except Exception as e:
        print(f"Error during batch processing: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
#  FILE-UPLOAD INFERENCE ENDPOINT
# -----------------------------------------------------------
@app.route("/process_audio_inference", methods=["POST"])
def process_audio_inference():
    """
    API endpoint for audio file upload inference (ASR + SER).
    Saves result to a JSON file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    import tempfile, uuid, json as _json, os as _os
    ext = _os.path.splitext(audio_file.filename)[1]
    tmp_path = _os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ext)
    audio_file.save(tmp_path)

    output_file = _os.path.join("inference_results", str(uuid.uuid4()) + ".json")
    _os.makedirs("inference_results", exist_ok=True)

    seg_paths = []  # for cleanup when doing sentence-wise

    try:
        # ——— reload models based on user's selection ———
        model_choice = request.form.get("model_choice", "").strip()
        lang = _infer_lang_from_name(model_choice)
        if not model_choice:
            return jsonify({"error": "No model_choice provided"}), 400

        ckpt_root = "models/checkpoints"
        asr_dir = _os.path.join(ckpt_root, f"{model_choice}_asr")
        ser_dir = _os.path.join(ckpt_root, f"{model_choice}_ser")
        asr_file_pt = _os.path.join(ckpt_root, f"{model_choice}_asr.pt")
        ser_file_pt = _os.path.join(ckpt_root, f"{model_choice}_ser.pt")

        if _looks_like_hf_model_folder(asr_dir):
            asr_ckpt_dir = asr_dir
        elif _os.path.isfile(asr_file_pt):
            asr_ckpt_dir = asr_file_pt
        else:
            return jsonify({"error": f"No ASR checkpoint found for '{model_choice}'"}), 400

        if _looks_like_hf_model_folder(ser_dir):
            ser_ckpt_dir = ser_dir
        elif _os.path.isfile(ser_file_pt):
            ser_ckpt_dir = ser_file_pt
        else:
            return jsonify({"error": f"No SER checkpoint found for '{model_choice}'"}), 400

        # reload globals with proper device and n_emotions
        global asr_model, asr_processor, ser_model, ser_feature_extractor
        asr_model, asr_processor = load_asr_model(
            asr_ckpt_dir,
            device=device,
            lang=lang,
            merge_peft_on_load=True,
        )
        ser_model, ser_feature_extractor = load_ser_model(
            ser_ckpt_dir,
            n_emotions=6,
            device=device,
            lang=lang,
        )

        # NEW: sentence-wise option via WebRTC VAD segmentation
        sentencewise = request.form.get("sentencewise") == "on"
        if not sentencewise:
            transcription = infer_asr(tmp_path, asr_model, asr_processor, device)
            emotion = infer_ser(tmp_path, ser_model, ser_feature_extractor, device)

            result = {
                "transcription": transcription,
                "emotion": emotion,
                "output_file": output_file
            }
        else:
            segs = vad_sentence_segments(tmp_path)  # [(start, end), ...]


            # >>> FIX: use robust loader (handles m4a/mp3/flac/ogg via ffmpeg)
            x = load_audio(tmp_path, sr=16000)   # <-- CHANGED (was sf.read(...))
            if x is None or len(x) == 0:
                raise ValueError(f"Could not decode audio for VAD: {tmp_path}")
            sr = 16000

            if x.ndim > 1:
                x = x.mean(axis=-1)

            results = []
            for (s, e) in segs:
                i0, i1 = int(s * sr), int(e * sr)
                seg = x[i0:i1].astype("float32")
                seg_path = _os.path.join(_tmp.gettempdir(), f"seg_{_uuid.uuid4().hex}.wav")
                sf.write(seg_path, seg, sr)
                seg_paths.append(seg_path)

                asr_txt = infer_asr(seg_path, asr_model, asr_processor, device)
                emo = infer_ser(seg_path, ser_model, ser_feature_extractor, device)
                results.append({
                    "start_sec": float(s),
                    "end_sec": float(e),
                    "transcription": asr_txt,
                    "emotion": emo
                })

            result = {"segments": results, "output_file": output_file}

        with open(output_file, "w", encoding="utf-8") as f:
            _json.dump(result, f, indent=2)

        return jsonify(result), 200

    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        for p in seg_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# -----------------------------------------------------------
#  NEW: METRICS ENDPOINT (ASR + SER)
# -----------------------------------------------------------
@app.route("/last_metrics", methods=["GET"])
def last_metrics():
    """
    Returns combined ASR/SER metrics JSON for the given model_choice.
    Looks under models/checkpoints/<name>_{asr,ser}/metrics.json
    """
    model_choice = request.args.get("model_choice", "").strip()
    if not model_choice:
        return jsonify({"error": "model_choice is required"}), 400

    ckpt_root = "models/checkpoints"
    asr_dir = os.path.join(ckpt_root, f"{model_choice}_asr")
    ser_dir = os.path.join(ckpt_root, f"{model_choice}_ser")
    out = {"model_choice": model_choice, "asr": None, "ser": None}

    try:
        asr_metrics_p = os.path.join(asr_dir, "metrics.json")
        if os.path.isfile(asr_metrics_p):
            with open(asr_metrics_p, "r", encoding="utf-8") as f:
                out["asr"] = json.load(f)
    except Exception:
        pass

    try:
        ser_metrics_p = os.path.join(ser_dir, "metrics.json")
        if os.path.isfile(ser_metrics_p):
            with open(ser_metrics_p, "r", encoding="utf-8") as f:
                out["ser"] = json.load(f)
    except Exception:
        pass

    if out["asr"] is None and out["ser"] is None:
        return jsonify({"error": "No metrics found for this checkpoint"}), 404
    return jsonify(out)


# -----------------------------------------------------------
#  REAL-TIME STREAMING FOR UNITY (WebSocket)
# -----------------------------------------------------------
stored_results = []  # Optional: Keep a list in memory
streaming_model_loaded = {"name": None}  # Global tracker for currently loaded checkpoint

# --- Streaming segmentation (VAD) state ---
VAD = webrtcvad.Vad(2)  # 0..3, 2 is a good default
STREAM_BUFFERS = defaultdict(list)        # sid -> [np.float32 chunks]
STREAM_SILENCE_MS = defaultdict(int)     # sid -> accumulated ms of silence
FRAME_MS = 30
ENDPOINT_SIL_MS = 600
TARGET_SR = 16000

# --- NEW: utterance state & thresholds for robust endpointing ---
IN_UTT = defaultdict(bool)
VOICED_MS = defaultdict(int)
UTT_MS = defaultdict(int)
MIN_UTT_MS = 700
MIN_VOICED_MS = 300
MAX_UTT_MS = 15000

# ──────────────────────────────────────────────────────────────
# NEW: Prefix-commit state + helpers (LCP over tokenized text)
# ──────────────────────────────────────────────────────────────
ASR_PREFIX_STATE = defaultdict(lambda: {
    "last_tokens": [],
    "committed_tokens": [],
})

# ──────────────────────────────────────────────────────────────
# NEW: SER smoothing state (per-session EMA + majority vote)
# ──────────────────────────────────────────────────────────────
SER_SMOOTHERS = defaultdict(
    lambda: SERSmoother(
        n_classes=6,
        ema_half_life_s=1.0,
        majority_k=5,
        dt_s_default=0.5
    )
)

# ──────────────────────────────────────────────────────────────
# NEW: Per-session window index + JSONL logging setup
# ──────────────────────────────────────────────────────────────
WIN_IDX = defaultdict(int)
LOG_DIR = "logs"
JSONL_PATH = os.path.join(LOG_DIR, "inference_windows.jsonl")
DEFAULT_BEAM_WIDTH = int(os.getenv("SPEEMO_ASR_BEAM", "0"))         # 0 → greedy
DEFAULT_LM_ALPHA = float(os.getenv("SPEEMO_ASR_LM_ALPHA", "0.6"))
DEFAULT_LM_BETA = float(os.getenv("SPEEMO_ASR_LM_BETA", "1.2"))


def _append_jsonl_line(record: dict):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _tokenize_asr_text(text: str) -> List[str]:
    return [t for t in text.strip().split() if t]


def _lcp_tokens(a: List[str], b: List[str]) -> List[str]:
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


WINDOW_SEC = 1.2


def _current_window_bytes(sid: str, tail_sec: float = WINDOW_SEC) -> bytes:
    frames = STREAM_BUFFERS[sid]
    if not frames:
        return b""
    hop = int(FRAME_MS * TARGET_SR / 1000)
    frames_needed = max(1, int(tail_sec * 1000 / FRAME_MS))
    slice_frames = frames[-frames_needed:] if len(frames) > frames_needed else frames
    x = np.concatenate(slice_frames).astype(np.float32) if slice_frames else np.zeros(0, dtype=np.float32)
    return x.astype(np.float32, copy=False).tobytes()


# -----------------------------------------------------------
#  >>> FIXED: Robust byte→float32 mono 16k decoder for real-time chunks
# -----------------------------------------------------------

def _decode_to_f32_mono_16k(blob: bytes, fmt: Optional[str] = None) -> np.ndarray:
    """
    If client says it's a container (webm/ogg/mp4/m4a/mp3/wav/aac/opus), decode via ffmpeg/pydub.
    Only try the raw float32 fast-path when format is *not* explicitly a container.
    Return mono float32 at 16 kHz.
    """
    fmt = (fmt or "").lower().strip()
    container_like = {"webm", "ogg", "mp4", "m4a", "mp3", "wav", "aac", "opus"}


    try_raw_first = fmt not in container_like

    # 1) Raw float32 fast-path (only if not told it's a container)
    if try_raw_first and blob and (len(blob) % 4 == 0):
        try:
            raw = np.frombuffer(blob, dtype=np.float32)
            # reject NaN/Inf/huge values
            if raw.size > 160 and np.isfinite(raw).all() and np.max(np.abs(raw)) <= 10.0:
                return raw.astype(np.float32, copy=False)
        except Exception:
            pass

    # 2) Container decode via ffmpeg (pydub)
    bio = BytesIO(blob)
    if fmt == "opus":
        trial_formats = ["webm", "ogg", "mp4", "m4a", "wav", "mp3"]
    elif fmt:
        trial_formats = [fmt]
    else:
        trial_formats = ["webm", "ogg", "mp4", "m4a", "mp3", "wav", "flac"]

    for f in trial_formats:
        try:
            seg = AudioSegment.from_file(bio, format=f)
            seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
            arr = np.array(seg.get_array_of_samples())
            sw = seg.sample_width  # bytes per sample
            if sw == 1:
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            elif sw == 2:
                arr = arr.astype(np.float32) / 32768.0
            elif sw == 4:
                arr = arr.astype(np.float32) / 2147483648.0
            else:
                arr = arr.astype(np.float32)
            return arr
        except Exception:
            bio.seek(0)

    raise ValueError("Unsupported container/codec or missing ffmpeg.")


def _endpoint_sentences(sid: str, x16k_f32: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if x16k_f32.size == 0:
        return out

    hop = int(FRAME_MS * TARGET_SR / 1000)
    buf = STREAM_BUFFERS[sid]
    silence = STREAM_SILENCE_MS[sid]
    in_utt = IN_UTT[sid]
    voiced_ms = VOICED_MS[sid]
    utt_ms = UTT_MS[sid]

    i = 0
    while i + hop <= x16k_f32.size:
        frame = x16k_f32[i:i + hop]
        # sanitize for VAD: finite & clipped to [-1, 1] before scaling
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        frame = np.clip(frame, -1.0, 1.0)
        pcm16 = (frame * 32767.0).astype(np.int16).tobytes()
        voiced = VAD.is_speech(pcm16, TARGET_SR)

        if not in_utt:
            if voiced:
                in_utt = True
                buf.clear()
                buf.append(frame)
                silence = 0
                voiced_ms = FRAME_MS
                utt_ms = FRAME_MS
        else:
            buf.append(frame)
            utt_ms += FRAME_MS
            if voiced:
                voiced_ms += FRAME_MS
                silence = 0
            else:
                silence += FRAME_MS

            if silence >= ENDPOINT_SIL_MS or utt_ms >= MAX_UTT_MS:
                trim_frames = min(len(buf), max(0, silence // FRAME_MS))
                effective = buf[:-trim_frames] if trim_frames > 0 else buf

                if (utt_ms - silence) >= MIN_UTT_MS and voiced_ms >= MIN_VOICED_MS and len(effective) > 0:
                    sent = np.concatenate(effective).astype(np.float32)
                    peak = float(np.max(np.abs(sent))) if sent.size else 0.0
                    rms = float(np.sqrt(np.mean(sent ** 2))) if sent.size else 0.0
                    if peak > 1e-3 and rms > 1e-4:
                        sent = sent / max(peak, 1e-6) * 0.9
                        out.append(sent)

                buf.clear()
                in_utt = False
                silence = 0
                voiced_ms = 0
                utt_ms = 0

        i += hop

    STREAM_BUFFERS[sid] = buf
    STREAM_SILENCE_MS[sid] = silence
    IN_UTT[sid] = in_utt
    VOICED_MS[sid] = voiced_ms
    UTT_MS[sid] = utt_ms
    return out


@socketio.on('unity_audio_chunk')
def handle_unity_audio_chunk(data):
    """
    Receives mic chunks (raw f32 or container), does VAD-based sentence endpointing,
    decodes each completed sentence with ASR + SER, emits JSON results.
    """
    try:
        audio_bytes = data.get("audio", None)
        model_choice = data.get("model_choice", None)
        meta = data.get("meta", {"format": "f32le", "sr": 16000, "ch": 1})

        if not audio_bytes or not model_choice:
            emit("asr_emotion_result", {"error": "Missing audio or model_choice."})
            return

        # ensure the right checkpoint is in memory
        if streaming_model_loaded["name"] != model_choice:
            try:
                load_checkpoint_from_name(model_choice)
                streaming_model_loaded["name"] = model_choice
                print(f"✅ Loaded streaming model: {model_choice}")
            except Exception as e:
                emit("asr_emotion_result", {"error": f"Model load failed: {str(e)}"})
                return

        # >>> FIX: decode bytes robustly (raw f32 → container fallback)
        fmt = (meta.get("format") or "").lower()
        if fmt == "opus":
            fmt = "ogg"  # opus often inside ogg/webm
        x16k = _decode_to_f32_mono_16k(audio_bytes, fmt)

        # VAD segmentation → sentence chunks
        sid = flask_request.sid
        sentences = _endpoint_sentences(sid, x16k)

        # prefix-commit for in-progress utterance
        if IN_UTT[sid]:
            win_bytes = _current_window_bytes(sid, tail_sec=WINDOW_SEC)
            if win_bytes:
                hyp_text = streaming_infer_asr(win_bytes, asr_model, asr_processor, device)
                hyp_tokens = _tokenize_asr_text(hyp_text)
                state = ASR_PREFIX_STATE[sid]
                lcp = _lcp_tokens(state["last_tokens"], hyp_tokens)
                new_stable = lcp[len(state["committed_tokens"]):]
                if new_stable:
                    partial_text = " ".join(new_stable)
                    emit("asr_emotion_result", {
                        "transcription": partial_text,
                        "emotion": None,
                        "final": False
                    }, room=sid)
                    state["committed_tokens"].extend(new_stable)
                state["last_tokens"] = hyp_tokens

        for sent in sentences:
            if np.max(np.abs(sent)) < 1e-3 or np.sqrt(np.mean(sent ** 2)) < 1e-4:
                continue

            WIN_IDX[sid] += 1
            win_idx = WIN_IDX[sid]

            sent_bytes = sent.tobytes()

            beam = DEFAULT_BEAM_WIDTH
            alpha = DEFAULT_LM_ALPHA
            beta = DEFAULT_LM_BETA

            asr_t0 = time.perf_counter()
            transcription = _stream_asr(
                sent_bytes,
                asr_model,
                asr_processor,
                device,
                beam_width=beam,
                alpha=alpha,
                beta=beta
            )
            asr_e2e_ms = (time.perf_counter() - asr_t0) * 1000.0

            state = ASR_PREFIX_STATE[sid]
            final_tokens = _tokenize_asr_text(transcription)
            committed = state["committed_tokens"]
            if committed and final_tokens[:len(committed)] == committed:
                suffix_tokens = final_tokens[len(committed):]
                if suffix_tokens:
                    emit("asr_emotion_result", {
                        "transcription": " ".join(suffix_tokens),
                        "emotion": None,
                        "final": False
                    }, room=sid)

            ser_probs_t0 = time.perf_counter()
            raw_probs = streaming_infer_ser_probs(
                sent_bytes,
                ser_model,
                ser_feature_extractor,
                device
            )
            ser_probs_ms = (time.perf_counter() - ser_probs_t0) * 1000.0

            smoother = SER_SMOOTHERS[sid]
            approx_dt_s = max(1e-3, len(sent) / float(TARGET_SR))

            smooth_t0 = time.perf_counter()
            smoothed_dist = smoother.update_probs(raw_probs, dt_s=approx_dt_s)
            smoother.update_label(int(np.argmax(raw_probs)))
            majority_dist = smoother.majority_distribution()
            if majority_dist is not None:
                final_dist = majority_dist
            else:
                final_dist = smoothed_dist
            ser_smooth_ms = (time.perf_counter() - smooth_t0) * 1000.0

            smoothed_emotion_id = int(np.argmax(final_dist))
            smoothed_emotion = EMOTION_MAP.get(smoothed_emotion_id, "unknown")

            ser_unsmoothed_t0 = time.perf_counter()
            emotion = _stream_ser(sent_bytes, ser_model, ser_feature_extractor, device)
            ser_unsmoothed_ms = (time.perf_counter() - ser_unsmoothed_t0) * 1000.0

            result = {
                "transcription": transcription,
                "emotion": smoothed_emotion,
                "final": True,
                "unsmoothed_emotion": emotion
            }
            stored_results.append(result)
            emit("asr_emotion_result", result)

            _append_jsonl_line({
                "mode": "stream",
                "sid": sid,
                "win_idx": win_idx,
                "len_ms": round(len(sent) * 1000.0 / TARGET_SR, 3),
                "t_asr_e2e_ms": round(asr_e2e_ms, 3),
                "t_ser_probs_ms": round(ser_probs_ms, 3),
                "t_ser_smooth_ms": round(ser_smooth_ms, 3),
                "t_ser_unsmoothed_ms": round(ser_unsmoothed_ms, 3),
                "beam": int(beam),
                "alpha": float(alpha),
                "beta": float(beta),
                "ema": float(getattr(smoother, "ema_half_life_s", 0.0)),
            })

            ASR_PREFIX_STATE.pop(sid, None)
            SER_SMOOTHERS[sid].reset()

    except Exception as e:
        print("❌ Error in handle_unity_audio_chunk:", e)
        emit("asr_emotion_result", {"error": str(e)})


# -----------------------------------------------------------
#  MAIN
# -----------------------------------------------------------
if __name__ == '__main__':
    print("Starting Flask + SocketIO app...")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
