# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import os 
import re
import glob
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import json
from transformers import Wav2Vec2Processor
from model_loader import load_asr_model, load_ser_model
from pydub import AudioSegment
import soundfile as sf
import webrtcvad
from collections import deque
import math
import time
from io import BytesIO  # NEW: for decoding webm/mp3/m4a bytes via pydub

# NEW: beam-search decoder support
try:
    from pyctcdecode import build_ctcdecoder
    _USE_CTCDECODER = True
except ImportError:
    build_ctcdecoder = None
    _USE_CTCDECODER = False

# make sure pydub can find ffmpeg (optional but helpful on Windows)
from pydub.utils import which
# NEW: discover both ffmpeg and ffprobe, honor common env vars
_ffmpeg = which("ffmpeg") or os.getenv("FFMPEG_BINARY") or os.getenv("FFMPEG_PATH")
_ffprobe = which("ffprobe") or os.getenv("FFPROBE_BINARY") or os.getenv("FFPROBE_PATH")
if _ffmpeg:
    AudioSegment.converter = _ffmpeg  # pydub will call this ffmpeg binary
    AudioSegment.ffmpeg = _ffmpeg     # explicit for newer pydub
if _ffprobe:
    AudioSegment.ffprobe = _ffprobe

# ──────────────────────────────────────────────────────────────────────────────
# SER smoothing helper (EMA + majority)
# ──────────────────────────────────────────────────────────────────────────────
class SERSmoother:
    """
    Keeps streaming SER smoothing state.
    - EMA over probability vectors using half-life parameterization.
    - Majority vote over last k labels (returns a distribution when queried).
    """
    def __init__(self, n_classes: int, ema_half_life_s: float = None,
                 majority_k: int = 0, dt_s_default: float = None):
        self.n_classes = n_classes
        self.ema_half_life_s = ema_half_life_s
        self.majority_k = int(majority_k or 0)
        self.dt_s_default = dt_s_default
        self.prev_probs = None
        self.labels = deque(maxlen=self.majority_k if self.majority_k > 0 else 0)

    @staticmethod
    def _lambda_from_half_life(half_life_s: float, dt_s: float) -> float:
        if half_life_s is None or half_life_s <= 0 or dt_s is None or dt_s <= 0:
            return None
        # standard half-life parameterization
        return 1.0 - math.exp(-math.log(2.0) * (dt_s / half_life_s))

    def update_probs(self, probs: np.ndarray, dt_s: float = None) -> np.ndarray:
        if probs.ndim != 1 or probs.shape[0] != self.n_classes:
            raise ValueError("probs must be a 1D array of length n_classes")

        if self.ema_half_life_s and self.ema_half_life_s > 0:
            lam = self._lambda_from_half_life(
                self.ema_half_life_s,
                dt_s or self.dt_s_default
            )
            if lam is None or self.prev_probs is None:
                smoothed = probs
            else:
                smoothed = lam * probs + (1.0 - lam) * self.prev_probs
            self.prev_probs = smoothed
            return smoothed
        else:
            self.prev_probs = probs
            return probs

    def update_label(self, label: int) -> None:
        if self.majority_k > 0:
            self.labels.append(int(label))

    def majority_distribution(self) -> np.ndarray:
        if self.majority_k <= 0 or len(self.labels) == 0:
            return None
        counts = np.bincount(list(self.labels),
                             minlength=self.n_classes).astype(np.float32)
        total = float(counts.sum())
        if total <= 0:
            return np.ones(self.n_classes, dtype=np.float32) / float(self.n_classes)
        return counts / total

    def reset(self):
        self.prev_probs = None
        self.labels.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window SER (with optional smoothing)
# ──────────────────────────────────────────────────────────────────────────────
def sliding_window_emotion(
    model,
    waveform: torch.Tensor,
    max_frames: int,
    hop: int = None,
    device="cpu",
    smoothing_mode: str = None,     # None | "ema" | "majority"
    ema_half_life_s: float = None,  # seconds; if smoothing_mode == "ema"
    majority_k: int = 0,            # if smoothing_mode == "majority"
    sample_rate: int = 16000,
    return_timings: bool = False
):
    if hop is None:
        hop = max_frames // 2

    L = waveform.shape[-1]
    if L < max_frames:
        padded = F.pad(waveform, (0, max_frames - L))
        windows = [padded]
    else:
        windows = []
        for start in range(0, L, hop):
            end = start + max_frames
            if end > L:
                start = L - max_frames
                end = L
            windows.append(waveform[start:end])

    probs = []
    t_enc_ms = 0.0
    t_dec_ms = 0.0

    model.eval()
    with torch.no_grad():
        for w in windows:
            inp = w.unsqueeze(0).to(device)  # [1, T]
            tic = time.perf_counter()
            output = model(inp)
            logits = output.logits if hasattr(output, "logits") else output
            # NEW: guard against BaseModelOutput / non-tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"[sliding_window_emotion] Expected Tensor/logits from SER model, "
                    f"got {type(logits)}. Check SER model / LoRA loading."
                )
            if logits.dim() == 3:
                logits = logits.mean(dim=1)
            t_enc_ms += (time.perf_counter() - tic) * 1000.0
            probs.append(torch.softmax(logits, dim=-1).cpu())

    # no smoothing → mean
    if smoothing_mode is None:
        tic = time.perf_counter()
        out_tensor = torch.stack(probs, dim=0).mean(dim=0)
        t_dec_ms += (time.perf_counter() - tic) * 1000.0
        if return_timings:
            return out_tensor, {
                "t_enc_ms": t_enc_ms,
                "t_dec_ms": t_dec_ms,
                "ema_half_life_s": None,
                "ema_decay": None,
                "smoothing_mode": "mean",
            }
        return out_tensor

    # smoothing modes
    np_probs = [p.squeeze(0).numpy() for p in probs]
    n_classes = np_probs[0].shape[0]
    dt_s = hop / float(sample_rate) if sample_rate and hop else None
    smoother = SERSmoother(
        n_classes=n_classes,
        ema_half_life_s=(ema_half_life_s if smoothing_mode == "ema" else None),
        majority_k=(majority_k if smoothing_mode == "majority" else 0),
        dt_s_default=dt_s,
    )

    last_smoothed = None
    tic = time.perf_counter()
    for arr in np_probs:
        last_smoothed = smoother.update_probs(arr, dt_s)
        if smoothing_mode == "majority":
            smoother.update_label(int(arr.argmax()))
    t_dec_ms += (time.perf_counter() - tic) * 1000.0

    if smoothing_mode == "ema":
        ema_decay = (
            SERSmoother._lambda_from_half_life(ema_half_life_s, dt_s)
            if ema_half_life_s
            else None
        )
        out = last_smoothed if last_smoothed is not None else np_probs[-1]
        out_tensor = torch.from_numpy(out).float()
        if return_timings:
            return out_tensor, {
                "t_enc_ms": t_enc_ms,
                "t_dec_ms": t_dec_ms,
                "ema_half_life_s": ema_half_life_s,
                "ema_decay": ema_decay,
                "smoothing_mode": "ema",
            }
        return out_tensor

    if smoothing_mode == "majority":
        dist = smoother.majority_distribution()
        if dist is None:
            out = np.mean(np_probs, axis=0)
            mode = "mean"
        else:
            out = dist
            mode = "majority"
        out_tensor = torch.from_numpy(out).float()
        if return_timings:
            return out_tensor, {
                "t_enc_ms": t_enc_ms,
                "t_dec_ms": t_dec_ms,
                "ema_half_life_s": None,
                "ema_decay": None,
                "smoothing_mode": mode,
            }
        return out_tensor

    # fallback
    out_tensor = torch.stack(probs, dim=0).mean(dim=0)
    if return_timings:
        return out_tensor, {
            "t_enc_ms": t_enc_ms,
            "t_dec_ms": t_dec_ms,
            "ema_half_life_s": None,
            "ema_decay": None,
            "smoothing_mode": "mean",
        }
    return out_tensor


# ──────────────────────────────────────────────────────────────────────────────
# ASR decoder / LM config
# ──────────────────────────────────────────────────────────────────────────────
ASR_DECODER = None
LM_MODEL_PATH = os.path.join("models", "lm", "4gram.arpa")

# Optional: cache config for decoder
_DECODER_CFG = None  # (tokens_hash, lm_path, alpha, beta)

# Env defaults
DEFAULT_BEAM_WIDTH = int(os.getenv("SPEEMO_ASR_BEAM", "0"))        # 0 → greedy
DEFAULT_LM_ALPHA = float(os.getenv("SPEEMO_ASR_LM_ALPHA", "0.6"))
DEFAULT_LM_BETA = float(os.getenv("SPEEMO_ASR_LM_BETA", "1.2"))
LM_MODEL_PATH = os.getenv("SPEEMO_KENLM_ARPA", LM_MODEL_PATH)


EMOTION_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
}


def _labels_from_processor(processor: Wav2Vec2Processor):
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    tokens = [tok for tok, _ in sorted_vocab]
    # Turn HF word_delimiter_token (often '|') into space for pyctcdecode
    wdt = getattr(processor.tokenizer, "word_delimiter_token", None)
    if wdt and wdt in tokens:
        idx = tokens.index(wdt)
        tokens[idx] = " "
    return tokens


def _get_ctc_decoder(processor: Wav2Vec2Processor,
                     kenlm_model_path: str,
                     alpha: float,
                     beta: float):
    """
    Build/reuse a pyctcdecode decoder if LM + pyctcdecode are available.
    """
    global ASR_DECODER, _DECODER_CFG
    if (not _USE_CTCDECODER) or (not os.path.exists(kenlm_model_path)):
        return None

    tokens = _labels_from_processor(processor)
    tokens_hash = hash(tuple(tokens))
    cfg = (tokens_hash, kenlm_model_path, float(alpha), float(beta))

    if ASR_DECODER is None or _DECODER_CFG != cfg:
        ASR_DECODER = build_ctcdecoder(tokens, kenlm_model_path,
                                       alpha=alpha, beta=beta)
        _DECODER_CFG = cfg
    return ASR_DECODER


# ──────────────────────────────────────────────────────────────────────────────
# JSONL logging helpers/state
# ──────────────────────────────────────────────────────────────────────────────
_LOG_DIR = "logs"
_JSONL_PATH = os.path.join(_LOG_DIR, "inference_windows.jsonl")

_ASR_FILE_WIN_IDX = 0
_ASR_STREAM_WIN_IDX = 0
_SER_FILE_WIN_IDX = 0
_SER_STREAM_WIN_IDX = 0


def _append_infer_log(record: dict):
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # never crash on logging
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Audio loading  (robust: m4a/mp3/flac/ogg via pydub+ffmpeg, wav via librosa)
# ──────────────────────────────────────────────────────────────────────────────
def load_audio(file_path, sr=16000):
    """
    Load mono audio at a target sample rate.
    - WAV/FLAC/OGG: try librosa first
    - Everything else (e.g., M4A/MP3/MP4/AAC/WEBM) → pydub + ffmpeg (with explicit format hint)
    - If librosa fails for any reason, fall back to pydub
    Returns float32 numpy array in [-1, 1].
    """
    ext = os.path.splitext(file_path)[1].lower()
    # NEW: map tricky extensions to container/codec name ffmpeg expects
    _FORMAT_HINT = {
        ".m4a": "mp4",
        ".mp4": "mp4",
        ".aac": "aac",
        ".mp3": "mp3",
        ".webm": "webm",
        ".ogg": "ogg",
        ".wav": "wav",
        ".flac": "flac",
    }

    def _pydub_decode(path, fmt_hint=None):
        # let ffmpeg auto-detect OR use a safe explicit format for tricky containers
        if fmt_hint:
            seg = AudioSegment.from_file(path, format=fmt_hint)
        else:
            seg = AudioSegment.from_file(path)
        if seg.channels != 1:
            seg = seg.set_channels(1)
        if seg.frame_rate != sr:
            seg = seg.set_frame_rate(sr)
        # scale integer samples to [-1,1] float32
        arr = np.array(seg.get_array_of_samples())
        sw = seg.sample_width  # bytes per sample
        if sw == 1:
            # 8-bit unsigned -> center and scale
            arr = (arr.astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            arr = arr.astype(np.float32) / 32768.0
        elif sw == 4:
            arr = arr.astype(np.float32) / 2147483648.0
        else:
            arr = arr.astype(np.float32)
        return arr

    try:
        # Prefer librosa only for formats libsndfile usually supports
        if ext in (".wav", ".flac", ".ogg"):
            y, _ = librosa.load(file_path, sr=sr, mono=True)
            return y.astype(np.float32)
        # Everything else (e.g., .m4a, .mp3, .aac, .mp4, .webm) → pydub/ffmpeg
        return _pydub_decode(file_path, _FORMAT_HINT.get(ext))
    except Exception as e_lib:
        # Fallback: try pydub regardless of extension, try with and without hint
        try:
            return _pydub_decode(file_path, _FORMAT_HINT.get(ext))
        except Exception:
            try:
                return _pydub_decode(file_path, None)
            except Exception as e_pydub:
                print(f"Error loading audio file {file_path}: librosa={e_lib} | pydub={e_pydub}")
                return None

# NEW: robust bytes decoder for streaming (handles raw float32 or container bytes like WEBM/MP3/M4A)
def decode_audio_bytes(audio_bytes: bytes, sr: int = 16000, fmt_hint: str = None) -> np.ndarray:
    """
    Try interpreting bytes as raw float32 first; if that fails, decode via pydub/ffmpeg.
    Returns mono float32 np.array at given sr.
    """
    # Attempt raw float32 path (Unity / custom clients sometimes send raw PCM32F)
    try:
        if len(audio_bytes) % 4 == 0:
            raw = np.frombuffer(audio_bytes, dtype=np.float32)
            # Heuristic: if it's too tiny or all zeros, treat as invalid and try container
            if raw.size > 160 and not (np.max(raw) == 0.0 and np.min(raw) == 0.0):
                return raw.astype(np.float32, copy=False)
    except Exception:
        pass

    # Container decode path (webm/opus, mp3, m4a, wav, ...)
    bio = BytesIO(audio_bytes)
    # If caller didn't hint, try a common-first cascade
    trial_formats = [fmt_hint] if fmt_hint else ["webm", "ogg", "mp4", "m4a", "mp3", "wav", "flac"]
    for fmt in trial_formats:
        try:
            seg = AudioSegment.from_file(bio, format=fmt) if fmt else AudioSegment.from_file(bio)
            seg = seg.set_channels(1).set_frame_rate(sr)
            arr = np.array(seg.get_array_of_samples())
            sw = seg.sample_width
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
            continue
    raise ValueError("Unsupported container/codec in audio_bytes. Ensure ffmpeg is installed and on PATH.")

# ──────────────────────────────────────────────────────────────────────────────
# Sentence-wise VAD segmentation (uses load_audio; works for m4a/wav/mp3)
# ──────────────────────────────────────────────────────────────────────────────
MAX_FRAMES = 4 * 16000  # used also for SER sliding windows

def vad_sentence_segments(path,
                          target_sr=16000,
                          frame_ms=30,
                          end_sil_ms=600,
                          min_speech_ms=300,
                          merge_gap_ms=200):
    """
    Return [(start_sec, end_sec), ...] sentence-like segments using WebRTC VAD.

    Uses load_audio() so it works for WAV, M4A, MP3 (requires ffmpeg for compressed).
    """
    audio = load_audio(path, sr=target_sr)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Could not load audio for VAD from: {path}")

    x = audio.astype("float32")
    sr = target_sr

    # WebRTC VAD expects 16-bit PCM mono at 8/16/32/48 kHz and 10/20/30 ms frames
    # (we convert each float frame back to int16 before calling is_speech).
    vad = webrtcvad.Vad(2)
    hop = int(frame_ms * sr / 1000)

    voiced_flags = []
    frames = []

    if hop <= 0 or len(x) < hop:
        # Too short; just treat as one segment
        return [(0.0, float(len(x) / sr))]

    for i in range(0, len(x) - hop + 1, hop):
        frame = x[i:i + hop]
        pcm16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        voiced = vad.is_speech(pcm16, sr)
        voiced_flags.append(voiced)
        frames.append((i, i + hop))

    segs = []
    start = None
    sil = 0
    for (s, e), v in zip(frames, voiced_flags):
        if v and start is None:
            start, sil = s, 0
        if start is not None:
            sil = 0 if v else sil + frame_ms
            if sil >= end_sil_ms:
                if (e - start) >= int(min_speech_ms * sr / 1000):
                    end = e - int(sil * sr / 1000)
                    segs.append((start / sr, end / sr))
                start, sil = None, 0

    # merge close segments
    merged = []
    for s, e in segs:
        if not merged:
            merged.append([s, e])
            continue
        if s - merged[-1][1] <= merge_gap_ms / 1000.0:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(float(s), float(e)) for s, e in merged] or [(0.0, float(len(x) / sr))]


# ──────────────────────────────────────────────────────────────────────────────
# NEW: ASR sentence-VAD helper (segment → decode → join)
# ──────────────────────────────────────────────────────────────────────────────
def infer_asr_with_sentence_vad(
    audio_path: str,
    model,
    processor,
    device,
    frame_ms: int = 30,
    end_sil_ms: int = 600,
    min_speech_ms: int = 300,
    merge_gap_ms: int = 200,
    join_with: str = " ",
    return_segments: bool = False,
    min_chars: int = 1,
):
    """
    Segment the audio into sentence-like chunks via WebRTC VAD and decode each chunk.
    Returns a single joined transcript by default, or (transcript, segments) if return_segments=True.
    Each segment item: {"start": s, "end": e, "text": t}
    """
    # Load once to slice quickly
    audio = load_audio(audio_path, sr=16000)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Audio file '{audio_path}' could not be loaded or is empty.")

    # Find segments
    seg_times = vad_sentence_segments(
        audio_path,
        target_sr=16000,
        frame_ms=frame_ms,
        end_sil_ms=end_sil_ms,
        min_speech_ms=min_speech_ms,
        merge_gap_ms=merge_gap_ms,
    )

    # Prepare decoder (reuse if available)
    decoder = None
    if _USE_CTCDECODER and os.path.exists(LM_MODEL_PATH):
        decoder = _get_ctc_decoder(
            processor,
            LM_MODEL_PATH,
            alpha=DEFAULT_LM_ALPHA,
            beta=DEFAULT_LM_BETA,
        )

    pieces = []
    seg_records = []
    t_enc_ms_total = 0.0
    t_dec_ms_total = 0.0

    model.eval()
    for (s, e) in seg_times:
        i0 = max(0, int(s * 16000))
        i1 = min(len(audio), int(e * 16000))
        if i1 <= i0:
            continue
        seg = audio[i0:i1]

        # Encode
        inputs = processor(
            seg,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            tic = time.perf_counter()
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"[infer_asr_with_sentence_vad] Expected Tensor logits, got {type(logits)}"
                )
            t_enc_ms = (time.perf_counter() - tic) * 1000.0
            t_enc_ms_total += t_enc_ms

        # Decode
        if decoder is not None:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            tic = time.perf_counter()
            beam = DEFAULT_BEAM_WIDTH if DEFAULT_BEAM_WIDTH > 0 else 100
            text = decoder.decode(probs, beam_width=beam).strip()
            t_dec_ms = (time.perf_counter() - tic) * 1000.0
        else:
            tic = time.perf_counter()
            pred_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            t_dec_ms = (time.perf_counter() - tic) * 1000.0
        t_dec_ms_total += t_dec_ms

        if len(text) >= min_chars:
            pieces.append(text)
            seg_records.append({"start": float(s), "end": float(e), "text": text})

    transcript = join_with.join(pieces).strip()

    _append_infer_log({
        "task": "asr",
        "mode": "file_vad",
        "win_idx": None,
        "audio_path": audio_path,
        "t_enc_ms": round(t_enc_ms_total, 3),
        "t_dec_ms": round(t_dec_ms_total, 3),
        "e2e_ms": round(t_enc_ms_total + t_dec_ms_total, 3),
        "beam": DEFAULT_BEAM_WIDTH if DEFAULT_BEAM_WIDTH > 0 else None,
        "alpha": DEFAULT_LM_ALPHA,
        "beta": DEFAULT_LM_BETA,
        "ema": None,
        "segments": len(seg_records),
    })

    if return_segments:
        return transcript, seg_records
    return transcript


# ──────────────────────────────────────────────────────────────────────────────
# ASR: file-based inference
# ──────────────────────────────────────────────────────────────────────────────
def infer_asr(audio_path, model, processor, device, sentence_vad: bool = False, vad_params: dict = None):
    """
    Performs ASR on a single file.

    - Uses KenLM + pyctcdecode if available.
    - Falls back to greedy decoding otherwise.
    - If LM decoding collapses (empty string), falls back to greedy.

    If sentence_vad=True, segments the audio into sentence-like chunks first and
    decodes per-chunk, joining the texts (see infer_asr_with_sentence_vad).
    """
    if sentence_vad:
        vp = vad_params or {}
        return infer_asr_with_sentence_vad(
            audio_path, model, processor, device, **vp
        )

    if model is None or processor is None:
        raise RuntimeError("❌ No ASR model loaded. Please select a checkpoint.")

    audio = load_audio(audio_path)
    if audio is None or len(audio) == 0:
        raise ValueError(
            f"Audio file '{audio_path}' could not be loaded or is empty."
        )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    global _ASR_FILE_WIN_IDX
    _ASR_FILE_WIN_IDX += 1
    t_enc_ms = 0.0
    t_dec_ms = 0.0

    model.eval()
    with torch.no_grad():
        tic = time.perf_counter()
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        if not hasattr(outputs, "logits"):
            raise TypeError(
                f"[infer_asr] Expected `outputs.logits` from ASR model, got {type(outputs)}. "
                f"Check that you're using a Wav2Vec2ForCTC-based checkpoint."
            )
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"[infer_asr] Expected Tensor logits from ASR model, got {type(logits)}. "
                f"Check ASR model / LoRA loading."
            )
        t_enc_ms = (time.perf_counter() - tic) * 1000.0

    decoder = None
    if _USE_CTCDECODER and os.path.exists(LM_MODEL_PATH):
        decoder = _get_ctc_decoder(
            processor,
            LM_MODEL_PATH,
            alpha=DEFAULT_LM_ALPHA,
            beta=DEFAULT_LM_BETA,
        )

    if decoder is not None:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        tic = time.perf_counter()
        beam = DEFAULT_BEAM_WIDTH if DEFAULT_BEAM_WIDTH > 0 else 100
        transcription = decoder.decode(probs, beam_width=beam).strip()
        t_dec_ms = (time.perf_counter() - tic) * 1000.0
        dec_mode = f"beam+kenlm({beam})"

        # Safety net: if LM collapses to empty, use greedy instead
        if not transcription:
            tic = time.perf_counter()
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(
                pred_ids, skip_special_tokens=True
            )[0].strip()
            t_dec_ms = (time.perf_counter() - tic) * 1000.0
            dec_mode += "->greedy_fallback"
    else:
        tic = time.perf_counter()
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0].strip()
        t_dec_ms = (time.perf_counter() - tic) * 1000.0
        dec_mode = "greedy"

    _append_infer_log({
        "task": "asr",
        "mode": "file",
        "win_idx": _ASR_FILE_WIN_IDX,
        "audio_path": audio_path,
        "t_enc_ms": round(t_enc_ms, 3),
        "t_dec_ms": round(t_dec_ms, 3),
        "e2e_ms": round(t_enc_ms + t_dec_ms, 3),
        "beam": DEFAULT_BEAM_WIDTH if DEFAULT_BEAM_WIDTH > 0 else None,
        "alpha": DEFAULT_LM_ALPHA,
        "beta": DEFAULT_LM_BETA,
        "ema": None,
        "decoder": dec_mode,
    })

    return transcription


# ──────────────────────────────────────────────────────────────────────────────
# ASR: convenience wrapper — prefer merged/root via model_loader; fallback to EN base
# ──────────────────────────────────────────────────────────────────────────────
def infer_asr_auto(audio_path: str,
                   asr_run_dir: str,
                   device: str = "cpu",
                   lang: str = "en",
                   fallback_to_base_on_empty: bool = True) -> str:
    """
    1) Load ASR with model_loader (which prefers <run>/merged → root → checkpoint-*).
    2) Decode the file.
    3) If transcript is empty, optionally fall back to the EN base model to keep the UI useful.
    """
    try:
        # model_loader.load_asr_model already prefers merged/root/nested.
        asr_model, asr_proc = load_asr_model(asr_run_dir, device=device, lang=lang, merge_peft_on_load=False)
    except Exception as e:
        print(f"[ASR] load from '{asr_run_dir}' failed ({e}). Falling back to EN base model.")
        asr_model, asr_proc = load_asr_model('en', device=device, lang='en')
        return infer_asr(audio_path, asr_model, asr_proc, device)

    txt = infer_asr(audio_path, asr_model, asr_proc, device)

    if fallback_to_base_on_empty and (not txt or txt.strip() == ""):
        try:
            print("[ASR] empty transcript, falling back to EN base model for UI")
            base_model, base_proc = load_asr_model('en', device=device, lang='en')
            txt2 = infer_asr(audio_path, base_model, base_proc, device)
            if txt2 and txt2.strip():
                return txt2
        except Exception as e:
            print(f"[ASR] EN base fallback failed: {e}")
    return txt


# ──────────────────────────────────────────────────────────────────────────────
# SER: file-based inference
# ──────────────────────────────────────────────────────────────────────────────
def infer_ser(audio_path, model, feature_extractor, device):
    """
    Performs SER on a single file, using sliding-window if it's long.
    """
    if model is None or feature_extractor is None:
        raise RuntimeError("❌ No SER model loaded. Please select a checkpoint.")

    audio = load_audio(audio_path)
    if audio is None or len(audio) == 0:
        raise ValueError(
            f"Audio file '{audio_path}' could not be loaded or is empty."
        )

    waveform = torch.tensor(audio, dtype=torch.float32)

    global _SER_FILE_WIN_IDX
    _SER_FILE_WIN_IDX += 1
    t_enc_ms = 0.0
    t_dec_ms = 0.0
    ema_field = None
    ema_decay = None
    smoothing_mode = "none"

    if waveform.shape[-1] <= MAX_FRAMES:
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        x = inputs.input_values.to(device)
        model.eval()
        with torch.no_grad():
            tic = time.perf_counter()
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output
            # NEW: guard
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"[infer_ser] Expected Tensor/logits from SER model, "
                    f"got {type(logits)}. Check SER model / LoRA loading."
                )
            t_enc_ms = (time.perf_counter() - tic) * 1000.0
        if logits.dim() == 3:
            logits = logits.mean(dim=1)
        emotion_id = torch.argmax(logits, dim=-1).item()
    else:
        scores, tim = sliding_window_emotion(
            model,
            waveform,
            MAX_FRAMES,
            device=device,
            return_timings=True,
        )
        t_enc_ms = tim["t_enc_ms"]
        t_dec_ms = tim["t_dec_ms"]
        smoothing_mode = tim.get("smoothing_mode", "mean")
        ema_field = tim.get("ema_half_life_s", None)
        ema_decay = tim.get("ema_decay", None)
        emotion_id = scores.argmax().item()

    _append_infer_log({
        "task": "ser",
        "mode": "file",
        "win_idx": _SER_FILE_WIN_IDX,
        "audio_path": audio_path,
        "t_enc_ms": round(t_enc_ms, 3),
        "t_dec_ms": round(t_dec_ms, 3),
        "e2e_ms": round(t_enc_ms + t_dec_ms, 3),
        "beam": None,
        "alpha": None,
        "beta": None,
        "ema": ema_field if ema_field is not None else ema_decay,
        "ema_half_life_s": ema_field,
        "ema_decay": ema_decay,
        "smoothing": smoothing_mode,
    })

    return EMOTION_MAP.get(emotion_id, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference helper
# ──────────────────────────────────────────────────────────────────────────────
def batch_infer(
    audio_dir,
    asr_model,
    asr_processor,
    ser_model,
    ser_feature_extractor,
    device,
):
    results = []
    for fn in os.listdir(audio_dir):
        if fn.lower().endswith(".wav"):
            path = os.path.join(audio_dir, fn)
            try:
                transcription = infer_asr(path, asr_model, asr_processor, device)
                emotion = infer_ser(path, ser_model, ser_feature_extractor, device)
                results.append({
                    "file": fn,
                    "transcription": transcription,
                    "emotion": emotion,
                })
            except Exception as ex:
                print(f"Error processing {fn}: {ex}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Real-time ASR (streaming) — UPDATED SIGNATURE
# ──────────────────────────────────────────────────────────────────────────────
def streaming_infer_asr(
    audio_bytes,
    model,
    processor,
    device,
    beam_width=None,
    alpha=None,
    beta=None,
):
    """
    Real-time ASR decoding from raw float32 audio bytes OR container bytes (webm/mp3/m4a…).

    - Accepts optional beam_width / alpha / beta so existing callers that pass
      these (e.g. main_app.py) no longer break.
    - Uses KenLM + pyctcdecode if available and beam_width > 0.
    - Falls back to greedy decoding.
    - If LM decoding collapses (empty), falls back to greedy.
    """
    if model is None or processor is None:
        raise RuntimeError("❌ No ASR model loaded. Please select a checkpoint.")

    # NEW: tolerate container bytes (webm/mp3/m4a) in addition to raw float32
    samples = decode_audio_bytes(audio_bytes, sr=16000)

    inputs = processor(
        [samples],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    global _ASR_STREAM_WIN_IDX
    _ASR_STREAM_WIN_IDX += 1
    t_enc_ms = 0.0
    t_dec_ms = 0.0

    model.eval()
    with torch.no_grad():
        tic = time.perf_counter()
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        if not hasattr(outputs, "logits"):
            raise TypeError(
                f"[streaming_infer_asr] Expected `outputs.logits` from ASR model, "
                f"got {type(outputs)}. Check that you're using a Wav2Vec2ForCTC-based checkpoint."
            )
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"[streaming_infer_asr] Expected Tensor logits from ASR model, "
                f"got {type(logits)}. Check ASR model / LoRA loading."
            )
        t_enc_ms = (time.perf_counter() - tic) * 1000.0

    # Resolve decoding hyperparameters
    use_beam = beam_width is not None and beam_width > 0
    dec_alpha = alpha if alpha is not None else DEFAULT_LM_ALPHA
    dec_beta = beta if beta is not None else DEFAULT_LM_BETA

    decoder = None
    if _USE_CTCDECODER and os.path.exists(LM_MODEL_PATH):
        decoder = _get_ctc_decoder(
            processor,
            LM_MODEL_PATH,
            alpha=dec_alpha,
            beta=dec_beta,
        )

    # Beam + LM if available and requested
    if decoder is not None and use_beam:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        tic = time.perf_counter()
        transcription = decoder.decode(probs, beam_width=beam_width).strip()
        t_dec_ms = (time.perf_counter() - tic) * 1000.0
        dec_mode = f"beam+kenlm({beam_width})"

        # Fallback to greedy if LM decoding collapses
        if not transcription:
            tic = time.perf_counter()
            pred = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(
                pred, skip_special_tokens=True
            )[0].strip()
            t_dec_ms = (time.perf_counter() - tic) * 1000.0
            dec_mode += "->greedy_fallback"
    else:
        # Pure greedy decoding
        tic = time.perf_counter()
        pred = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(
            pred, skip_special_tokens=True
        )[0].strip()
        t_dec_ms = (time.perf_counter() - tic) * 1000.0
        dec_mode = "greedy"

    _append_infer_log({
        "task": "asr",
        "mode": "stream",
        "win_idx": _ASR_STREAM_WIN_IDX,
        "t_enc_ms": round(t_enc_ms, 3),
        "t_dec_ms": round(t_dec_ms, 3),
        "e2e_ms": round(t_enc_ms + t_dec_ms, 3),
        "beam": beam_width if (beam_width and beam_width > 0) else None,
        "alpha": dec_alpha,
        "beta": dec_beta,
        "ema": None,
        "decoder": dec_mode,
    })

    return transcription


# ──────────────────────────────────────────────────────────────────────────────
# Real-time SER (streaming)
# ──────────────────────────────────────────────────────────────────────────────
def streaming_infer_ser(audio_bytes, model, feature_extractor, device):
    """
    Real-time SER decoding from raw float32 audio bytes.
    """
    if model is None or feature_extractor is None:
        raise RuntimeError("❌ No SER model loaded. Please select a checkpoint.")

    # NEW: accept container bytes here too
    samples = decode_audio_bytes(audio_bytes, sr=16000)

    inputs = feature_extractor(
        [samples],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    x = inputs.input_values.to(device)

    global _SER_STREAM_WIN_IDX
    _SER_STREAM_WIN_IDX += 1
    t_enc_ms = 0.0
    t_dec_ms = 0.0
    ema_field = None
    ema_decay = None

    model.eval()
    with torch.no_grad():
        tic = time.perf_counter()
        output = model(x)
        logits = output.logits if hasattr(output, "logits") else output
        # NEW: guard
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"[streaming_infer_ser] Expected Tensor/logits from SER model, "
                f"got {type(logits)}. Check SER model / LoRA loading."
            )
        t_enc_ms = (time.perf_counter() - tic) * 1000.0

    if logits.dim() == 3:
        logits = logits.mean(dim=1)

    emotion_id = torch.argmax(logits, dim=-1).item()

    _append_infer_log({
        "task": "ser",
        "mode": "stream",
        "win_idx": _SER_STREAM_WIN_IDX,
        "t_enc_ms": round(t_enc_ms, 3),
        "t_dec_ms": round(t_dec_ms, 3),
        "e2e_ms": round(t_enc_ms + t_dec_ms, 3),
        "beam": None,
        "alpha": None,
        "beta": None,
        "ema": None,
        "ema_half_life_s": ema_field,
        "ema_decay": ema_decay,
        "smoothing": "none",
    })

    return EMOTION_MAP.get(emotion_id, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Probabilities variant for streaming SER
# ──────────────────────────────────────────────────────────────────────────────
def streaming_infer_ser_probs(audio_bytes, model, feature_extractor, device) -> np.ndarray:
    """
    Real-time SER: returns a probability vector (numpy float32 [C]) for the given audio window.
    No smoothing here; intended to be combined with SERSmoother by the caller.
    """
    if model is None or feature_extractor is None:
        raise RuntimeError("❌ No SER model loaded. Please select a checkpoint.")

    # NEW: accept container bytes here too
    samples = decode_audio_bytes(audio_bytes, sr=16000)

    inputs = feature_extractor(
        [samples],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    x = inputs.input_values.to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        logits = output.logits if hasattr(output, "logits") else output

    # NEW: guard
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"[streaming_infer_ser_probs] Expected Tensor/logits from SER model, "
            f"got {type(logits)}. Check SER model / LoRA loading."
        )

    if logits.dim() == 3:
        logits = logits.mean(dim=1)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].astype(np.float32)
    return probs
