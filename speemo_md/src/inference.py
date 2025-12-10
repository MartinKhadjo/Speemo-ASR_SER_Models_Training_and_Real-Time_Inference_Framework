import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import json
from transformers import Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer
from model_loader import load_asr_model, load_ser_model, load_neural_lm
from pydub import AudioSegment

# Beam-search decoder (pyctcdecode) — optional
try:
    from pyctcdecode import build_ctcdecoder
    _USE_CTCDECODER = True
except ImportError:
    build_ctcdecoder = None
    _USE_CTCDECODER = False


def sliding_window_emotion(model, waveform: torch.Tensor, max_frames: int, hop: int = None, device="cpu"):
    """
    Slice long waveforms into overlapping windows, run SER, then average softmax scores.
    """
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
    model.eval()
    with torch.no_grad():
        for w in windows:
            inp = w.unsqueeze(0).to(device)  # [1, T]
            output = model(inp)
            logits = output.logits if hasattr(output, "logits") else output  # [1, C] or [1,1,C]
            if logits.dim() == 3:
                logits = logits.mean(dim=1)  # → [1, C]
            probs.append(torch.softmax(logits, dim=-1).cpu())

    return torch.stack(probs, dim=0).mean(dim=0)  # [C]


# Global decoder and LM path
ASR_DECODER = None
LM_MODEL_PATH = os.path.join("models", "lm", "4gram.arpa")

# Shallow-fusion neural LM (optional)
NEURAL_LM: AutoModelForCausalLM = None
LM_TOKENIZER: AutoTokenizer = None
LM_ALPHA: float = 0.5
LM_NAME: str = "gpt2-medium"

EMOTION_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
}

# -------------------- NEW: Robust ASR chunking to avoid OOM/timeouts --------------------
CHUNK_SEC = 20
CHUNK_SAMPLES = 16000 * CHUNK_SEC
OVERLAP = int(0.5 * CHUNK_SAMPLES)  # 50% overlap to reduce word cuts

# -------------------- Streaming ASR ring buffer state (for microphone/Unity) --------------------
STREAM_SAMPLE_RATE = 16000
STREAM_MIN_SEC = 0.7   # minimum audio length before running ASR on streaming data
STREAM_MAX_SEC = 15.0  # keep at most last 15 seconds in buffer
STREAM_ASR_BUFFER = np.zeros(0, dtype=np.float32)
LAST_STREAM_TRANSCRIPTION = ""  # remember last non-empty transcript for streaming

# -------------------- Streaming SER ring buffer state (separate from ASR) --------------------
STREAM_SER_SAMPLE_RATE = 16000
STREAM_SER_MIN_SEC = 1.0    # or 1.5s for more stable emotion, tune if needed
STREAM_SER_MAX_SEC = 10.0
STREAM_SER_BUFFER = np.zeros(0, dtype=np.float32)

# --- SER smoothing / calibration / hysteresis globals (for streaming SER only) ---
STREAM_SER_PROBS = np.zeros(len(EMOTION_MAP), dtype=np.float32)  # EMA of class probs
SER_EMA_ALPHA = 0.7            # EMA smoothing factor (0→no smoothing, 1→frozen)
SER_TEMP = 0.7                 # temperature for logits (<1 → sharper, >1 → smoother)
SER_CONF_THRESHOLD = 0.55      # confidence threshold for switching emotion
SER_ENERGY_THRESHOLD = 0.005   # simple VAD: treat below as silence
LAST_STREAM_EMOTION = None     # last non-neutral/stable emotion label
LAST_SER_CONFIDENCE = 0.0      # last confidence for hysteresis


# -------------------- NEW: unified loader for any audio → 16k mono float32 --------------------
def load_audio_16k_mono(path: str):
    import numpy as np
    try:
        # librosa first (handles many codecs via soundfile/audioread)
        y, _ = librosa.load(path, sr=16000, mono=True)
        return y.astype(np.float32), 16000
    except Exception:
        # fallback: soundfile, then resample + downmix if needed
        import soundfile as sf
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            if np.ndim(y) > 1:
                y = y.mean(axis=1)
            return y.astype(np.float32), 16000
        except Exception:
            # final resort: pydub/ffmpeg
            try:
                seg = AudioSegment.from_file(path)
                seg = seg.set_channels(1).set_frame_rate(16000)
                samples = np.array(seg.get_array_of_samples()).astype(np.float32)
                max_val = float(1 << (8 * seg.sample_width - 1))
                if max_val > 0:
                    samples /= max_val
                return samples, 16000
            except Exception as e:
                print(f"[load_audio_16k_mono] Failed to decode {path}: {e}")
                return None, 16000


def load_audio(file_path, sr=16000):
    """
    Load an audio file at a specified sampling rate (mono).

    Robust 3-step fallback:
      1) librosa (soundfile backend first, then audioread for exotic codecs)
      2) soundfile/libsndfile (WAV/FLAC/OGG/AIFF/...); resample if needed
      3) pydub/ffmpeg (last resort); resample + normalize by sample width

    Returns:
      np.ndarray(float32, mono, sr=sr) or None on failure.
    """
    # 1) Try librosa first
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        pass

    # 2) Try soundfile/libsndfile
    try:
        import soundfile as sf
        y, orig_sr = sf.read(file_path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if orig_sr != sr:
            # Use librosa’s resampler
            y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
        return y.astype(np.float32)
    except Exception:
        pass

    # 3) Last resort: pydub (requires ffmpeg in PATH)
    try:
        seg = AudioSegment.from_file(file_path)
        seg = seg.set_channels(1).set_frame_rate(sr)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)

        # Normalize by bit depth (handles 16/24/32-bit)
        max_val = float(1 << (8 * seg.sample_width - 1))
        if max_val > 0:
            samples /= max_val

        return samples
    except Exception as e:
        print(f"[load_audio] Failed to decode {file_path}: {e}")
        return None


# -------------------- NEW: robust decoder for streaming payloads --------------------
def _decode_streaming_payload(audio_payload, expected_dtype=None):
    """
    Decode streaming payload coming from Socket.IO / Unity.

    - If it's bytes/bytearray/memoryview: interpret as raw PCM16 or float32.
    - If it's a list/tuple of numbers (e.g. from JSON): convert via np.array.
    - Returns a 1D float32 numpy array in [-1, 1] for int16 payloads or raw
      float32 values for float32 payloads.
    """
    # Case 1: bytes-like (Socket.IO / Unity usually sends this)
    if isinstance(audio_payload, (bytes, bytearray, memoryview)):
        buf = bytes(audio_payload)  # ensure bytes

        # If dtype explicitly requested, use it
        if expected_dtype is not None:
            if expected_dtype == "int16":
                # trim to multiple of 2 bytes
                cut = len(buf) - (len(buf) % 2)
                if cut <= 0:
                    print("[streaming] int16 payload too short after trim; returning zeros")
                    return np.zeros(0, dtype=np.float32)
                buf = buf[:cut]
                arr = np.frombuffer(buf, dtype=np.int16)
                audio = arr.astype(np.float32) / 32768.0
                print(f"[streaming] decoded int16 bytes -> {audio.size} samples")
                return audio

            elif expected_dtype == "float32":
                # trim to multiple of 4 bytes
                cut = len(buf) - (len(buf) % 4)
                if cut <= 0:
                    print("[streaming] float32 payload too short after trim; returning zeros")
                    return np.zeros(0, dtype=np.float32)
                buf = buf[:cut]
                audio = np.frombuffer(buf, dtype=np.float32)
                print(f"[streaming] decoded float32 bytes -> {audio.size} samples")
                return audio

            else:
                raise ValueError(f"Unsupported expected_dtype={expected_dtype!r}")

        # Otherwise auto-detect: prefer float32, fallback to int16 if needed
        if len(buf) >= 4 and (len(buf) % 4 == 0):
            cut = len(buf) - (len(buf) % 4)
            buf32 = buf[:cut]
            arr32 = np.frombuffer(buf32, dtype=np.float32)
            if arr32.size > 0 and np.all(np.isfinite(arr32)) and np.all(np.abs(arr32) <= 1.0 + 1e-3):
                print(f"[streaming] auto-detected float32 bytes -> {arr32.size} samples")
                return arr32.astype(np.float32)

        if len(buf) >= 2 and (len(buf) % 2 == 0):
            cut = len(buf) - (len(buf) % 2)
            buf16 = buf[:cut]
            arr16 = np.frombuffer(buf16, dtype=np.int16)
            audio = arr16.astype(np.float32) / 32768.0
            print(f"[streaming] auto-detected int16 bytes -> {audio.size} samples")
            return audio

        # If we get here, we couldn't make sense of the payload
        print(f"[streaming] bytes payload had weird length={len(buf)}; returning zeros")
        return np.zeros(0, dtype=np.float32)

    # Case 2: list/tuple of numbers (e.g. from JS: Array.from(pcm16/pcm32))
    if isinstance(audio_payload, (list, tuple)):
        arr = np.array(audio_payload)
        if arr.size == 0:
            print("[streaming] empty list/tuple payload; returning zeros")
            return np.zeros(0, dtype=np.float32)

        # If within [-2, 2] and finite, assume float32 samples
        if np.all(np.isfinite(arr)) and np.all(np.abs(arr) <= 2.0):
            print(f"[streaming] decoded float list/tuple -> {arr.size} samples")
            return arr.astype(np.float32)

        # Otherwise assume int16 PCM
        arr = arr.astype(np.int16)
        audio = arr.astype(np.float32) / 32768.0
        print(f"[streaming] decoded int16 list/tuple -> {audio.size} samples")
        return audio

    # Fallback: unsupported type
    print(f"[streaming] Unsupported payload type: {type(audio_payload)}; returning zeros")
    return np.zeros(0, dtype=np.float32)


def _maybe_build_ctcdecoder(processor):
    """
    Build and cache a pyctcdecode decoder if the ARPA LM is present.
    """
    global ASR_DECODER
    if ASR_DECODER is not None:
        return ASR_DECODER

    if not _USE_CTCDECODER:
        return None
    if not os.path.exists(LM_MODEL_PATH):
        return None

    # Build labels list from tokenizer vocab sorted by ID
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    tokens = [tok for tok, _ in sorted_vocab]

    ASR_DECODER = build_ctcdecoder(tokens, LM_MODEL_PATH)
    return ASR_DECODER


def _maybe_load_neural_lm(device):
    """
    Load and cache a transformer LM + tokenizer for shallow fusion rescoring.
    """
    global NEURAL_LM, LM_TOKENIZER
    if NEURAL_LM is None or LM_TOKENIZER is None:
        NEURAL_LM, LM_TOKENIZER = load_neural_lm(LM_NAME, device)
    return NEURAL_LM, LM_TOKENIZER


def infer_asr(audio_path, model, processor, device):
    """
    File-based ASR with optional CTC beam search (pyctcdecode) and shallow fusion LM.
    Now robust to long audio by chunking into ~20s windows with 50% overlap.
    """
    if model is None or processor is None:
        raise RuntimeError("❌ No ASR model loaded. Please select a checkpoint.")

    # --- use unified robust loader (returns float32 mono @16k + sr)
    audio, _sr = load_audio_16k_mono(audio_path)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Audio file '{audio_path}' could not be loaded or is empty.")

    # Build chunks (single chunk if short enough)
    if len(audio) <= CHUNK_SAMPLES:
        chunks = [audio]
    else:
        chunks = []
        start = 0
        L = len(audio)
        while start < L:
            end = min(start + CHUNK_SAMPLES, L)
            chunks.append(audio[start:end])
            if end >= L:
                break
            start = end - OVERLAP  # overlap to avoid cutting words

    texts = []
    decoder = _maybe_build_ctcdecoder(processor)

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            # Prepare inputs for this chunk
            inputs = processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Forward
            output = model(input_values, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, "logits") else output  # [1, T, V]

            # Decode
            if decoder is not None:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [T, V]
                # Try shallow fusion re-score if neural LM is available
                best_text = None
                if probs.shape[0] > 0:
                    try:
                        beams = decoder.decode_beams(probs, beam_width=100)
                        _maybe_load_neural_lm(device)
                        best_score = None
                        for beam in beams:
                            text = beam.text
                            ctc_score = beam.ctc_score
                            inputs_lm = LM_TOKENIZER(text, return_tensors="pt").to(device)
                            lm_outputs = NEURAL_LM(**inputs_lm, labels=inputs_lm["input_ids"])
                            lm_score = -lm_outputs.loss.item() * inputs_lm["input_ids"].size(1)
                            fused = ctc_score + LM_ALPHA * lm_score
                            if best_score is None or fused > best_score:
                                best_score, best_text = fused, text
                    except Exception:
                        # Fallback to plain LM decoding if beam rescoring fails
                        best_text = decoder.decode(probs)
                texts.append((best_text or "").strip())
            else:
                # Greedy fallback
                pred_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                texts.append(text)

    # Join chunk texts
    transcription = " ".join(t for t in texts if t).strip()
    return transcription


# Window length (in raw samples @ 16kHz)
MAX_FRAMES = 4 * 16000


def infer_ser(audio_path, model, feature_extractor, device):
    """
    File-based SER; uses sliding windows if utterance is longer than MAX_FRAMES.
    """
    if model is None or feature_extractor is None:
        raise RuntimeError("❌ No SER model loaded. Please select a checkpoint.")

    # --- use unified robust loader (returns float32 mono @16k + sr)
    audio, _sr = load_audio_16k_mono(audio_path)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Audio file '{audio_path}' could not be loaded or is empty.")

    waveform = torch.tensor(audio, dtype=torch.float32)

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
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output
        if logits.dim() == 3:
            logits = logits.mean(dim=1)
        emotion_id = torch.argmax(logits, dim=-1).item()
    else:
        scores = sliding_window_emotion(model, waveform, MAX_FRAMES, device=device)
        emotion_id = scores.argmax().item()

    return EMOTION_MAP.get(emotion_id, "unknown")


def batch_infer(audio_dir, asr_model, asr_processor, ser_model, ser_feature_extractor, device):
    """
    Batch ASR+SER inference over a directory of WAV files.
    """
    results = []
    for fn in os.listdir(audio_dir):
        if fn.lower().endswith(".wav"):
            path = os.path.join(audio_dir, fn)
            try:
                transcription = infer_asr(path, asr_model, asr_processor, device)
                emotion = infer_ser(path, ser_model, ser_feature_extractor, device)
                results.append(
                    {
                        "file": fn,
                        "transcription": transcription,
                        "emotion": emotion,
                    }
                )
            except Exception as ex:
                print(f"Error processing {fn}: {ex}")
    return results


def streaming_infer_asr(audio_bytes, model, processor, device):
    """
    Real-time ASR from streaming audio payload (bytes or list of PCM samples).

    The payload is expected to be PCM16 or float32, either:
      - bytes / bytearray / memoryview of int16/float32 samples, or
      - a list of int16/float values from JS like Array.from(pcm16/pcm32).

    We decode safely and trim buffers so NumPy never throws
    "buffer size must be a multiple of element size".
    Additionally, we accumulate several chunks in a ring buffer so that
    very short microphone chunks still yield meaningful transcripts.

    We also keep track of the last non-empty transcript so the frontend
    doesn't get flooded with "" results.
    """
    if model is None or processor is None:
        raise RuntimeError("❌ No ASR model loaded. Please select a checkpoint.")

    global STREAM_ASR_BUFFER, LAST_STREAM_TRANSCRIPTION

    # Decode incoming chunk into float32 in [-1, 1]
    chunk = _decode_streaming_payload(audio_bytes)
    print(f"[streaming_asr] received chunk with {chunk.size} samples")

    if chunk.size == 0:
        print("[streaming_asr] empty chunk after decoding; returning last transcript")
        return LAST_STREAM_TRANSCRIPTION

    # Append to ring buffer
    STREAM_ASR_BUFFER = np.concatenate([STREAM_ASR_BUFFER, chunk])

    sample_rate = STREAM_SAMPLE_RATE
    min_samples = int(STREAM_MIN_SEC * sample_rate)
    max_samples = int(STREAM_MAX_SEC * sample_rate)

    # Keep only last max_seconds of audio
    if STREAM_ASR_BUFFER.size > max_samples:
        STREAM_ASR_BUFFER = STREAM_ASR_BUFFER[-max_samples:]

    print(
        f"[streaming_asr] buffer size={STREAM_ASR_BUFFER.size}, "
        f"min_samples={min_samples}, max_samples={max_samples}"
    )

    # If not enough audio accumulated yet, don't decode
    if STREAM_ASR_BUFFER.size < min_samples:
        print("[streaming_asr] not enough audio accumulated yet; returning last transcript")
        return LAST_STREAM_TRANSCRIPTION

    # Work on a copy so we don't block appending
    audio = STREAM_ASR_BUFFER.copy()

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_values, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output  # [B, T, V]

    decoder = _maybe_build_ctcdecoder(processor)

    if decoder is not None:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        if probs.shape[0] == 0:
            print("[streaming_asr] decoder probs empty; returning last transcript")
            return LAST_STREAM_TRANSCRIPTION

        try:
            beams = decoder.decode_beams(probs, beam_width=100)
            _maybe_load_neural_lm(device)
            best_score, best_text = None, None
            for beam in beams:
                text = beam.text
                ctc_score = beam.ctc_score

                inputs_lm = LM_TOKENIZER(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    lm_outputs = NEURAL_LM(**inputs_lm, labels=inputs_lm["input_ids"])
                lm_score = -lm_outputs.loss.item() * inputs_lm["input_ids"].size(1)

                fused = ctc_score + LM_ALPHA * lm_score
                if best_score is None or fused > best_score:
                    best_score, best_text = fused, text

            transcription = best_text or ""
        except Exception as e:
            print(f"[streaming_asr] beam decoding failed ({e}); falling back to plain decoder")
            transcription = decoder.decode(probs)
    else:
        pred = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred, skip_special_tokens=True)[0]

    # Final cleanup + greedy fallback if everything above produced blanks
    transcription = transcription.strip()
    if not transcription:
        print("[streaming_asr] decoded empty transcript; falling back to greedy and/or last transcript")
        pred = torch.argmax(logits, dim=-1)
        greedy = processor.batch_decode(pred, skip_special_tokens=True)[0].strip()
        if greedy:
            transcription = greedy
        else:
            transcription = LAST_STREAM_TRANSCRIPTION

    LAST_STREAM_TRANSCRIPTION = transcription
    print(f"[streaming_asr] final transcript='{transcription}'")
    return transcription


def streaming_infer_ser(audio_bytes, model, feature_extractor, device):
    """
    Real-time SER from streaming audio payload (bytes or list of PCM samples).

    Payload is treated as PCM16/float32 and decoded to float32 in [-1, 1],
    then accumulated in a separate SER ring buffer. We only run the model
    once we have at least STREAM_SER_MIN_SEC seconds of audio, and at most
    STREAM_SER_MAX_SEC seconds are kept.

    Additional real-time tricks:
      - simple VAD (energy-based) to ignore silence/noise
      - multi-window (multi-scale) SER over several recent window lengths
      - temperature scaling of logits
      - probability-level EMA smoothing
      - confidence-based hysteresis to avoid jittery label flips
    """
    if model is None or feature_extractor is None:
        raise RuntimeError("❌ No SER model loaded. Please select a checkpoint.")

    global STREAM_SER_BUFFER, STREAM_SER_PROBS, LAST_STREAM_EMOTION, LAST_SER_CONFIDENCE

    # Decode current chunk
    chunk = _decode_streaming_payload(audio_bytes)
    print(f"[streaming_ser] received chunk with {chunk.size} samples")

    if chunk.size == 0:
        print("[streaming_ser] empty chunk after decoding; returning last emotion or 'neutral'")
        return LAST_STREAM_EMOTION or "neutral"

    # Append to SER ring buffer
    STREAM_SER_BUFFER = np.concatenate([STREAM_SER_BUFFER, chunk])

    sr = STREAM_SER_SAMPLE_RATE
    min_samples = int(STREAM_SER_MIN_SEC * sr)
    max_samples = int(STREAM_SER_MAX_SEC * sr)

    # Keep only last max_seconds
    if STREAM_SER_BUFFER.size > max_samples:
        STREAM_SER_BUFFER = STREAM_SER_BUFFER[-max_samples:]

    print(
        f"[streaming_ser] buffer size={STREAM_SER_BUFFER.size}, "
        f"min_samples={min_samples}, max_samples={max_samples}"
    )

    # Not enough audio yet → don't update emotion
    if STREAM_SER_BUFFER.size < min_samples:
        print("[streaming_ser] not enough audio accumulated yet; returning last emotion or 'neutral'")
        return LAST_STREAM_EMOTION or "neutral"

    # Work on a copy so we don't block updates
    audio = STREAM_SER_BUFFER.copy()

    # ------------------ Simple VAD (energy-based silence detection) ------------------
    energy = float(np.mean(np.abs(audio))) if audio.size > 0 else 0.0
    print(f"[streaming_ser] energy={energy:.6f}")
    if energy < SER_ENERGY_THRESHOLD:
        print("[streaming_ser] low energy, treating as silence; returning last emotion or 'neutral'")
        return LAST_STREAM_EMOTION or "neutral"

    # ------------------ Multi-window / multi-scale SER ------------------
    # Use several time scales over the same buffer (e.g., last 1s, 2.5s, 5s)
    windows_sec = [1.0, 2.5, 5.0]
    probs_list = []

    model.eval()

    for w_sec in windows_sec:
        win_samples = int(w_sec * sr)
        if audio.size < win_samples:
            continue

        segment = audio[-win_samples:]

        inputs = feature_extractor(
            segment,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        x = inputs.input_values.to(device)

        with torch.no_grad():
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output

        if logits.dim() == 3:
            logits = logits.mean(dim=1)

        # Temperature scaling
        if SER_TEMP is not None and SER_TEMP > 0:
            logits = logits / SER_TEMP

        # Get probabilities for this window
        probs_win = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probs_list.append(probs_win)

    if not probs_list:
        # Should not really happen because of min_samples, but just in case
        print("[streaming_ser] no valid windows for SER; returning last emotion or 'neutral'")
        return LAST_STREAM_EMOTION or "neutral"

    # Average probabilities across all windows
    raw_probs = np.mean(np.stack(probs_list, axis=0), axis=0)

    # ------------------ EMA smoothing over time ------------------
    if STREAM_SER_PROBS.shape[0] != raw_probs.shape[0]:
        # In case num_classes changed for some reason
        STREAM_SER_PROBS = np.zeros_like(raw_probs, dtype=np.float32)

    STREAM_SER_PROBS = SER_EMA_ALPHA * STREAM_SER_PROBS + (1.0 - SER_EMA_ALPHA) * raw_probs

    # Final probabilities after smoothing
    smoothed_probs = STREAM_SER_PROBS
    best_id = int(smoothed_probs.argmax())
    confidence = float(smoothed_probs[best_id])

    print(f"[streaming_ser] raw_probs={raw_probs}, smoothed_probs={smoothed_probs}, "
          f"best_id={best_id}, confidence={confidence:.4f}")

    # ------------------ Confidence threshold + hysteresis ------------------
    if confidence < SER_CONF_THRESHOLD and LAST_STREAM_EMOTION is not None:
        # Not confident enough → keep previous emotion
        emotion = LAST_STREAM_EMOTION
        print(f"[streaming_ser] confidence below threshold ({SER_CONF_THRESHOLD}); "
              f"keeping last emotion='{emotion}'")
    else:
        emotion = EMOTION_MAP.get(best_id, "unknown")
        LAST_STREAM_EMOTION = emotion
        LAST_SER_CONFIDENCE = confidence
        print(f"[streaming_ser] updating emotion='{emotion}' (id={best_id}, conf={confidence:.4f})")

    return emotion
