# Speemo-ASR and SER Training and Inference Framework for Audiofile-based and Real-Time Inference. Martin Khadjavian © 

import os
import librosa
import numpy as np
import soundfile as sf

def pitch_shift_correct(audio, sr=16000, n_steps=2):
    """
    Shifts the pitch of the audio by n_steps semitones.
    """
    try:
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"❌ Error in pitch shifting: {e}")
        return audio

def add_noise(audio, noise_level=0.02):
    """
    Adds random noise to the audio.
    """
    try:
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    except Exception as e:
        print(f"❌ Error in adding noise: {e}")
        return audio

def time_stretch_correct(audio, rate=1.1):
    """
    Stretches the audio by a given rate (rate > 1 speeds up (shorter), rate < 1 slows down (longer)).
    """
    try:
        return librosa.effects.time_stretch(y=audio, rate=rate)
    except Exception as e:
        print(f"❌ Error in time stretching: {e}")
        return audio

def synthesize_emotional_audio(text, emotion, tts_model, output_path):
    """
    Stub for a TTS-based emotional speech synthesis function.
    """
    print(f"🔊 Synthesizing '{emotion}' TTS audio for text: '{text}' -> {output_path}")
    # Implementation would go here if you have a TTS model.



def augment_data(
    input_dir,
    output_dir,
    tts_model=None,
    text="Example text",
    emotion="happy",
    sr=None
):
    """
    Augments audio data by applying pitch shifting, noise, time stretching,
    and optionally generating synthetic emotional audio.
    Only processes original files (i.e., skips anything already suffixed).

    Parameters:
        input_dir:   directory of raw audio files
        output_dir:  directory to write augmented files
        tts_model:   optional TTS model to synthesize emotion
        text:        text for TTS synthesis
        emotion:     emotion label
        sr:          optional sample rate (None = use native)
    """
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory {input_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔄 Augmenting audio files from {input_dir} to {output_dir}")

    file_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            # 1) Skip non-audio files
            if not file.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                continue

            base_name, ext = os.path.splitext(file)

            # 2) Skip already-augmented files by suffix
            if any(base_name.endswith(suf) for suf in ("_noise", "_pitch", "_stretch")):
                continue

            file_path = os.path.join(root, file)
            pitch_path     = os.path.join(output_dir, f"{base_name}_pitch{ext}")
            noise_path     = os.path.join(output_dir, f"{base_name}_noise{ext}")
            stretch_path   = os.path.join(output_dir, f"{base_name}_stretch{ext}")
            synthetic_path = os.path.join(output_dir, f"{base_name}_synthetic{ext}")

            # 3) If *all* three core augmentations exist, skip
            if (os.path.exists(pitch_path)
             and os.path.exists(noise_path)
             and os.path.exists(stretch_path)):
                print(f"⚠ {file} already augmented. Skipping.")
                continue

            # 4) Load original audio once
            try:
                audio, detected_sr = librosa.load(file_path, sr=sr)
                print(f"✅ Processing file: {file_path} at {detected_sr} Hz")
                if sr and detected_sr != sr:
                    print(f"⚠ Warning: {file} was resampled from {detected_sr} to {sr} Hz")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue

            # 5) Pitch shift
            if not os.path.exists(pitch_path):
                try:
                    aug = pitch_shift_correct(audio, detected_sr, n_steps=2)
                    sf.write(pitch_path, aug, detected_sr)
                except Exception as e:
                    print(f"❌ Error creating pitch file for {file}: {e}")

            # 6) Add noise
            if not os.path.exists(noise_path):
                try:
                    aug = add_noise(audio, noise_level=0.02)
                    sf.write(noise_path, aug, detected_sr)
                except Exception as e:
                    print(f"❌ Error creating noise file for {file}: {e}")

            # 7) Time stretch
            if not os.path.exists(stretch_path):
                try:
                    aug = time_stretch_correct(audio, rate=1.1)
                    sf.write(stretch_path, aug, detected_sr)
                except Exception as e:
                    print(f"❌ Error creating stretch file for {file}: {e}")

            # 8) Optional TTS‐based synthetic audio
            if tts_model and not os.path.exists(synthetic_path):
                try:
                    synthesize_emotional_audio(text, emotion, tts_model, synthetic_path)
                except Exception as e:
                    print(f"❌ Error creating synthetic file for {file}: {e}")

            file_count += 1

    if file_count == 0:
        print("✅ No new files were augmented. (Either none found or all are already augmented.)")
    else:
        print(f"✅ Augmentation completed. {file_count} files processed.")


