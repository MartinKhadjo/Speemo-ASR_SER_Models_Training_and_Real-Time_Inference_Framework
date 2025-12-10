import os
import librosa
import csv
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import soundfile as sf
from sklearn.metrics import confusion_matrix, classification_report
from augmentation import augment_data



# -- Import the updated augment_data if needed for direct usage
from augmentation import augment_data

# NEW: Reconfigure standard output to use UTF-8 so that Unicode emoji can be printed
import sys
sys.stdout.reconfigure(encoding='utf-8')

# at top of file, after imports
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
#

# ------------------------------------------------------------------------------
# 2. Audio Loading and Feature Extraction
# ------------------------------------------------------------------------------
def load_audio(file_path, sr=16000):
    """
    Loads an audio file and resamples it to the specified sampling rate.
    Skips files that are too short (< 0.1 seconds).
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        if len(audio) < sr * 0.1:
            print(f"⚠ Audio file {file_path} is too short and will be skipped.")
            return None
        return audio
    except Exception as e:
        print(f"❌ Error loading audio file {file_path}: {e}")
        return None






# ------------------------------------------------------------------------------
# 3. Preprocessing for ASR, Language ID, Emotion
# ------------------------------------------------------------------------------

def load_transcript_map(raw_asr_dir):
    """
    Reads **all** TSV manifests under the Common Voice clips directory
    and builds a { filename → sentence } map.
    """
    transcript_map = {}
    # Walk up two levels to hit the directory containing *all* the .tsv files
    search_root = os.path.dirname(os.path.dirname(raw_asr_dir))

    for root, _, files in os.walk(search_root):
        for fname in files:
            if not fname.lower().endswith(".tsv"):
                continue
            manifest = os.path.join(root, fname)
            with open(manifest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Common Voice TSVs always use 'path' for the audio filename
                    fn   = row.get("path")
                    # Some older TSVs use 'text' instead of 'sentence'
                    sent = row.get("sentence") or row.get("text")
                    if not fn or not sent:
                        continue
                    # 🔁 store by Common Voice ID only (strip extension)
                    key = os.path.splitext(fn)[0]
                    transcript_map[key] = sent

    return transcript_map




def preprocess_asr(data_dir, processed_dir, backbone_ckpt=None, device="cpu"):
    """
    Copies raw + augmented ASR audio into a processed directory and
    generates a labels.csv manifest, correctly handling augmentation suffixes.
    """
    import os
    import shutil
    import csv

    # Ensure output directory exists
    os.makedirs(processed_dir, exist_ok=True)
    print(f"🔄 Copying raw ASR audio from {data_dir} to {processed_dir}")

    # Build a map from common‐voice ID → transcript
    raw_asr_dir    = os.path.dirname(data_dir)
    transcript_map = load_transcript_map(raw_asr_dir)

    copied, skipped = 0, 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith((".wav", ".mp3")):
                continue
            src = os.path.join(root, file)

            # Strip augmentation suffixes before lookup
            name_no_ext = os.path.splitext(file)[0]
            for suf in ("_noise", "_pitch", "_stretch"):
                if name_no_ext.endswith(suf):
                    name_no_ext = name_no_ext[:-len(suf)]
                    break

            transcript = transcript_map.get(name_no_ext, "").lower()
            if not transcript:
                print(f"⚠ No transcript for {file}, skipping.")
                skipped += 1
                continue

            dst = os.path.join(processed_dir, file)
            shutil.copy(src, dst)
            copied += 1

    # Write out the manifest, again stripping suffixes on each filename
    csv_path = os.path.join(processed_dir, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcript"])
        for wav in sorted(os.listdir(processed_dir)):
            if not wav.lower().endswith((".wav", ".mp3")):
                continue
            base = os.path.splitext(wav)[0]
            lookup = base
            for suf in ("_noise", "_pitch", "_stretch"):
                if lookup.endswith(suf):
                    lookup = lookup[:-len(suf)]
                    break
            transcript = transcript_map.get(lookup, "").lower()
            writer.writerow([wav, transcript])

    print(f"🎯 ASR copy summary:  ✅ {copied} files  ⚠ {skipped} files\n")




def preprocess_emotion(data_dir, processed_dir, backbone_ckpt=None, device="cpu"):
    """
    Copies raw + augmented SER audio into processed_dir and writes labels.csv,
    consuming a single “augmented” folder per language and unifying into a
    single global label space.
    """
    

    # 6-way target label space (shared across all SER datasets)
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
    EMO2ID   = { emo:i for i, emo in enumerate(EMOTIONS) }

    # Dataset-specific canonicalization into that space
    _EMODB_MAP    = {'W': "angry",  'A': "fear",    'T': "sad",
                     'F': "happy",  'N': "neutral", 'L': "neutral",
                     'E': "disgust"}
    _CREMAD_MAP   = {'ANG': "angry",  'DIS': "disgust", 'FEA': "fear",
                     'HAP': "happy",  'NEU': "neutral", 'SAD': "sad"}
    _RAVDESS_MAP  = {'01': "neutral", '02': "calm",     '03': "happy",
                     '04': "sad",     '05': "angry",    '06': "fear",
                     '07': "disgust", '08': "surprise"}
    _REYERSON_MAP = _RAVDESS_MAP.copy()
    _IEMO_FILES   = ['ang.csv', 'hap.csv', 'neu.csv', 'sad.csv']


    os.makedirs(processed_dir, exist_ok=True)

    # ── 0) We assume data_dir is already the language‐specific “…/augmented” folder
    data_source = data_dir

    # ── 1) Detect language by parent folder (“de” or “en”)
    lang     = os.path.basename(os.path.dirname(data_source)).lower()
    raw_root = os.path.dirname(os.path.dirname(data_source))

    # ── 2) Build IEMOCAP label map if English
    label_map = {}
    if lang == "en":
        iemocap_dir = os.path.join(raw_root, "en", "IEMOCAP")
        if os.path.isdir(iemocap_dir):
            print("ℹ️  Detected IEMOCAP — parsing its CSV labels…")
            for csv_name in _IEMO_FILES:
                path = os.path.join(iemocap_dir, csv_name)
                if not os.path.exists(path):
                    continue
                tag = os.path.splitext(csv_name)[0]  # 'ang','hap','neu','sad'
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        label_map[row[0].strip()] = tag

    # ── 3) Copy all augmented + original files into processed_dir
    copied = 0
    for root, _, files in os.walk(data_source):
        for fn in files:
            if fn.lower().endswith((".wav", ".mp3")):
                shutil.copy(os.path.join(root, fn),
                            os.path.join(processed_dir, fn))
                copied += 1
    print(f"ℹ️  Copied {copied} files into {processed_dir!r}")

    # ── 4) Write labels.csv using unified EMO2ID ───────────────────────────────
    out_csv = os.path.join(processed_dir, "labels.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for fn in sorted(os.listdir(processed_dir)):
            if not fn.lower().endswith((".wav", ".mp3")):
                continue

            # strip augmentation suffix
            name = os.path.splitext(fn)[0]
            for suf in ("_noise", "_pitch", "_stretch", "_shift"):
                if name.endswith(suf):
                    name = name[:-len(suf)]
                    break

            canon = None

            # 1) IEMOCAP lookup (English only)
            if lang == "en" and name in label_map:
                canon = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}[label_map[name]]

            # 2) CREMA-D: 3rd underscore field
            if canon is None and "_" in name:
                parts = name.split("_")
                if len(parts) >= 3:
                    canon = _CREMAD_MAP.get(parts[2])

            # 3) REYERSON / RAVDESS: 3rd hyphen field
            if canon is None and "-" in name:
                parts = name.split("-")
                if len(parts) >= 3:
                    code = parts[2]
                    canon = _REYERSON_MAP.get(code) or _RAVDESS_MAP.get(code)

            # 4) EMODB: 6th character of filename (German only)
            if canon is None and lang == "de" and len(name) > 5:
                code = name[5].upper()
                canon = _EMODB_MAP.get(code)
                if code and canon is None:
                    print(f"⚠️ Unknown EMODB code '{code}' in {fn}")

            lbl = EMO2ID.get(canon, "-1")
            writer.writerow([fn, lbl])

    print(f"✅ Done: labels.csv generated with {copied} entries in {processed_dir!r}")


















# ------------------------------------------------------------------------------
# 4a. Generic  train/val splitter for ASR & Emotion
# ------------------------------------------------------------------------------
import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(input_dir, output_dir, test_size=0.2, seed=42, exts=(".wav", ".mp3"), suffixes=("_noise","_pitch","_stretch")):
    """
    Splits audio into train (left in place) and val (moved to output_dir),
    grouping each original + its augmented variants (_noise, _pitch, _stretch)
    so they never get separated.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"No directory at {input_dir}")

    # 1) Gather all files
    all_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    ]
    if not all_files:
        raise ValueError(f"No audio files found in {input_dir}")

    # 2) Build grouping: root_name → [filenames]
    groups = {}
    for fn in all_files:
        name, ext = os.path.splitext(fn)
        # strip any augmentation suffix
        root = name
        for suf in suffixes:
            if root.endswith(suf):
                root = root[:-len(suf)]
                break
        groups.setdefault(root, []).append(fn)

    # 3) Decide which roots go to val
    roots = sorted(groups.keys())
    # adjust test_size if very few groups
    if len(roots) <= 5:
        test_size = min(test_size, (len(roots) - 1) / len(roots))
    train_roots, val_roots = train_test_split(roots, test_size=test_size, random_state=seed)

    # 4) Move all files in val_roots to output_dir
    os.makedirs(output_dir, exist_ok=True)
    moved = 0
    for root in val_roots:
        for fn in groups[root]:
            src = os.path.join(input_dir, fn)
            dst = os.path.join(output_dir, fn)
            shutil.move(src, dst)
            moved += 1

    total = sum(len(files) for files in groups.values())
    print(f"Split {input_dir} → {output_dir}: moved {moved} files ({len(val_roots)} groups), "
          f"{total - moved} files remain ({len(train_roots)} groups) for training.")




# ------------------------------------------------------------------------------
# 4b. Master Preprocess Function
# ------------------------------------------------------------------------------
def preprocess_data(
    skip_preprocessing_asr: bool = False,
    skip_preprocessing_emotion: bool = False,
    skip_splitting: bool = False,
):
    """
    Preprocesses raw + augmented data for ASR and SER and
    organizes everything into train/val splits.
    """
    print("🔄 Starting full preprocessing pipeline…")

    # ── Paths ─────────────────────────────────────────────────────────────────────
    raw_asr_en       = "data/raw/asr/en/clips"
    raw_asr_de       = "data/raw/asr/de/clips"
    augmented_asr_en = "data/raw/asr/en/augmented"
    augmented_asr_de = "data/raw/asr/de/augmented"
    proc_asr_en      = "data/processed/train/asr/en"
    proc_asr_de      = "data/processed/train/asr/de"

    raw_emotion_en   = "data/raw/emotion/en"
    raw_emotion_de   = "data/raw/emotion/de"
    augmented_ser_en = "data/raw/emotion/en/augmented"
    augmented_ser_de = "data/raw/emotion/de/augmented"
    proc_emotion_en  = "data/processed/train/emotion/en"
    proc_emotion_de  = "data/processed/train/emotion/de"

    # ── ASR ──────────────────────────────────────────────────────────────────────
    if not skip_preprocessing_asr:
        # 1a) copy originals into the “augmented” dir
        os.makedirs(augmented_asr_en, exist_ok=True)
        for f in os.listdir(raw_asr_en):
            if f.lower().endswith((".wav", ".mp3")):
                shutil.copy(
                    os.path.join(raw_asr_en, f),
                    os.path.join(augmented_asr_en, f)
                )
        os.makedirs(augmented_asr_de, exist_ok=True)
        for f in os.listdir(raw_asr_de):
            if f.lower().endswith((".wav", ".mp3")):
                shutil.copy(
                    os.path.join(raw_asr_de, f),
                    os.path.join(augmented_asr_de, f)
                )

        # 1b) then generate the pitch/noise/stretch variants in the same dirs
        augment_data(raw_asr_en, augmented_asr_en)
        augment_data(raw_asr_de, augmented_asr_de)

        # 2) preprocess that augmented directory (now contains both original & aug)
        preprocess_asr(
            data_dir=augmented_asr_en,
            processed_dir=proc_asr_en,
            backbone_ckpt="facebook/wav2vec2-base-960h",
            device=device,
        )
        preprocess_asr(
            data_dir=augmented_asr_de,
            processed_dir=proc_asr_de,
            backbone_ckpt="facebook/wav2vec2-base-960h",
            device=device,
        )
    else:
        print("⚠ Skipping ASR preprocessing.")


    # ── SER (Emotion) ───────────────────────────────────────────────────────────
    raw_emotion_en   = "data/raw/emotion/en"
    raw_emotion_de   = "data/raw/emotion/de"
    proc_emotion_en  = "data/processed/train/emotion/en"
    proc_emotion_de  = "data/processed/train/emotion/de"

    if not skip_preprocessing_emotion:
        # — German side: one 'augmented' folder under raw_emotion_de
        de_aug        = os.path.join(raw_emotion_de, "augmented")
        first_time_de = not os.path.isdir(de_aug) or not os.listdir(de_aug)
        os.makedirs(de_aug, exist_ok=True)

        if first_time_de:
            for root, _, files in os.walk(raw_emotion_de):
                # skip the augmented folder itself
                if os.path.abspath(root).startswith(os.path.abspath(de_aug)):
                    continue
                for fn in files:
                    if fn.lower().endswith((".wav", ".mp3")):
                        src = os.path.join(root, fn)
                        dst = os.path.join(de_aug, fn)
                        shutil.copy(src, dst)
            print(f"ℹ️  Augmenting German emotion data in {de_aug}")
            augment_data(raw_emotion_de, de_aug)
        else:
            print(f"ℹ️  German augmented folder already exists at {de_aug}, skipping.")

        preprocess_emotion(
            data_dir=de_aug,
            processed_dir=proc_emotion_de,
            backbone_ckpt="facebook/wav2vec2-base-960h",
            device=device,
        )

        # — English side: one 'augmented' folder under raw_emotion_en
        en_aug        = os.path.join(raw_emotion_en, "augmented")
        first_time_en = not os.path.isdir(en_aug) or not os.listdir(en_aug)
        os.makedirs(en_aug, exist_ok=True)

        if first_time_en:
            for root, _, files in os.walk(raw_emotion_en):
                # skip the augmented folder itself
                if os.path.abspath(root).startswith(os.path.abspath(en_aug)):
                    continue
                for fn in files:
                    if fn.lower().endswith((".wav", ".mp3")):
                        src = os.path.join(root, fn)
                        dst = os.path.join(en_aug, fn)
                        shutil.copy(src, dst)
            print(f"ℹ️  Augmenting English emotion data in {en_aug}")
            augment_data(raw_emotion_en, en_aug)
        else:
            print(f"ℹ️  English augmented folder already exists at {en_aug}, skipping.")

        preprocess_emotion(
            data_dir=en_aug,
            processed_dir=proc_emotion_en,
            backbone_ckpt="facebook/wav2vec2-base-960h",
            device=device,
        )
    else:
        print("⚠ Skipping SER preprocessing.")


    # ── Train/Val Split ───────────────────────────────────────────────────────────
    if not skip_splitting:
        print("🔀 Splitting into train/val…")
        # ASR splits
        split_data(proc_asr_en, "data/processed/val/asr/en")
        split_data(proc_asr_de, "data/processed/val/asr/de")
        # SER (emotion) splits
        split_data(proc_emotion_en, "data/processed/val/emotion/en")
        split_data(proc_emotion_de, "data/processed/val/emotion/de")

        # ── Copy the manifests so val/…/labels.csv exists ───────────────────────────
        shutil.copy(
            os.path.join(proc_asr_en,      "labels.csv"),
            os.path.join("data/processed/val/asr/en",      "labels.csv")
        )
        shutil.copy(
            os.path.join(proc_asr_de,      "labels.csv"),
            os.path.join("data/processed/val/asr/de",      "labels.csv")
        )
        shutil.copy(
            os.path.join(proc_emotion_en,  "labels.csv"),
            os.path.join("data/processed/val/emotion/en",  "labels.csv")
        )
        shutil.copy(
            os.path.join(proc_emotion_de,  "labels.csv"),
            os.path.join("data/processed/val/emotion/de",  "labels.csv")
        )
    else:
        print("⚠ Skipping train/val splitting as requested.")




def load_asr_manifest(csv_path, data_dir):
    """
    Loads an ASR manifest CSV (filename, transcript) and returns
    a list of full‐path audio filenames and a parallel list of transcripts.
    """
    import csv
    filenames, transcripts = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(os.path.join(data_dir, row["filename"]))
            transcripts.append(row["transcript"])
    return filenames, transcripts


def evaluate_confusion_matrix(ground_truth, predictions, target_names=None):
    """
    Computes and prints the confusion matrix and classification report.
    """
    cm = confusion_matrix(ground_truth, predictions)
    report = classification_report(ground_truth, predictions, target_names=target_names)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
# NEW: End of Utility Functions

if __name__ == "__main__":
    preprocess_data()
