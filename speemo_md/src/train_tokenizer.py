import os
import json
import numpy as np
from collections import Counter
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

def train_all_tokenizers(
    langs,
    input_root="data/processed/train/asr",
    vocab_size=50,
    out_root="models/pretrained"
):
    """
    Train and save a character-level CTC tokenizer + processor for each language.

    langs: list of language codes (e.g. ["en","de"])
    input_root: where your per-lang ASR .npy files live
    vocab_size: target vocab size for the new tokenizer
    out_root: root under which to save each lang's tokenizer+processor
    """
    # ── 1) Locate a locally cached Wav2Vec2 processor to get its feature_extractor ──
    fe = None
    for lang in langs:
        local_dir = os.path.join(out_root, lang)
        if os.path.isdir(local_dir):
            proc = Wav2Vec2Processor.from_pretrained(local_dir)
            fe = proc.feature_extractor
            break
    if fe is None:
        raise RuntimeError(
            "No local Wav2Vec2 processor found under 'models/pretrained/<lang>'. "
            "Run load_multitask_model at least once to cache one."
        )

    # ── 2) Loop per language, gather transcripts, train & save ──
    for lang in langs:
        transcripts = []
        folder = os.path.join(input_root, lang)
        if not os.path.isdir(folder):
            print(f"⚠ no ASR dir for {lang}, skipping tokenizer")
            continue

        for fn in os.listdir(folder):
            if not fn.endswith(".npy"):
                continue
            try:
                item = np.load(os.path.join(folder, fn), allow_pickle=True).item()
                txt = item.get("transcript", "").strip().lower()
                if txt:
                    transcripts.append(txt)
            except Exception:
                continue

        if not transcripts:
            print(f"⚠ no transcripts for {lang}, skipping tokenizer")
            continue

        # ── Build character vocabulary manually ──
        char_counter = Counter()
        for line in transcripts:
            char_counter.update(list(line))
        chars = sorted(char_counter)

        # Special tokens
        vocab_list = ["<pad>", "<unk>", "|"] + chars
        vocab_dict = {ch: i for i, ch in enumerate(vocab_list)}

        # Save vocab.json
        out = os.path.join(out_root, lang)
        os.makedirs(out, exist_ok=True)
        vocab_path = os.path.join(out, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2)

        # ── Initialize tokenizer from vocab.json ──
        tok = Wav2Vec2CTCTokenizer(
            vocab_file=vocab_path,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|"
        )
        tok.save_pretrained(out)

        proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
        proc.save_pretrained(out)

        print(f"✔ tokenizer+processor saved for {lang} ({len(transcripts)} utts)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Wav2Vec2 CTC tokenizer for one or more languages.")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code(s), comma-separated (e.g., 'en,de')"
    )
    parser.add_argument("--input_root", type=str, default="data/processed/train/asr")
    parser.add_argument("--vocab_size", type=int, default=50)
    parser.add_argument("--out_root", type=str, default="models/pretrained")

    args = parser.parse_args()

    langs = [l.strip() for l in args.language.split(",")]
    train_all_tokenizers(
        langs=langs,
        input_root=args.input_root,
        vocab_size=args.vocab_size,
        out_root=args.out_root
    )
