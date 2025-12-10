#!/usr/bin/env python3
import argparse
import subprocess
import json
import torch
import sys
import os
import shlex

# Ensure src/ is on the import path (harmless if it doesn't exist)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_data       # data preprocessing
from augmentation import augment_data           # data augmentation
from inference import infer_asr, infer_ser, batch_infer
from main_app import app, socketio


def run_preprocess(args):
    preprocess_data(
        skip_preprocessing_asr=args.skip_asr,
        skip_preprocessing_emotion=args.skip_ser,
        skip_splitting=args.skip_splitting
    )

def run_augment(args):
    augment_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text=args.text,
        emotion=args.emotion
    )

def run_train(args):
    # Prefer src/train.py if present; fall back to train.py in project root
    _ws = os.path.dirname(__file__)
    _candidates = [os.path.join(_ws, "src", "train.py"),
                   os.path.join(_ws, "train.py")]
    for _cand in _candidates:
        if os.path.exists(_cand):
            script = _cand
            break
    else:
        raise FileNotFoundError(
            f"Could not find train.py. Tried: {', '.join(_candidates)}"
        )

    # NOTE: Do NOT use torchrun here. The SLURM script is the only launcher.
    cmd = [
        "python3",
        script,
        f"--device={args.device}",
        f"--phase={args.phase}",            # pass through the phase flag
        # ASR args
        f"--asr_learning_rate={args.asr_lr}",
        f"--asr_batch_size={args.asr_bs}",
        f"--asr_epochs={args.asr_epochs}",
        f"--asr_patience={args.asr_patience}",
        f"--asr_checkpoint={args.asr_ckpt}",
        f"--asr_lang={args.asr_lang}",
        # SER args
        f"--ser_learning_rate={args.ser_lr}",
        f"--ser_batch_size={args.ser_bs}",
        f"--ser_epochs={args.ser_epochs}",
        f"--ser_dropout={args.ser_dropout}",
        f"--ser_patience={args.ser_patience}",
        f"--ser_checkpoint={args.ser_ckpt}",
        f"--ser_lang={args.ser_lang}",
    ]
    print("Running training:", " ".join(map(shlex.quote, cmd)))

    # Ensure checkpoint directories exist (save targets)
    for d in [args.asr_ckpt, args.ser_ckpt]:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    subprocess.run(cmd, check=True)

def run_infer(args):
    result = {
        "transcription": infer_asr(args.file, None, None, args.device),
        "emotion":      infer_ser(args.file, None, None, args.device)
    }
    print(json.dumps(result, indent=2))

def run_batch_infer(args):
    results = batch_infer(
        args.audio_dir, None, None, None, None, args.device
    )
    print(json.dumps(results, indent=2))

def run_serve(args):
    # import only when we actually want to serve the web dashboard
    socketio.run(
        app,
        debug=args.debug,
        host=args.host,
        port=args.port
    )

def main():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for speech_transcription_project"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p = sub.add_parser("preprocess", help="Run full preprocessing pipeline")
    p.add_argument("--skip_asr",       action="store_true")
    p.add_argument("--skip_ser",       action="store_true")
    p.add_argument("--skip_splitting", action="store_true")
    p.set_defaults(func=run_preprocess)

    # augment
    a = sub.add_parser("augment", help="Augment data")
    a.add_argument("--input_dir",   required=True)
    a.add_argument("--output_dir",  required=True)
    a.add_argument("--text",        default=None)
    a.add_argument("--emotion",     default="happy")
    a.set_defaults(func=run_augment)

    # train (now just a thin wrapper that calls train.py directly)
    tr = sub.add_parser("train", help="Train ASR + SER models")
    tr.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--phase", choices=["all", "asr", "ser"], default="all",
                    help="Train only ASR, only SER, or both.")
    tr.add_argument("--asr_lr",      type=float, required=True)
    tr.add_argument("--asr_bs",      type=int,   required=True)
    tr.add_argument("--asr_epochs",  type=int,   required=True)
    tr.add_argument("--asr_patience",type=int,   required=True)
    tr.add_argument("--asr_ckpt",    required=True)
    tr.add_argument("--asr_lang",    choices=["en","de"], default="en")
    tr.add_argument("--ser_lr",      type=float, required=True)
    tr.add_argument("--ser_bs",      type=int,   required=True)
    tr.add_argument("--ser_epochs",  type=int,   required=True)
    tr.add_argument("--ser_dropout", type=float, required=True)
    tr.add_argument("--ser_patience",type=int,   required=True)
    tr.add_argument("--ser_ckpt",    required=True)
    tr.add_argument("--ser_lang",    choices=["en","de"], default="en")
    tr.set_defaults(func=run_train)

    # infer
    inf = sub.add_parser("infer", help="Infer on a single audio file")
    inf.add_argument("--file",   required=True)
    inf.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    inf.set_defaults(func=run_infer)

    # batch-infer
    bi = sub.add_parser("batch-infer", help="Batch inference on a directory")
    bi.add_argument("--audio_dir", required=True)
    bi.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    bi.set_defaults(func=run_batch_infer)

    # serve
    srv = sub.add_parser("serve", help="Launch the web dashboard")
    srv.add_argument("--host",  default="0.0.0.0")
    srv.add_argument("--port",  type=int, default=5000)
    srv.add_argument("--debug", action="store_true")
    srv.set_defaults(func=run_serve)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
