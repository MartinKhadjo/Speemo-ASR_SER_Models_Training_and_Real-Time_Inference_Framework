import os
import sys
import glob
import subprocess, getpass, stat
import threading
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import tempfile, uuid, json, os

import torch
import psutil
import shutil
# make sure you have this import too (for chmod)
import stat

# -------------------- NEW IMPORTS FOR SLURM API --------------------
import re
import time
import pathlib
from jinja2 import Template
# ------------------------------------------------------------------

# Ensure your `src/` folder is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from inference import infer_asr, infer_ser, batch_infer
from model_loader import load_asr_model, load_ser_model
from preprocessing import preprocess_data
from augmentation import augment_data
from model_loader import _ensure_pretrained_backbone

SER_NUM_CLASSES = 6  # keep in sync with preprocessing.EMOTIONS & inference.EMOTION_MAP



# Slurm binaries (bound into the container by runserve.sh)
# IMPORTANT: do NOT use shutil.which here (it may find /usr/local_host/bin/...).
# --- Slurm resolver (use env from runserve.sh, fallback to PATH, then /usr/bin/*) ---
def resolve_slurm():
    sbatch = os.environ.get("SBATCH") or shutil.which("sbatch") or "/usr/bin/sbatch"
    squeue = os.environ.get("SQUEUE") or shutil.which("squeue") or "/usr/bin/squeue"
    sacct  = os.environ.get("SACCT")  or shutil.which("sacct")  or "/usr/bin/sacct"
    return sbatch, squeue, sacct

# One-time visibility on startup:
try:
    print("Resolved Slurm tools at startup:", resolve_slurm(), "HPCWORK=", os.environ.get("HPCWORK"), flush=True)
except Exception as _e:
    pass


# -------------------- NEW CONSTANTS + HELPERS FOR SLURM API --------------------
HPCWORK = os.environ.get("HPCWORK", f"/hpcwork/{os.environ.get('USER','user')}")
SPEEMO_DIR = os.path.join(HPCWORK, "speemo_md_0.005")
SCRIPTS_DIR = os.path.join(HPCWORK, "scripts")
SLURM_TEMPLATE = os.path.join(SPEEMO_DIR, "templates", "slurm_template.sh")
# on-disk template file (not the Jinja in templates/)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

def _render_slurm(vars_dict: dict) -> str:
    with open(SLURM_TEMPLATE, "r") as f:
        t = Template(f.read())
    return t.render(**vars_dict)

def _num(x, default, cast=float):
    try:
        return cast(x)
    except Exception:
        return default
# -------------------------------------------------------------------------------

# ==================== FIX: define app/socketio BEFORE any @app.* decorators ====================
app = Flask(__name__)
# --- NEW: upload size guard + JSON unicode ---
app.config.update(
    MAX_CONTENT_LENGTH=256 * 1024 * 1024,  # 256 MB
)
socketio = SocketIO(app, cors_allowed_origins="*")
# ===============================================================================================


# -------------------- NEW: SLURM SUBMIT + STATUS API --------------------
@app.post("/api/submit-train")
def submit_train():
    data = request.get_json(force=True)

    # ---- sanitize + defaults (keep this tight; never render arbitrary strings to shell) ----
    vars_dict = {
        "user": os.environ.get("USER", "user"),
        "hpc": HPCWORK,
        "device": data.get("device", "cuda"),           # "cuda" or "cpu"
        "time_limit": data.get("time_limit", "04:00:00"),
        "nodes": _num(data.get("nodes", 1), 1, int),
        "gpus":  _num(data.get("gpus", 1), 1, int),
        "cpus":  _num(data.get("cpus", 8), 8, int),
        "mem_gb": _num(data.get("mem_gb", 32), 32, int),

        # ASR
        "asr_learning_rate": _num(data.get("asr_learning_rate", 1e-5), 1e-5, float),
        "asr_batch_size": _num(data.get("asr_batch_size", 4), 4, int),
        "asr_epochs": _num(data.get("asr_epochs", 10), 10, int),
        "asr_patience": _num(data.get("asr_patience", 3), 3, int),
        "asr_lang": data.get("asr_lang", "en"),

        # SER
        "ser_learning_rate": _num(data.get("ser_learning_rate", 1e-5), 1e-5, float),
        "ser_batch_size": _num(data.get("ser_batch_size", 8), 8, int),
        "ser_epochs": _num(data.get("ser_epochs", 10), 10, int),
        "ser_patience": _num(data.get("ser_patience", 3), 3, int),
        "ser_dropout": _num(data.get("ser_dropout", 0.2), 0.2, float),
        "ser_lang": data.get("ser_lang", "en"),

        # checkpoints
        "checkpoint_name": data.get("checkpoint_name") or time.strftime("run_%Y%m%d_%H%M%S"),
        "existing_checkpoint": data.get("existing_checkpoint", ""),
        # optional networking & partition niceness can be added here
    }

    # Render, write job file
    script_text = _render_slurm(vars_dict)
    jobfile = os.path.join(SCRIPTS_DIR, f"train_{vars_dict['checkpoint_name']}.sbatch")
    with open(jobfile, "w") as f:
        f.write(script_text)
    os.chmod(jobfile, 0o750)

    # Submit via wrapper created by runserve.sh
    sbatch_bin = os.environ.get("SBATCH", "sbatch")
    try:
        res = subprocess.run([sbatch_bin, jobfile], check=True, capture_output=True, text=True)
        out = res.stdout.strip() + "\n" + res.stderr.strip()
        m = re.search(r"Submitted batch job (\d+)", out)
        jobid = m.group(1) if m else None
        return jsonify({
            "ok": True,
            "job_id": jobid,
            "message": out,
            "log_hint": f"{HPCWORK}/logs/train.{jobid}.out" if jobid else None
        })
    except subprocess.CalledProcessError as e:
        return jsonify({"ok": False, "error": e.stdout + e.stderr}), 500

@app.get("/api/job-status")
def job_status():
    jobid = request.args.get("jobid", "").strip()
    if not jobid.isdigit():
        return jsonify({"ok": False, "error": "Invalid jobid"}), 400

    squeue_bin = os.environ.get("SQUEUE", "squeue")
    try:
        res = subprocess.run(
            [squeue_bin, "-j", jobid, "-o", "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"],
            capture_output=True, text=True, check=True
        )
        return jsonify({"ok": True, "stdout": res.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"ok": False, "error": e.stdout + e.stderr}), 500
# ----------------------------------------------------------------------


# Declare global models
global asr_model, asr_processor, ser_model, ser_feature_extractor

# UTF-8 logging
sys.stdout.reconfigure(encoding="utf-8")

# Supported languages for transcription
transcription_languages = ["en", "de"]

# Device for Flask-side inference (GPU if available)
# --- REPLACED with robust chooser to avoid sm_61 / low-VRAM crashes ---
def choose_device():
    # Prefer CPU if CUDA looks risky (old SM or tiny VRAM)
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory
            if major < 7 or total < 4 * 1024**3:
                print(f"[Flask] Falling back to CPU (SM {major}.{minor}, VRAM {total/1e9:.1f} GB)")
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception as e:
        print(f"[Flask] CUDA probe failed: {e}")
    return torch.device("cpu")

device = choose_device()
print(f"[Flask] Using device for inference: {device}")


# --- Helper: load checkpoints by base name for streaming/inference ---
def load_checkpoint_from_name(base_name: str):
    """
    Load globals asr_model/asr_processor/ser_model/ser_feature_extractor
    from models/checkpoints/<base>_{asr,ser}[.pt or dir].
    """
    ckpt_root = os.path.join(os.path.dirname(__file__), "models", "checkpoints")
    asr_pt = os.path.join(ckpt_root, f"{base_name}_asr.pt")
    ser_pt = os.path.join(ckpt_root, f"{base_name}_ser.pt")

    asr_dir = asr_pt if os.path.isfile(asr_pt) else os.path.join(ckpt_root, f"{base_name}_asr")
    ser_dir = ser_pt if os.path.isfile(ser_pt) else os.path.join(ckpt_root, f"{base_name}_ser")

    global asr_model, asr_processor, ser_model, ser_feature_extractor
    asr_model, asr_processor = load_asr_model(asr_dir, device=device)
    ser_model, ser_feature_extractor = load_ser_model(ser_dir, n_emotions=SER_NUM_CLASSES, device=device)

# === NEW: robust checkpoint helpers (don’t remove/replace old funcs, just add) ===
def _ckpt_root() -> str:
    return os.path.join(os.path.dirname(__file__), "models", "checkpoints")

def _list_checkpoint_bases():
    """
    Return a sorted list of base names for which we find either *_asr(.pt|/) or *_ser(.pt|/).
    """
    root = _ckpt_root()
    os.makedirs(root, exist_ok=True)
    bases = set()
    for name in os.listdir(root):
        if name.endswith("_asr"):
            bases.add(name[:-4])
        elif name.endswith("_ser"):
            bases.add(name[:-4])
        elif name.endswith("_asr.pt"):
            bases.add(name[:-7])
        elif name.endswith("_ser.pt"):
            bases.add(name[:-7])
    return sorted(bases)

def _resolve_ckpt_paths(base):
    """
    Resolve concrete ASR/SER paths (state-dict .pt or directory).
    Returns (asr_path, ser_path) or (None, None) if missing.
    """
    root = _ckpt_root()
    asr_pt = os.path.join(root, f"{base}_asr.pt")
    ser_pt = os.path.join(root, f"{base}_ser.pt")
    asr_dir = os.path.join(root, f"{base}_asr")
    ser_dir = os.path.join(root, f"{base}_ser")

    asr_path = asr_pt if os.path.isfile(asr_pt) else (asr_dir if os.path.exists(asr_dir) else None)
    ser_path = ser_pt if os.path.isfile(ser_pt) else (ser_dir if os.path.exists(ser_dir) else None)
    return asr_path, ser_path

def _pick_latest_base():
    """
    Pick the latest checkpoint base by mtime (considers *_asr(.pt|/) and *_ser(.pt|/)).
    """
    root = _ckpt_root()
    candidates = {}
    for base in _list_checkpoint_bases():
        asr, ser = _resolve_ckpt_paths(base)
        mtimes = []
        for p in (asr, ser):
            if p and os.path.exists(p):
                try:
                    mtimes.append(os.path.getmtime(p))
                except Exception:
                    pass
        if mtimes:
            candidates[base] = max(mtimes)
    if not candidates:
        return None
    return max(candidates, key=candidates.get)



# -----------------------------------------------------------
#  LOAD THE ASR AND SER MODELS + PROCESSORS
# -----------------------------------------------------------
# -----------------------------------------------------------
#  LOAD THE ASR AND SER MODELS + PROCESSORS (with new-checkpoint support)
# -----------------------------------------------------------
@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    global asr_model, asr_processor, ser_model, ser_feature_extractor

    # 1) Grab form inputs
    existing = request.form.get("existing_checkpoint", "").strip()
    new_name = request.form.get("checkpoint_name", "").strip()
    asr_lang = request.form.get("asr_lang", "en")
    ser_lang = request.form.get("ser_lang", "en")

    # 2) Decide base name and whether it’s new
    if existing:
        base = existing
        is_new = False
    elif new_name:
        base = new_name
        is_new = True
    else:
        return jsonify({"error": "Please select or name a checkpoint."}), 400

    # 3) Prepare checkpoint directories
    ckpt_root = "models/checkpoints"
    os.makedirs(ckpt_root, exist_ok=True)
    asr_dir = os.path.join(ckpt_root, f"{base}_asr")
    ser_dir = os.path.join(ckpt_root, f"{base}_ser")

    # 4) If it’s a new run, bootstrap from the pretrained backbones
    if is_new:
        from model_loader import _ensure_pretrained_backbone

        # ── ASR ────────────────────────────────────────────────────────────────
        # Copy raw backbone into asr_dir
        backbone_asr = _ensure_pretrained_backbone(asr_lang)
        shutil.copytree(backbone_asr, asr_dir, dirs_exist_ok=True)
        # Load fresh ASR model + processor, then save them into asr_dir
        m_asr, p_asr = load_asr_model(asr_lang, device=device, lang=asr_lang)
        m_asr.save_pretrained(asr_dir)
        p_asr.save_pretrained(asr_dir)

        # ── SER ────────────────────────────────────────────────────────────────
        backbone_ser = _ensure_pretrained_backbone(ser_lang)
        shutil.copytree(backbone_ser, ser_dir, dirs_exist_ok=True)
        m_ser, fe_ser = load_ser_model(ser_lang, n_emotions=SER_NUM_CLASSES, device=device,
                                       dropout=0.5, lang=ser_lang)
        m_ser.save_pretrained(ser_dir)        # saves `ser_head.pt`
        fe_ser.save_pretrained(ser_dir)

    # 5) Now load from the checkpoint folders
    try:
        asr_model, asr_processor = load_asr_model(asr_dir, device=device, lang=asr_lang)
        ser_model, ser_feature_extractor = load_ser_model(
            ser_dir, n_emotions=SER_NUM_CLASSES, device=device, dropout=0.5, lang=ser_lang
        )
        return jsonify({"message": f"✅ Loaded checkpoint: {base}"}), 200

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return jsonify({"error": str(e)}), 500




# -----------------------------------------------------------
#  RENDERING: HOMEPAGE
# -----------------------------------------------------------
@app.route("/")
def index():
    # use the folder inside your bind-mount at /workspace
    return render_template("index.html", checkpoints=_list_checkpoint_bases())

# --- NEW: health check for front-end quick probe ---
@app.route("/health")
def health():
    return jsonify({"ok": True, "device": str(device)}), 200




# -----------------------------------------------------------
#  TRAINING
# -----------------------------------------------------------
def stream_training_logs(process):
    """Read process stdout line by line and emit via SocketIO."""
    for line in process.stdout:
        line = line.strip()
        if line:
            print("TRAIN LOG:", line)
            socketio.emit("training_logs", {"log": line})
    process.stdout.close()


# -------------------- NEW: real-time log streaming from Slurm log file --------------------
def stream_training_logs_file(log_path, job_id):
    """
    Tail a Slurm log file and emit new lines via Socket.IO for real-time logs.
    This reuses the same 'training_logs' event the frontend can subscribe to:
      socket.on('training_logs', (payload) => { ... });
    """
    try:
        # Wait until the log file exists (job may not have started yet)
        while not os.path.exists(log_path):
            time.sleep(1.0)

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            # Start from the beginning so the UI sees the full history
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    # No new data; wait a bit and try again
                    time.sleep(1.0)
                    f.seek(where)
                    continue

                line = line.rstrip("\n")
                if line:
                    print("TRAIN LOG (tail):", line)
                    socketio.emit("training_logs", {"log": line, "job_id": job_id})
    except Exception as e:
        print(f"[stream_training_logs_file] error while tailing {log_path}: {e}")
# ----------------------------------------------------------------------


@app.route('/train', methods=['POST'])
def train():
    """
    Trigger ASR/SER training via Slurm:
    - Render slurm_template.sh -> $HPCWORK/scripts/train_<id>.sh
    - sbatch --parsable (returns just the jobid)
    - Respond 202 with jobid

    Additionally, start a background thread to tail the Slurm log file
    and push lines via Socket.IO ('training_logs') for real-time logs.
    """
    try:
        # 1) Collect and sanitize inputs (pass everything through to the template)
        params = request.form.to_dict(flat=True)

        def as_int(name, default):
            try:
                return int(request.form.get(name, default))
            except Exception:
                return int(default)

        # Resolve user/HPC paths (works with --cleanenv inside Apptainer)
        user = (os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser())
        hpc  = os.environ.get("HPCWORK", f"/hpcwork/{user}")
        job_id = uuid.uuid4().hex[:8]

        # Ensure 'device' default (GPU queue expects CUDA)
        params.setdefault("device", "cuda")

        # Add/override fields our template expects
        params.update({
            "user": user,
            "job_id": job_id,
            "hpc": hpc,
            "existing_checkpoint": request.form.get("existing_checkpoint", ""),
            "checkpoint_name":     request.form.get("checkpoint_name", ""),
            "gpus":       as_int("gpus", 1),
            "cpus":       as_int("cpus", 8),
            "mem_gb":     as_int("mem_gb", 64),
            "time_limit": request.form.get("time_limit", "48:00:00"),
        })

        # --- normalize checkbox booleans to 'on'/'off' strings ---
        for k in ("skip_preprocessing_asr", "skip_preprocessing_emotion",
                  "skip_splitting", "skip_tokenizer", "skip_preprocessing"):
            params[k] = "on" if request.form.get(k) == "on" else "off"

        # --- fan-out the master toggle to individual flags ---
        if params["skip_preprocessing"] == "on":
            params["skip_preprocessing_asr"] = "on"
            params["skip_preprocessing_emotion"] = "on"
            params["skip_splitting"] = "on"

        # 2) Ensure host-side dirs exist
        scripts_dir = os.path.join(hpc, "scripts")
        logs_dir    = os.path.join(hpc, "logs")
        os.makedirs(scripts_dir, exist_ok=True)
        os.makedirs(logs_dir,    exist_ok=True)

        # 3) Render SBATCH script (LF newlines), verify zsh shebang, chmod +x
        sbatch_text = render_template('slurm_template.sh', **params).strip() + "\n"
        script_path = os.path.join(scripts_dir, f"train_{job_id}.sh")
        with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(sbatch_text)

        if not sbatch_text.startswith("#!/usr/bin/zsh"):
            raise RuntimeError("SBATCH script must start with #!/usr/bin/zsh")

        os.chmod(script_path, 0o755)

        # 4) Resolve Slurm tools now (use wrapper paths from runserve.sh)
        sbatch_bin, squeue_bin, sacct_bin = resolve_slurm()
        print("Resolved Slurm tools (submit):", (sbatch_bin, squeue_bin, sacct_bin),
              "HPCWORK=", hpc, flush=True)

        if not os.path.exists(sbatch_bin):
            raise FileNotFoundError(
                f"sbatch not found at '{sbatch_bin}'. Check runserve.sh binds/export."
            )

        # 5) Submit and capture just the jobid
        result = subprocess.run(
            [sbatch_bin, '--parsable', script_path],
            capture_output=True, text=True, check=True
        )
        jobid = result.stdout.strip().splitlines()[-1]

        # --- NEW: start background thread to tail the Slurm log file and emit via Socket.IO ---
        log_path = os.path.join(logs_dir, f"train.{jobid}.out")
        tail_thread = threading.Thread(
            target=stream_training_logs_file,
            args=(log_path, jobid),
            daemon=True,
        )
        tail_thread.start()
        # --------------------------------------------------------------------------------------

        return jsonify({"job_id": jobid, "script": script_path}), 202

    except subprocess.CalledProcessError as e:
        # Surface the real Slurm stderr to the UI
        return jsonify({
            "error": "sbatch failed",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except Exception as e:
        app.logger.error(f"❌ Error submitting SLURM job: {e}")
        return jsonify({"error": str(e)}), 500


# ---- Logs ----
@app.route('/train_logs/<job_id>')
def train_logs(job_id):
    """
    Return the last 100 lines from the Slurm stdout file for this job.
    Our batch script writes logs to $HPCWORK/logs/train.%J.out.

    NOTE: For true real-time logs, the frontend can also listen to the
    'training_logs' Socket.IO event; this HTTP endpoint remains as a
    fallback / initial history loader.
    """
    import os, getpass
    from flask import jsonify

    user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
    hpc  = os.environ.get("HPCWORK", f"/hpcwork/{user}")
    log_path = os.path.join(hpc, "logs", f"train.{job_id}.out")

    if not os.path.exists(log_path):
        # Job may not have started writing yet
        return ('', 204)

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()[-100:]
        return jsonify(lines)
    except Exception as e:
        app.logger.error(f"Error reading log file {log_path}: {e}")
        return ('', 204)


# ---- Status ----
@app.route('/train_status/<job_id>')
def train_status(job_id):
    """
    Report Slurm job state. Try squeue first (live PD/R/etc).
    If not found there, fall back to sacct for terminal states
    (COMPLETED/FAILED/CANCELLED/etc).
    """
    try:
        # Resolve tools each time (picks up wrapper paths)
        _, squeue_bin, sacct_bin = resolve_slurm()
        print("Resolved Slurm tools (status):", (squeue_bin, sacct_bin),
              "HPCWORK=", os.environ.get("HPCWORK"), flush=True)

        # 1) Live queue (no header; just the state via %T)
        q = subprocess.run(
            [squeue_bin, "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=10
        )
        state = (q.stdout or "").strip()
        if state:
            return jsonify({"status": state})

        # 2) Not in queue anymore → accounting (historical terminal states)
        a = subprocess.run(
            [sacct_bin, "-j", job_id, "-X", "-n", "-o", "State"],
            capture_output=True, text=True, timeout=10
        )
        out = (a.stdout or "").strip()
        if out:
            first = out.splitlines()[0].strip().split()[0]
            if first:
                return jsonify({"status": first})

        # 3) Unknown / not found yet
        return jsonify({"status": "UNKNOWN"})

    except FileNotFoundError as e:
        app.logger.error(f"Slurm tool missing for job {job_id}: {e}")
        return jsonify({"status": "UNKNOWN", "error": "Slurm tools not available"}), 202
    except subprocess.TimeoutExpired:
        return jsonify({"status": "UNKNOWN", "error": "query timeout"}), 202
    except Exception as e:
        app.logger.error(f"Status query error for job {job_id}: {e}")
        return jsonify({"status": "UNKNOWN"}), 202




# -----------------------------------------------------------
#  PREPROCESS
# -----------------------------------------------------------
@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # master skip-all
        skip_all = request.form.get("skip_preprocessing", "off") == "on"
        if skip_all:
            return jsonify({"message": "Skipped all preprocessing (skip_preprocessing=on)."}), 200

        # granular flags
        skip_asr       = request.form.get("skip_preprocessing_asr", "off") == "on"
        skip_ser       = request.form.get("skip_preprocessing_emotion", "off") == "on"
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
#  FILE-UPLOAD INFERENCE ENDPOINT (Already Existing)
# -----------------------------------------------------------
@app.route("/process_audio_inference", methods=["POST"])
def process_audio_inference():
    """
    API endpoint for audio file upload inference (ASR + SER).
    Saves result to a JSON file.

    NOTE: The front-end select is outside the form; accept either 'model_choice'
    or 'model_choice_inference'. If neither provided, fall back to the latest
    checkpoint available under models/checkpoints.
    """
    # --- ensure we have a file payload ---
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request (multipart/form-data expected with field 'file')."}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # --- resolve model choice robustly ---
    model_choice = (request.form.get("model_choice", "").strip()
                    or request.form.get("model_choice_inference", "").strip())

    if not model_choice:
        # Try to pick the latest checkpoint automatically
        auto = _pick_latest_base()
        if not auto:
            return jsonify({"error": "No model_choice provided and no checkpoints found."}), 400
        print(f"[Inference] No model_choice provided; using latest checkpoint: {auto}")
        model_choice = auto

    # Check that the checkpoint paths exist before loading
    asr_ckpt_path, ser_ckpt_path = _resolve_ckpt_paths(model_choice)
    if not asr_ckpt_path or not os.path.exists(asr_ckpt_path):
        return jsonify({"error": f"ASR checkpoint not found for base '{model_choice}'"}), 404
    if not ser_ckpt_path or not os.path.exists(ser_ckpt_path):
        return jsonify({"error": f"SER checkpoint not found for base '{model_choice}'"}), 404

    ext = os.path.splitext(audio_file.filename)[1]
    tmp_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ext)
    audio_file.save(tmp_path)

    output_file = os.path.join("inference_results", str(uuid.uuid4()) + ".json")
    os.makedirs("inference_results", exist_ok=True)

    try:
        # ——— reload models with proper device ———
        global asr_model, asr_processor, ser_model, ser_feature_extractor
        asr_model, asr_processor = load_asr_model(asr_ckpt_path, device=device)
        ser_model, ser_feature_extractor = load_ser_model(ser_ckpt_path, n_emotions=SER_NUM_CLASSES, device=device)

        # run inference
        transcription = infer_asr(tmp_path, asr_model, asr_processor, device)
        emotion       = infer_ser(tmp_path, ser_model, ser_feature_extractor, device)

        result = {
            "transcription": transcription,
            "emotion": emotion,
            "output_file": output_file,
            "model_choice": model_choice
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return jsonify(result), 200

    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



# -----------------------------------------------------------
#  NEW: REAL-TIME STREAMING FOR UNITY (WebSocket)
# -----------------------------------------------------------
stored_results = []  # Optional: Keep a list in memory

streaming_model_loaded = {"name": None}  # Global tracker for currently loaded checkpoint

# -------------------- NEW: Throttling for websocket emits --------------------
LAST_EMIT_TS = 0.0
EMIT_INTERVAL = 1.0  # seconds between websocket emits to avoid spamming the frontend


@socketio.on('unity_audio_chunk')
def handle_unity_audio_chunk(data):
    """
    Receives raw PCM samples from Unity **or browser microphone** in real-time
    (either as bytes-like PCM or as a list of floats/ints),
    dynamically loads the selected checkpoint,
    and returns transcription + emotion.
    """
    try:
        audio_bytes = data.get("audio", None)
        model_choice = data.get("model_choice", None)

        if audio_bytes is None or not model_choice:
            emit("asr_emotion_result", {"error": "Missing audio or model_choice."})
            return

        # IMPORTANT:
        # - If coming from the browser AudioContext, `audio` is a list of float32 samples in [-1, 1].
        #   In that case, we leave it as a Python list and let inference._decode_streaming_payload
        #   handle it (it treats list/tuple as raw samples).
        #
        # - If some client sends actual bytes/memoryview, we keep it bytes-like so that
        #   _decode_streaming_payload can treat it as PCM16/float32 as needed.
        #
        # So we ONLY normalize memoryview -> bytes; we DO NOT convert generic lists to bytes
        # anymore, to avoid "bytes must be in range(0, 256)" / "'float' object cannot be
        # interpreted as an integer" for float sample arrays.

        # If it's a memoryview, convert to bytes
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()

        # Reload model only if different from the current one
        if streaming_model_loaded["name"] != model_choice:
            try:
                load_checkpoint_from_name(model_choice)
                streaming_model_loaded["name"] = model_choice
                print(f"✅ Loaded streaming model: {model_choice}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint {model_choice}: {e}")
                emit("asr_emotion_result", {"error": f"Model load failed: {str(e)}"})
                return

        # Perform streaming inference
        from inference import streaming_infer_asr, streaming_infer_ser
        transcription = streaming_infer_asr(audio_bytes, asr_model, asr_processor, device)
        emotion = streaming_infer_ser(audio_bytes, ser_model, ser_feature_extractor, device)

        result = {
            "transcription": transcription,
            "emotion": emotion
        }

        stored_results.append(result)

        # -------------------- NEW: throttle websocket emissions --------------------
        global LAST_EMIT_TS
        now = time.time()
        if now - LAST_EMIT_TS < EMIT_INTERVAL:
            print(f"[streaming] throttling emit; dt={now - LAST_EMIT_TS:.3f}s < {EMIT_INTERVAL}s")
            return
        LAST_EMIT_TS = now
        # ------------------------------------------------------------------------

        emit("asr_emotion_result", result)

    except Exception as e:
        print("❌ Error in handle_unity_audio_chunk:", e)
        emit("asr_emotion_result", {"error": str(e)})


# -----------------------------------------------------------
#  MAIN
# -----------------------------------------------------------
if __name__ == '__main__':
    print("Starting Flask + SocketIO app...")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
