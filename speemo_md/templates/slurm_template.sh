#!/usr/bin/zsh
# ---------- Required Slurm header (RWTH needs zsh + SBATCH lines first) ----------
# NOTE: This template is expected at: /hpcwork/<user>/speemo_md_0.005/slurm_template_sh

{% set use_gpu = (device | default('cuda')) == 'cuda' %}

#SBATCH --job-name={{ (checkpoint_name or existing_checkpoint) | default('speemo') }}
#SBATCH --time={{ time_limit | default('02:00:00') }}
#SBATCH --nodes={{ nodes | default('1') }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{ cpus | default('8') }}
#SBATCH --mem={{ mem_gb | default('32') }}G
#SBATCH --partition={{ 'c23g' if use_gpu else 'c23ms' }}       # GPU: c23g, CPU: c23ms
{% if use_gpu -%}
#SBATCH --gres=gpu:hopper:{{ gpus | default('1') }}            # GPUs per node (H100 "Hopper")
{%- endif %}
#SBATCH --constraint=Rocky9
#SBATCH --chdir={{ hpc }}/speemo_md_0.005
#SBATCH --export=NONE
#SBATCH --output={{ hpc }}/logs/train.%J.out
#SBATCH --error={{ hpc }}/logs/train.%J.err

set -euo pipefail

# Load Apptainer if needed (safe even if it's already on PATH)
module load apptainer 2>/dev/null || true

# Paths
export USER="{{ user }}"
export HPCWORK="{{ hpc }}"
export SIF="{{ hpc }}/containers/speemo.sif"
export WS="/workspace"

# Hugging Face cache on HPCWORK
export HF_HOME="{{ hpc }}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Writable Numba cache for librosa (bind into container)
export NUMBA_CACHE_DIR="{{ hpc }}/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# Ensure dirs exist
mkdir -p "$HPCWORK/logs" "$HPCWORK/scripts" "$HPCWORK/speemo_md_0.005"/{data,models}

# Make sure no stray bind-path env leaks into Apptainer (processed outside the container)
unset APPTAINER_BINDPATH SINGULARITY_BINDPATH

# -------------------- Cluster / multi-node rendezvous setup --------------------
# Use SLURM's nodelist to pick a stable rendezvous endpoint (rank-0 hostname)
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${SLURM_NNODES:-{{ nodes | default('1') }}}"
# GPUs per node requested (fallback to template default if SLURM var not present)
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-{{ gpus | default('1') }}}"

echo "[DIST] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"

# Some NCCL/QoL defaults (safe on RWTH)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# If your fabric requires a specific NIC, set NIC name via template var `nic` (e.g., 'ib0' or 'eno1')
{% if nic is defined and nic %}
export NCCL_SOCKET_IFNAME="{{ nic }}"
{% endif %}

# -------------------- Optional preprocessing (checkboxes render "on") --------------------
{% set skip_all = (skip_preprocessing | default('off')) == 'on' %}

{% if not skip_all and (skip_preprocessing_asr != 'on' or skip_preprocessing_emotion != 'on' or skip_splitting != 'on') %}
echo "[PRE] Running preprocessing with flags: skip_asr={{ skip_preprocessing_asr }}, skip_ser={{ skip_preprocessing_emotion }}, skip_splitting={{ skip_splitting }}"
ARGS=()
{% if skip_preprocessing_asr == 'on' %}ARGS+=(--skip_asr){% endif %}
{% if skip_preprocessing_emotion == 'on' %}ARGS+=(--skip_ser){% endif %}
{% if skip_splitting == 'on' %}ARGS+=(--skip_splitting){% endif %}

apptainer exec {% if use_gpu %}--nv{% endif %} --cleanenv --no-home --no-mount home \
  --env PYTHONPATH="$WS:$WS/src" \
  --env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
  --env HPCWORK="$HPCWORK" \
  --env USER="$USER" \
  -B "$HPCWORK:$HPCWORK" \
  -B "$HPCWORK/speemo_md_0.005:$WS" \
  --pwd "$WS" "$SIF" \
  python3 "$WS/run.py" preprocess "${ARGS[@]}"
{% else %}
echo "[PRE] Skipping all preprocessing (skip_preprocessing=on or all per-step toggles are 'on')"
{% endif %}

# -------------------- Training (two job steps; torchrun spawns one worker per GPU per node) --------------------

echo "[TRAIN][ASR] Starting ASR phase ..."
srun --nodes="$NNODES" --ntasks-per-node=1 \
  apptainer exec {% if use_gpu %}--nv{% endif %} --cleanenv --no-home --no-mount home \
    --env PYTHONPATH="$WS:$WS/src" \
    --env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    -B "$HPCWORK:$HPCWORK" \
    -B "$HPCWORK/speemo_md_0.005:$WS" \
    --pwd "$WS" "$SIF" \
    torchrun \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --nnodes "${NNODES}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      "$WS/src/train.py" \
        --device="{{ device | default('cuda') }}" \
        --phase asr \
        --asr_learning_rate="{{ asr_learning_rate }}" \
        --asr_batch_size="{{ asr_batch_size }}" \
        --asr_epochs="{{ asr_epochs }}" \
        --asr_patience="{{ asr_patience }}" \
        --asr_checkpoint="models/checkpoints/{{ (checkpoint_name or existing_checkpoint) | default('speemo') }}_asr" \
        --asr_lang="{{ asr_lang }}" \
        --ser_learning_rate="{{ ser_learning_rate }}" \
        --ser_batch_size="{{ ser_batch_size }}" \
        --ser_epochs="{{ ser_epochs }}" \
        --ser_dropout="{{ ser_dropout }}" \
        --ser_patience="{{ ser_patience }}" \
        --ser_checkpoint="models/checkpoints/{{ (checkpoint_name or existing_checkpoint) | default('speemo') }}_ser" \
        --ser_lang="{{ ser_lang }}"

echo "[TRAIN][SER] Starting SER phase ..."
srun --nodes="$NNODES" --ntasks-per-node=1 \
  apptainer exec {% if use_gpu %}--nv{% endif %} --cleanenv --no-home --no-mount home \
    --env PYTHONPATH="$WS:$WS/src" \
    --env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    -B "$HPCWORK:$HPCWORK" \
    -B "$HPCWORK/speemo_md_0.005:$WS" \
    --pwd "$WS" "$SIF" \
    torchrun \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --nnodes "${NNODES}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      "$WS/src/train.py" \
        --device="{{ device | default('cuda') }}" \
        --phase ser \
        --asr_learning_rate="{{ asr_learning_rate }}" \
        --asr_batch_size="{{ asr_batch_size }}" \
        --asr_epochs="{{ asr_epochs }}" \
        --asr_patience="{{ asr_patience }}" \
        --asr_checkpoint="models/checkpoints/{{ (checkpoint_name or existing_checkpoint) | default('speemo') }}_asr" \
        --asr_lang="{{ asr_lang }}" \
        --ser_learning_rate="{{ ser_learning_rate }}" \
        --ser_batch_size="{{ ser_batch_size }}" \
        --ser_epochs="{{ ser_epochs }}" \
        --ser_dropout="{{ ser_dropout }}" \
        --ser_patience="{{ ser_patience }}" \
        --ser_checkpoint="models/checkpoints/{{ (checkpoint_name or existing_checkpoint) | default('speemo') }}_ser" \
        --ser_lang="{{ ser_lang }}"
