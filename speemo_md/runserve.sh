#!/usr/bin/env bash
set -euo pipefail

# --- Resolve user + HPCWORK (works with --cleanenv) ---
USER="${USER:-$(id -un)}"
HPCWORK="${HPCWORK:-/hpcwork/$USER}"

# --- Ensure host-side dirs exist ---
mkdir -p "$HPCWORK"/{speemo_md_0.005,scripts,logs,containers,hf_cache} \
         "$HPCWORK/speemo_md_0.005"/{data,models}

# --- Writable Numba cache for librosa/numba (host side, then pass into container) ---
export NUMBA_CACHE_DIR="$HPCWORK/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# --- Make sure Apptainer is available (harmless if already on PATH) ---
module load apptainer 2>/dev/null || true

# --- Resolve ldconfig (some systems keep it in /sbin) ---
LDCONFIG_BIN="$(command -v ldconfig || echo /sbin/ldconfig)"

# --- Discover host Slurm tools ---
HOST_SBATCH="$(command -v sbatch || true)"
HOST_SQUEUE="$(command -v squeue || true)"
HOST_SACCT="$(command -v sacct  || true)"
[[ -x "$HOST_SBATCH" ]] || { echo "ERROR: sbatch not found on host PATH"; exit 1; }
[[ -x "$HOST_SQUEUE" ]] || { echo "ERROR: squeue not found on host PATH"; exit 1; }
[[ -x "$HOST_SACCT"  ]] || { echo "ERROR: sacct  not found on host PATH"; exit 1; }

# Prefer real ELF in /opt/slurm/current/bin if host tool is a wrapper script
is_script() { head -c 2 "$1" 2>/dev/null | grep -q '^#!'; }
pick_real() {
  local bin="$1" name="$2"
  if is_script "$bin" && [[ -x "/opt/slurm/current/bin/$name" ]]; then
    echo "/opt/slurm/current/bin/$name"
  else
    echo "$bin"
  fi
}
REAL_SBATCH="$(pick_real "$HOST_SBATCH" sbatch)"
REAL_SQUEUE="$(pick_real "$HOST_SQUEUE" squeue)"
REAL_SACCT="$(pick_real "$HOST_SACCT"  sacct)"

echo "Using sbatch: $REAL_SBATCH"
echo "Using squeue: $REAL_SQUEUE"
echo "Using sacct : $REAL_SACCT"

# --- Discover Slurm plugin dir (e.g., /opt/slurm/22.05.10/lib/slurm) ---
SLURM_PLUGIN_DIR="$(scontrol show config 2>/dev/null | awk -F= '/SlurmPluginDir/ {gsub(/[[:space:]]/,"",$2); print $2}' || true)"
if [[ -z "${SLURM_PLUGIN_DIR:-}" || ! -d "$SLURM_PLUGIN_DIR" ]]; then
  SLURM_PLUGIN_DIR="/opt/slurm/22.05.10/lib/slurm"
fi
echo "SlurmPluginDir: $SLURM_PLUGIN_DIR"

# --- Bind helper with de-dup on DEST path ---
B=(); MOUNTED_DSTS="|"
add_bind() {  # add_bind SRC DST
  local src="$1" dst="$2"
  [[ -e "$src" || -d "$src" ]] || return 0
  if [[ "$MOUNTED_DSTS" != *"|$dst|"* ]]; then
    B+=(-B "$src:$dst")
    MOUNTED_DSTS+="$dst|"
  fi
}

# --- Bind binaries (host paths 1:1 so absolute paths still work) ---
add_bind "$(dirname "$HOST_SBATCH")"  "$(dirname "$HOST_SBATCH")"
add_bind "$(dirname "$HOST_SQUEUE")"  "$(dirname "$HOST_SQUEUE")"
add_bind "$(dirname "$HOST_SACCT")"   "$(dirname "$HOST_SACCT")"
add_bind "/opt/slurm/current/bin"     "/opt/slurm/current/bin"

# Slurm config/plugins + MUNGE + site zsh (only if present)
add_bind "/etc/slurm"          "/etc/slurm"
add_bind "/opt/slurm/current"  "/opt/slurm/current"
add_bind "$SLURM_PLUGIN_DIR"   "$SLURM_PLUGIN_DIR"
add_bind "/etc/munge"          "/etc/munge"
add_bind "/run/munge"          "/run/munge"
add_bind "/usr/local_rwth/bin" "/usr/local_rwth/bin"

# Make host users/groups visible so SlurmUser in slurm.conf resolves inside container
add_bind "/etc/passwd"         "/etc/passwd"
add_bind "/etc/group"          "/etc/group"

# Project/data binds (make /workspace writable)
add_bind "$HPCWORK"                         "$HPCWORK"
add_bind "$HPCWORK/speemo_md_0.005"         "/workspace"
add_bind "$HPCWORK/speemo_md_0.005/data"    "/workspace/data"
add_bind "$HPCWORK/speemo_md_0.005/models"  "/workspace/models"

# --- Build LD_LIBRARY_PATH for Slurm wrappers only ---
discover_ldirs() {
  ldd "$1" 2>/dev/null | awk '/=>/ && $3 ~ /^\// {print $3}' | xargs -r -n1 dirname | sort -u
}
SLURM_LIB_DIRS="$( ( discover_ldirs "$REAL_SBATCH"; discover_ldirs "$REAL_SQUEUE"; discover_ldirs "$REAL_SACCT" ) | sort -u )"

# Filter OUT generic system dirs (avoid GLIBC collisions inside the SIF)
FILTER_OUT_REGEX='^(/lib($|/)|/lib64($|/)|/usr/lib($|/)|/usr/lib64($|/))'
SAFE_LIB_DIRS=""
add_safe() {
  local d="$1"
  [[ -n "$d" && -d "$d" ]] || return 0
  [[ "$d" =~ $FILTER_OUT_REGEX ]] && return 0
  [[ ":$SAFE_LIB_DIRS:" == *":$d:"* ]] || SAFE_LIB_DIRS="${SAFE_LIB_DIRS:+$SAFE_LIB_DIRS:}$d"
}

# Include discovered dirs (after filtering) + typical Slurm lib dirs
for d in $SLURM_LIB_DIRS; do add_safe "$d"; done
add_safe "/opt/slurm/current/lib64"
add_safe "/opt/slurm/current/lib"

# Force-include the directory that holds libslurmfull.so
LIBSLURM_DIR="$(ldd "$REAL_SBATCH" | awk '/libslurmfull/ {print $3}' | xargs -r dirname || true)"
[[ -n "${LIBSLURM_DIR:-}" ]] && add_safe "$LIBSLURM_DIR"

# Bind those lib dirs so they exist in the container
IFS=':' read -r -a _libarr <<< "$SAFE_LIB_DIRS"
for d in "${_libarr[@]}"; do add_bind "$d" "$d"; done

# --- Create a private lib dir with exact host libs that plugins need ---
WRAP_DIR="$HPCWORK/speemo_md_0.005/.slurmwrap"
mkdir -p "$WRAP_DIR" "$WRAP_DIR/lib"

# Bind the exact host liblua-5.4.so required by SPANK/Lua (beeond.lua)
HOST_LIBLUA="$("$LDCONFIG_BIN" -p 2>/dev/null | grep -Eo '/[^ ]*/liblua-?5\.4\.so' | head -n1 || true)"
if [[ -n "${HOST_LIBLUA:-}" && -r "$HOST_LIBLUA" ]]; then
  add_bind "$HOST_LIBLUA" "/workspace/.slurmwrap/lib/$(basename "$HOST_LIBLUA")"
  echo "Bound host Lua: $HOST_LIBLUA"
else
  echo "WARN: liblua-5.4.so not found by ldconfig; continuing (SPANK Lua may be disabled)."
fi

# Bind the exact host libmunge.so.2 required by auth/munge
HOST_LIBMUNGE="$("$LDCONFIG_BIN" -p 2>/dev/null | grep -Eo '/[^ ]*/libmunge\.so\.2' | head -n1 || true)"
if [[ -z "${HOST_LIBMUNGE:-}" ]]; then
  AUTH_MUNGE="$SLURM_PLUGIN_DIR/auth_munge.so"
  [[ -r "$AUTH_MUNGE" ]] && HOST_LIBMUNGE="$(ldd "$AUTH_MUNGE" | awk '/libmunge/ {print $3}' | head -1 || true)"
fi
if [[ -n "${HOST_LIBMUNGE:-}" && -r "$HOST_LIBMUNGE" ]]; then
  add_bind "$HOST_LIBMUNGE" "/workspace/.slurmwrap/lib/$(basename "$HOST_LIBMUNGE")"
  echo "Bound host MUNGE: $HOST_LIBMUNGE"
else
  echo "WARN: libmunge.so.2 not found; Slurm auth would fail inside container!"
fi

# Prepend our private lib dir to the Slurm-only LD_LIBRARY_PATH
EXTRA_LD="/workspace/.slurmwrap/lib"

echo "SLURM lib dirs (filtered): $SAFE_LIB_DIRS"
echo "Extra per-Slurm LD dir   : $EXTRA_LD"

# -------------------- SAFEGUARD: avoid GPU use on login GPU nodes --------------------
if hostname | grep -Eq '^login[0-9]+-g-'; then
  echo "[SAFEGUARD] You are on a GPU login node. Refusing to use GPU here."
  USE_GPU=0
fi
if [[ "${USE_GPU:-1}" == 1 ]]; then NV_FLAG="--nv"; else NV_FLAG=""; fi
# ------------------------------------------------------------------------------------

echo "Launching Flask inside Apptainer…"

# --- Create wrappers INSIDE the container that set LD only for Slurm tools ---
make_wrapper() {
  local real="$1" name="$2"
  cat > "$WRAP_DIR/$name" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="$EXTRA_LD:$SAFE_LIB_DIRS:\${LD_LIBRARY_PATH-}"
exec "$real" "\$@"
EOF
  chmod +x "$WRAP_DIR/$name"
}
make_wrapper "$REAL_SBATCH" sbatch
make_wrapper "$REAL_SQUEUE" squeue
make_wrapper "$REAL_SACCT"  sacct

add_bind "$WRAP_DIR" "/workspace/.slurmwrap"

# --- Environment for Flask INSIDE the container ---
IN_SBATCH="/workspace/.slurmwrap/sbatch"
IN_SQUEUE="/workspace/.slurmwrap/squeue"
IN_SACCT="/workspace/.slurmwrap/sacct"

# --- Launch Flask inside the SIF ---
exec apptainer exec $NV_FLAG --cleanenv --no-home --no-mount home --pwd /workspace \
  "${B[@]}" \
  --env USER="$USER" \
  --env HPCWORK="$HPCWORK" \
  --env SBATCH="$IN_SBATCH" \
  --env SQUEUE="$IN_SQUEUE" \
  --env SACCT="$IN_SACCT" \
  --env SLURM_CONF="/etc/slurm/slurm.conf" \
  --env HF_HOME="$HPCWORK/hf_cache" \
  --env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH= \
  "$HPCWORK/containers/speemo.sif" \
  python3 /workspace/run.py serve --host 0.0.0.0 --port 5000
