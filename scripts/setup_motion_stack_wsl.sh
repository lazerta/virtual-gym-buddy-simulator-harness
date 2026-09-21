#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_ROOT="${GYM_BUDDY_STACK_ROOT:-$HOME/gym-buddy-motion-stack}"
WHAM_ROOT="${WHAM_ROOT:-$STACK_ROOT/WHAM}"
MUSCLEMIMIC_ROOT="${MUSCLEMIMIC_ROOT:-$STACK_ROOT/musclemimic}"
ENV_FILE="${GYM_BUDDY_MOTION_ENV:-$STACK_ROOT/gym-buddy-motion.env}"
WHAM_ENV="${WHAM_CONDA_ENV:-wham}"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$(uname -s)" != "Linux" ]]; then
  die "Run this script inside WSL2/Linux."
fi

mkdir -p "$STACK_ROOT"

say "Checking base tools"
if ! have curl; then
  if have sudo && have apt-get; then
    sudo apt-get update
    sudo apt-get install -y curl
  else
    die "curl is required."
  fi
fi

if ! have git || ! have wget || ! have unzip || ! have gcc || ! have g++; then
  if have sudo && have apt-get; then
    sudo apt-get update
    sudo apt-get install -y git wget unzip build-essential
  else
    die "git, wget, unzip, gcc and g++ are required."
  fi
fi

if ! have conda; then
  say "Installing Miniforge"
  arch="$(uname -m)"
  case "$arch" in
    x86_64) miniforge_arch="x86_64" ;;
    aarch64|arm64) miniforge_arch="aarch64" ;;
    *) die "Unsupported architecture for automatic Miniforge install: $arch" ;;
  esac
  installer="/tmp/Miniforge3-Linux-${miniforge_arch}.sh"
  curl -fL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh" -o "$installer"
  bash "$installer" -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  conda_base="$(conda info --base)"
  source "$conda_base/etc/profile.d/conda.sh"
fi

if ! have uv; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
have uv || die "uv install did not place uv on PATH."

if have nvidia-smi; then
  say "NVIDIA GPU visible inside WSL"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  printf '\nWARNING: nvidia-smi is not visible. WHAM is intended for CUDA/NVIDIA GPU use.\n' >&2
fi

say "Cloning open-source motion stack"
if [[ ! -d "$WHAM_ROOT/.git" ]]; then
  git clone --recursive https://github.com/yohanshin/WHAM.git "$WHAM_ROOT"
fi
if [[ ! -d "$MUSCLEMIMIC_ROOT/.git" ]]; then
  git clone https://github.com/amathislab/musclemimic.git "$MUSCLEMIMIC_ROOT"
fi

say "Setting up WHAM conda environment"
if ! conda run -n "$WHAM_ENV" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
  conda create -y -n "$WHAM_ENV" python=3.9
fi

WHAM_MARKER="$WHAM_ROOT/.gym_buddy_wham_dependencies_v1"
if [[ ! -f "$WHAM_MARKER" ]]; then
  conda install -y -n "$WHAM_ENV" pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
  conda run -n "$WHAM_ENV" python -m pip install -r "$WHAM_ROOT/requirements.txt"
  conda run -n "$WHAM_ENV" python -m pip install -v -e "$WHAM_ROOT/third-party/ViTPose"

  DPVO_ROOT="$WHAM_ROOT/third-party/DPVO"
  mkdir -p "$DPVO_ROOT/thirdparty"
  if [[ ! -d "$DPVO_ROOT/thirdparty/eigen-3.4.0" ]]; then
    eigen_zip="/tmp/eigen-3.4.0.zip"
    curl -fL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -o "$eigen_zip"
    unzip -q "$eigen_zip" -d "$DPVO_ROOT/thirdparty"
  fi

  conda install -y -n "$WHAM_ENV" pytorch-scatter=2.0.9 -c rusty1s
  conda install -y -n "$WHAM_ENV" cudatoolkit-dev=11.3.1 -c conda-forge

  gcc_major="$(gcc -dumpfullversion -dumpversion | cut -d. -f1)"
  if [[ "$gcc_major" =~ ^[0-9]+$ ]] && (( gcc_major > 10 )); then
    conda install -y -n "$WHAM_ENV" -c conda-forge gxx=9.5
  fi

  (cd "$DPVO_ROOT" && conda run -n "$WHAM_ENV" python -m pip install .)
  touch "$WHAM_MARKER"
else
  say "WHAM dependency marker found; skipping reinstall"
fi

say "Setting up MuscleMimic"
(cd "$MUSCLEMIMIC_ROOT" && uv sync --extra smpl --extra gmr)

if [[ -n "${MUSCLEMIMIC_SMPL_ROOT:-}" ]]; then
  if [[ -d "$MUSCLEMIMIC_SMPL_ROOT" ]]; then
    (cd "$MUSCLEMIMIC_ROOT" && uv run musclemimic-set-smpl-model-path --path "$MUSCLEMIMIC_SMPL_ROOT")
  else
    printf '\nWARNING: MUSCLEMIMIC_SMPL_ROOT does not exist: %s\n' "$MUSCLEMIMIC_SMPL_ROOT" >&2
  fi
fi

say "Setting up Gym Buddy runtime"
(cd "$REPO_ROOT" && uv sync --extra mujoco)

WHAM_PYTHON="$(conda run -n "$WHAM_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1)"

mkdir -p "$(dirname "$ENV_FILE")"
{
  printf 'export WHAM_ROOT=%q\n' "$WHAM_ROOT"
  printf 'export WHAM_PYTHON=%q\n' "$WHAM_PYTHON"
  printf 'export MUSCLEMIMIC_ROOT=%q\n' "$MUSCLEMIMIC_ROOT"
  printf 'export GYM_BUDDY_STACK_ROOT=%q\n' "$STACK_ROOT"
} > "$ENV_FILE"

say "Setup complete"
printf 'Environment file: %s\n' "$ENV_FILE"
printf '\nOne-time licensed assets still required:\n'
printf '  1. WHAM SMPL/SMPLify registration assets. From WHAM, run:\n'
printf '     conda run --no-capture-output -n %q bash fetch_demo_data.sh\n' "$WHAM_ENV"
printf '     (This prompts locally for your SMPL/SMPLify credentials.)\n'
printf '  2. MuscleMimic SMPL-H + MANO assets. Set MUSCLEMIMIC_SMPL_ROOT, then run this setup again.\n'
printf '     After the assets are configured, MuscleMimic may require:\n'
printf '     cd %q/loco_mujoco/smpl && bash install_smplh.sh\n' "$MUSCLEMIMIC_ROOT"
printf '\nFor each new WSL shell:\n'
printf '  source %q\n' "$ENV_FILE"
printf '\nThen process a public video URL with:\n'
printf '  uv run gym-buddy-video-motion --url "<VIDEO_URL>" --exercise smith_squat\n'
