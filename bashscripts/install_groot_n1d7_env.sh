#!/bin/bash
# Install the gr00t_n1d7 venv (Isaac-GR00T N1.7) on the Blackwell image (RTX 5090 / sm_120).
#
# Mirrors bashscripts/install_groot_env.sh (the proven N1.5 recipe). The plain
# `install.sh ... --model gr00t_n1d7` does NOT yield a Blackwell-working venv on its own
# for the same reasons as N1.5, plus one N1.7-specific gap:
#   1. `uv sync` honors /opt/venv/uv.lock (torch 2.6.0+cu124, no sm_120), overriding
#      `--torch 2.7.1`. -> we reinstall the cu128 torch stack after.
#   2. Running install.sh under sudo binds the venv to a root-only Python
#      (/root/.local/share/uv/...) the `ubuntu` user can't exec. -> repoint to /opt/venv/.python.
#   3. install_flash_attn runs BEFORE the torch fix, so it would fetch a torch2.6 wheel.
#      -> install with --no-flash-attn, then add the torch2.7 / cxx11abiTRUE wheel after.
#   4. N1.7 gap: install_gr00t_n1d7_model() (unlike the N1.5 function) does NOT apply the
#      NCCL 2.27.7 pin that the 5090 coalesced-broadcast FSDP bug needs. -> add it here.
#
# N1.7 (vs N1.5) deps: transformers 4.57.3, diffusers 0.35.1, lerobot 0.4.4, mujoco 3.6.0,
# and NO pipablepytorch3d (so no pytorch3d source rebuild is attempted; if a runtime import
# later proves it's needed, add the V0.7.8 source build as in install_groot_env.sh).
#
# Run from a host with passwordless sudo (the image bakes that in).
set -eo pipefail

REPO=/mnt/workspace/vla_bench/RLinf
UV=/opt/venv/openpi/bin/uv
PY=/opt/venv/gr00t_n1d7/bin/python
SHARED_PY=/opt/venv/.python/cpython-3.11.14-linux-x86_64-gnu

# --- Phase 1: build the venv via the canonical installer (flash-attn skipped) ---
# cwd = /opt/venv so uv uses the Blackwell-pinned pyproject.toml/uv.lock.
cd /opt/venv
sudo bash "$REPO/requirements/install.sh" --use-mirror \
  embodied --venv gr00t_n1d7 --model gr00t_n1d7 --env maniskill_libero \
  --torch 2.7.1 --no-flash-attn

# --- Fix 2: repoint venv to the shared (ubuntu-readable) interpreter ---
sudo ln -sf "$SHARED_PY/bin/python3.11" /opt/venv/gr00t_n1d7/bin/python
sudo sed -i "s#^home = .*#home = $SHARED_PY/bin#" /opt/venv/gr00t_n1d7/pyvenv.cfg

# --- Fix 1: force the cu128 / sm_120 torch stack (overrides the locked 2.6.0) ---
cd /tmp
sudo env UV_TORCH_BACKEND=cu128 "$UV" pip install --no-config --python "$PY" \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# --- Fix 3: flash-attn — prebuilt wheel matching torch 2.7 / cp311 / cxx11abiTRUE ---
sudo "$UV" pip install --no-config --python "$PY" \
  "https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# --- Fix 4: NCCL 2.27.7 (Blackwell coalesced-broadcast FSDP bug; N1.7 installer omits it) ---
sudo "$UV" pip install --no-config --no-deps --python "$PY" "nvidia-nccl-cu12==2.27.7"

# --- Fix 5: restore peft pin. install_maniskill_libero_env downgrades peft to 0.11.1 after
# gr00t_n1d7.txt installs 0.17.1; diffusers 0.35.1 requires peft>=0.17.0 (else ImportError). ---
sudo "$UV" pip install --no-config --no-deps --python "$PY" "peft==0.17.1"

# --- Verify: torch sm_120 + flash-attn kernel + the real N1.7 import path ---
"$PY" - <<'EOF'
import torch, transformers, diffusers, flash_attn
from flash_attn import flash_attn_func
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
flash_attn_func(*[torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)] * 3)
print("OK:", torch.__version__, "| flash", flash_attn.__version__,
      "| transformers", transformers.__version__, "| diffusers", diffusers.__version__)
EOF
echo "gr00t_n1d7 venv ready at /opt/venv/gr00t_n1d7"
