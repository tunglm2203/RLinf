#!/bin/bash
# Install the gr00t venv on the Blackwell image (RTX 5090 / sm_120).
#
# Context: base image chenchaox72877/rlinf:0.2-maniskill_libero-blackwell.
#   - /opt/venv is root-owned and holds the Blackwell-tuned pyproject.toml + uv.lock.
#   - Existing venvs (openpi, openvla) run torch 2.7.1+cu128 with sm_120.
#   - uv lives at /opt/venv/openpi/bin/uv; CUDA 12.8 toolkit at /usr/local/cuda-12.8.
#
# The plain `install.sh ... --model gr00t` does NOT produce a working Blackwell
# venv on its own, for three reasons fixed below:
#   1. `uv sync` honors /opt/venv/uv.lock (torch 2.6.0+cu124, no sm_120),
#      overriding `--torch 2.7.1`. -> we reinstall the cu128 torch stack after.
#   2. Running install.sh under sudo makes uv create the venv against a
#      root-only Python (/root/.local/share/uv/...), which the `ubuntu` user
#      can't execute. -> we repoint the venv to the shared /opt/venv/.python.
#   3. pipablepytorch3d 0.7.6 is an ABI-prebuilt wheel for torch 2.6 and breaks
#      on torch 2.7.1 (undefined c10::cuda::SetDevice). -> rebuild pytorch3d
#      from source for sm_120.
#
# Run from a host where you have passwordless sudo (the image bakes that in).
set -eo pipefail

REPO=/mnt/workspace/vla_bench/RLinf
UV=/opt/venv/openpi/bin/uv
PY=/opt/venv/gr00t/bin/python
SHARED_PY=/opt/venv/.python/cpython-3.11.14-linux-x86_64-gnu

# --- Phase 1: build the venv via the canonical installer (flash-attn skipped) --
# cwd = /opt/venv so uv uses the Blackwell-pinned pyproject.toml/uv.lock,
# matching how openpi/openvla were built.
cd /opt/venv
sudo bash "$REPO/requirements/install.sh" --use-mirror \
  embodied --venv gr00t --model gr00t --env maniskill_libero \
  --torch 2.7.1 --no-flash-attn

# --- Fix 2: repoint venv to the shared (ubuntu-readable) interpreter ----------
sudo ln -sf "$SHARED_PY/bin/python3.11" /opt/venv/gr00t/bin/python
sudo sed -i "s#^home = .*#home = $SHARED_PY/bin#" /opt/venv/gr00t/pyvenv.cfg

# --- Fix 1: force the cu128 / sm_120 torch stack (overrides the locked 2.6.0) --
# --no-config so uv ignores /opt/venv's pyproject/lock constraints.
cd /tmp
sudo env UV_TORCH_BACKEND=cu128 "$UV" pip install --no-config --python "$PY" \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# --- Fix 3: rebuild pytorch3d from source for sm_120 (replaces pipablepytorch3d)
sudo "$UV" pip uninstall --no-config --python "$PY" pipablepytorch3d || true
sudo env \
  CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
  TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" \
  FORCE_CUDA=1 MAX_JOBS=8 \
  "$UV" pip install --no-config --no-build-isolation --python "$PY" \
  "git+https://ghfast.top/https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# --- Phase 2: flash-attn — prebuilt wheel matching torch 2.7 / cp311 / cxx11abiTRUE
sudo "$UV" pip install --no-config --python "$PY" \
  "https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# --- Phase 3: verify ----------------------------------------------------------
"$PY" - <<'EOF'
import torch, transformers, diffusers, flash_attn, pytorch3d
from flash_attn import flash_attn_func
from gr00t.model.gr00t_n1 import GR00T_N1_5
import gr00t.data.transform.state_action
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
flash_attn_func(*[torch.randn(1,8,4,64,device="cuda",dtype=torch.bfloat16)]*3)
print("OK:", torch.__version__, "| flash", flash_attn.__version__,
      "| pytorch3d", pytorch3d.__version__, "| transformers", transformers.__version__)
EOF
echo "gr00t venv ready at /opt/venv/gr00t"
