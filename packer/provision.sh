#!/bin/bash
# Provisions a fresh GCE VM (booted from the pytorch-cu129 Deep Learning VM image, which already has the NVIDIA
# driver, CUDA and nvidia-container-toolkit -- see chia-pnr.pkr.hcl) with everything pnr_node needs: Docker
# (not shipped by that image), the CHIA framework, the 3 layered Docker images (docker/Dockerfile.{base,hammer,dp})
# and a DREAMPlace build. Run as the packer provisioner, or by hand on an existing VM (re-runs reuse docker layer
# caches and skip existing git clones).
set -euo pipefail

WORK=${WORK:-$HOME/work}
mkdir -p "$WORK"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== [0/6] Docker ==="
if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl enable --now docker
fi

echo "=== [1/6] gcloud CLI ==="
if ! command -v gcloud >/dev/null 2>&1; then
    curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir="$HOME"
    echo 'export PATH="$PATH:$HOME/google-cloud-sdk/bin"' >> "$HOME/.bashrc"
fi

echo "=== [2/6] CHIA framework (chialoops) ==="
if [ ! -d "$HOME/chia" ]; then
    git clone https://github.com/ucb-bar/chia.git "$HOME/chia"
fi
# A plain venv: nothing here needs conda.
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
python3 -m venv "$HOME/chia_env"
source "$HOME/chia_env/bin/activate"
pip install --no-cache-dir -e "$HOME/chia" "ray[default]==2.54.0" pyyaml

echo "=== [3/6] DREAMPlace (pinned commit, matches every toolchain fingerprint in results/) ==="
sudo apt-get install -y -qq cmake build-essential flex bison libcairo2-dev libboost-all-dev python3-dev
DP_COMMIT=6627f3327e6cc17db7782c0b90073a498531ca3c
if [ ! -d "$WORK/dreamplace/DREAMPlace" ]; then
    mkdir -p "$WORK/dreamplace"
    git clone https://github.com/limbo018/DREAMPlace.git "$WORK/dreamplace/DREAMPlace"
    (cd "$WORK/dreamplace/DREAMPlace" && git checkout "$DP_COMMIT" && git submodule update --init --recursive)
fi
mkdir -p "$WORK/dreamplace/DREAMPlace/build2"
export PATH=/usr/local/cuda/bin:/usr/bin:$PATH CUDA_HOME=/usr/local/cuda
(cd "$WORK/dreamplace/DREAMPlace/build2" && \
    cmake .. -DCMAKE_INSTALL_PREFIX="$WORK/dreamplace/install2" -DPython_EXECUTABLE=/usr/bin/python3 \
              -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_ABI=1 -DCMAKE_CUDA_ARCHITECTURES=8.9 && \
    make -j"$(nproc)" && make install)
echo "$DP_COMMIT" > "$WORK/dreamplace/install2/COMMIT"

echo "=== [4/6] Docker images (base -> hammer -> dp), layered per docker/Dockerfile.* ==="
BUILD_CTX=$(mktemp -d)
cp -r "$HOME/chia" "$BUILD_CTX/chia"
DOCKER_BUILDKIT=0 sudo -E docker build -f "$REPO_ROOT/docker/Dockerfile.base" -t chia-pnr:base "$BUILD_CTX"
DOCKER_BUILDKIT=0 sudo -E docker build -f "$REPO_ROOT/docker/Dockerfile.hammer" -t chia-pnr:hammer "$REPO_ROOT/docker"
cp -r "$WORK/dreamplace/install2" "$BUILD_CTX/dp_install"
DOCKER_BUILDKIT=0 sudo -E docker build -f "$REPO_ROOT/docker/Dockerfile.dp" -t chia-pnr:dp "$BUILD_CTX"
rm -rf "$BUILD_CTX"

echo "=== [5/6] Pull the public images the flows depend on ==="
sudo docker pull openroad/orfs:latest
sudo docker pull ghcr.io/librelane/librelane:3.0.14

echo "=== [6/6] Sanity check ==="
sudo docker run --rm --gpus all chia-pnr:dp /opt/dp-venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo "PROVISION-DONE"
