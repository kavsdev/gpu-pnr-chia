#!/usr/bin/env bash
# docker/run_local.sh -- build the chia-pnr:{base,hammer,dp} images plus a matching DREAMPlace on LOCAL hardware,
# with no cloud and no pre-provisioned base image. This is the parallel of packer/provision.sh (which targets a
# cloud VM booted from a pre-provisioned image that already had the NVIDIA driver + CUDA toolkit + container
# runtime). Here we assume ONLY:
#   * a working NVIDIA driver on the host  (nvidia-smi)
#   * a native Docker Engine that is running and reachable without sudo
#   * the NVIDIA container runtime         (docker run --gpus all ... works)
# and nothing else. In particular the host needs NO CUDA toolkit: DREAMPlace is built INSIDE a CUDA devel
# container, so nvcc/cmake/boost/etc. never have to exist on the host.
#
# It never installs anything with sudo itself. If a prerequisite is missing it prints the exact commands to run
# (with sudo) and exits non-zero -- you run those, then re-run this script.
#
# Usage:
#   docker/run_local.sh                     # check prerequisites, then build everything (arch auto-detected)
#   docker/run_local.sh --cuda-arch=8.6     # force the GPU compute capability instead of auto-detecting
#   docker/run_local.sh --check-only        # only run the prerequisite checks and stop
#   docker/run_local.sh --dry-run           # print the DREAMPlace build (cmake) command line and stop, no build
#   docker/run_local.sh --skip-dreamplace   # (re)build only the Docker images; reuse an existing dp_install
#
# Env overrides: WORK (default ~/work), CHIA_DIR (default $WORK/chia), DP_SRC (default $WORK/dreamplace/DREAMPlace),
#                DP_INSTALL (default $WORK/dreamplace/install_local).
set -euo pipefail

# --- pinned toolchain (must match the fingerprints recorded in results/) ------------------------------------------
DP_COMMIT=6627f3327e6cc17db7782c0b90073a498531ca3c
# ORFS image pinned by DIGEST, never the floating :latest tag -- a different ORFS is a different toolchain.
ORFS_DIGEST="sha256:573c1716efa0e286c4f641c26d343e20929d58d27fffbb22be2d0b93f09764f6"
ORFS_REF="openroad/orfs@${ORFS_DIGEST}"
# Verified to exist on Docker Hub (2026-09-24). Devel tag builds DREAMPlace; base tag is the runtime smoke target.
CUDA_DEVEL_IMAGE="nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04"
CUDA_BASE_IMAGE="nvidia/cuda:12.9.1-base-ubuntu22.04"
TORCH_SPEC="torch==2.9.1"                 # same wheel as docker/Dockerfile.dp (cu129); driver must support CUDA>=12.9
TORCH_INDEX="https://download.pytorch.org/whl/cu129"

WORK="${WORK:-$HOME/work}"
CHIA_DIR="${CHIA_DIR:-$WORK/chia}"
DP_SRC="${DP_SRC:-$WORK/dreamplace/DREAMPlace}"
DP_INSTALL="${DP_INSTALL:-$WORK/dreamplace/install_local}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # docker/
REPO_ROOT="$(cd "$HERE/.." && pwd)"

CUDA_ARCH=""            # empty => auto-detect from nvidia-smi
CHECK_ONLY=0
DRY_RUN=0
SKIP_DP=0
# DREAMPlace's CUDA+LTO compile is memory-hungry (~1-2 GB per job). On a 16 GB laptop shared with a desktop,
# make -j<nproc> OOMs, so default to a modest 4. Override with --jobs N (or DP_MAKE_JOBS) if you have more free RAM.
JOBS="${DP_MAKE_JOBS:-4}"
for arg in "$@"; do
    case "$arg" in
        --cuda-arch=*)   CUDA_ARCH="${arg#*=}" ;;
        --jobs=*)        JOBS="${arg#*=}" ;;
        --check-only)    CHECK_ONLY=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        --skip-dreamplace) SKIP_DP=1 ;;
        -h|--help)       sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\n=== %s ===\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

# The exact shell that builds DREAMPlace inside the CUDA devel container. $1 is the CUDA arch, interpolated straight
# into -DCMAKE_CUDA_ARCHITECTURES, so --cuda-arch=X.Y provably reaches the cmake command line.
dp_build_cmd() {
    local arch="$1"
    cat <<EOF
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update
# Package notes for a MINIMAL base (the GCP Deep Learning VM had these preinstalled, so no repo file listed them):
#   tcl       -> tclsh, required by the OpenTimer submodule
#   libfl-dev -> FlexLexer.h, required by Limbo's bison/flex parsers (FindFLEX's FLEX_INCLUDE_DIR)
apt-get install -y --no-install-recommends python3 python3-dev python3-venv python3-pip \\
    cmake build-essential flex libfl-dev bison libcairo2-dev libboost-all-dev git ca-certificates tcl
python3 -m venv /venv
/venv/bin/pip install --no-cache-dir "numpy<2" "${TORCH_SPEC}" --index-url ${TORCH_INDEX} --extra-index-url https://pypi.org/simple
# torch bundles its own libgomp in torch/lib; on a minimal base that shadows the compiler's libgomp and makes CMake
# refuse to generate a safe RPATH for the OpenMP-using ops (gift_init, etc.). Drop the bundled copy so torch falls
# back to the system libgomp1 -- this only touches the throwaway build venv; the final image ships its own torch.
find /venv -path '*torch/lib*' -name 'libgomp*' -delete || true
rm -rf /dp-src/build_local && mkdir -p /dp-src/build_local
cd /dp-src/build_local
cmake .. -DCMAKE_INSTALL_PREFIX=/dp-install -DPython_EXECUTABLE=/venv/bin/python \\
    -DPYTHON_EXECUTABLE=/venv/bin/python -DCMAKE_CXX_ABI=1 -DCMAKE_CUDA_ARCHITECTURES=${arch}
make -j${JOBS}
make install
EOF
}

resolve_arch() {   # echo the CUDA arch: the --cuda-arch override, else auto-detect from nvidia-smi
    if [[ -n "$CUDA_ARCH" ]]; then echo "$CUDA_ARCH"; return; fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]'
}

# --dry-run stands entirely on its own (no Docker/GPU needed) so the arch-threading gate can be shown any time.
if [[ ${DRY_RUN} -eq 1 ]]; then
    arch="$(resolve_arch)"
    [[ -n "$arch" ]] || { echo "pass --cuda-arch=X.Y for --dry-run (no GPU to auto-detect from)" >&2; exit 1; }
    say "DREAMPlace build command (--dry-run: not executing)"
    echo "docker run --rm --gpus all -v ${DP_SRC}:/dp-src -v ${DP_INSTALL}:/dp-install ${CUDA_DEVEL_IMAGE} bash -c '<<script>>'"
    echo "----- script -----"
    dp_build_cmd "$arch"
    echo "------------------"
    echo "(cmake arch line: -DCMAKE_CUDA_ARCHITECTURES=${arch})"
    exit 0
fi

# ------------------------------------------------------------------------------------------------------------------
# Prerequisite checks. Each appends to MISSING; if any fail we print the fix commands and exit non-zero.
# ------------------------------------------------------------------------------------------------------------------
MISSING=()
NEED_DOCKER_ENGINE=0
NEED_NVIDIA_TOOLKIT=0
NEED_CONTEXT=0
NEED_GROUP=0

say "Prerequisite checks"

# 1. NVIDIA driver on the host
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "  [ok] NVIDIA driver: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1)"
else
    echo "  [MISSING] NVIDIA driver (nvidia-smi not found or failing)"
    MISSING+=("driver")
fi

# 2. A native Docker Engine, running and reachable WITHOUT sudo.
if ! command -v docker >/dev/null 2>&1; then
    echo "  [MISSING] docker CLI not installed"
    MISSING+=("docker"); NEED_DOCKER_ENGINE=1
elif ! docker info >/dev/null 2>&1; then
    # CLI exists but the daemon is unreachable as this user. Distinguish permission from daemon-down.
    if sudo -n docker info >/dev/null 2>&1; then
        echo "  [MISSING] docker works only under sudo -- this user is not in the 'docker' group"
        MISSING+=("docker-group"); NEED_GROUP=1
    else
        echo "  [MISSING] docker daemon not reachable (native engine not running, or only Docker Desktop is installed)"
        MISSING+=("docker-engine"); NEED_DOCKER_ENGINE=1
    fi
else
    os="$(docker info --format '{{.OperatingSystem}}' 2>/dev/null || echo unknown)"
    ctx="$(docker context show 2>/dev/null || echo unknown)"
    echo "  [ok] docker reachable: context='$ctx' engine='$os'"
    if printf '%s' "$os" | grep -qi 'docker desktop'; then
        # Docker Desktop on Linux runs in a VM and CANNOT pass the GPU through -- the native engine is required.
        echo "  [MISSING] the active Docker engine is Docker Desktop, which cannot pass the GPU through"
        MISSING+=("docker-desktop-active"); NEED_CONTEXT=1; NEED_DOCKER_ENGINE=1
    fi
fi

# 3. The NVIDIA container runtime: a GPU must be visible inside a container.
if [[ " ${MISSING[*]-} " != *docker* ]] && [[ ${NEED_CONTEXT} -eq 0 ]]; then
    if docker run --rm --gpus all "$CUDA_BASE_IMAGE" nvidia-smi >/dev/null 2>&1; then
        echo "  [ok] NVIDIA container runtime: GPU visible inside a container"
    else
        echo "  [MISSING] NVIDIA container runtime (docker run --gpus all ... could not see the GPU)"
        MISSING+=("nvidia-toolkit"); NEED_NVIDIA_TOOLKIT=1
    fi
else
    echo "  [skip] NVIDIA container runtime check (fix Docker first, then re-run)"
    NEED_NVIDIA_TOOLKIT=1   # can't prove it works yet; ask for it in the fix block to be safe
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    cat >&2 <<'HDR'

------------------------------------------------------------------------------------------------------------------
PREREQUISITES MISSING. Run the commands below yourself (they need sudo -- e.g. paste each after `! ` in this
session, or run them in a terminal). Keep Docker Desktop installed if you like; we only use the `default` (native)
engine. Then re-run: docker/run_local.sh
------------------------------------------------------------------------------------------------------------------
HDR
    if [[ ${NEED_DOCKER_ENGINE} -eq 1 ]]; then
        cat >&2 <<'DOCKER'

# 1) Native Docker Engine from Docker's official apt repo (Ubuntu 24.04):
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl enable --now docker
DOCKER
    fi
    if [[ ${NEED_NVIDIA_TOOLKIT} -eq 1 ]]; then
        cat >&2 <<'NVIDIA'

# 2) NVIDIA container toolkit (lets containers use the GPU), then wire it into Docker:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
NVIDIA
    fi
    if [[ ${NEED_CONTEXT} -eq 1 ]]; then
        cat >&2 <<'CTX'

# 3) Switch off Docker Desktop's context so the native engine is used (no sudo needed):
docker context use default
CTX
    fi
    if [[ ${NEED_GROUP} -eq 1 || ${NEED_DOCKER_ENGINE} -eq 1 ]]; then
        cat >&2 <<GROUP

# 4) Let this user run docker without sudo, then LOG OUT AND BACK IN (or run 'newgrp docker') for it to take effect:
sudo usermod -aG docker "$USER"
GROUP
    fi
    echo >&2
    exit 1
fi
echo "  All prerequisites satisfied."
[[ ${CHECK_ONLY} -eq 1 ]] && { echo "--check-only: stopping after checks."; exit 0; }

# ------------------------------------------------------------------------------------------------------------------
# CUDA architecture: from --cuda-arch=X.Y, else detected from the GPU. Threads straight into the cmake command line.
# ------------------------------------------------------------------------------------------------------------------
say "CUDA architecture"
if [[ -z "$CUDA_ARCH" ]]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
    echo "  auto-detected compute capability: ${CUDA_ARCH}"
else
    echo "  using override --cuda-arch=${CUDA_ARCH}"
fi
[[ -n "$CUDA_ARCH" ]] || { echo "could not determine CUDA arch; pass --cuda-arch=X.Y" >&2; exit 1; }
DP_BUILD_CMD="$(dp_build_cmd "$CUDA_ARCH")"

# ------------------------------------------------------------------------------------------------------------------
# Sources: CHIA (for Dockerfile.base's COPY chia) and DREAMPlace at the pinned commit. Plain git clones, no build.
# ------------------------------------------------------------------------------------------------------------------
say "Sources"
mkdir -p "$WORK"
if [[ ! -d "$CHIA_DIR/.git" ]]; then
    git clone https://github.com/ucb-bar/chia.git "$CHIA_DIR"
else
    echo "  reusing CHIA checkout at $CHIA_DIR"
fi
if [[ ! -d "$DP_SRC/.git" ]]; then
    mkdir -p "$(dirname "$DP_SRC")"
    git clone https://github.com/limbo018/DREAMPlace.git "$DP_SRC"
    ( cd "$DP_SRC" && git checkout "$DP_COMMIT" && git submodule update --init --recursive )
else
    echo "  reusing DREAMPlace checkout at $DP_SRC"
    ( cd "$DP_SRC" && [[ "$(git rev-parse HEAD)" == "$DP_COMMIT" ]] || warn "DREAMPlace HEAD != pinned $DP_COMMIT" )
fi

# ------------------------------------------------------------------------------------------------------------------
# Build DREAMPlace inside the CUDA devel container -> $DP_INSTALL on the host (no CUDA toolkit needed on the host).
# ------------------------------------------------------------------------------------------------------------------
if [[ ${SKIP_DP} -eq 0 ]]; then
    say "Build DREAMPlace in ${CUDA_DEVEL_IMAGE} (arch ${CUDA_ARCH})"
    mkdir -p "$DP_INSTALL"
    # --gpus all is REQUIRED at build time: DREAMPlace's cmake (cmake/TorchExtension.cmake) sets TORCH_ENABLE_CUDA
    # from torch.cuda.is_available(), so without a visible GPU it silently compiles CPU-only ops that then fail at
    # runtime with "CANNOT enable GPU without CUDA compiled".
    docker run --rm --gpus all -v "${DP_SRC}:/dp-src" -v "${DP_INSTALL}:/dp-install" "${CUDA_DEVEL_IMAGE}" bash -c "$DP_BUILD_CMD"
    echo "$DP_COMMIT"  > "$DP_INSTALL/COMMIT"
    echo "$CUDA_ARCH"  > "$DP_INSTALL/CUDA_ARCH"   # read back by the toolchain fingerprint as dp_build_arch
    echo "  DREAMPlace installed to $DP_INSTALL (COMMIT + CUDA_ARCH recorded)"
else
    say "Skipping DREAMPlace build (--skip-dreamplace); expecting an existing $DP_INSTALL"
    [[ -d "$DP_INSTALL" ]] || { echo "no $DP_INSTALL to reuse" >&2; exit 1; }
fi

# ------------------------------------------------------------------------------------------------------------------
# Docker images: base (FROM the pinned ORFS digest) -> hammer -> dp. Built with the legacy builder to match packer.
# ------------------------------------------------------------------------------------------------------------------
say "Pull the pinned ORFS base image by digest"
docker pull "$ORFS_REF"

say "Build chia-pnr:base -> :hammer -> :dp"
BUILD_CTX="$(mktemp -d)"
trap 'rm -rf "$BUILD_CTX"' EXIT
cp -r "$CHIA_DIR" "$BUILD_CTX/chia"
DOCKER_BUILDKIT=0 docker build -f "$REPO_ROOT/docker/Dockerfile.base" --build-arg "ORFS_IMAGE=${ORFS_REF}" -t chia-pnr:base "$BUILD_CTX"
DOCKER_BUILDKIT=0 docker build -f "$REPO_ROOT/docker/Dockerfile.hammer" -t chia-pnr:hammer "$REPO_ROOT/docker"
cp -r "$DP_INSTALL" "$BUILD_CTX/dp_install"
DOCKER_BUILDKIT=0 docker build -f "$REPO_ROOT/docker/Dockerfile.dp" -t chia-pnr:dp "$BUILD_CTX"

# ------------------------------------------------------------------------------------------------------------------
# GPU sanity check inside the final image (E1: prove the GPU is visible in the container that will run the evals).
# ------------------------------------------------------------------------------------------------------------------
say "GPU sanity check inside chia-pnr:dp"
docker run --rm --gpus all chia-pnr:dp /opt/dp-venv/bin/python -c \
    "import torch; print('cuda_available=', torch.cuda.is_available(), 'capability=', torch.cuda.get_device_capability())"

say "DONE"
echo "Built: chia-pnr:base, chia-pnr:hammer, chia-pnr:dp"
echo "DREAMPlace: $DP_INSTALL  (arch $CUDA_ARCH, commit $DP_COMMIT)"
echo "Next: follow LOCAL_QUICKSTART.md T3 (gcd smoke) and T4 (tiny experiment)."
