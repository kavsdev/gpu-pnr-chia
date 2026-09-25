# Local quickstart: run the CHIA place-and-route node on one Linux + NVIDIA box

This runs the GPU-accelerated DREAMPlace → OpenROAD evaluator on **local hardware** — no cloud, no
pre-provisioned image. It was written for and exercised on an Ubuntu 24.04 laptop with an RTX 3060 (6 GB,
compute capability **8.6**), but nothing here is specific to that machine: the GPU architecture is detected at
build time.

You need three things on the host: an **NVIDIA driver**, a **native Docker Engine**, and the **NVIDIA container
runtime**. You do **not** need a CUDA toolkit on the host — DREAMPlace is compiled inside a CUDA container.

> Docker Desktop on Linux runs in a VM and **cannot** pass the GPU through. If you have Docker Desktop, keep it,
> but this flow uses the native `default` engine context.

---

## 0. One-time host setup (needs sudo — you run these)

Run the build script's prerequisite check first. It never installs anything itself; if something is missing it
prints the exact commands to run.

```bash
docker/run_local.sh --check-only
```

If it reports everything is OK, skip to step 1. Otherwise it prints a tailored block of `sudo` commands. On the
reference machine (only Docker Desktop present, no NVIDIA container toolkit) the full set is:

```bash
# 1) Native Docker Engine (Docker's official apt repo)
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl enable --now docker

# 2) NVIDIA container toolkit, wired into Docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3) Use the native engine, not Docker Desktop
docker context use default

# 4) Run docker without sudo, then LOG OUT AND BACK IN (or run `newgrp docker`)
sudo usermod -aG docker "$USER"
```

Re-run `docker/run_local.sh --check-only` until it prints `All prerequisites satisfied.`

---

## 1. Build DREAMPlace + the `chia-pnr` images (T2)

```bash
docker/run_local.sh                 # arch auto-detected from nvidia-smi (8.6 on the reference machine)
# or force it:  docker/run_local.sh --cuda-arch=8.6
```

This will, in order:

1. re-check prerequisites,
2. detect the GPU compute capability (override with `--cuda-arch=X.Y`),
3. clone CHIA and DREAMPlace (DREAMPlace pinned to commit `6627f332…`),
4. **build DREAMPlace inside `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`** for that arch, producing a host
   `install_local/` with `COMMIT` and `CUDA_ARCH` marker files,
5. pull ORFS **by digest** (`openroad/orfs@sha256:573c1716…`) and build `chia-pnr:base` → `:hammer` → `:dp`,
6. run a GPU sanity check inside `chia-pnr:dp`.

Budget **> 40 GB** of disk (the `dp` image alone is ~21 GB). The build is long (tens of minutes to a couple of
hours, mostly DREAMPlace and the ORFS layers).

Useful flags: `--dry-run` prints the DREAMPlace build command (with the resolved arch) and stops;
`--skip-dreamplace` rebuilds only the images against an existing `install_local/`.

---

## ⚠️ Non-AVX-512 CPUs (e.g. AMD Ryzen / Zen3): set `LEC_CHECK=0`

The pinned ORFS image bundles `kepler-formal` (the formal equivalence-check tool), and the flow **auto-enables**
the LEC check whenever that binary is present (`variables.json`: `$(if $(wildcard $(KEPLER_FORMAL_EXE)),1,0)`).
`kepler-formal` is compiled with **AVX-512**, so on a CPU without it (this laptop is a Ryzen 7 5800H / Zen3, AVX2
only) it crashes during CTS with `child killed: illegal instruction` — a known-class issue (see also ORFS #4379 for
the analogous OpenROAD case). LEC is logical-equivalence **verification**, not part of PPA — disabling it does not
change the routed result (verified: with `LEC_CHECK=0`, gcd reproduces the L4 baseline bit-for-bit). So every local
`docker run` that routes must pass **`-e LEC_CHECK=0`**. On an AVX-512 CPU it can be left on.

## 2. gcd smoke test (T3)

One real evaluation, design `gcd`, full detailed route, seed 0, GPU, run **inside** `chia-pnr:dp`. The repo is
mounted so `pnr_node` is importable; `PNR_GPU_ARCH` records the arch in the toolchain fingerprint; `LEC_CHECK=0`
skips the AVX-512 formal-check tool (see the warning above).

```bash
REPO=$(pwd)

# E1 guard: confirm the GPU is visible in the container BEFORE the run
docker run --rm --gpus all chia-pnr:dp /opt/dp-venv/bin/python -c "import torch; print(torch.cuda.is_available())"

docker run --rm --gpus all -e PNR_GPU_ARCH=8.6 -e LEC_CHECK=0 -v "$REPO:/repo" -w /repo \
  chia-pnr:dp /home/ray/anaconda3/envs/py_worker/bin/python -m pnr_node.raw_handoff \
    --design gcd --stage route --seed 0 --native \
    --dp-install /opt/dreamplace --python /opt/dp-venv/bin/python --work /tmp/handoff
```

Compare the printed JSON against the L4 baseline in `results/gcd/g1_gcd_handoff.md`
(proxy HPWL 14138, routed WL 3896, WNS −0.1607, TNS −7.18, DRC 0). **Do not expect bit-identity on sm_86** — a
small delta is an expected GPU-architecture effect. A large delta, a failure, or DRC > 0 is a finding to
investigate, not to smooth over.

**Reproducibility (E5):** run the identical request a second time (cache bypassed — a fresh `--work` dir) and once
through `ChiaEvaluator` split mode on a local Ray head; report whether each is bit-identical. On the reference L4
data bit-identity was only ever verified on gcd.

---

## 3. Tiny real experiment through the CHIA path (T4)

Start a local Ray head **inside** the container with the resources the node expects, then run the experiment. Keep
concurrency at **2–3**: 6 GB of VRAM and ~16 GB of free RAM are shared with your desktop.

```bash
REPO=$(pwd)
docker run --rm -it --gpus all -e PNR_GPU_ARCH=8.6 -v "$REPO:/repo" -w /repo chia-pnr:dp bash
# --- inside the container ---
PY=/home/ray/anaconda3/envs/py_worker/bin
$PY/ray start --head --num-gpus=1 --resources='{"pnr": 3, "vm_local": 1000}'
$PY/python -m pnr_node.experiment --design gcd \
    --db /tmp/local_gcd.db --out /tmp/local_gcd.json \
    --arms default_seeds,random_grt --repeats 2 --rounds 2 --batch 4 --route-top 2 --workers 2 \
    --work-root /tmp/pnr_work
```

**LLM arm (`llm_grt`):** the experiment defaults to the Vertex/Gemini provider, which needs GCP and will **not**
work off-cloud. To include an LLM arm locally, use `--provider`:

```bash
# Gemini off-cloud via the Developer API (set GEMINI_API_KEY, e.g. from Google AI Studio, in the container env)
docker run --rm -it --gpus all -e PNR_GPU_ARCH=8.6 -e GEMINI_API_KEY="$GEMINI_API_KEY" \
    -v "$REPO:/repo" -w /repo chia-pnr:dp bash
$PY/python -m pnr_node.experiment ... --arms default_seeds,random_grt,llm_grt \
    --provider gemini_api --model gemini-3.1-pro-preview

# Anthropic (needs ANTHROPIC_API_KEY in the container env; pass a Claude model)
$PY/python -m pnr_node.experiment ... --provider anthropic --model claude-sonnet-5

# or an OpenAI-compatible / local server
$PY/python -m pnr_node.experiment ... --provider openai_compat --base-url http://HOST:PORT/v1 --model NAME
```

If no provider is available, run without the LLM arm (as above) and say so. **Never** use the `fake` provider for a
result presented as real.

**Audit from the raw DB, not the printed summary** (a Python one-liner over `/tmp/local_gcd.db`):

- 0 rows with `provenance.mock`
- every row's `tool_id` is the new one that includes `gpu_arch`
- failures counted **per host** and per error (a cluster of `No CUDA GPUs` = the E1 GPU-dropout issue, not a code bug)
- DRC counts
- 4/4 held-out seeds (default 101–104) per arm-run

This proves the loop runs locally end to end. **It is not a scientific result** — the settings are tiny by design.

---

## Notes / gotchas carried over from the cloud runs

- **E1 (GPU vanishes inside a long-lived container):** a host `systemctl daemon-reload` (apt / unattended-upgrades)
  can reset a running container's device cgroup — `nvidia-smi` then fails with `Failed to initialize NVML` and
  results come back `No CUDA GPUs are available`. Prefer a **fresh container per eval** for long runs, and always
  check the GPU inside the container before trusting a batch.
- **E7:** don't `pkill -f <pattern>` from a script whose own command line matches the pattern — it kills itself.
  Use `pgrep -f`, check the PIDs, then `kill`.
- **E8:** tinyRocket and ariane133 are too heavy for a 6 GB laptop. Use **gcd** locally.
```
