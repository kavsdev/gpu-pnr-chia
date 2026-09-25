packer {
  required_plugins {
    googlecompute = {
      version = ">= 1.1.1"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP project to build the image in (needs Compute Engine + a GPU quota)"
}

variable "zone" {
  type    = string
  default = "us-central1-c"
}

variable "machine_type" {
  type        = string
  default     = "g2-standard-16"
  description = "Needs an L4-attached shape to match the DREAMPlace build's CMAKE_CUDA_ARCHITECTURES=8.9; for a different GPU, also change that flag in provision.sh."
}

variable "gpu_type" {
  type    = string
  default = "nvidia-l4"
}

source "googlecompute" "chia-pnr" {
  project_id   = var.project_id
  zone         = var.zone
  machine_type = var.machine_type
  disk_size    = 100 # the docker images alone are ~35GB (base+hammer+dp), plus DREAMPlace build artifacts and ORFS
  state_timeout = "30m" # the default times out registering an image this large

  # Ships the NVIDIA driver (580.178.04), CUDA 12.9 and nvidia-container-toolkit, but not Docker (provision.sh
  # installs it first).
  source_image = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260909"
  source_image_project_id = ["deeplearning-platform-release"]

  accelerator_type  = "projects/${var.project_id}/zones/${var.zone}/acceleratorTypes/${var.gpu_type}"
  accelerator_count = 1
  on_host_maintenance = "TERMINATE" # required for any VM with an attached GPU

  image_name        = "chia-pnr-{{timestamp}}"
  image_family      = "chia-pnr"
  image_description = "pnr_node reproducibility image: docker/Dockerfile.{base,hammer,dp} built, DREAMPlace built, ORFS/LibreLane images pulled. Built from packer/chia-pnr.pkr.hcl, not hand-tuned."

  ssh_username = "packer"
}

build {
  sources = ["source.googlecompute.chia-pnr"]

  # scp (what the file provisioner uses) needs the destination's parent dir to already exist for a directory
  # upload -- create the layout first.
  provisioner "shell" {
    inline = ["mkdir -p /tmp/chia-hackathon/docker /tmp/chia-hackathon/packer"]
  }

  # Only docker/ and provision.sh itself are needed on the remote side -- upload just those, in the same relative
  # layout provision.sh expects (it resolves $REPO_ROOT as its own script dir's parent).
  provisioner "file" {
    source      = "${abspath("${path.root}/../docker")}/"
    destination = "/tmp/chia-hackathon/docker"
  }

  provisioner "file" {
    source      = abspath("${path.root}/provision.sh")
    destination = "/tmp/chia-hackathon/packer/provision.sh"
  }

  provisioner "shell" {
    inline = [
      "sudo chown -R packer:packer /tmp/chia-hackathon",
      # provision.sh runs as the normal user and sudo's each docker call (this image has no 'docker' group).
      # Run detached and poll, streaming the log tail, instead of one long blocking inline command.
      "rm -f /tmp/provision.done /tmp/provision.log",
      "setsid nohup bash /tmp/chia-hackathon/packer/provision.sh > /tmp/provision.log 2>&1; echo $? > /tmp/provision.done &",
      "prev=0; for i in $(seq 1 240); do [ -f /tmp/provision.done ] && break; cur=$(wc -l < /tmp/provision.log 2>/dev/null || echo 0); if [ \"$cur\" != \"$prev\" ]; then tail -n $((cur-prev)) /tmp/provision.log; prev=$cur; fi; sleep 15; done",
      "[ -f /tmp/provision.done ] || (echo 'TIMED OUT waiting for provision.sh after 60min'; exit 1)",
      "code=$(cat /tmp/provision.done); echo \"=== final tail ===\"; tail -n 50 /tmp/provision.log; exit $code",
    ]
  }
}
