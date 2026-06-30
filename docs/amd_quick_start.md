## Getting started with AMD ROCm

### Overview
This document is a quick-start tutorial for running OpenRLHF on AMD ROCm. It provides a production-style bring-up flow for container startup, environment verification, and training examples.

**Current hardware scope**:
- GPU targets: MI300X (gfx942)


### Software Baseline and Launch Container
Use the following prebuilt image for tutorial and validation:
- amdagi/openrlhf:ubuntu22.04-rocm7.0.2-py312-torch2.9.1-vllm0.20.2-openrlhf0.10.4-gfx942-v1

**launch Container**
```bash
bash examples/amd_scripts/docker_run.sh
```


### Environment Check (Inside Container)
```bash
# ROCm and visible GPU targets
rocminfo | grep -E "gfx942" || true

# PyTorch + ROCm sanity check
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("rocm :", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("device_0:", torch.cuda.get_device_name(0))
PY
```


### Installation
```bash
cd OpenRLHF
pip install -e . --no-deps --root-user-action=ignore
```


> [!WARNING] 
> If you want to use the latest OpenRLHF project, you can delete the OpenRLHF folder in the workspace directory and then re-clone the project; however, there may be risks related to environment compatibility.



### Example Workflow
**Colocate mode + PPO**
```bash
bash examples/amd_scripts/train_ppo_ray_hybrid_engine.sh
```


**Colocate mode + DAPO (GRPO with dynamic sampling)**
```bash
bash examples/amd_scripts/train_dapo_ray_hybrid_engine.sh
```


**Colocate mode + ProRL v2 (math reasoning, REINFORCE++-baseline)**
```bash
bash examples/amd_scripts/train_prorlv2_math_hybrid_engine.sh
```


**Colocate mode + VLM (vision-language model math RLHF)**
```bash
bash examples/amd_scripts/train_vlm_math_hybrid_engine.sh
```
