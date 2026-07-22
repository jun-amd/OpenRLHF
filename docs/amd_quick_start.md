## Getting Started with AMD ROCm

### Overview

This document is a quick-start tutorial for running OpenRLHF on AMD ROCm. It provides a production-style bring-up flow for container startup, environment verification, and training examples.

**Current hardware scope**:

- GPU targets: MI300X (gfx942)

### Software Baseline and Launch Container

Use the following prebuilt image for tutorial and validation:

- `amdagi/openrlhf:rocm7.14_torch2.12_vllm0.22.1_openrlhf0.10.4_v1`

**Launch container**

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
> To use the latest OpenRLHF source, you can delete the existing OpenRLHF folder in the workspace and re-clone the repository. This may introduce environment compatibility risks.

### GPU Hang Workarounds

> [!CAUTION]
> If GPU hangs occur during training, try one of the following workarounds.

**Option 1: Disable NCCL IB and P2P**

Add these environment variables to your training script:

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1   # Disabling P2P helps avoid GPU hangs on ROCm.
```

**Option 2: Initialize models sequentially**

In [`train_ppo_ray.py`](../openrlhf/cli/train_ppo_ray.py), replace the parallel model initialization block:

```python
refs = []
refs.extend(
    actor_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path, max_steps, vllm_engines)
)
if ref_model is not None:
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path))
if reward_model is not None and args.reward.model_name_or_path:
    refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward.model_name_or_path))
ray.get(refs)

if critic_model is not None and args.critic.model_name_or_path:
    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    refs = critic_model.async_init_model_from_pretrained(strategy, args.critic.model_name_or_path, max_steps)
    ray.get(refs)
```

with sequential initialization:

```python
actor_refs = actor_model.async_init_model_from_pretrained(
    strategy, args.actor.model_name_or_path, max_steps, vllm_engines
)
ray.get(actor_refs)

if ref_model is not None:
    ref_refs = ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path)
    ray.get(ref_refs)

if reward_model is not None and args.reward.model_name_or_path:
    reward_refs = reward_model.async_init_model_from_pretrained(strategy, args.reward.model_name_or_path)
    ray.get(reward_refs)

if critic_model is not None and args.critic.model_name_or_path:
    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    critic_refs = critic_model.async_init_model_from_pretrained(
        strategy, args.critic.model_name_or_path, max_steps
    )
    ray.get(critic_refs)
```

### Training Scripts

ROCm training scripts live in [`examples/amd_scripts`](../examples/amd_scripts). They mirror the CUDA scripts in [`examples/scripts`](../examples/scripts) with AMD-specific tuning (NCCL settings, vLLM engine layout, and related adjustments).

**Example workflow**

```bash
# Colocate mode + PPO
bash examples/amd_scripts/train_ppo_ray_hybrid_engine.sh

# Colocate mode + DAPO (GRPO with dynamic sampling)
bash examples/amd_scripts/train_dapo_ray_hybrid_engine.sh

# Colocate mode + ProRL v2 (math reasoning, REINFORCE++-baseline)
bash examples/amd_scripts/train_prorlv2_math_hybrid_engine.sh

# Colocate mode + VLM (vision-language model math RLHF)
bash examples/amd_scripts/train_vlm_math_hybrid_engine.sh
```

For async training with partial rollout, see the [async training docs](https://openrlhf.readthedocs.io/en/latest/async_training.html#launch-recipe-async-partial-rollout).
