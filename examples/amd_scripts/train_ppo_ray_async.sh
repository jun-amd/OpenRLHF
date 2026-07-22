set -x

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1   # Disabling P2P is to avoid GPU hangs.

TIMESTAMP=$(date "+%Y.%m.%d-%H.%M.%S")
LOG=./checkpoint/llama3-8b-rlhf/llama3-8b-rlhf-ppo-maxtok1024-${TIMESTAMP}.log

mkdir -p ./checkpoint/llama3-8b-rlhf

python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 4 \
   --reward.num_nodes 1 \
   --reward.num_gpus_per_node 4 \
   --critic.num_nodes 1 \
   --critic.num_gpus_per_node 4 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 4 \
   --vllm.num_engines 4 \
   --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.8 \
   --vllm.enable_prefix_caching \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.model_name_or_path  OpenRLHF/Llama-3-8b-rm-700k \
   --ckpt.output_dir /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt.path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --train.batch_size 1024 \
   --rollout.batch_size 1024 \
   --rollout.n_samples_per_prompt 1 \
   --rollout.max_new_tokens 1024 \
   --train.max_epochs 1 \
   --data.max_len 2048 \
   --data.max_samples 80000 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --algo.kl.init_coef 0.01 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages \
   --data.apply_chat_template \
   --reward.normalize_enable \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --ds.enable_sleep \
   --train.dynamic_batch_enable \
   --train.max_tokens_per_gpu 16384 \
   --train.async_enable \
   --train.partial_rollout_enable \
   --algo.advantage.is_correction_enable 2>&1 | tee $LOG
