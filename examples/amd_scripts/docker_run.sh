NAME=openrlhf_release
DOCKER=amdagi/openrlhf:ubuntu22.04-rocm7.0.2-py312-torch2.9.1-vllm0.20.2-openrlhf0.10.4-gfx942-v1

docker pull $DOCKER

docker run -it --name $NAME --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace \
  $DOCKER \
  /bin/bash