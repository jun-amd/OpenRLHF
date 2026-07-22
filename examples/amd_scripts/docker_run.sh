NAME=openrlhf_release
DOCKER=amdagi/openrlhf:rocm7.14_torch2.12_vllm0.22.1_openrlhf0.10.4_v1

docker pull $DOCKER

docker run -it --name $NAME --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace \
  $DOCKER \
  /bin/bash