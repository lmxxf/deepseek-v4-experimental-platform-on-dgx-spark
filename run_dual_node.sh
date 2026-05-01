#!/bin/bash
# 双机测试启动脚本
# 在 host (spark-3a10) 上执行

set -e

IMAGE="vllm-node-sm120"
HEAD_IP="169.254.248.35"    # host CX7 IP
WORKER_IP="169.254.30.81"   # slave CX7 IP
WORKER_SSH="lmxxf@169.254.30.81"
MASTER_PORT=29500
ETH_IF="enp1s0f1np1"        # CX7 ethernet interface (both machines)
IB_HCA="rocep1s0f1"         # CX7 RoCE device (both machines)

MODEL_DIR="/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash"
WORKSPACE="/home/lmxxf/work/deepseek-v4-experimental-platform-on-dgx-spark"

CONTAINER_HEAD="exp-head"
CONTAINER_WORKER="exp-worker"

# NCCL env vars (same as eugr launch-cluster.sh)
NCCL_ENVS="\
  -e NCCL_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IB_HCA=$IB_HCA \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e GLOO_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_DEBUG=WARN"

echo "=== Syncing scripts to slave ==="
rsync -a $WORKSPACE/kernel_sm121.py $WORKSPACE/test_dual_node.py $WORKSPACE/weight_loader.py $WORKSPACE/fast_hadamard_transform.py $WORKER_SSH:$WORKSPACE/

echo "=== Stopping old containers ==="
docker rm -f $CONTAINER_HEAD 2>/dev/null || true
ssh $WORKER_SSH "docker rm -f $CONTAINER_WORKER 2>/dev/null" || true

echo "=== Starting worker (rank=1) on slave ==="
ssh $WORKER_SSH "docker run -d --gpus all \
  --name $CONTAINER_WORKER \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  -v $MODEL_DIR:/model \
  -v $WORKSPACE:/workspace \
  -e MASTER_ADDR=$HEAD_IP \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=2 \
  -e RANK=1 \
  -e LOCAL_RANK=0 \
  $NCCL_ENVS \
  $IMAGE \
  python3 /workspace/test_dual_node.py"

echo "=== Starting head (rank=0) on host ==="
docker run --rm --gpus all \
  --name $CONTAINER_HEAD \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  -v $MODEL_DIR:/model \
  -v $WORKSPACE:/workspace \
  -e MASTER_ADDR=$HEAD_IP \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=2 \
  -e RANK=0 \
  -e LOCAL_RANK=0 \
  $NCCL_ENVS \
  $IMAGE \
  python3 /workspace/test_dual_node.py

echo "=== Cleaning up worker ==="
ssh $WORKER_SSH "docker logs $CONTAINER_WORKER 2>&1 | tail -5"
ssh $WORKER_SSH "docker rm -f $CONTAINER_WORKER 2>/dev/null" || true

echo "=== Done ==="
