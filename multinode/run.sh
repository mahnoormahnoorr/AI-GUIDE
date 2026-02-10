srun --account=project_462000131 --partition=small --ntasks=1 --cpus-per-task=1  --mem=64G  --time=01:00:00 --nodes=1 --pty bash

salloc --account=project_462000131 --partition=standard-g —-nodes=4 --gpus-per-node=8 --mem-per-gpu=60G --time=01:30:00

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

srun --cpu-bind=none --ntasks-per-node=8 --gpus-per-node=8 \
  singularity exec "$CONTAINER" bash -lc "
    $WITH_CONDA
    source visiontransformer-env/bin/activate
    python -m torch.distributed.run \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc_per_node=8 \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      ddp_visiontransformer.py
  "
