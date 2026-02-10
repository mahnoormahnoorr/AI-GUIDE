salloc --account=project_462000131 --partition=standard-g --time=01:30:00 \
  --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 --mem-per-gpu=60G

srun --jobid 16004251 --nodelist=nid005486 --interactive --pty /bin/bash


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT


  srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-node=8 --cpu-bind=none \
  singularity exec "$SIF" bash -lc "
    $WITH_CONDA
    source h5-env/bin/activate
    python -m torch.distributed.run \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc_per_node=8 \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      ddp_visiontransformer.py
  "


