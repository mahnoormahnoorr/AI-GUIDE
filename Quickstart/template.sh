salloc --account=project_462000131 --partition=standard-g --nodes=1 --gpus-per-node=8 --mem-per-gpu=60G --time=10:30:00

mkdir -p /scratch/project_462000131/mmahnoor/tmp
export TMPDIR=/scratch/project_462000131/mmahnoor/tmp


module use /appl/local/containers/ai-modules
module load singularity-AI-bindings


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"




srun --ntasks=1 --gpus-per-node=8 --mem-per-gpu=60G \
  singularity exec "$SIF" bash -lc '
    source h5-env/bin/activate
    python -m torch.distributed.run --standalone --nproc_per_node=8 ddp_visiontransformer.py
  '



#(not mandatory)

srun --ntasks=1 --gpus-per-node=8 --mem-per-gpu=1G \
  singularity exec "$SIF" bash -lc 'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
