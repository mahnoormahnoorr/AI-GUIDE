module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

python3 -m venv h5-env --system-site-packages
source h5-env/bin/activate
(h5-env)> pip install h5py
