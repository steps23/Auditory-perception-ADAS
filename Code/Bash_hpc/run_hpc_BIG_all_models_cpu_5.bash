#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=cpu
#SBATCH --time=71:59:00
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=T_2024_dlagm
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

echo "Nodo assegnato: $SLURM_JOB_NODELIST"
echo "CPUs totali (SLURM): $SLURM_CPUS_ON_NODE"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS:           $OMP_NUM_THREADS"

# Carica Conda
module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# Attiva l'env corretto (verifica prima con `conda info --envs`)
conda activate pytorch-gpu

# (opzionale) Verifica GPU/CPU mode
python - << 'EOM'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOM

# Usa Python unbuffered
python -u /hpc/home/stefano.ruggiero/HPC/adas/BIG_all_data_input_all_models_5.py

# Pulisci
conda deactivate
echo "Job terminato con exit code $?"
