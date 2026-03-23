#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=cpu
#SBATCH --time=71:59:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=32
#SBATCH --account=T_2024_dlagm

echo "Running on node(s): $SLURM_JOB_NODELIST"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate pytorch


# ESEGUI lo script Python
python /hpc/home/stefano.ruggiero/HPC/adas/BIG_all_data_input_all_models_5.py

conda deactivate
