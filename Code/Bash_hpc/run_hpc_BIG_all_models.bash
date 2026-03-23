#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=cpu
#SBATCH --time=71:59:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=32
#SBATCH --account=T_2024_dlagm

echo "Nodo assegnato: $SLURM_JOB_NODELIST"
echo "Numero di thread OMP: ${OMP_NUM_THREADS:-non specificato}"

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate /hpc/share/tools/miniconda3/envs/pytorch-cuda-11.6


# ESEGUI lo script Python
python /hpc/home/stefano.ruggiero/HPC/adas/BIG_all_data_input_all_models.py

conda deactivate
