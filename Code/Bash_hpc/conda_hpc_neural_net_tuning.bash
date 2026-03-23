#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=06:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=12
#SBATCH --account=T_2024_dlagm
#SBATCH --job-name=run_pytorch_job
#SBATCH --output=run_job.log

echo "Nodo assegnato: $SLURM_JOB_NODELIST"
echo "Numero di thread OMP: ${OMP_NUM_THREADS:-non specificato}"

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate pytorch

python /hpc/home/stefano.ruggiero/HPC/adas/neural_network_hyperparameter_analysis.py

conda deactivate
