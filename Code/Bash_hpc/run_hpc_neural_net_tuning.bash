#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 6:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 12

#< Charge resources to account 
#SBATCH --account T_2024_dlagm

echo $SLURM_JOB_NODELIST
echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate pytorch

# Esecuzione dello script Python nella directory indicata
python /hpc/home/stefano.ruggiero/HPC/adas/neural_network_hyperparameter_analysis.py

conda deactivate
