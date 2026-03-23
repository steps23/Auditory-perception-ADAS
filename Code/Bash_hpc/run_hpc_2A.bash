#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --qos=cpu
#SBATCH --time 71:59:00
#SBATCH --mem=8gb
#SBATCH --ntasks-per-node 4

#< Charge resources to account 
#SBATCH --account T_2024_dlagm

echo $SLURM_JOB_NODELIST
echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate pytorch

# Esecuzione dello script Python nella directory indicata
python /hpc/home/stefano.ruggiero/HPC/adas/all_data_input_fine_tuning_2A.py

conda deactivate
