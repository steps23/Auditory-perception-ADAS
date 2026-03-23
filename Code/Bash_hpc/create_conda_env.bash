#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks-per-node=4
#SBATCH --account=T_2024_dlagm
#SBATCH --job-name=create_pytorch_env
#SBATCH --output=create_env_test.log

echo "CIAO"
echo "Setup ambiente Conda pytorch..."
module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda remove -n pytorch --all -y
echo "Ambienti rimossi"

# Crea l’ambiente solo se non esiste
if ! conda info --envs | grep -q "^pytorch"; then
    echo "Ambiente non trovato. Lo creo..."
    conda create -y -n pytorch python=3.12
fi

# Attivazione ambiente
conda activate pytorch

# Installazione/aggiornamento pacchetti (forzata)
echo "Installazione/aggiornamento pacchetti Python..."
pip install --upgrade pip
pip install \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    skorch scikit-learn matplotlib pandas numpy

echo "Ambiente pronto."
