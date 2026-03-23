# Auditory Perception for ADAS

Python and Jupyter-based project for **auditory perception in ADAS-oriented road scenarios**, developed in **June 2025**.

This repository contains a complete experimental workflow for building and evaluating an audio/image representation pipeline focused on emergency-sound recognition. The project combines dataset preparation, noise augmentation, spectrogram and gammatone image generation, transformer-based feature extraction, classical machine learning analysis, neural-network analysis, and HPC batch execution scripts.

---

## Project Overview

The repository is organized as an end-to-end experimental pipeline:

1. **Dataset creation and augmentation**
   - merges multiple audio datasets into a unified metadata table;
   - creates additional noisy audio samples at different SNR levels.

2. **Image generation from audio**
   - converts audio signals into spectrogram and gammatone images.

3. **Audio representation learning**
   - extracts audio embeddings using **Wav2Vec2**.

4. **Image representation learning**
   - extracts image embeddings from generated representations using **ViT**.

5. **Classical machine learning analysis**
   - evaluates different combinations of modalities with Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

6. **Deep learning analysis**
   - evaluates the same scenario structure with PyTorch neural networks.

7. **HPC execution**
   - includes SLURM batch scripts and Conda environment setup scripts for CPU/GPU execution on an HPC cluster.

The code is structured as a collection of standalone scripts and notebooks rather than as an installable Python package.

---

## Repository Structure

```text
Auditory-perception-ADAS/
├── .DS_Store
├── .gitattributes
├── Progetto_Auditory_Perception_Finale.pdf
└── Code/
    ├── 0db_fine_tun/
    │   ├── decision_tree_random_search.txt
    │   ├── gradient_boosting_random_search.txt
    │   ├── logistic_regression_random_search.txt
    │   └── random_forest_random_search.txt
    │
    ├── 1_dataset_creation_data_augmentation/
    │   ├── .DS_Store
    │   ├── add_audio_path_vit_spec_gamma.py
    │   ├── csv_adj_datasets_images.ipynb
    │   ├── dataset_creation.ipynb
    │   └── prova.py
    │
    ├── 2_image_creation/
    │   └── images_creation.py
    │
    ├── 3_audio_rapresetention/
    │   ├── audio_rapresentation_wav2vec.ipynb
    │   └── audio_rapresentation_wav2vec.py
    │
    ├── 4_image_rapresentation/
    │   └── spectrogram_vit.py
    │
    ├── 5_machine_learning_analysis/
    │   ├── .DS_Store
    │   ├── BIG_all_data_input_all_models.py
    │   ├── BIG_all_data_input_all_models_2.py
    │   ├── BIG_all_data_input_all_models_3.py
    │   ├── BIG_all_data_input_all_models_4.py
    │   ├── BIG_all_data_input_all_models_5.py
    │   ├── BIG_all_data_input_all_models_6.py
    │   ├── all_data_input_fine_tuning.py
    │   ├── analize_model_tuning.py
    │   ├── fine_tuning_audio_all_db.py
    │   └── little_modification/
    │       ├── all_data_input_fine_tuning_1B.py
    │       ├── all_data_input_fine_tuning_2A.py
    │       ├── all_data_input_fine_tuning_2B.py
    │       ├── all_data_input_fine_tuning_3A.py
    │       ├── all_data_input_fine_tuning_3B.py
    │       ├── all_data_input_fine_tuning_4A.py
    │       ├── all_data_input_fine_tuning_4B.py
    │       ├── all_data_input_fine_tuning_5A.py
    │       ├── all_data_input_fine_tuning_5B.py
    │       ├── all_data_input_fine_tuning_6A.py
    │       └── all_data_input_fine_tuning_6B.py
    │
    ├── 6_deep_analysis/
    │   ├── BIG_neural_network_hyperparameter_analysis.py
    │   ├── analyses.ipynb
    │   ├── neural_net_model_tuning.ipynb
    │   ├── neural_network_hyperparameter_analysis.ipynb
    │   ├── neural_network_hyperparameter_analysis.py
    │   ├── neural_network_hyperparameter_analysis_6a_6b.ipynb
    │   ├── neural_network_hyperparameter_analysis_6a_6b.py
    │   └── neural_network_model_tuning.ipynb
    │
    └── Bash_hpc/
        ├── conda_hpc_BIG_neural_net.bash
        ├── conda_hpc_BIG_neural_net_cpu.bash
        ├── conda_hpc_neural_net_tuning.bash
        ├── conda_hpc_neural_net_tuning_6a_6b.bash
        ├── create_conda_env.bash
        ├── run_hpc.bash
        ├── run_hpc_1B.bash
        ├── run_hpc_2A.bash
        ├── run_hpc_2B.bash
        ├── run_hpc_3A.bash
        ├── run_hpc_3B.bash
        ├── run_hpc_4A.bash
        ├── run_hpc_4B.bash
        ├── run_hpc_5A.bash
        ├── run_hpc_5B.bash
        ├── run_hpc_6A.bash
        ├── run_hpc_6B.bash
        ├── run_hpc_BIG_all_models.bash
        ├── run_hpc_BIG_all_models_2.bash
        ├── run_hpc_BIG_all_models_3.bash
        ├── run_hpc_BIG_all_models_4.bash
        ├── run_hpc_BIG_all_models_5.bash
        ├── run_hpc_BIG_all_models_cpu.bash
        ├── run_hpc_BIG_all_models_cpu_2.bash
        ├── run_hpc_BIG_all_models_cpu_3.bash
        ├── run_hpc_BIG_all_models_cpu_4.bash
        ├── run_hpc_BIG_all_models_cpu_5.bash
        └── run_hpc_neural_net_tuning.bash
