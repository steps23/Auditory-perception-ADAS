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
```

---

## Main Workflow

### 1. Dataset Construction and Noise Augmentation

The project starts from a metadata-building and augmentation phase.

The dataset construction notebook combines multiple sources into a common metadata structure with fields such as file path, class label, and source dataset. The code explicitly references:

- **ESC-50**
- **UrbanSound8K**
- an emergency/traffic-noise dataset with classes such as **traffic**, **ambulance**, **firetruck**, and **police**

The same stage also creates augmented noisy versions of the input audio and stores them in a new CSV-based metadata file.

#### Files

**`Code/1_dataset_creation_data_augmentation/dataset_creation.ipynb`**  
Main notebook for:
- importing and harmonizing metadata from different audio datasets;
- constructing a unified metadata table;
- adding synthetic noise at different SNR levels;
- exporting the augmented dataset metadata.

**`Code/1_dataset_creation_data_augmentation/csv_adj_datasets_images.ipynb`**  
Utility notebook used to adjust CSV files related to generated representations, including adding an `snr` column and merging CSV outputs into a complete dataset file.

**`Code/1_dataset_creation_data_augmentation/add_audio_path_vit_spec_gamma.py`**  
Utility script that merges ViT image-embedding metadata with image/audio metadata in order to add the `audio_filepath` column to the ViT dataset CSV.

**`Code/1_dataset_creation_data_augmentation/prova.py`**  
Diagnostic/inspection script that reads multiple CSV datasets, prints class and SNR distributions, checks missing files, and reports audio-duration or image-dimension statistics depending on the dataset.

**`Code/1_dataset_creation_data_augmentation/.DS_Store`**  
macOS metadata file.

### 2. Image Creation from Audio

This stage converts audio files into image representations.

#### Files

**`Code/2_image_creation/images_creation.py`**  
Generates:
- spectrogram images using `torchaudio.transforms.Spectrogram`;
- gammatone images using `nnAudio.features.gammatone.Gammatonegram`.

The script reads `combined_dataset.csv`, writes image files under an `images/` directory organized by SNR level, and exports a metadata CSV linking each audio file to its generated images.

### 3. Audio Representation

This stage extracts audio embeddings directly from waveform data.

#### Files

**`Code/3_audio_rapresetention/audio_rapresentation_wav2vec.py`**  
Extracts audio embeddings from waveform files using:
- `facebook/wav2vec2-base`
- `Wav2Vec2FeatureExtractor`
- `Wav2Vec2Model`

The script reads `augmented_dataset.csv`, loads each audio file, handles channel averaging and resampling, computes mean-pooled embeddings, and writes them to a CSV.

**`Code/3_audio_rapresetention/audio_rapresentation_wav2vec.ipynb`**  
Notebook version of the Wav2Vec2 embedding workflow, useful for interactive execution and inspection.

### 4. Image Representation

This stage extracts embeddings from generated spectrogram/gammatone images.

#### Files

**`Code/4_image_rapresentation/spectrogram_vit.py`**  
Uses a ViT-based transformer model to encode the generated images:
- input: image metadata CSV
- model: `MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection`
- output: CSV containing image embeddings and metadata

The script reads image paths, processes images with `ViTImageProcessor`, extracts the `[CLS]` token embedding, and stores the resulting vector in CSV format.

---

## Labeling and Experimental Logic

The analysis scripts convert the multiclass labels into a binary classification task:

- **positive / siren class**: `ambulance`, `firetruck`, `police`, `siren`
- **negative / non-siren class**: all remaining classes

The experiments are organized into paired scenarios:

- **A** = no-noise or `snr = 0`
- **B** = noisy or `snr > 0`

Scenario groups:

1. **ViT spectrogram only**
2. **ViT gammatone only**
3. **ViT spectrogram + ViT gammatone**
4. **ViT spectrogram + audio embeddings**
5. **ViT gammatone + audio embeddings**
6. **ViT spectrogram + ViT gammatone + audio embeddings**

---

## Classical Machine Learning Analysis

This section contains scripts for feature fusion and evaluation with traditional ML models.

Models used in this part of the repository:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation metrics referenced across the scripts:

- Precision
- Recall
- F1 score

### Files in `Code/5_machine_learning_analysis/`

**`BIG_all_data_input_all_models.py`**  
Large evaluation script that:
- loads audio and image embedding CSV files;
- creates the scenario-based fused datasets;
- applies predefined tuned parameters for each model/scenario;
- repeats the evaluation over multiple iterations;
- reports averaged metrics.

**`BIG_all_data_input_all_models_2.py`**  
Numbered variant of the large classical-model experiment script.

**`BIG_all_data_input_all_models_3.py`**  
Numbered variant of the large classical-model experiment script.

**`BIG_all_data_input_all_models_4.py`**  
Numbered variant of the large classical-model experiment script.

**`BIG_all_data_input_all_models_5.py`**  
Numbered variant of the large classical-model experiment script.

**`BIG_all_data_input_all_models_6.py`**  
Numbered variant of the large classical-model experiment script.

**`all_data_input_fine_tuning.py`**  
Main hyperparameter-search script for the scenario-based multimodal experiments. It uses `RandomizedSearchCV` to tune:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

It also defines the six main scenario families used throughout the repository.

**`analize_model_tuning.py`**  
Classical-model tuning script focused on audio embeddings, including hyperparameter search and export of best-parameter dictionaries to text files.

**`fine_tuning_audio_all_db.py`**  
Classical-model tuning script focused on audio embeddings across multiple SNR conditions, including separate analyses for `3 dB`, `10 dB`, `20 dB`, and the combined dataset.

**`.DS_Store`**  
macOS metadata file.

### Files in `Code/5_machine_learning_analysis/little_modification/`

This subfolder contains specialized per-scenario fine-tuning variants.

- **`all_data_input_fine_tuning_1B.py`** – fine-tuning script specialized for Scenario 1B.
- **`all_data_input_fine_tuning_2A.py`** – fine-tuning script specialized for Scenario 2A.
- **`all_data_input_fine_tuning_2B.py`** – fine-tuning script specialized for Scenario 2B.
- **`all_data_input_fine_tuning_3A.py`** – fine-tuning script specialized for Scenario 3A.
- **`all_data_input_fine_tuning_3B.py`** – fine-tuning script specialized for Scenario 3B.
- **`all_data_input_fine_tuning_4A.py`** – fine-tuning script specialized for Scenario 4A.
- **`all_data_input_fine_tuning_4B.py`** – fine-tuning script specialized for Scenario 4B.
- **`all_data_input_fine_tuning_5A.py`** – fine-tuning script specialized for Scenario 5A.
- **`all_data_input_fine_tuning_5B.py`** – fine-tuning script specialized for Scenario 5B.
- **`all_data_input_fine_tuning_6A.py`** – fine-tuning script specialized for Scenario 6A.
- **`all_data_input_fine_tuning_6B.py`** – fine-tuning script specialized for Scenario 6B.

---

## Saved Hyperparameter Outputs

The repository contains text files storing selected parameter configurations for 0 dB tuning experiments.

### Files in `Code/0db_fine_tun/`

- **`decision_tree_random_search.txt`** – stored Decision Tree parameter set for the 0 dB audio-tuning workflow.
- **`gradient_boosting_random_search.txt`** – stored Gradient Boosting parameter set for the 0 dB audio-tuning workflow.
- **`logistic_regression_random_search.txt`** – stored Logistic Regression parameter set for the 0 dB audio-tuning workflow.
- **`random_forest_random_search.txt`** – stored Random Forest parameter set for the 0 dB audio-tuning workflow.

These files act as persistent outputs from earlier hyperparameter-search runs.

---

## Deep Learning Analysis

This section contains PyTorch-based neural-network experiments for the same scenario logic used in the classical pipeline.

### Files in `Code/6_deep_analysis/`

**`BIG_neural_network_hyperparameter_analysis.py`**  
Main large-scale neural-network analysis script. It:
- constructs scenario-specific datasets from the embedding CSV files;
- defines a multilayer PyTorch network with BatchNorm and Dropout;
- trains with Adam and BCEWithLogitsLoss;
- uses early stopping and repeated iterations;
- reports precision, recall, and F1 score across scenarios.

**`analyses.ipynb`**  
Notebook used for additional experiment analysis and/or result inspection.

**`neural_net_model_tuning.ipynb`**  
Notebook for neural-network configuration experiments.

**`neural_network_hyperparameter_analysis.ipynb`**  
Notebook version of the neural-network hyperparameter analysis workflow.

**`neural_network_hyperparameter_analysis.py`**  
Python script version of the neural-network hyperparameter analysis workflow.

**`neural_network_hyperparameter_analysis_6a_6b.ipynb`**  
Notebook focused on scenario group `6A/6B`.

**`neural_network_hyperparameter_analysis_6a_6b.py`**  
Python script focused on scenario group `6A/6B`.

**`neural_network_model_tuning.ipynb`**  
Notebook dedicated to neural-network model tuning.

---

## HPC / SLURM Execution Scripts

The repository includes a large set of `.bash` scripts intended for HPC execution through SLURM. These scripts typically:

- define SLURM resources;
- load `miniconda3`;
- activate a Conda environment;
- launch a specific Python experiment script.

### Environment / Setup Scripts

**`Code/Bash_hpc/create_conda_env.bash`**  
Creates or recreates a Conda environment and installs the main Python packages used by the project.

**`Code/Bash_hpc/conda_hpc_BIG_neural_net.bash`**  
GPU-oriented launcher for the large neural-network analysis.

**`Code/Bash_hpc/conda_hpc_BIG_neural_net_cpu.bash`**  
CPU-oriented launcher for the large neural-network analysis.

**`Code/Bash_hpc/conda_hpc_neural_net_tuning.bash`**  
Launcher for neural-network tuning.

**`Code/Bash_hpc/conda_hpc_neural_net_tuning_6a_6b.bash`**  
Launcher for neural-network tuning focused on scenarios `6A/6B`.

### Generic or Scenario Launchers

- **`Code/Bash_hpc/run_hpc.bash`** – general-purpose HPC run script.
- **`Code/Bash_hpc/run_hpc_1B.bash`** – launcher for `all_data_input_fine_tuning_1B.py`.
- **`Code/Bash_hpc/run_hpc_2A.bash`** – launcher for `all_data_input_fine_tuning_2A.py`.
- **`Code/Bash_hpc/run_hpc_2B.bash`** – launcher for `all_data_input_fine_tuning_2B.py`.
- **`Code/Bash_hpc/run_hpc_3A.bash`** – launcher for `all_data_input_fine_tuning_3A.py`.
- **`Code/Bash_hpc/run_hpc_3B.bash`** – launcher for `all_data_input_fine_tuning_3B.py`.
- **`Code/Bash_hpc/run_hpc_4A.bash`** – launcher for `all_data_input_fine_tuning_4A.py`.
- **`Code/Bash_hpc/run_hpc_4B.bash`** – launcher for `all_data_input_fine_tuning_4B.py`.
- **`Code/Bash_hpc/run_hpc_5A.bash`** – launcher for `all_data_input_fine_tuning_5A.py`.
- **`Code/Bash_hpc/run_hpc_5B.bash`** – launcher for `all_data_input_fine_tuning_5B.py`.
- **`Code/Bash_hpc/run_hpc_6A.bash`** – launcher for `all_data_input_fine_tuning_6A.py`.
- **`Code/Bash_hpc/run_hpc_6B.bash`** – launcher for `all_data_input_fine_tuning_6B.py`.

### Large Aggregate Experiment Launchers

- **`Code/Bash_hpc/run_hpc_BIG_all_models.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_2.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_3.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_4.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_5.bash`**

These scripts launch the numbered large-scale classical-model experiment variants.

### CPU Aggregate Experiment Launchers

- **`Code/Bash_hpc/run_hpc_BIG_all_models_cpu.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_cpu_2.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_cpu_3.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_cpu_4.bash`**
- **`Code/Bash_hpc/run_hpc_BIG_all_models_cpu_5.bash`**

These scripts provide CPU-only launch variants for the large aggregate classical-model experiments.

### Additional Neural-Network Launcher

**`Code/Bash_hpc/run_hpc_neural_net_tuning.bash`**  
HPC launcher for neural-network tuning jobs.

---

## Root-Level Files

- **`.gitattributes`** – Git text-normalization configuration.
- **`.DS_Store`** – macOS metadata file.
- **`Progetto_Auditory_Perception_Finale.pdf`** – final project document stored at the repository root.

---

## External Data Expectations

The code references external datasets and local/HPC file paths rather than bundling all required data directly inside the repository.

Examples visible in the scripts include:

- dataset folders such as `ESC-50-master` and `UrbanSound8K`;
- local CSV inputs such as `combined_dataset.csv`, `augmented_dataset.csv`, and image/audio embedding CSVs;
- HPC absolute paths under `/hpc/home/...`.

For reuse on another machine, file paths and dataset locations should therefore be adapted to the local environment.

---

## Main Dependencies

Based on the imports and execution scripts, the project uses the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `torchvision`
- `torchaudio`
- `transformers`
- `scikit-learn`
- `tqdm`
- `Pillow`
- `nnAudio`
- `librosa`
- `mutagen`
- `soundfile`
- `plotly`
- `skorch` (referenced by the environment setup workflow)

---

## Typical Experimental Flow

A typical execution sequence is:

1. Build the unified metadata and augmented dataset.
2. Generate spectrogram and gammatone images from audio.
3. Extract Wav2Vec2 audio embeddings.
4. Extract ViT image embeddings.
5. Merge metadata where needed to keep audio/image alignment.
6. Run classical machine-learning tuning and scenario analyses.
7. Run neural-network scenario analyses.
8. Submit jobs through the HPC batch scripts when large-scale execution is required.
