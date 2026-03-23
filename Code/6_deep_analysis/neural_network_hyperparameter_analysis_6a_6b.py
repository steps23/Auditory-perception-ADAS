import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# FUNZIONI UTILI

def parse_vector(vec_str):
    """Splitta la stringa su ';' e converte ciascun valore in float."""
    return [float(num_str) for num_str in vec_str.split(";")]

# FUNZIONI PER CREARE I DATASET (VARI SCENARI)

def dataset_solo_vit_spec(df_spec):
    X_list = df_spec['vector'].apply(parse_vector).tolist()
    X = np.array(X_list)
    y = df_spec['class'].values
    return X, y

def dataset_solo_vit_gam(df_gam):
    X_list = df_gam['vector'].apply(parse_vector).tolist()
    X = np.array(X_list)
    y = df_gam['class'].values
    return X, y

def dataset_vit_spec_plus_vit_gam(df_spec, df_gam):
    df_merge = pd.merge(df_spec, df_gam, on='audio_filepath', suffixes=('_spec','_gam'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_spec = parse_vector(row["vector_spec"])
        vec_gam  = parse_vector(row["vector_gam"])
        combined = vec_spec + vec_gam
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values
    return X, y

def dataset_vit_spec_plus_audio(df_spec, df_audio):
    df_merge = pd.merge(df_spec, df_audio, on='audio_filepath', suffixes=('_vit','_audio'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_vit   = parse_vector(row["vector_vit"])
        vec_audio = parse_vector(row["vector_audio"])
        combined  = vec_vit + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_vit'].values  # oppure 'class_audio'
    return X, y

def dataset_vit_gam_plus_audio(df_gam, df_audio):
    df_merge = pd.merge(df_gam, df_audio, on='audio_filepath', suffixes=('_vit','_audio'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_vit   = parse_vector(row["vector_vit"])
        vec_audio = parse_vector(row["vector_audio"])
        combined  = vec_vit + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_vit'].values
    return X, y

def dataset_spec_plus_gam_plus_audio(df_spec, df_gam, df_audio):
    df_spec_gam = pd.merge(df_spec, df_gam, on='audio_filepath', suffixes=('_spec','_gam'))
    df_merge = pd.merge(df_spec_gam, df_audio, on='audio_filepath', suffixes=('_sg','_audio'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_spec  = parse_vector(row["vector_spec"])
        vec_gam   = parse_vector(row["vector_gam"])
        # Usa la colonna 'vector' al posto di 'vector_audio'
        vec_audio = parse_vector(row["vector"])
        combined  = vec_spec + vec_gam + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values  # si assume che le classi coincidano
    return X, y


# MODELLO NEURALE CON PYTORCH

class MyNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=1, neurons=32, dropout_rate=0.0):
        super(MyNetwork, self).__init__()
        layers = []
        # Primo layer con specifica dell'input
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # Layer nascosti
        for i in range(num_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        # Layer di output: 2 neuroni per classificazione binaria
        layers.append(nn.Linear(neurons, 2))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# FUNZIONE PER L'ANALISI DEGLI IPERPARAMETRI DELLA RETE NEURALE CON PYTORCH

def nn_model_tuning(X_train, y_train, X_val, y_val):
    input_dim = X_train.shape[1]
    
    # Imposta il device su "cuda" se disponibile, altrimenti "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizzo del device: {device}")
    
    net = NeuralNetClassifier(
        module=MyNetwork,
        module__input_dim=input_dim,
        optimizer=torch.optim.Adam,
        max_epochs=20,
        batch_size=32,
        lr=0.01,
        device=device,
        verbose=0
    )
    
    param_dist = {
        'module__num_layers': [1, 2, 3],
        'module__neurons': [32, 64, 128],
        'module__dropout_rate': [0.0, 0.2, 0.5],
        'optimizer': [torch.optim.Adam, torch.optim.RMSprop],
        'max_epochs': [20, 30, 50],
        'batch_size': [16, 32, 64],
        'lr': [0.001, 0.01, 0.1]
    }
    
    random_search = RandomizedSearchCV(
        estimator=net,
        param_distributions=param_dist,
        n_iter=50,
        random_state=42,
        n_jobs=-1
    )
    
    # Convertiamo i dati nel formato richiesto da PyTorch (float32 per X, int64 per y)
    random_search.fit(X_train.astype(np.float32), y_train.astype(np.int64))
    
    y_pred = random_search.predict(X_val.astype(np.float32))
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print("\n===== Risultati per Neural Network (PyTorch con GPU) =====")
    print("Precision:", precision, "Recall:", recall, "F1:", f1)
    print("Migliori iperparametri:", random_search.best_params_)

# FUNZIONE PER CREARE TRAIN/TEST/VALIDATION E LANCIARE L'ANALISI

def run_nn_analysis(X, y, scenario_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, stratify=y_train, random_state=42)

    print("\n----------------------------------------")
    print(f"SCENARIO: {scenario_name}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    nn_model_tuning(X_train, y_train, X_val, y_val)

# MAIN: Analisi della rete neurale per ciascuno scenario

def main():
    # LETTURA DATASET
    df_audio = pd.read_csv("all_audio_wav2vec_dataset_complete.csv")
    # Creiamo la classificazione binaria
    df_audio['class'] = df_audio['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)
    
    df_vit = pd.read_csv("spectrogram_vit_with_audio.csv")
    df_vit['class'] = df_vit['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)
    
    # Separiamo i dati per tipo di trasformazione
    df_vit_spec = df_vit[df_vit['transform_type'] == 'spectrogram'].copy()
    df_vit_gam  = df_vit[df_vit['transform_type'] == 'gammatone'].copy()
    
    # Filtriamo per SNR
    df_audio_no = df_audio[df_audio['snr'] == 0].copy()
    df_audio_noise = df_audio[df_audio['snr'] > 0].copy()

    df_vit_spec_no = df_vit_spec[df_vit_spec['snr'] == 0].copy()
    df_vit_spec_noise = df_vit_spec[df_vit_spec['snr'] > 0].copy()

    df_vit_gam_no = df_vit_gam[df_vit_gam['snr'] == 0].copy()
    df_vit_gam_noise = df_vit_gam[df_vit_gam['snr'] > 0].copy()

    # ---------------------------
    # SCENARIO 1: SOLO VIT SPETTROGRAMMA
    # (A) no noise
    X, y = dataset_solo_vit_spec(df_vit_spec_no)
    #run_nn_analysis(X, y, "Scenario 1A: vit spettrogramma, snr=0")

    # (B) con noise
    X, y = dataset_solo_vit_spec(df_vit_spec_noise)
    #run_nn_analysis(X, y, "Scenario 1B: vit spettrogramma, snr>0")

    # ---------------------------
    # SCENARIO 2: SOLO VIT GAMMATONE
    # (A) no noise
    X, y = dataset_solo_vit_gam(df_vit_gam_no)
    #run_nn_analysis(X, y, "Scenario 2A: vit gammatone, snr=0")

    # (B) con noise
    X, y = dataset_solo_vit_gam(df_vit_gam_noise)
    #run_nn_analysis(X, y, "Scenario 2B: vit gammatone, snr>0")

    # ---------------------------
    # SCENARIO 3: VIT SPETTRO + VIT GAMMA
    # (A) no noise
    X, y = dataset_vit_spec_plus_vit_gam(df_vit_spec_no, df_vit_gam_no)
    #run_nn_analysis(X, y, "Scenario 3A: vit spettro + gam, snr=0")

    # (B) con noise
    X, y = dataset_vit_spec_plus_vit_gam(df_vit_spec_noise, df_vit_gam_noise)
    #run_nn_analysis(X, y, "Scenario 3B: vit spettro + gam, snr>0")

    # ---------------------------
    # SCENARIO 4: VIT SPETTRO + AUDIO
    # (A) no noise
    X, y = dataset_vit_spec_plus_audio(df_vit_spec_no, df_audio_no)
    #run_nn_analysis(X, y, "Scenario 4A: vit spettro + audio, snr=0")

    # (B) con noise
    X, y = dataset_vit_spec_plus_audio(df_vit_spec_noise, df_audio_noise)
    #run_nn_analysis(X, y, "Scenario 4B: vit spettro + audio, snr>0")

    # ---------------------------
    # SCENARIO 5: VIT GAMMA + AUDIO
    # (A) no noise
    X, y = dataset_vit_gam_plus_audio(df_vit_gam_no, df_audio_no)
    #run_nn_analysis(X, y, "Scenario 5A: vit gammatone + audio, snr=0")

    # (B) con noise
    X, y = dataset_vit_gam_plus_audio(df_vit_gam_noise, df_audio_noise)
    #run_nn_analysis(X, y, "Scenario 5B: vit gammatone + audio, snr>0")

    # ---------------------------
    # SCENARIO 6: VIT SPETTRO + GAMMA + AUDIO
    # (A) no noise
    X, y = dataset_spec_plus_gam_plus_audio(df_vit_spec_no, df_vit_gam_no, df_audio_no)
    run_nn_analysis(X, y, "Scenario 6A: vit spettro+gam+audio, snr=0")

    # (B) con noise
    X, y = dataset_spec_plus_gam_plus_audio(df_vit_spec_noise, df_vit_gam_noise, df_audio_noise)
    run_nn_analysis(X, y, "Scenario 6B: vit spettro+gam+audio, snr>0")

if __name__ == "__main__":
    main()
