import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# =====================
# FUNZIONI DI UTILITÀ
# =====================

def parse_vector(vec_str):
    """Splitta la stringa su ';' e converte ciascun valore in float."""
    return [float(x) for x in vec_str.split(";")]

# --------------------------------------------------
# FUNZIONI PER LA CREAZIONE DEI DATASET (SCENARI)
# --------------------------------------------------

def dataset_solo_vit_spec(df_spec):
    X = np.array(df_spec['vector'].apply(parse_vector).tolist())
    y = df_spec['class'].values
    return X, y

def dataset_solo_vit_gam(df_gam):
    X = np.array(df_gam['vector'].apply(parse_vector).tolist())
    y = df_gam['class'].values
    return X, y

def dataset_vit_spec_plus_vit_gam(df_spec, df_gam):
    df_merge = pd.merge(df_spec, df_gam, on='audio_filepath', suffixes=('_spec', '_gam'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_spec = parse_vector(row["vector_spec"])
        vec_gam = parse_vector(row["vector_gam"])
        combined = vec_spec + vec_gam
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values
    return X, y

def dataset_vit_spec_plus_audio(df_spec, df_audio):
    df_merge = pd.merge(df_spec, df_audio, on='audio_filepath', suffixes=('_vit', '_audio'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_vit = parse_vector(row["vector_vit"])
        vec_audio = parse_vector(row["vector_audio"])
        combined = vec_vit + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_vit'].values  
    return X, y

def dataset_vit_gam_plus_audio(df_gam, df_audio):
    df_merge = pd.merge(df_gam, df_audio, on='audio_filepath', suffixes=('_vit', '_audio'))
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
    df_spec_gam = pd.merge(df_spec, df_gam, on='audio_filepath', suffixes=('_spec', '_gam'))
    df_merge = pd.merge(df_spec_gam, df_audio, on='audio_filepath', suffixes=('_sg', '_audio'))
    X_list = []
    for idx, row in df_merge.iterrows():
        vec_spec  = parse_vector(row["vector_spec"])
        vec_gam   = parse_vector(row["vector_gam"])
        # Si assume che la colonna 'vector' contenga le features audio
        vec_audio = parse_vector(row["vector"])
        combined  = vec_spec + vec_gam + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values  
    return X, y

# ===============================================
# DEFINIZIONE DEL MODELLO NEURALE CON PYTORCH
# ===============================================

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        # Architettura con BatchNorm e Dropout
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x  # BCEWithLogitsLoss gestisce la sigmoid

# =====================================================
# FUNZIONE DI TRAINING CON EARLY STOPPING E VALUTAZIONE
# =====================================================

def model_tuning_nn(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=32, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    model = NeuralNet(input_size).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Conversione dei dati in tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Ciclo di training con early stopping
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Calcola la loss di validazione
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()
        model.train()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    # Carica i pesi del miglior modello
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# =====================================================
# FUNZIONE PER ESEGUIRE UN'ITERAZIONE SINGOLA (split + training + test)
# =====================================================

def run_iteration(X, y, scenario_name, seed):
    # Utilizziamo il seed per garantire che, in questa iterazione, lo split sia lo stesso per tutti gli scenari
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    
    model = model_tuning_nn(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=32, patience=10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
        y_test_np = y_test_tensor.cpu().numpy().astype(int)
    
    prec = precision_score(y_test_np, predictions, average='binary')
    rec  = recall_score(y_test_np, predictions, average='binary')
    f1   = f1_score(y_test_np, predictions, average='binary')
    return prec, rec, f1

# =======================================
# LETTURA DEI DATASET E PREPARAZIONE LABEL
# =======================================

df_audio = pd.read_csv("../all_audio_wav2vec_dataset_complete.csv")
df_audio['class'] = df_audio['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)

df_vit = pd.read_csv("../spectrogram_vit_with_audio.csv")
df_vit['class'] = df_vit['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)

df_vit_spec = df_vit[df_vit['transform_type'] == 'spectrogram'].copy()
df_vit_gam  = df_vit[df_vit['transform_type'] == 'gammatone'].copy()

# ========================
# FUNZIONE MAIN (ITERAZIONI PER TUTTI GLI SCENARI)
# ========================

def main():
    num_iterations = 100  # Numero di iterazioni (cicli esterni)
    
    # Filtriamo i DataFrame in base a snr
    df_audio_no = df_audio[df_audio['snr'] == 0].copy()
    df_audio_noise = df_audio[df_audio['snr'] > 0].copy()

    df_vit_spec_no = df_vit_spec[df_vit_spec['snr'] == 0].copy()
    df_vit_spec_noise = df_vit_spec[df_vit_spec['snr'] > 0].copy()

    df_vit_gam_no = df_vit_gam[df_vit_gam['snr'] == 0].copy()
    df_vit_gam_noise = df_vit_gam[df_vit_gam['snr'] > 0].copy()
    
    # Definiamo una lista di scenari:
    # Ogni elemento è una tupla (nome_scenario, funzione_dataset, [lista di DataFrame necessari])
    scenarios = [
        ("Scenario 1A: Solo ViT Spettrogramma, snr=0", dataset_solo_vit_spec, [df_vit_spec_no]),
        ("Scenario 1B: Solo ViT Spettrogramma, snr>0", dataset_solo_vit_spec, [df_vit_spec_noise]),
        ("Scenario 2A: Solo ViT Gammatone, snr=0", dataset_solo_vit_gam, [df_vit_gam_no]),
        ("Scenario 2B: Solo ViT Gammatone, snr>0", dataset_solo_vit_gam, [df_vit_gam_noise]),
        ("Scenario 3A: ViT Spettrogramma + Gammatone, snr=0", dataset_vit_spec_plus_vit_gam, [df_vit_spec_no, df_vit_gam_no]),
        ("Scenario 3B: ViT Spettrogramma + Gammatone, snr>0", dataset_vit_spec_plus_vit_gam, [df_vit_spec_noise, df_vit_gam_noise]),
        ("Scenario 4A: ViT Spettrogramma + Audio, snr=0", dataset_vit_spec_plus_audio, [df_vit_spec_no, df_audio_no]),
        ("Scenario 4B: ViT Spettrogramma + Audio, snr>0", dataset_vit_spec_plus_audio, [df_vit_spec_noise, df_audio_noise]),
        ("Scenario 5A: ViT Gammatone + Audio, snr=0", dataset_vit_gam_plus_audio, [df_vit_gam_no, df_audio_no]),
        ("Scenario 5B: ViT Gammatone + Audio, snr>0", dataset_vit_gam_plus_audio, [df_vit_gam_noise, df_audio_noise]),
        ("Scenario 6A: Spettrogramma + Gammatone + Audio, snr=0", dataset_spec_plus_gam_plus_audio, [df_vit_spec_no, df_vit_gam_no, df_audio_no]),
        ("Scenario 6B: Spettrogramma + Gammatone + Audio, snr>0", dataset_spec_plus_gam_plus_audio, [df_vit_spec_noise, df_vit_gam_noise, df_audio_noise])
    ]
    
    # Struttura per accumulare i risultati: per ogni scenario, per ogni iterazione
    results = {scenario[0]: {"precision": [], "recall": [], "f1": []} for scenario in scenarios}
    
    # Ciclo esterno: iterazioni
    for iteration in range(num_iterations):
        print(f"\n========================================")
        print(f"Iterazione {iteration+1}/{num_iterations} (seed={iteration})")
        print(f"========================================")
        # Per ogni scenario, usiamo lo stesso seed (l'iterazione corrente)
        for scenario in scenarios:
            scenario_name, dataset_func, dfs = scenario
            X, y = dataset_func(*dfs)
            prec, rec, f1 = run_iteration(X, y, scenario_name, seed=iteration)
            results[scenario_name]["precision"].append(prec)
            results[scenario_name]["recall"].append(rec)
            results[scenario_name]["f1"].append(f1)
            print(f"{scenario_name} -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Calcola le medie per ogni scenario
    print("\n========================================")
    print("Risultati medi per ogni scenario:")
    print("========================================")
    for scenario_name, metrics in results.items():
        avg_prec = np.mean(metrics["precision"])
        avg_rec = np.mean(metrics["recall"])
        avg_f1 = np.mean(metrics["f1"])
        print(f"{scenario_name} -> Average Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
