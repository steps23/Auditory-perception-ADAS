import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")

# =====================
# FUNZIONI UTILI
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
        vec_vit   = parse_vector(row["vector_vit"])
        vec_audio = parse_vector(row["vector_audio"])
        combined  = vec_vit + vec_audio
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
        # Qui si assume che la colonna 'vector' contenga le feature audio
        vec_audio = parse_vector(row["vector"])
        combined  = vec_spec + vec_gam + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values  
    return X, y

# =====================================================
# PARAMETRI PREDEFINITI PER I MODELLI (per ogni scenario)
# =====================================================

predefined_params = {
    'logistic_regression': {
        "Scenario 5B: vit gammatone + audio, snr>0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 3B: vit spettro + gam, snr>0": {'solver': 'liblinear', 'penalty': 'l1', 'max_iter': 70, 'class_weight': None},
        "Scenario 2B: vit gammatone, snr>0": {'solver': 'liblinear', 'penalty': 'l1', 'max_iter': 80, 'class_weight': None},
        "Scenario 1B: vit spettrogramma, snr>0": {'solver': 'saga', 'penalty': 'l1', 'max_iter': 200, 'class_weight': None},
        "Scenario 4B: vit spettro + audio, snr>0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 6B: vit spettro+gam+audio, snr>0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 1A: vit spettrogramma, snr=0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 2A: vit gammatone, snr=0": {'solver': 'liblinear', 'penalty': 'l1', 'max_iter': 130, 'class_weight': None},
        "Scenario 3A: vit spettro + gam, snr=0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 4A: vit spettro + audio, snr=0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None},
        "Scenario 5A: vit gammatone + audio, snr=0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 130, 'class_weight': None},
        "Scenario 6A: vit spettro+gam+audio, snr=0": {'solver': 'saga', 'penalty': 'l2', 'max_iter': 250, 'class_weight': None}
    },
    'decision_tree': {
        "Scenario 5B: vit gammatone + audio, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 3B: vit spettro + gam, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 2B: vit gammatone, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 1B: vit spettrogramma, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 4B: vit spettro + audio, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 6B: vit spettro+gam+audio, snr>0": {'min_samples_split': 9, 'min_samples_leaf': 11, 'max_features': None, 'max_depth': 10, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 1A: vit spettrogramma, snr=0": {'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': None, 'max_depth': 5, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 2A: vit gammatone, snr=0": {'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': None, 'max_depth': 5, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 3A: vit spettro + gam, snr=0": {'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': None, 'max_depth': 5, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 4A: vit spettro + audio, snr=0": {'min_samples_split': 3, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': 18, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 5A: vit gammatone + audio, snr=0": {'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': None, 'max_depth': 11, 'criterion': 'entropy', 'class_weight': None, 'ccp_alpha': 0.0},
        "Scenario 6A: vit spettro+gam+audio, snr=0": {'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': None, 'max_depth': 5, 'criterion': 'gini', 'class_weight': None, 'ccp_alpha': 0.0}
    },
    'random_forest': {
        "Scenario 5B: vit gammatone + audio, snr>0": {'n_estimators': 200, 'min_samples_split': 16, 'min_samples_leaf': 9, 'max_features': 'sqrt', 'max_depth': 17, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 3B: vit spettro + gam, snr>0": {'n_estimators': 200, 'min_samples_split': 16, 'min_samples_leaf': 9, 'max_features': 'sqrt', 'max_depth': 17, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 2B: vit gammatone, snr>0": {'n_estimators': 70, 'min_samples_split': 19, 'min_samples_leaf': 10, 'max_features': 'log2', 'max_depth': 25, 'criterion': 'entropy', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False},
        "Scenario 1B: vit spettrogramma, snr>0": {'n_estimators': 350, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 'sqrt', 'max_depth': 21, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 4B: vit spettro + audio, snr>0": {'n_estimators': 350, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 'sqrt', 'max_depth': 21, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 6B: vit spettro+gam+audio, snr>0": {'n_estimators': 350, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': 'sqrt', 'max_depth': 21, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 1A: vit spettrogramma, snr=0": {'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'max_depth': 19, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False},
        "Scenario 2A: vit gammatone, snr=0": {'n_estimators': 200, 'min_samples_split': 16, 'min_samples_leaf': 9, 'max_features': 'sqrt', 'max_depth': 17, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': True},
        "Scenario 3A: vit spettro + gam, snr=0": {'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'max_depth': 19, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False},
        "Scenario 4A: vit spettro + audio, snr=0": {'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'max_depth': 19, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False},
        "Scenario 5A: vit gammatone + audio, snr=0": {'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'max_depth': 19, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False},
        "Scenario 6A: vit spettro+gam+audio, snr=0": {'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'max_depth': 19, 'criterion': 'log_loss', 'class_weight': 'balanced', 'ccp_alpha': 0.0, 'bootstrap': False}
    },
    'gradient_boosting': {
        "Scenario 5B: vit gammatone + audio, snr>0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 3B: vit spettro + gam, snr>0": {'subsample': 0.7, 'n_estimators': 500, 'min_samples_leaf': 13, 'max_features': 'log2', 'max_depth': 5, 'loss': 'log_loss', 'learning_rate': 0.3},
        "Scenario 2B: vit gammatone, snr>0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 1B: vit spettrogramma, snr>0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 4B: vit spettro + audio, snr>0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 6B: vit spettro+gam+audio, snr>0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 1A: vit spettrogramma, snr=0": {'subsample': 0.7, 'n_estimators': 500, 'min_samples_leaf': 13, 'max_features': 'log2', 'max_depth': 5, 'loss': 'log_loss', 'learning_rate': 0.3},
        "Scenario 2A: vit gammatone, snr=0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 3A: vit spettro + gam, snr=0": {'subsample': 0.7, 'n_estimators': 500, 'min_samples_leaf': 13, 'max_features': 'log2', 'max_depth': 5, 'loss': 'log_loss', 'learning_rate': 0.3},
        "Scenario 4A: vit spettro + audio, snr=0": {'subsample': 1.0, 'n_estimators': 400, 'min_samples_leaf': 17, 'max_features': None, 'max_depth': 5, 'loss': 'exponential', 'learning_rate': 0.3},
        "Scenario 5A: vit gammatone + audio, snr=0": {'subsample': 0.7, 'n_estimators': 500, 'min_samples_leaf': 13, 'max_features': 'log2', 'max_depth': 5, 'loss': 'log_loss', 'learning_rate': 0.3},
        "Scenario 6A: vit spettro+gam+audio, snr=0": {'subsample': 0.7, 'n_estimators': 500, 'min_samples_leaf': 13, 'max_features': 'log2', 'max_depth': 5, 'loss': 'log_loss', 'learning_rate': 0.3}
    }
}

# --------------------------------------------------
# FUNZIONE PER ESEGUIRE UNA SINGOLA ITERAZIONE (split + training + validazione)
# --------------------------------------------------

def run_iteration_sklearn(X, y, scenario_name, seed):
    # Split con lo stesso seed per garantire comparazione equa all'interno dell'iterazione
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    
    results_iteration = {}
    
    # Logistic Regression
    params_lr = predefined_params['logistic_regression'][scenario_name]
    model_lr = LogisticRegression(**params_lr)
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_val)
    prec_lr = precision_score(y_val, y_pred, average='binary')
    rec_lr = recall_score(y_val, y_pred, average='binary')
    f1_lr = f1_score(y_val, y_pred, average='binary')
    results_iteration['logistic_regression'] = (prec_lr, rec_lr, f1_lr)
    
    # Decision Tree
    params_dt = predefined_params['decision_tree'][scenario_name]
    model_dt = tree.DecisionTreeClassifier(**params_dt)
    model_dt.fit(X_train, y_train)
    y_pred = model_dt.predict(X_val)
    prec_dt = precision_score(y_val, y_pred, average='binary')
    rec_dt = recall_score(y_val, y_pred, average='binary')
    f1_dt = f1_score(y_val, y_pred, average='binary')
    results_iteration['decision_tree'] = (prec_dt, rec_dt, f1_dt)
    
    # Random Forest
    params_rf = predefined_params['random_forest'][scenario_name]
    model_rf = RandomForestClassifier(**params_rf)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    prec_rf = precision_score(y_val, y_pred, average='binary')
    rec_rf = recall_score(y_val, y_pred, average='binary')
    f1_rf = f1_score(y_val, y_pred, average='binary')
    results_iteration['random_forest'] = (prec_rf, rec_rf, f1_rf)
    
    # Gradient Boosting
    params_gb = predefined_params['gradient_boosting'][scenario_name]
    model_gb = GradientBoostingClassifier(**params_gb)
    model_gb.fit(X_train, y_train)
    y_pred = model_gb.predict(X_val)
    prec_gb = precision_score(y_val, y_pred, average='binary')
    rec_gb = recall_score(y_val, y_pred, average='binary')
    f1_gb = f1_score(y_val, y_pred, average='binary')
    results_iteration['gradient_boosting'] = (prec_gb, rec_gb, f1_gb)
    
    return results_iteration

# ========================
# LETTURA DEI DATASET E PREPARAZIONE LABEL
# ========================

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
    num_iterations = 10  # Numero di iterazioni
    results = {}  # Accumula i risultati per ogni scenario e per ogni modello
    
    # Definizione degli scenari: ogni tupla contiene (nome_scenario, funzione_dataset, lista di DataFrame necessari)
    scenarios = [
        ("Scenario 1A: vit spettrogramma, snr=0", dataset_solo_vit_spec, [df_vit_spec[df_vit_spec['snr']==0]]),
        ("Scenario 1B: vit spettrogramma, snr>0", dataset_solo_vit_spec, [df_vit_spec[df_vit_spec['snr']>0]]),
        ("Scenario 2A: vit gammatone, snr=0", dataset_solo_vit_gam, [df_vit_gam[df_vit_gam['snr']==0]]),
        ("Scenario 2B: vit gammatone, snr>0", dataset_solo_vit_gam, [df_vit_gam[df_vit_gam['snr']>0]]),
        ("Scenario 3A: vit spettro + gam, snr=0", dataset_vit_spec_plus_vit_gam, [df_vit_spec[df_vit_spec['snr']==0], df_vit_gam[df_vit_gam['snr']==0]]),
        ("Scenario 3B: vit spettro + gam, snr>0", dataset_vit_spec_plus_vit_gam, [df_vit_spec[df_vit_spec['snr']>0], df_vit_gam[df_vit_gam['snr']>0]]),
        ("Scenario 4A: vit spettro + audio, snr=0", dataset_vit_spec_plus_audio, [df_vit_spec[df_vit_spec['snr']==0], df_audio[df_audio['snr']==0]]),
        ("Scenario 4B: vit spettro + audio, snr>0", dataset_vit_spec_plus_audio, [df_vit_spec[df_vit_spec['snr']>0], df_audio[df_audio['snr']>0]]),
        ("Scenario 5A: vit gammatone + audio, snr=0", dataset_vit_gam_plus_audio, [df_vit_gam[df_vit_gam['snr']==0], df_audio[df_audio['snr']==0]]),
        ("Scenario 5B: vit gammatone + audio, snr>0", dataset_vit_gam_plus_audio, [df_vit_gam[df_vit_gam['snr']>0], df_audio[df_audio['snr']>0]]),
        ("Scenario 6A: vit spettro+gam+audio, snr=0", dataset_spec_plus_gam_plus_audio, [df_vit_spec[df_vit_spec['snr']==0], df_vit_gam[df_vit_gam['snr']==0], df_audio[df_audio['snr']==0]]),
        ("Scenario 6B: vit spettro+gam+audio, snr>0", dataset_spec_plus_gam_plus_audio, [df_vit_spec[df_vit_spec['snr']>0], df_vit_gam[df_vit_gam['snr']>0], df_audio[df_audio['snr']>0]])
    ]
    
    # Inizializzo il dizionario dei risultati
    results = {scenario[0]: {
                    'logistic_regression': {'precision':[], 'recall':[], 'f1':[]},
                    'decision_tree': {'precision':[], 'recall':[], 'f1':[]},
                    'random_forest': {'precision':[], 'recall':[], 'f1':[]},
                    'gradient_boosting': {'precision':[], 'recall':[], 'f1':[]}
                } for scenario in scenarios}
    
    # Ciclo esterno: iterazioni
    for iteration in range(num_iterations):
        print(f"\n========================================")
        print(f"Iterazione {iteration+1}/{num_iterations} (seed={iteration})")
        print(f"========================================")
        # Per ogni scenario, utilizziamo lo stesso seed per lo split
        for scenario in scenarios:
            scenario_name, dataset_func, dfs = scenario
            X, y = dataset_func(*dfs)
            iter_results = run_iteration_sklearn(X, y, scenario_name, seed=iteration)
            for model_name, (prec, rec, f1) in iter_results.items():
                results[scenario_name][model_name]['precision'].append(prec)
                results[scenario_name][model_name]['recall'].append(rec)
                results[scenario_name][model_name]['f1'].append(f1)
            print(f"{scenario_name}:")
            for model_name, metrics in iter_results.items():
                print(f"  {model_name} -> Precision: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}, F1: {metrics[2]:.4f}")
    
    # Calcolo e stampa dei risultati medi per ogni scenario e modello
    print("\n========================================")
    print("Risultati medi per ogni scenario e modello:")
    print("========================================")
    for scenario_name, models in results.items():
        print(scenario_name)
        for model_name, metrics in models.items():
            avg_prec = np.mean(metrics['precision'])
            avg_rec = np.mean(metrics['recall'])
            avg_f1 = np.mean(metrics['f1'])
            print(f"  {model_name} -> Average Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
