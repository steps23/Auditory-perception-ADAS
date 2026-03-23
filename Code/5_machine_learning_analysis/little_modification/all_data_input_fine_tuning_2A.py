import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# FUNZIONI UTILI

def parse_vector(vec_str):
    """Splitta la stringa su ';' e converte ciascun valore in float."""
    floats = [float(num_str) for num_str in vec_str.split(";")]
    return floats

def model_tuning(model_name: str, X_train, y_train, X_val, y_val):
    """Esegue un RandomizedSearchCV per il modello richiesto e stampa i risultati."""
    if model_name == 'logistic_regression':
        base_model = LogisticRegression()
        params = {
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],  
            'max_iter': [40, 45, 50, 70, 80, 100, 120, 130, 150, 160, 170, 180, 200, 250, 300, 350],
            'class_weight': [None, 'balanced']
        }
        n_iterations = 80
    elif model_name == 'decision_tree':
        base_model = tree.DecisionTreeClassifier()
        params = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': list(range(2, 21)),      
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 0.01, 0.05, 0.1]
        }
        n_iterations = 100
    elif model_name == 'random_forest':
        base_model = RandomForestClassifier()
        params = {
            'n_estimators': [20, 50, 70, 100, 150, 200, 250, 300, 350, 400],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': list(range(2, 31)),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf': list(range(1, 21)),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 0.01, 0.05, 0.1]  # pruning
        }
        n_iterations = 100
    elif model_name == 'gradient_boosting':
        base_model = GradientBoostingClassifier()
        params = {
            'loss': ['log_loss', 'exponential'],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 350, 400, 500],
            'subsample': [0.5, 0.7, 1.0],
            'min_samples_leaf': list(range(1, 21)),
            'max_depth': list(range(2, 11)),
            'max_features': [None, 'sqrt', 'log2']
        }
        n_iterations = 100
    else:
        sys.exit('ERRORE: modello non riconosciuto -> ' + model_name)
    
    model_random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=params,
        n_iter=n_iterations,
        random_state=42,
        n_jobs=-1
    )
    model_random_search.fit(X_train, y_train)

    y_pred = model_random_search.predict(X_val)
    precision = precision_score(y_val, y_pred, average='binary')
    recall = recall_score(y_val, y_pred, average='binary')
    f1 = f1_score(y_val, y_pred, average='binary')
    print(f"\n===== Risultati per {model_name} =====")
    print("Precision:", precision, " Recall:", recall, " F1:", f1)
    print("Migliori parametri:", model_random_search.best_params_)


# LETTURA DATASET

# Carichiamo il dataset audio e modifichiamo la colonna 'class' per ottenere una classificazione binaria
df_audio = pd.read_csv("all_audio_wav2vec_dataset_complete.csv")
# print(df_audio["class"].unique())
df_audio['class'] = df_audio['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)
# -> col 'audio_filepath', 'class', 'snr', 'vector'

df_vit = pd.read_csv("spectrogram_vit_with_audio.csv")
# -> col 'audio_filepath', 'class', 'snr', 'transform_type', 'vector'
df_vit['class'] = df_vit['class'].apply(lambda x: 1 if x in ['ambulance', 'firetruck', 'police', 'siren'] else 0)


# Suddividiamo in due subset: 
df_vit_spec = df_vit[df_vit['transform_type'] == 'spectrogram'].copy()
df_vit_gam  = df_vit[df_vit['transform_type'] == 'gammatone'].copy()


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
    # assumiamo che le colonne 'class_spec' e 'class_gam' coincidano
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
    y = df_merge['class_vit'].values  # o class_audio
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
        vec_audio = parse_vector(row["vector_audio"])
        combined  = vec_spec + vec_gam + vec_audio
        X_list.append(combined)
    X = np.array(X_list)
    y = df_merge['class_spec'].values  # ipotizziamo che coincidano
    return X, y


# FUNZIONE PER CREARE TRAIN/TEST/VAL E LANCIARE I MODELLI

def run_analysis(X, y, scenario_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, stratify=y_train)

    print("\n----------------------------------------")
    print(f"SCENARIO: {scenario_name}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    for modelname in ['logistic_regression','decision_tree','random_forest','gradient_boosting']:
        model_tuning(modelname, X_train, y_train, X_val, y_val)


# MAIN: 6 scenari x 2 situazioni (snr=0 vs snr>0)
def main():
    # 1) Filtriamo i DF per "no noise" (snr=0) e "con noise" (snr>0)
    df_audio_no = df_audio[df_audio['snr'] == 0].copy()
    df_audio_noise = df_audio[df_audio['snr'] > 0].copy()

    df_vit_spec_no = df_vit_spec[df_vit_spec['snr'] == 0].copy()
    df_vit_spec_noise = df_vit_spec[df_vit_spec['snr'] > 0].copy()

    df_vit_gam_no = df_vit_gam[df_vit_gam['snr'] == 0].copy()
    df_vit_gam_noise = df_vit_gam[df_vit_gam['snr'] > 0].copy()

    # ---------------------------
    # SCENARIO 2: SOLO VIT GAMMATONE
    # (A) no noise
    X, y = dataset_solo_vit_gam(df_vit_gam_no)
    run_analysis(X, y, "Scenario 2A: vit gammatone, snr=0")

   

if __name__ == "__main__":
    main()
