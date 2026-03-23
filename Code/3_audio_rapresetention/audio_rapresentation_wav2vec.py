import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Configurazione
INPUT_CSV = "augmented_dataset.csv"      
OUTPUT_CSV = "audio_noises_representation_wav2vec.csv"
MODEL_NAME = "facebook/wav2vec2-base"

def main():
    # 1) Leggiamo il CSV di input
    df = pd.read_csv(INPUT_CSV)

    if "filepath" not in df.columns:
        raise ValueError("Nel CSV deve esserci almeno la colonna 'filepath' con il path del file .wav.")

    if "snr" not in df.columns:
        df["snr"] = 0
    if "class" not in df.columns:
        df["class"] = None

    # 2) Carichiamo modello e feature extractor
    print(f"[INFO] Caricamento del modello {MODEL_NAME}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Disabilitiamo gradient checkpointing per rimuovere il warning
    model.config.gradient_checkpointing = False

    # Forziamo l'uso della CPU
    device = torch.device("cpu")
    model.to(device)

    # 3) Lista per i risultati
    output_rows = []

    # 4) Iteriamo con tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio (CPU)"):
        audio_path = row["filepath"]
        snr_val = row["snr"]
        label_val = row["class"]

        if not os.path.isfile(audio_path):
            print(f"[AVVISO] File non trovato: {audio_path}")
            continue

        # Carichiamo l'audio => waveform shape [channels, time]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Se multi-canale, media sui canali => shape [1, time]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Esempio: waveform.shape = [1, time] (mono)
        # Rimuoviamo la dimensione channels => shape [time]
        waveform = waveform.squeeze(0)  # shape: [time]

        # Resample se necessario
        expected_sample_rate = feature_extractor.sampling_rate
        if sample_rate != expected_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, expected_sample_rate)
            waveform = resampler(waveform)
            sample_rate = expected_sample_rate
        
        # 4c) Passiamo un "batch" di 1 esempio => [waveform]
        # shape => (samples,) per un singolo esempio
        inputs = feature_extractor(
            [waveform.numpy()],  # lista con un solo array 1D
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # spostiamo su CPU
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)  # last_hidden_state shape: [batch=1, seq_len, hidden_dim]
            hidden_states = outputs.last_hidden_state
        
        # Pooling (media) sul tempo => [1, hidden_dim]
        embedding_mean = hidden_states.mean(dim=1)
        embedding_vector = embedding_mean.squeeze(0)  # => [hidden_dim]

        # Convertiamo in lista di float
        vector_list = embedding_vector.cpu().numpy().tolist()
        vector_str = ";".join([f"{x:.6f}" for x in vector_list])

        # Salviamo nel CSV
        output_rows.append({
            "audio_filepath": audio_path,
            "snr": snr_val,
            "class": label_val,
            "vector": vector_str
        })

    # 5) Scriviamo il CSV
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] CSV '{OUTPUT_CSV}' creato con le rappresentazioni audio.")

if __name__ == "__main__":
    main()