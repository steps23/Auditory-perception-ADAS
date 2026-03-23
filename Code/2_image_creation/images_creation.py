import os
import pandas as pd
import torch
import torchaudio
import numpy as np

# Per salvare direttamente le immagini (senza creare figure)
import matplotlib.image as mpimg

# Per mostrare la barra di avanzamento
from tqdm import tqdm

# Se usi la trasformazione Gammatonegram (nnAudio >= 0.3.3)
from nnAudio.features.gammatone import Gammatonegram

INPUT_CSV = "combined_dataset.csv"   # CSV di input con le colonne: filepath, snr, class, ...
OUTPUT_CSV = "combined_images_dataset.csv"     # CSV di output con info sulle immagini generate
IMAGES_BASE_DIR = "images"            # Cartella base in cui salvare tutte le immagini

# Parametri per lo spettrogramma (Torchaudio)
spectrogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=1024,
    hop_length=512,
    power=2.0
)

# Funzioni di trasformazione

def compute_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Calcola lo spettrogramma di un segnale audio [channels, time].
    Restituisce un tensore [freq, time].
    """
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    spec = spectrogram_transform(waveform)  # [1, freq, time]
    spec = spec.squeeze(0)                  # [freq, time]
    return spec

def compute_gammatonegram(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Calcola il gammatonegram di un segnale usando nnAudio (classe Gammatonegram).
    Restituisce un tensore [freq, time].
    """
    gammatone_transform = Gammatonegram(
        sr=sample_rate,
        n_fft=1024,
        hop_length=512,
        center=True,
        pad_mode='reflect',
        power=2.0
    )
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    
    gamma = gammatone_transform(waveform)  # [1, freq, time]
    gamma = gamma.squeeze(0)               # [freq, time]
    return gamma

# Salvataggio di un tensore 2D come immagine PNG (senza figure Matplotlib)

def save_2d_tensor_as_image(tensor_2d: torch.Tensor, out_path: str):
    """
    Salva un tensore 2D [freq, time] come immagine PNG,
    usando direttamente imsave (matplotlib.image.imsave).
    Applichiamo una scala logaritmica (dB) e usiamo cmap='jet'.
    """
    data = tensor_2d.cpu().numpy()
    data = np.where(data <= 1e-10, 1e-10, data)  # evitiamo log(0)
    data_db = 10.0 * np.log10(data)

    # Crea la cartella d'uscita se non esiste
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Salvataggio diretto
    mpimg.imsave(out_path, data_db, cmap='jet', origin='lower')

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # Verifica la presenza di colonne
    if 'filepath' not in df.columns:
        raise ValueError("Nel CSV deve esserci almeno la colonna 'filepath'.")
    
    # Se 'snr' non esiste, creala come 0
    if 'snr' not in df.columns:
        df['snr'] = 0
    
    # Se 'class' non esiste, creala come None
    if 'class' not in df.columns:
        df['class'] = None
    
    rows_for_images_csv = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
        audio_path = row['filepath']
        snr_value = row['snr']
        label = row['class']
        
        if not os.path.isfile(audio_path):
            print(f"[AVVISO] File audio non trovato: {audio_path}")
            continue
        
        # Carica l'audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Determiniamo la sottocartella in base a SNR
        if pd.isna(snr_value) or snr_value == 0:
            snr_folder = "no_noise"
        else:
            snr_folder = f"snr_{snr_value}db"
        
        # Salvataggio Spettrogramma
        spec = compute_spectrogram(waveform)
        
        audio_basename = os.path.basename(audio_path)
        file_no_ext, _ = os.path.splitext(audio_basename)
        
        out_dir_spectrogram = os.path.join(IMAGES_BASE_DIR, snr_folder, "spectrogram")
        spec_filename = f"{file_no_ext}_spec.png"
        spec_path = os.path.join(out_dir_spectrogram, spec_filename)
        
        save_2d_tensor_as_image(spec, spec_path)
        
        rows_for_images_csv.append({
            "audio_filepath": audio_path,
            "image_filepath": spec_path,
            "snr": snr_value,
            "class": label,
            "transform_type": "spectrogram"
        })
        
        # Salvataggio Gammatonegram
        gamma = compute_gammatonegram(waveform, sample_rate)
        
        out_dir_gammatone = os.path.join(IMAGES_BASE_DIR, snr_folder, "gammatone")
        gamma_filename = f"{file_no_ext}_gammatone.png"
        gamma_path = os.path.join(out_dir_gammatone, gamma_filename)
        
        save_2d_tensor_as_image(gamma, gamma_path)
        
        rows_for_images_csv.append({
            "audio_filepath": audio_path,
            "image_filepath": gamma_path,
            "snr": snr_value,
            "class": label,
            "transform_type": "gammatone"
        })

    print("Immagini processate, procedo con creazione file .csv")
        
    # Creiamo il CSV delle immagini
    images_df = pd.DataFrame(rows_for_images_csv)
    images_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Creato il CSV {OUTPUT_CSV} con le informazioni sulle immagini.")

if __name__ == "__main__":
    main()
