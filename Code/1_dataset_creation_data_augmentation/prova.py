import pandas as pd
import os
import soundfile as sf
from PIL import Image

# Specifica i CSV da analizzare e una descrizione
csv_files = {
    "Combined metadata": "../combined_dataset.csv",
    "Augmented audio": "../augmented_dataset.csv",
    "All audio embeddings": "../all_audio_wav2vec_dataset_complete.csv",
    "Combined images metadata": "../combined_images_dataset.csv",
    "Spectrogram ViT embeddings": "../spectrogram_vit_with_audio.csv"
}

for title, csv_path in csv_files.items():
    print(f"\n=== Dataset: {title} ===")
    if not os.path.isfile(csv_path):
        print(f"File non trovato: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    print(f"• Totale righe: {len(df)}")
    
    # Distribuzione delle classi
    if 'class' in df.columns:
        print("\n  Class distribution:")
        print(df['class'].value_counts())
    
    # Provenienza dei dati (dataset originale)
    if 'dataset' in df.columns:
        print("\n  Source dataset distribution:")
        print(df['dataset'].value_counts())
    
    # Distribuzione dei livelli SNR
    if 'snr' in df.columns:
        print("\n  SNR distribution:")
        print(df['snr'].value_counts().sort_index())
    
    # Verifica file audio/immagine esistenti
    file_col = None
    if 'filepath' in df.columns:
        file_col = 'filepath'
    elif 'audio_filepath' in df.columns:
        file_col = 'audio_filepath'
    
    if file_col:
        exists = df[file_col].apply(os.path.exists)
        missing = len(df) - exists.sum()
        print(f"\n  Missing files: {missing} / {len(df)}")
        
        # Statistiche di durata per gli audio
        if title in ["Combined metadata", "Augmented audio", "All audio embeddings"]:
            durations = []
            for f in df[file_col]:
                if os.path.exists(f):
                    try:
                        info = sf.info(f)
                        durations.append(info.frames / info.samplerate)
                    except:
                        pass
            if durations:
                s = pd.Series(durations)
                print("\n  Audio duration (seconds):")
                print(s.describe())
        
        # Statistiche di dimensioni per le immagini
        if title in ["Combined images metadata", "Spectrogram ViT embeddings"] and 'image_filepath' in df.columns:
            dims = []
            for f in df['image_filepath']:
                if os.path.exists(f):
                    try:
                        img = Image.open(f)
                        dims.append(img.size)
                    except:
                        pass
            if dims:
                df_dims = pd.DataFrame(dims, columns=['width', 'height'])
                print("\n  Image dimensions:")
                print(df_dims.describe())

print("\n✓ Analisi completata.")
