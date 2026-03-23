import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Per ViT
from transformers import (
    ViTModel,
    ViTImageProcessor  # o ViTFeatureExtractor se hai una versione più vecchia di transformers
)

INPUT_CSV = "all_images_dataset_complete.csv" # CSV con colonna "image_filepath", "snr", "class", "transform_type"
OUTPUT_CSV = "spect_gamma_vit_dataset.csv"    # CSV finale con embedding e metadati
MODEL_NAME = "MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection"

def main():
    # Leggiamo il CSV
    df = pd.read_csv(INPUT_CSV)
    
    # Verifichiamo che ci sia la colonna con il percorso delle immagini
    if "image_filepath" not in df.columns:
        raise ValueError("Nel CSV deve essere presente la colonna 'image_filepath' con il path delle immagini.")

    # Se 'snr', 'class', o 'transform_type' non esistono, le creiamo come None
    if "snr" not in df.columns:
        df["snr"] = 0
    if "class" not in df.columns:
        df["class"] = None
    if "transform_type" not in df.columns:
        df["transform_type"] = None
    
    # Carichiamo il modello ViT e il processor
    print(f"[INFO] Caricamento del modello {MODEL_NAME} ...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)  
    
    model = ViTModel.from_pretrained(MODEL_NAME)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Lista per salvare i risultati
    output_rows = []

    # Iteriamo sulle righe del CSV, usando tqdm per mostrare l'avanzamento
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing spectrograms with ViT"):
        audio_path= row["audio_filepath"]
        img_path = row["image_filepath"]
        snr_val = row["snr"]
        label_val = row["class"]
        transform_type_val = row["transform_type"]

        if not os.path.isfile(img_path):
            print(f"[AVVISO] Immagine non trovata: {img_path}")
            continue

        # Carichiamo l'immagine con PIL
        # Convertiamola in RGB nel caso fosse grayscale
        image = Image.open(img_path).convert("RGB")

        # Pre-processamento immagine con il processor di ViT (resize, crop, normalizzazione...)
        inputs = processor(images=image, return_tensors="pt")
        
        # Spostiamo su CPU
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Passiamo i dati al modello
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state.shape = [batch_size=1, seq_len, hidden_dim]
            # outputs.pooler_output.shape     = [1, hidden_dim] se disponibile

        # Estraiamo un vettore di embedding
        # prendiamo il token [CLS], che è l'indice 0 lungo seq_len
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        embedding_vector = cls_embedding.squeeze(0)        
        
        # Convertiamo in lista di float
        vector_list = embedding_vector.cpu().numpy().tolist()
        vector_str = ";".join([f"{x:.6f}" for x in vector_list])

        # CSV finale
        output_rows.append({
            "audio_filepath": audio_path,
            "image_filepath": img_path,
            "snr": snr_val,
            "class": label_val,
            "transform_type": transform_type_val,
            "vector": vector_str
        })

    # Creiamo un DataFrame e salviamo
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Creato il CSV '{OUTPUT_CSV}' con le embedding ViT dei tuoi spettrogrammi.")

if __name__ == "__main__":
    main()
