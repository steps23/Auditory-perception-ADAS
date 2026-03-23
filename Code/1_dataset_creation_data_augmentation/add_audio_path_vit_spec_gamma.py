import pandas as pd

def main():
    # Nomi dei file CSV
    VIT_CSV = "spectrogram_vit_dataset.csv"                  # CSV che contiene l'embedding ViT
    IMAGES_CSV = "all_images_dataset_complete.csv"    # CSV originale con image_filepath + audio_filepath
    OUTPUT_CSV = "spectrogram_vit_with_audio.csv"     # Nome del CSV di output

    # 1) Leggiamo i dataset
    df_vit = pd.read_csv(VIT_CSV)
    df_img = pd.read_csv(IMAGES_CSV)

    # 2) Eseguiamo il merge per aggiungere la colonna audio_filepath
    #    come chiave di unione usiamo "image_filepath" (presente in entrambi).
    #    Usiamo how='left' per mantenere tutte le righe di spectrogram_vit.csv,
    #    aggiungendo la colonna audio_filepath (se esiste corrispondenza in df_img).
    df_merged = pd.merge(
        df_vit,
        df_img[["image_filepath", "audio_filepath"]],
        on="image_filepath",
        how="left"
    )

    # 3) Salviamo il nuovo CSV, ora con la colonna "audio_filepath"
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Creato il file {OUTPUT_CSV} con la colonna 'audio_filepath' aggiunta.")

if __name__ == "__main__":
    main()
