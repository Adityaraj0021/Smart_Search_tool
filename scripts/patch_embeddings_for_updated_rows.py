import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_analytics_vidhya_courses.csv')
EMBED_PATH = os.path.join(BASE_DIR, 'embeddings.npy')

# Titles whose combined_text were updated by the other script
TARGET_TITLES = [
    "Introduction to SQL[Do not Delete]",
    "Essentials of Excel",
    "The Complete Power BI Blueprint",
    "Case Study - Data Analysis using SQL",
    "DHS 2024 Sessions",
]


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(EMBED_PATH):
        raise SystemExit(f"Embeddings not found: {EMBED_PATH}")

    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMBED_PATH)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Find indices to update
    updated_indices = []
    texts_to_encode = []
    for i, title in enumerate(df['Title'].astype(str)):
        if title in TARGET_TITLES:
            updated_indices.append(i)
            texts_to_encode.append(df.at[i, 'combined_text'])

    if not updated_indices:
        print('No matching titles found; nothing to update.')
        return

    print(f'Updating embeddings for indices: {updated_indices}')

    # encode
    new_embs = model.encode(texts_to_encode, show_progress_bar=False)

    # ensure shapes align
    if len(new_embs.shape) == 1:
        new_embs = np.expand_dims(new_embs, 0)

    if embeddings.shape[0] < max(updated_indices) + 1:
        raise SystemExit('Embeddings array smaller than expected; cannot patch')

    for idx, emb in zip(updated_indices, new_embs):
        embeddings[idx] = emb

    np.save(EMBED_PATH, embeddings)
    print(f'Patched {len(updated_indices)} embeddings and saved to {EMBED_PATH}')


if __name__ == '__main__':
    main()
