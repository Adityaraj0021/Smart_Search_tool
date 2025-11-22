from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv

app = FastAPI(title="Analytics Vidhya Smart Search API")

BASE_DIR = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE_DIR, "cleaned_analytics_vidhya_courses.csv")
EMBED_PATH = os.path.join(BASE_DIR, "embeddings.npy")

model = None
embeddings = None
df = None


@app.on_event("startup")
def load_resources():
    global model, embeddings, df
    # load .env into environment first so OPENAI_API_KEY can be set from a file
    load_dotenv()

    # fallback: try a secrets.json file in the project root
    if not os.getenv("OPENAI_API_KEY"):
        secrets_path = os.path.join(BASE_DIR, "secrets.json")
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "OPENAI_API_KEY" in data:
                    os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]
            except Exception:
                # if secrets file is malformed, continue without raising here
                pass
    if not os.path.exists(DATA_CSV):
        raise RuntimeError(f"Missing data file: {DATA_CSV}")
    if not os.path.exists(EMBED_PATH):
        raise RuntimeError(f"Missing embeddings file: {EMBED_PATH}")

    df = pd.read_csv(DATA_CSV)
    embeddings = np.load(EMBED_PATH)
    model = SentenceTransformer('all-MiniLM-L6-v2')


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 500


@app.post("/search")
def search(req: SearchRequest):
    q_emb = model.encode([req.query])[0]
    sims = cosine_similarity([q_emb], embeddings)[0]

    df2 = df.copy()
    df2['similarity'] = sims
    # sort by similarity descending
    df_sorted = df2.sort_values('similarity', ascending=False)

    # Define placeholders to filter out (common values from scraped data)
    def is_placeholder_title(t: str) -> bool:
        if not isinstance(t, str):
            return True
        tl = t.strip().lower()
        return tl == '' or 'not found' in tl or 'title not found' in tl

    def is_placeholder_text(s: str) -> bool:
        if not isinstance(s, str):
            return True
        sl = s.strip().lower()
        return sl == '' or 'not found' in sl or sl.startswith('descript found') or sl.startswith('description not found')

    # First try to collect top_k entries with non-placeholder title and description
    records = []
    for _, row in df_sorted.iterrows():
        title = row.get('Title', '')
        combined = row.get('combined_text', '')
        if not is_placeholder_title(title) and not is_placeholder_text(combined):
            records.append({
                'title': title,
                'combined_text': combined,
                'similarity': float(row['similarity'])
            })
        if len(records) >= req.top_k:
            break

    # If not enough valid records, fill from remaining sorted rows (allow placeholders)
    if len(records) < req.top_k:
        for _, row in df_sorted.iterrows():
            title = row.get('Title', '')
            combined = row.get('combined_text', '')
            rec = {
                'title': title if isinstance(title, str) else '',
                'combined_text': combined if isinstance(combined, str) else '',
                'similarity': float(row['similarity'])
            }
            # avoid duplicates
            if rec in records:
                continue
            records.append(rec)
            if len(records) >= req.top_k:
                break

    return {'results': records}


@app.post("/generate")
def generate(req: GenerateRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set. Provide via environment or secrets.json")

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=0.7,
        )
        return {"generated": resp['choices'][0]['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "info": "Use /search and /generate endpoints"}
