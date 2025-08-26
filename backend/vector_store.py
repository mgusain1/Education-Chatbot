import pandas as pd
from openai import OpenAI
import faiss
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pathlib import Path
import time

# Load API key
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

# Load datasets
df = pd.read_csv("C:/Users/gusai/OneDrive/Desktop/Allinonechatbot/data/chatbot_universities_dataset.csv")
scorecard = pd.read_csv("C:/Users/gusai/OneDrive/Desktop/Allinonechatbot/data/Most-Recent-Cohorts-Institution_05192025.csv", low_memory=True)


# Pull just the useful SAT/ACT data using UNITID
sat_act_df = scorecard[["UNITID", "SAT_AVG", "ACTCMMID"]].copy()
sat_act_df.columns = ["UNITID", "sat_avg", "actcmmid"]

# Merge by UNITID to get SAT and ACT without messing with name matching
df = df.merge(sat_act_df, on="UNITID", how="left")

# Format for embedding
def format_row(row):
    sat = row['sat_avg'] if pd.notna(row.get('sat_avg')) else 'N/A'
    act = row['actcmmid'] if pd.notna(row.get('actcmmid')) else 'N/A'
    return (
        f"{row['name']}, located in {row['city']}, {row['state']}, "
        f"is a {'public' if row['control_type'] == 1 else 'private'} university. "
        f"In-state tuition is ${row['tuition_in_state']}, "
        f"out-of-state tuition is ${row['tuition_out_state']}. "
        f"Undergrad programs: {row['offers_undergrad']}, "
        f"Graduate programs: {row['offers_grad']}. "
        f"Website: {row['website']}. "
        f"Avg SAT: {sat}, ACT: {act}."
    )

texts = df.apply(format_row, axis=1).tolist()

# Batch embedding
BATCH_SIZE = 100
vectors = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
    batch = texts[i:i + BATCH_SIZE]
    retries = 3
    success = False
    while not success and retries > 0:
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-ada-002")
            batch_vectors = [r.embedding for r in response.data]
            vectors.extend(batch_vectors)
            success = True
        except Exception as e:
            print(f"[Error] Batch {i}-{i+BATCH_SIZE}: {e}")
            retries -= 1
            time.sleep(5)
    if not success:
        print(f"[Fatal] Failed to process batch {i}-{i+BATCH_SIZE} after retries.")
        break

# Convert and save FAISS
vectors_np = np.array(vectors).astype('float32')
dim = vectors_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors_np)
faiss.write_index(index, "university_index.faiss")
print("Non-null SAT count:", df['sat_avg'].notna().sum())
print("Non-null ACT count:", df['actcmmid'].notna().sum())
print(df[['name', 'sat_avg', 'actcmmid']].dropna().sample(5))
# Save metadata
df.to_csv("university_metadata.csv", index=False, columns=[
    'UNITID', 'name', 'city', 'state', 'control_type', 'website',
    'tuition_in_state', 'tuition_out_state', 'offers_undergrad',
    'offers_grad', 'sat_avg', 'actcmmid'  # <== THIS MUST BE HERE
])


print("✅ Vector store and metadata saved — SAT/ACT loaded using UNITID only, no fuzzy, no freeze.")
