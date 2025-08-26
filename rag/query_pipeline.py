import pandas as pd
from openai import OpenAI
import faiss
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pathlib import Path

#FAISS is what searches semantically
#CSV metadata returns real info
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

index = faiss.read_index("university_index.faiss")
df = pd.read_csv("university_metadata.csv")
print(df.tail(5)) 

def embeded_query(query: str):
    response = client.embeddings.create(
        input = query,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

def search_universitites(query: str,top_k=1):
    query_vector = embeded_query(query)
    distances, indices = index.search(query_vector,top_k)
    results = []
    for i in indices[0]:
        university = df.iloc[i]
        results.append({
            "name": str(university["name"]),
            "city": str(university["city"]),
            "state": str(university["state"]),
            "tuition_in_state": int(university["tuition_in_state"]),
            "tuition_out_state": int(university["tuition_out_state"]),
            "control_type": "Public" if int(university["control_type"]) == 1 else "Private",
            "undergrad": bool(university["offers_undergrad"]),
            "grad": bool(university["offers_grad"]),
            "website": str(university["website"])
        })

    return results

def get_admission_requirments(university:str)->str:
    uni_row = df[df['name'].str.lower()==university.lower()]
    sat = uni_row['sat_avg'].values[0] if not uni_row.empty else 'N/A'
    act = uni_row['actcmmid'].values[0] if not uni_row.empty else 'N/A'
    prompt = f"""
    You are an expert in U.S. college admissions. Provide a highly detailed report for admission requirements for both undergraduate and graduate programs at {university}.

    Include:
    - Typical unweighted GPA range
    - SAT and ACT score ranges and percentiles
    - Whether tests are optional or required
    - Number and type of recommendation letters
    - Common essay topics
    - Interview requirements
    - Acceptance rate
    - Application deadlines (if known)

    For reference, this university has an average SAT score of {sat} and ACT score of {act}.

    Then, generate a 5–6 line **ideal applicant profile** based on current standards — include example scores, extracurriculars, achievements, and traits typical of accepted candidates.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful assistant who knows about U.S. college admissions."},
            {"role":"user","content":prompt}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

if __name__=="__main__":
    query = input("Ask me about a university?")
    matches = search_universitites(query)
    print("\nTop matching universities:\n")
    for idx, uni in enumerate(matches, 1):
        print(f"{idx}. {uni['name']} ({uni['city']}, {uni['state']})")
        print(f"   Type: {uni['control_type']}")
        print(f"   In-State Tuition: ${uni['tuition_in_state']}")
        print(f"   Out-of-State Tuition: ${uni['tuition_out_state']}")
        print(f"   Offers Undergrad: {uni['undergrad']}, Grad: {uni['grad']}")
        print(f"   Website: {uni['website']}\n")

