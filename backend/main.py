from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
import faiss
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pathlib import Path
from rag.query_pipeline import embeded_query, search_universitites, get_admission_requirments
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

index = faiss.read_index("university_index.faiss")
df = pd.read_csv("university_metadata.csv")

class QueryInput(BaseModel):
    query: str
    
class UniversityInput(BaseModel):
    university:str
    
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
def ask_question(input: QueryInput):
    query = input.query
    results = search_universitites(query,top_k=1)
    return {"matches":results}

@app.post("/admission-requirements")
def admission_requirement(input: UniversityInput):
    results = get_admission_requirments(input.university)
    return {"requirements": results}
