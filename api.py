import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Graduation Project Recommendation API")

# ==============================
# Model
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# Request Models
# ==============================
class ProjectItem(BaseModel):
    id: str
    description: str   # 👈 نفس اسم الباك

class ProjectRequest(BaseModel):
    problem: str
    projects: List[ProjectItem]

# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}

@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem or not request.projects:
        return {"results": []}

    # 🔥 النصوص
    texts = [p.description for p in request.projects]
    texts.append(request.problem)

    # 🔥 embeddings
    embeddings = model.encode(texts)

    project_vecs = embeddings[:-1]
    user_vec = embeddings[-1].reshape(1, -1)

    # 🔥 similarity
    similarities = cosine_similarity(user_vec, project_vecs)[0]

    results = []

    for i, score in enumerate(similarities):
        if score < 0.2:  # فلترة بسيطة
            continue

        results.append({
            "id": request.projects[i].id,
            "similarity": round(float(score * 100), 2)
        })

    # ترتيب
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "results": results
    }