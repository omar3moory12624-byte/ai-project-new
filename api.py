import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Graduation Project AI API")

# ==============================
# Model (خفيف وسريع)
# ==============================
model = SentenceTransformer("all-mpnet-base-v2")
# ==============================
# Models
# ==============================
class ProjectItem(BaseModel):
    id: str
    description: str

class Idea(BaseModel):
    id: str
    title: str
    description: str
    specialization: List[str]
    tools: List[str]

class CheckRequest(BaseModel):
    problem: str
    projects: List[ProjectItem]

class RecommendRequest(BaseModel):
    student_specializations: List[str]
    ideas: List[Idea]

# ==============================
# Helper Functions
# ==============================
def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def calc_specialization_score(user_specs, idea_specs):
    matches = len(set(user_specs) & set(idea_specs))
    return matches / max(len(user_specs), 1)

# ==============================
# Home
# ==============================
@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}

# ==============================
# 1️⃣ CHECK DUPLICATION
# ==============================
@app.post("/check")
def check_duplication(request: CheckRequest):

    if not request.problem or not request.projects:
        return {"results": []}

    texts = [p.description for p in request.projects]
    texts.append(request.problem)

    embeddings = model.encode(texts)

    project_vecs = embeddings[:-1]
    user_vec = embeddings[-1].reshape(1, -1)

    # normalize
    project_vecs = normalize(project_vecs)
    user_vec = normalize(user_vec)

    similarities = cosine_similarity(user_vec, project_vecs)[0]

    results = []

    for i, score in enumerate(similarities):

        similarity_percent = float(score * 100)

        # تجاهل copy-paste
        if similarity_percent >= 98:
            continue

        if similarity_percent < 20:
            continue

        results.append({
            "id": request.projects[i].id,
            "similarity": round(similarity_percent, 2)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {"results": results}

# ==============================
# 2️⃣ RECOMMENDATION SYSTEM
# ==============================
@app.post("/recommend")
def recommend(request: RecommendRequest):

    if not request.student_specializations or not request.ideas:
        return {"recommendations": []}

    # 🧠 نحول التخصصات لجملة
    user_text = "Projects related to " + ", ".join(request.student_specializations)

    idea_texts = [idea.description for idea in request.ideas]

    texts = idea_texts + [user_text]

    embeddings = model.encode(texts)

    idea_vecs = embeddings[:-1]
    user_vec = embeddings[-1].reshape(1, -1)

    # normalize
    idea_vecs = normalize(idea_vecs)
    user_vec = normalize(user_vec)

    semantic_scores = cosine_similarity(user_vec, idea_vecs)[0]

    results = []

    for i, idea in enumerate(request.ideas):

        semantic_score = semantic_scores[i]

        # specialization match
        spec_score = calc_specialization_score(
            request.student_specializations,
            idea.specialization
        )

        # tools bonus
        tools_score = 0.1 if any(
            tool.lower() in idea.description.lower()
            for tool in idea.tools
        ) else 0

        # final score
        final_score = (
            semantic_score * 0.7 +
            spec_score * 0.25 +
            tools_score * 0.05
        )

        results.append({
            "id": idea.id,
            "score": round(float(final_score * 100), 2)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "recommendations": results[:5]  # top 5
    }