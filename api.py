import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Graduation Project Recommendation API")

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # أخف موديل
    return model

RECOMMEND_THRESHOLD = 0.55
DUPLICATION_THRESHOLD = 0.6
TOP_K_RECOMMEND = 3

class ProjectRequest(BaseModel):
    problem: str
    previousIdeas: list[str]

@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}

@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"error": "Problem text is empty"}

    model = get_model()

    try:
        user_embedding = model.encode(
            [request.problem],
            batch_size=1,
            convert_to_numpy=True
        )
    except Exception as e:
        return {"error": str(e)}

    if not request.previousIdeas:
        return {
            "recommendations": [],
            "status": "accepted",
            "duplicates": []
        }

    try:
        previous_embeddings = model.encode(
            request.previousIdeas[:10],  # limit عشان الرام
            batch_size=1,
            convert_to_numpy=True
        )
    except Exception as e:
        return {"error": str(e)}

    similarities = cosine_similarity(
        user_embedding,
        previous_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:TOP_K_RECOMMEND]

    recommendations = []
    duplicates = []

    for idx in top_indices:
        score = float(similarities[idx])
        idea_text = request.previousIdeas[idx]

        item = {
            "idea": idea_text,
            "similarity_percentage": round(score * 100, 2)
        }

        recommendations.append(item)

        if score >= DUPLICATION_THRESHOLD:
            duplicates.append(item)

    status = "rejected" if duplicates else "accepted"

    return {
        "recommendations": recommendations,
        "status": status,
        "duplicates": duplicates
    }