import numpy as np
from config import RECOMMEND_THRESHOLD, TOP_K_RECOMMEND

def recommend(user_embedding, external_embeddings, external_projects):

    # ✅ Normalize (مهم جدًا)
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    external_embeddings = external_embeddings / np.linalg.norm(external_embeddings, axis=1, keepdims=True)

    # ✅ cosine similarity
    similarities = (user_embedding @ external_embeddings.T)[0]

    # ✅ sort top K
    top_indices = np.argsort(similarities)[::-1][:TOP_K_RECOMMEND]

    results = []

    for idx in top_indices:
        score = similarities[idx]

        # optional threshold (مش mandatory)
        if score < RECOMMEND_THRESHOLD:
            continue

        proj = external_projects[idx]

        results.append({
            "project_name": proj["Project Name"],
            "year": proj["Year"],
            "specialization": proj["Specialization"],
            "tools": proj["Tools"],
            "description": proj["Introduction"],
            "similarity": round(float(score * 100), 2)  # 🔥 مهم تعرضه
        })

    return results