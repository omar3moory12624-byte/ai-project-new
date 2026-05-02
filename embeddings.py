from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # أخف موديل

def create_embeddings(projects):
    texts = [
        f"{p['Project Name']} {p['Introduction']}"
        for p in projects
    ]

    return model.encode(
        texts,
        batch_size=1,
        convert_to_numpy=True
    )

def get_similarity(vector, matrix):
    return cosine_similarity(vector, matrix)[0]